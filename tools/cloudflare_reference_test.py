"""
Cloudflare Reference-Image Test (stand-alone test tool, not part of the pipeline)

Tests whether FLUX.2 klein 4B on Cloudflare Workers AI can keep ONE character
face across different views, by giving it a reference picture of the character.
It also checks which sizes and a seed the service accepts.

For each view it sends the same reference picture plus a prompt that describes
the view, saves the result and reports speed and size. The token is never
printed. Reads CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN from the
environment or the .env file in the current folder.

Usage (project folder, virtual environment active):

    python tools/cloudflare_reference_test.py --reference PATH_TO_PICTURE [--face-box 0.40,0.05,0.66,0.35]
    python tools/cloudflare_reference_test.py --scene "..." --ref silas=PATH --ref place=PATH [--count 3]

Results go to output/reference_test/ (one picture per view plus a contact sheet).
"""

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

import requests
from PIL import Image

from cloudflare_image_test import ApiError
from cloudflare_image_test import get_credentials
from cloudflare_image_test import save_bytes_atomic

MODEL = "@cf/black-forest-labs/flux-2-klein-4b"
API_BASE = "https://api.cloudflare.com/client/v4"

# The service says reference pictures must be smaller than 512x512.
MAX_REFERENCE_SIDE = 448

CHARACTER = (
    "Image 0 shows the character Silas: an old lighthouse keeper with swept-back grey hair, "
    "a deeply lined weathered face with heavy furrowed brows and stern eyes, a long wavy grey beard "
    "reaching his chest, a dark charcoal-navy ribbed wool sweater (not pure black), olive-green trousers "
    "and dark brown lace-up boots. Keep exactly this face, hair, beard, body and clothes, with exactly the "
    "same colours. Do not copy the background or any objects from image 0."
)

FACE_NOTE = " Image 1 is a close-up of his face: keep exactly this face."

STYLE = "Anime film illustration, painterly, clean smooth lines, same art style as image 0."

# Fixed lighting and background, so the weather and time of day can never change between views.
SHEET = (
    "Character reference sheet, exactly one single figure, full body visible, plain neutral dark-grey "
    "studio background, soft even lighting, no scenery, no props, no text, no letters, no name label, "
    "no signature, no extra drawings."
)

VIEWS = {
    "front": "Front view, standing straight facing the camera, arms relaxed, neutral serious expression.",
    "three_quarter": "Three-quarter view, body turned about 45 degrees, looking toward the camera, neutral serious expression.",
    "side": "Side profile view, facing left, neutral serious expression.",
    "back": (
        "Back view: one single figure seen entirely from behind, hair and shoulders visible, face not visible, "
        "wearing the same dark charcoal-navy sweater and olive-green trousers."
    ),
    "happy": "Front view, a warm friendly smile, kind eyes, relaxed shoulders.",
    "thinking": (
        "Three-quarter view, his right hand raised to his chin, his left arm hanging relaxed at his side, "
        "exactly two arms and two hands, looking thoughtfully upward, thinking expression."
    ),
}

# Scene mode (a whole picture, not a character sheet).
SCENE_STYLE = "Vivid colourful anime film illustration, painterly, clean smooth lines, rich saturated colours, no text, no watermark."
SILAS_REF = (
    "Image {i} shows the character Silas: keep exactly his face, hair, beard, body, charcoal-navy sweater "
    "and olive-green trousers. Do not copy its plain background."
)
PLACE_REF = "Image {i} shows the place to draw: keep the same buildings, shapes and colours."

RETRYABLE_STATUS = (429, 500, 502, 503, 504)


def prepare_reference(path: Path) -> bytes:
    """Load a picture and shrink it so both sides are at most MAX_REFERENCE_SIDE (PNG bytes)."""
    try:
        picture = Image.open(path).convert("RGB")
    except (OSError, ValueError) as exc:
        raise ApiError(f"Cannot read the reference picture: {path}") from exc

    picture.thumbnail((MAX_REFERENCE_SIDE, MAX_REFERENCE_SIDE))
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG")
    return buffer.getvalue()


def build_prompt(view: str, with_face: bool = False) -> str:
    return f"{STYLE} {SHEET} {CHARACTER}{FACE_NOTE if with_face else ''} {VIEWS[view]}"


def prepare_face(path: Path, box: tuple) -> bytes:
    """Cut the face out of the reference (box = left, top, right, bottom as fractions 0-1) as PNG bytes."""
    left, top, right, bottom = box
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise ApiError("--face-box must be four fractions like 0.40,0.05,0.66,0.35 (left,top,right,bottom).")

    try:
        picture = Image.open(path).convert("RGB")
    except (OSError, ValueError) as exc:
        raise ApiError(f"Cannot read the reference picture: {path}") from exc

    width, height = picture.size
    face = picture.crop((round(left * width), round(top * height), round(right * width), round(bottom * height)))
    face.thumbnail((MAX_REFERENCE_SIDE, MAX_REFERENCE_SIDE))
    buffer = io.BytesIO()
    face.save(buffer, format="PNG")
    return buffer.getvalue()


def build_fields(prompt: str, width: int, height: int, seed) -> dict:
    fields = {"prompt": prompt, "width": str(width), "height": str(height)}
    if seed is not None:
        fields["seed"] = str(seed)
    return fields


def extract_image(response) -> bytes:
    """Image bytes from a raw image reply or a JSON reply with base64."""
    if response.headers.get("Content-Type", "").lower().startswith("image/"):
        return response.content

    try:
        data = json.loads(response.content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ApiError("The reply was neither an image nor JSON.") from exc

    if isinstance(data, dict) and data.get("success") is False:
        messages = "; ".join(str(item.get("message", item)) for item in data.get("errors") or [] if item)
        raise ApiError(f"Cloudflare reported an error: {messages or 'unknown'}")

    encoded = None
    if isinstance(data, dict):
        result = data.get("result")
        encoded = result.get("image") if isinstance(result, dict) else None
        encoded = encoded or data.get("image")

    if not encoded or not isinstance(encoded, str):
        raise ApiError("The reply contained no image field.")

    return base64.b64decode(encoded)


def request_picture(url, token, fields, references, post=requests.post, sleep=time.sleep) -> bytes:
    """Send one multipart request (retry only temporary failures, 3 tries)."""
    for attempt in range(1, 4):
        try:
            response = post(
                url,
                headers={"Authorization": f"Bearer {token}"},
                data=fields,
                files={
                    f"input_image_{index}": (f"reference_{index}.png", data, "image/png")
                    for index, data in enumerate(references)
                },
                timeout=180,
            )
        except requests.RequestException as exc:
            if attempt < 3:
                sleep(5 * attempt)
                continue
            raise ApiError(f"Network problem: {type(exc).__name__}") from exc

        if response.status_code < 400:
            return extract_image(response)

        if response.status_code in RETRYABLE_STATUS and attempt < 3:
            sleep(5 * attempt)
            continue

        raise ApiError(f"HTTP {response.status_code}: {response.text[:300]}")

    raise ApiError("The request failed.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Test character consistency with a reference picture.")
    parser.add_argument("--reference", default=None, help="Picture of the character (the face to keep).")
    parser.add_argument("--count", type=int, default=1, help="Pictures per view (or per scene); the seed goes up by 1 each.")
    parser.add_argument("--scene", default=None, help="Scene mode: describe a whole picture (needs --ref).")
    parser.add_argument(
        "--ref",
        action="append",
        default=None,
        help="Scene mode reference, silas=PATH or place=PATH (repeat up to 4 times).",
    )
    parser.add_argument("--views", default=",".join(VIEWS), help="Comma-separated: " + ",".join(VIEWS) + ".")
    parser.add_argument("--width", type=int, default=None, help="Default 1024 (sheet) or 1344 (scene).")
    parser.add_argument("--height", type=int, default=None, help="Default 1024 (sheet) or 768 (scene).")
    parser.add_argument(
        "--face-box",
        default=None,
        help="Optional face area of the reference as fractions left,top,right,bottom (sent as a second reference).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional seed (to test that it is accepted).")
    parser.add_argument("--output", default="output/reference_test")
    return parser.parse_args(argv)


def check_size(width: int, height: int) -> None:
    if not 256 <= width <= 1920 or not 256 <= height <= 1920:
        raise ApiError("Width and height must be between 256 and 1920.")


def check_count(count: int) -> None:
    if not 1 <= count <= 6:
        raise ApiError("--count must be between 1 and 6.")


def run(args, environ=None, env_file: Path = Path(".env"), post=requests.post, sleep=time.sleep) -> list:
    check_count(args.count)

    if args.scene:
        return run_scene(args, environ, env_file, post, sleep)

    if not args.reference:
        raise ApiError("Give --reference PATH (a picture of the character), or use --scene with --ref.")

    width = args.width or 1024
    height = args.height or 1024
    views = [item.strip() for item in args.views.split(",") if item.strip()]
    unknown = [item for item in views if item not in VIEWS]
    if unknown or not views:
        raise ApiError(f"Unknown view(s): {', '.join(unknown) or '(none)'}. Use: {', '.join(VIEWS)}.")

    check_size(width, height)

    account, token = get_credentials(environ, env_file)
    url = f"{API_BASE}/accounts/{account}/ai/run/{MODEL}"
    references = [prepare_reference(Path(args.reference))]

    if args.face_box:
        try:
            box = tuple(float(item) for item in args.face_box.split(","))
        except ValueError:
            box = ()
        if len(box) != 4:
            raise ApiError("--face-box must be four fractions like 0.40,0.05,0.66,0.35 (left,top,right,bottom).")
        references.append(prepare_face(Path(args.reference), box))

    out = Path(args.output)

    return generate_all(
        url, token, [(view, build_prompt(view, len(references) > 1)) for view in views],
        references, width, height, args, out, post, sleep,
    )


def generate_all(url, token, jobs, references, width, height, args, out, post, sleep) -> list:
    """Make args.count pictures for every (name, prompt) job, save them and a contact sheet."""
    results = []
    names = []

    for name, prompt in jobs:
        for number in range(args.count):
            seed = None if args.seed is None else args.seed + number
            label = name if args.count == 1 else f"{name}_{number + 1}"
            begin = time.monotonic()
            data = request_picture(url, token, build_fields(prompt, width, height, seed), references, post, sleep)
            seconds = time.monotonic() - begin

            picture = Image.open(io.BytesIO(data))
            picture.load()
            path = out / f"{label}.png"
            save_bytes_atomic(data, path)

            print(f"{label:>15}: {picture.width}x{picture.height}, {seconds:.1f} s -> {path}")
            results.append({"view": label, "size": [picture.width, picture.height], "seconds": round(seconds, 1)})
            names.append(label)

    sheet_path = make_sheet(out, names)
    print(f"Contact sheet: {sheet_path}")
    return results


def run_scene(args, environ, env_file: Path, post, sleep) -> list:
    """Make a whole picture from a scene description and up to 4 reference pictures."""
    items = args.ref or []

    if not items:
        raise ApiError("Scene mode needs at least one --ref silas=PATH or --ref place=PATH.")

    if len(items) > 4:
        raise ApiError("At most 4 reference pictures can be sent.")

    kinds = []
    paths = []

    for item in items:
        kind, _, file_name = item.partition("=")
        if kind not in ("silas", "place") or not file_name:
            raise ApiError("--ref must look like silas=PATH or place=PATH.")
        kinds.append(kind)
        paths.append(Path(file_name))

    width = args.width or 1344
    height = args.height or 768
    check_size(width, height)

    account, token = get_credentials(environ, env_file)
    url = f"{API_BASE}/accounts/{account}/ai/run/{MODEL}"
    references = [prepare_reference(file_path) for file_path in paths]

    notes = " ".join(
        (SILAS_REF if kind == "silas" else PLACE_REF).format(i=index) for index, kind in enumerate(kinds)
    )
    prompt = f"{SCENE_STYLE} {notes} {args.scene}"

    return generate_all(url, token, [("scene", prompt)], references, width, height, args, Path(args.output), post, sleep)


def make_sheet(out: Path, views: list) -> Path:
    """One picture with the reference views side by side (2 columns)."""
    pictures = [(view, Image.open(out / f"{view}.png").convert("RGB")) for view in views]
    thumb_width = 640
    cells = [(name, p.resize((thumb_width, round(thumb_width * p.height / p.width)))) for name, p in pictures]
    row_height = max(c.height for _, c in cells)
    rows = (len(cells) + 1) // 2
    sheet = Image.new("RGB", (2 * thumb_width + 18, rows * (row_height + 6) + 6), (30, 30, 30))

    for index, (_, cell) in enumerate(cells):
        sheet.paste(cell, (6 + (index % 2) * (thumb_width + 6), 6 + (index // 2) * (row_height + 6)))

    path = out / "sheet.png"
    buffer = io.BytesIO()
    sheet.save(buffer, format="PNG")
    save_bytes_atomic(buffer.getvalue(), path)
    return path


def main(argv=None) -> int:
    try:
        run(parse_args(argv))
    except ApiError as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())