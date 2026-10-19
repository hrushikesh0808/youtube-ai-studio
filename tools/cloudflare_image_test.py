"""
Cloudflare Image Test (stand-alone measurement tool, not part of the pipeline)

Generates test images with FLUX.1 schnell on Cloudflare Workers AI (free tier:
10,000 neurons per day) and reports speed, the real image size and whether the
picture is blank, so it can be compared with the local GPU test and with a
reference picture.

Credentials are read from the environment or from the .env file in the
current folder (never printed, never written anywhere):

    CLOUDFLARE_ACCOUNT_ID=...
    CLOUDFLARE_API_TOKEN=...

Usage (from the project folder, virtual environment active):

    python tools/cloudflare_image_test.py

Pictures are saved to output/benchmarks/ and results are appended to
output/benchmarks/results.jsonl.
"""

import argparse
import base64
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from image_speed_test import append_result, looks_blank

DEFAULT_MODEL = "@cf/black-forest-labs/flux-1-schnell"
DEFAULT_API_BASE = "https://api.cloudflare.com/client/v4"

DEFAULT_PROMPT = (
    "Anime film illustration, hand-painted background art style, a tall white "
    "lighthouse with a red lantern room and a glowing lamp on a rocky green island, "
    "dark stormy navy sky with heavy rain, crashing waves, clean linework, rich "
    "detailed rocks and moss, moody blue colour palette, cinematic wide shot, "
    "no people, no text"
)

MAX_PROMPT_LENGTH = 2048
MAX_STEPS = 8
REQUEST_TIMEOUT_SECONDS = 120

# Only temporary problems are retried; a wrong token or bad request is not.
RETRYABLE_STATUS = (429, 500, 502, 503, 504)
MAX_ATTEMPTS = 3
RETRY_WAIT_SECONDS = 5

# Cloudflare says 10,000 neurons per day are free.
FREE_DAILY_NEURONS = 10_000


class ApiError(Exception):
    """A failed request, with a message that is safe to show."""


def load_env_file(path: Path) -> dict:
    """Read KEY=VALUE lines from a .env file (comments and blanks ignored)."""
    values = {}
    if not path.is_file():
        return values

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")

    return values


def get_credentials(environ=None, env_file: Path = Path(".env")) -> tuple:
    """Return (account_id, token) from the environment, then from .env."""
    environ = os.environ if environ is None else environ
    file_values = load_env_file(env_file)

    account = environ.get("CLOUDFLARE_ACCOUNT_ID") or file_values.get("CLOUDFLARE_ACCOUNT_ID", "")
    token = environ.get("CLOUDFLARE_API_TOKEN") or file_values.get("CLOUDFLARE_API_TOKEN", "")

    missing = [
        name
        for name, value in (("CLOUDFLARE_ACCOUNT_ID", account), ("CLOUDFLARE_API_TOKEN", token))
        if not value
    ]
    if missing:
        raise ApiError("Missing in environment or .env: " + ", ".join(missing))

    return account, token


def build_url(account: str, model: str, api_base: str) -> str:
    return f"{api_base.rstrip('/')}/accounts/{account}/ai/run/{model}"


def extract_image_bytes(body: bytes, content_type: str) -> bytes:
    """Return image bytes from either a raw image or a JSON reply with base64."""
    if content_type.startswith("image/"):
        return body

    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ApiError("The reply was neither an image nor JSON.") from exc

    if isinstance(data, dict) and data.get("success") is False:
        errors = data.get("errors") or []
        messages = "; ".join(str(item.get("message", item)) for item in errors if item)
        raise ApiError(f"Cloudflare reported an error: {messages or 'unknown'}")

    encoded = None
    if isinstance(data, dict):
        result = data.get("result")
        if isinstance(result, dict):
            encoded = result.get("image")
        if encoded is None:
            encoded = data.get("image")

    if not encoded or not isinstance(encoded, str):
        raise ApiError("The reply contained no image field.")

    try:
        return base64.b64decode(encoded, validate=False)
    except ValueError as exc:
        raise ApiError("The image data could not be decoded.") from exc


def request_image(url: str, token: str, payload: dict, opener=urllib.request.urlopen, sleep=time.sleep) -> bytes:
    """POST one request; retry only temporary failures, a limited number of times."""
    body = json.dumps(payload).encode("utf-8")

    for attempt in range(1, MAX_ATTEMPTS + 1):
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )

        try:
            with opener(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                content_type = response.headers.get("Content-Type", "")
                return extract_image_bytes(response.read(), content_type)

        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]

            if exc.code in RETRYABLE_STATUS and attempt < MAX_ATTEMPTS:
                wait = RETRY_WAIT_SECONDS * attempt
                print(f"  HTTP {exc.code}, retrying in {wait} s (attempt {attempt}/{MAX_ATTEMPTS})")
                sleep(wait)
                continue

            hints = {
                401: "The API token is wrong or expired.",
                403: "The token lacks Workers AI permission.",
                404: "Account ID or model name is wrong.",
                429: "Daily free allowance may be used up; it resets at 00:00 UTC.",
            }
            raise ApiError(f"HTTP {exc.code}. {hints.get(exc.code, '')} {detail}".strip()) from exc

        except urllib.error.URLError as exc:
            if attempt < MAX_ATTEMPTS:
                wait = RETRY_WAIT_SECONDS * attempt
                print(f"  Network problem ({exc.reason}), retrying in {wait} s")
                sleep(wait)
                continue
            raise ApiError(f"Network problem: {exc.reason}") from exc

    raise ApiError("Request failed.")  # not reachable, keeps the function total


def describe_image(data: bytes) -> tuple:
    """Return (width, height, blank, extension) for the picture bytes."""
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        image.load()
        extension = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}.get(image.format, "img")
        return image.width, image.height, looks_blank(image), extension


def save_bytes_atomic(data: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Test image generation on Cloudflare Workers AI.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--steps", type=int, default=4, help=f"1 to {MAX_STEPS} (default 4).")
    parser.add_argument("--seed", type=int, default=None, help="Sent only if given (Cloudflare rejected it in testing).")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--width", type=int, default=None, help="Sent only if given (not documented by Cloudflare).")
    parser.add_argument("--height", type=int, default=None, help="Sent only if given (not documented by Cloudflare).")
    parser.add_argument("--outdir", default="output/benchmarks")
    return parser.parse_args(argv)


def validate_args(args) -> None:
    if not args.prompt.strip() or len(args.prompt) > MAX_PROMPT_LENGTH:
        raise ValueError(f"Prompt must be 1 to {MAX_PROMPT_LENGTH} characters.")
    if args.steps < 1 or args.steps > MAX_STEPS:
        raise ValueError(f"Steps must be between 1 and {MAX_STEPS}.")
    if args.runs < 1 or args.runs > 10:
        raise ValueError("Runs must be between 1 and 10.")
    if (args.width is None) != (args.height is None):
        raise ValueError("Give both --width and --height, or neither.")


def run(args, environ=None, env_file: Path = Path(".env"), opener=urllib.request.urlopen, sleep=time.sleep) -> dict:
    account, token = get_credentials(environ, env_file)
    source = os.environ if environ is None else environ
    api_base = source.get("CLOUDFLARE_API_BASE", DEFAULT_API_BASE)
    url = build_url(account, args.model, api_base)

    outdir = Path(args.outdir)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    timings, sizes, blanks = [], [], 0

    for index in range(args.runs):
        payload = {"prompt": args.prompt, "steps": args.steps}
        if args.seed is not None:
            payload["seed"] = args.seed + index
        if args.width is not None:
            payload["width"] = args.width
            payload["height"] = args.height

        begin = time.monotonic()
        data = request_image(url, token, payload, opener=opener, sleep=sleep)
        elapsed = time.monotonic() - begin

        width, height, blank, extension = describe_image(data)
        blanks += int(blank)
        timings.append(elapsed)
        sizes.append((width, height))

        name = f"{stamp}_cloudflare_flux_schnell_{width}x{height}_run{index + 1}.{extension}"
        save_bytes_atomic(data, outdir / name)
        print(f"  image {index + 1}/{args.runs}: {elapsed:.1f} s, {width}x{height}{'  <-- BLANK' if blank else ''}  ({name})")

    return {
        "time": stamp,
        "provider": "cloudflare",
        "model": args.model,
        "steps": args.steps,
        "width": sizes[0][0],
        "height": sizes[0][1],
        "seconds_per_image": round(sum(timings) / len(timings), 1),
        "blank_images": blanks,
        "runs": args.runs,
    }


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        validate_args(args)
        result = run(args)
    except Exception as exc:  # report a clear message instead of a traceback
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    append_result(Path(args.outdir) / "results.jsonl", result)
    print(json.dumps(result, indent=2))

    if result["blank_images"]:
        print("BLANK IMAGE: the service returned an empty picture; check the prompt and try again.")
    else:
        print(
            f"OK: {result['seconds_per_image']:.1f} s per image, picture size "
            f"{result['width']}x{result['height']}. Free allowance: {FREE_DAILY_NEURONS:,} neurons per day."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())