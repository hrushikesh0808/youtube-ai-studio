"""
Reference Pictures

Finds the approved reference pictures (character views, a close-up of the
character's face, and places) and chooses which of them go into each shot.

Folder layout under REFERENCE_DIR (default output/references):

    characters/<character id>/front.png three_quarter.png side.png back.png
                              happy.png thinking.png face.png
    places/<location id>.png          (.jpg, .jpeg and .webp also work)

Pure file lookups and small decisions: no network.
"""

import hashlib
import io
from pathlib import Path

from PIL import Image

from core.exceptions.exceptions import ConfigurationError

EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

# The service takes at most 4 reference pictures, each smaller than 512x512.
MAX_REFERENCES = 4
MAX_REFERENCE_SIDE = 448

VIEWS = ("front", "three_quarter", "side", "back", "happy", "thinking")

# Cameras close enough that the face must be kept exactly (the face close-up is added).
FACE_CAMERAS = ("close_up", "extreme_close_up", "medium", "wide", "over_the_shoulder", "low_angle", "point_of_view")

# Cameras that look at the character from the front.
FRONT_CAMERAS = ("close_up", "extreme_close_up", "point_of_view", "over_the_shoulder")

# Words in the shot action that mean the character is seen from behind.
BACK_PHRASES = ("from behind", "back to the camera", "his back", "back turned", "seen from the back")


def find_picture(folder: Path, stem: str) -> Path | None:
    """The picture folder/<stem>.<extension>, or None."""
    for extension in EXTENSIONS:
        path = Path(folder) / f"{stem}{extension}"

        if path.is_file():
            return path

    return None


def choose_view(design: dict) -> str:
    """Which character view fits the shot: back, front or three_quarter."""
    action = " ".join(str(design.get("action") or "").lower().split())

    if any(phrase in action for phrase in BACK_PHRASES):
        return "back"

    if design.get("camera") in FRONT_CAMERAS:
        return "front"

    return "three_quarter"


def _name_of(bible: dict, key: str, item_id: str) -> str:
    for item in bible.get(key, []):
        if item.get("id") == item_id:
            return item.get("name") or item_id

    return item_id


def select_references(bible: dict, design: dict, root: Path) -> tuple[list[dict], list[str]]:
    """
    Choose the reference pictures of one shot.

    Order (the most important first, the service takes at most 4): the first
    character's view, that character's face close-up (close shots), the place,
    then the views of other characters.

    Returns (references, missing). Each reference is a dict with kind
    ("character", "face" or "place"), id, name, view (characters) and path.
    missing lists the files that should exist but do not. A missing face
    close-up is not an error; it is simply left out.
    """
    root = Path(root)
    view = choose_view(design)
    camera = design.get("camera")
    character_ids = list(design.get("characters") or [])
    location_id = design.get("location")

    views = []
    faces = []
    missing = []

    for index, character_id in enumerate(character_ids):
        name = _name_of(bible, "characters", character_id)
        folder = root / "characters" / character_id
        path = find_picture(folder, view)

        if path is None:
            missing.append(str(folder / f"{view}.png"))
        else:
            views.append({"kind": "character", "id": character_id, "name": name, "view": view, "path": path})

        if index == 0 and camera in FACE_CAMERAS:
            face = find_picture(folder, "face")

            if face is not None:
                faces.append({"kind": "face", "id": character_id, "name": name, "view": "face", "path": face})

    places = []

    if location_id:
        path = find_picture(root / "places", location_id)

        if path is None:
            missing.append(str(root / "places" / f"{location_id}.png"))
        else:
            places.append(
                {"kind": "place", "id": location_id, "name": _name_of(bible, "locations", location_id), "view": "", "path": path}
            )

    ordered = views[:1] + faces + places + views[1:]

    return ordered[:MAX_REFERENCES], missing


def prepare_reference(path: Path) -> bytes:
    """Load a reference picture and shrink it so both sides are at most 448 pixels (PNG bytes)."""
    try:
        picture = Image.open(path).convert("RGB")
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f"Cannot read the reference picture: {path}") from exc

    picture.thumbnail((MAX_REFERENCE_SIDE, MAX_REFERENCE_SIDE))
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG")
    return buffer.getvalue()


def reference_key(prepared: list[bytes]) -> str:
    """Short fingerprint of a set of reference pictures, to notice when one changed."""
    digest = hashlib.sha1()

    for data in prepared:
        digest.update(hashlib.sha1(data).digest())

    return digest.hexdigest()[:12]