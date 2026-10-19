
"""
Shot Design Utilities

Rules, validation, resume-file handling and display helpers for the
per-shot design: for every shot, what the camera shows (action, location,
characters, props, shot-specific state, lighting, camera, motion).

Every id a design uses must exist in the approved story bible.
"""

import hashlib
import json
import re
from pathlib import Path

from core.exceptions.exceptions import ValidationError
from core.logging.logger import logger

CAMERAS = (
    "wide",
    "medium",
    "close_up",
    "extreme_close_up",
    "over_the_shoulder",
    "low_angle",
    "high_angle",
    "point_of_view",
)

# Motions that can be produced from a single still image.
MOTIONS = (
    "static",
    "slow_push_in",
    "slow_pull_out",
    "pan_left",
    "pan_right",
    "tilt_up",
    "tilt_down",
)

MAX_CHARACTERS_PER_SHOT = 3
MIN_ACTION_WORDS = 5
MAX_ACTION_CHARS = 350
MAX_STATE_CHARS = 250
MAX_LIGHTING_CHARS = 120

# Generated frames must never contain readable text, numbers or dates
# (image models render them badly; overlays are added during assembly).
_FORBIDDEN_IN_VISUALS = re.compile(r"\d|\b(text|letters|captions?|subtitles?)\b", re.IGNORECASE)

_NARRATION_SKIPPED_WORDS = {"the", "and", "old", "young", "mr", "mrs", "dr"}


def _text(value) -> str:
    """Return a whitespace-normalised string, or "" for non-strings."""
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def _enum_value(value) -> str:
    """Normalise an enum-like answer ("Close Up" -> "close_up")."""
    return re.sub(r"[^a-z0-9]+", "_", _text(value).lower()).strip("_")


def bible_ids(bible: dict) -> dict:
    """Return {"characters": [...ids], "locations": [...], "props": [...]}."""
    return {
        key: [item["id"] for item in bible.get(key, [])]
        for key in ("characters", "locations", "props")
    }


def build_shot_schema(bible: dict) -> dict:
    """
    Build the JSON Schema sent to Ollama for one shot design.

    Location, character and prop ids are restricted to the bible's ids, so
    the model cannot invent one. Property order is the order the model
    writes them in: the action comes first so the rest follows from it.
    """
    ids = bible_ids(bible)

    def id_list(values):
        items = {"type": "string"}
        if values:
            items["enum"] = values
        return {"type": "array", "items": items}

    return {
        "type": "object",
        "properties": {
            "action": {"type": "string"},
            "location": {"type": "string", "enum": ids["locations"]},
            "characters": id_list(ids["characters"]),
            "props": id_list(ids["props"]),
            "state_notes": {"type": "string"},
            "lighting": {"type": "string"},
            "camera": {"type": "string", "enum": list(CAMERAS)},
            "motion": {"type": "string", "enum": list(MOTIONS)},
        },
        "required": [
            "action",
            "location",
            "characters",
            "props",
            "state_notes",
            "lighting",
            "camera",
            "motion",
        ],
    }


def _clean_id_list(value, allowed, label, problems) -> list[str]:
    """Validate a list of bible ids; keep order, drop duplicates."""
    if value is None:
        return []

    if not isinstance(value, list):
        problems.append(f'"{label}" must be a list of ids.')
        return []

    result = []
    for item in value:
        item_id = _enum_value(item)
        if item_id not in allowed:
            problems.append(
                f'"{label}" contains unknown id \'{_text(item)}\'. '
                f"Valid ids: {', '.join(allowed) if allowed else '(none)'}."
            )
        elif item_id not in result:
            result.append(item_id)

    return result


def validate_shot_design(data: dict, bible: dict) -> dict:
    """
    Validate one shot design against the bible and return a cleaned copy.

    All problems are collected so they can be sent back to the model together.

    Raises:
        ValidationError: listing every problem found.
    """
    if not isinstance(data, dict):
        raise ValidationError("Shot design must be a JSON object.")

    ids = bible_ids(bible)
    problems = []

    action = _text(data.get("action"))
    if len(action.split()) < MIN_ACTION_WORDS:
        problems.append(
            f'"action" needs at least {MIN_ACTION_WORDS} words describing what is visible.'
        )
    elif len(action) > MAX_ACTION_CHARS:
        problems.append(f'"action" is longer than {MAX_ACTION_CHARS} characters.')

    location = _enum_value(data.get("location"))
    if location not in ids["locations"]:
        problems.append(
            f'"location" must be one of: {", ".join(ids["locations"])}; '
            f"got '{_text(data.get('location'))}'."
        )

    characters = _clean_id_list(data.get("characters"), ids["characters"], "characters", problems)
    if len(characters) > MAX_CHARACTERS_PER_SHOT:
        problems.append(f'"characters" may list at most {MAX_CHARACTERS_PER_SHOT} ids.')

    props = _clean_id_list(data.get("props"), ids["props"], "props", problems)

    state_notes = _text(data.get("state_notes"))
    if len(state_notes) > MAX_STATE_CHARS:
        problems.append(f'"state_notes" is longer than {MAX_STATE_CHARS} characters.')

    lighting = _text(data.get("lighting"))
    if not lighting:
        problems.append('"lighting" must be a short non-empty phrase.')
    elif len(lighting) > MAX_LIGHTING_CHARS:
        problems.append(f'"lighting" is longer than {MAX_LIGHTING_CHARS} characters.')

    camera = _enum_value(data.get("camera"))
    if camera not in CAMERAS:
        problems.append(f'"camera" must be one of: {", ".join(CAMERAS)}; got \'{_text(data.get("camera"))}\'.')

    motion = _enum_value(data.get("motion"))
    if motion not in MOTIONS:
        problems.append(f'"motion" must be one of: {", ".join(MOTIONS)}; got \'{_text(data.get("motion"))}\'.')

    for label, value in (("action", action), ("state_notes", state_notes), ("lighting", lighting)):
        if _FORBIDDEN_IN_VISUALS.search(value):
            problems.append(
                f'"{label}" must not mention readable text, letters, numbers or dates; '
                "describe writing as illegible marks."
            )

    if problems:
        raise ValidationError(
            "Shot design is invalid:\n" + "\n".join(f"- {p}" for p in problems)
        )

    return {
        "action": action,
        "location": location,
        "characters": characters,
        "props": props,
        "state_notes": state_notes,
        "lighting": lighting,
        "camera": camera,
        "motion": motion,
    }


def design_warnings(design: dict, narration: str, bible: dict) -> list[str]:
    """
    Soft checks that should not block a design: characters or props named
    in the narration that the design leaves out.
    """
    narration_lower = narration.lower()
    warnings = []

    for key, label in (("characters", "character"), ("props", "prop")):
        for item in bible.get(key, []):
            if item["id"] in design[key]:
                continue

            words = [
                w
                for w in re.findall(r"[a-z]+", item["name"].lower())
                if len(w) >= 3 and w not in _NARRATION_SKIPPED_WORDS
            ]

            if any(re.search(rf"\b{re.escape(w)}\b", narration_lower) for w in words):
                warnings.append(
                    f"The narration mentions {label} '{item['name']}' "
                    f"but the design does not include it."
                )

    return warnings


def inputs_hash(shots: list[dict], bible: dict) -> str:
    """
    Fingerprint of everything a design depends on (shot texts and bible).
    A saved design file is only reused while this still matches.
    """
    payload = json.dumps(
        {"shots": [[s["id"], s["narration"]] for s in shots], "bible": bible},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def progress_document(source: str, fingerprint: str, entries: list[dict], complete: bool) -> dict:
    """Build the JSON document saved to the designs file."""
    return {
        "source": source,
        "inputs_hash": fingerprint,
        "complete": complete,
        "shot_count": len(entries),
        "shots": entries,
    }


def load_progress(path: Path, fingerprint: str, expected_count: int) -> tuple[list[dict], bool]:
    """
    Load saved designs for resuming.

    Returns (entries, complete). A missing, unreadable or outdated file
    (shots or bible changed since it was written) gives ([], False).
    """
    path = Path(path)

    if not path.is_file():
        return [], False

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        entries = document["shots"]
        saved_hash = document["inputs_hash"]
        complete = bool(document.get("complete", False))
        if not isinstance(entries, list) or not all("shot_id" in e and "design" in e for e in entries):
            raise ValueError("unexpected structure")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("Ignoring unreadable design file %s (%s). Designing from scratch.", path.name, exc)
        return [], False

    if saved_hash != fingerprint:
        logger.warning(
            "Saved designs in %s were made for a different story or bible. "
            "Designing from scratch.",
            path.name,
        )
        return [], False

    entries = sorted(entries, key=lambda e: e["shot_id"])

    return entries, complete and len(entries) == expected_count


def format_design_table(entries: list[dict]) -> str:
    """Render the designs as a readable table."""
    header = f"{'#':>3}  {'location':<20} {'camera':<17} {'motion':<14} action"
    lines = ["=" * 100, header, "=" * 100]

    for entry in entries:
        design = entry["design"]
        action = design["action"]
        if len(action) > 38:
            action = action[:35] + "..."

        lines.append(
            f"{entry['shot_id']:>3}  {design['location']:<20.20} {design['camera']:<17} "
            f"{design['motion']:<14} {action}"
        )

    lines.append("=" * 100)
    return "\n".join(lines)

