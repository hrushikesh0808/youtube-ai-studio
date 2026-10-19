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

# Raw bible ids (snake_case) must not leak into the descriptive text.
_ID_IN_TEXT = re.compile(r"\b[a-z0-9]+(?:_[a-z0-9]+)+\b")

# Camera and shot-type words do not belong in the action: camera and motion
# are separate fields, and the text would contradict them.
_CAMERA_WORDS = re.compile(
    r"\b(camera|shots?|close[- ]?ups?|zoom\w*|pans?|panning|tilts?|tilting|angles?"
    r"|pushes in|pushing in|pulls? (?:back|out)|pulling (?:back|out))\b",
    re.IGNORECASE,
)

_NARRATION_SKIPPED_WORDS = {"the", "and", "old", "young", "mr", "mrs", "dr"}

# A person may be shown when the narration names them or uses one of these.
_PERSON_PRONOUNS = {"he", "him", "his", "himself", "she", "her", "hers", "herself", "they", "them", "their"}

# An object from the previous shot may stay when the narration says "it".
_OBJECT_PRONOUNS = {"it", "its", "itself", "they", "them", "their"}

# How many previous shots' cameras and motions may not be reused.
RECENT_SHOTS_AVOIDED = 2


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


def _name_words(name: str) -> list[str]:
    """Meaningful words of a bible name, for matching against narration."""
    return [
        w
        for w in re.findall(r"[a-z]+", name.lower())
        if len(w) >= 3 and w not in _NARRATION_SKIPPED_WORDS
    ]


def _mentions(tokens: set[str], name: str) -> bool:
    """
    True when the narration words mention a bible item by its name.

    A name word matches a narration word it starts with ("lamps"), or one
    that ends it ("shell" for "seashell").
    """
    for word in _name_words(name):
        for token in tokens:
            if token.startswith(word) or (len(token) >= 4 and word.endswith(token)):
                return True
    return False


def shot_allowances(bible: dict, narration: str, previous: dict | None) -> dict[str, list[str]]:
    """
    Which characters and props this shot may show, from its own narration.

    - A character is allowed when the narration names them or says he/she/they.
      A narration about a place or an object therefore shows nobody.
    - A prop is allowed when the narration names it. When the narration says
      "it" or "them", the previous shot's props are allowed too.
    """
    tokens = set(re.findall(r"[a-z]+", narration.lower()))

    has_person_pronoun = bool(tokens & _PERSON_PRONOUNS)
    has_object_pronoun = bool(tokens & _OBJECT_PRONOUNS)
    carried = set(previous["props"]) if previous and has_object_pronoun else set()

    return {
        "characters": [
            item["id"]
            for item in bible.get("characters", [])
            if has_person_pronoun or _mentions(tokens, item["name"])
        ],
        "props": [
            item["id"]
            for item in bible.get("props", [])
            if item["id"] in carried or _mentions(tokens, item["name"])
        ],
    }


def unnamed_props(bible: dict, shots: list[dict]) -> list[str]:
    """Names of props that no shot's narration mentions (they can never be shown)."""
    tokens = set()
    for shot in shots:
        tokens |= set(re.findall(r"[a-z]+", shot["narration"].lower()))

    return [item["name"] for item in bible.get("props", []) if not _mentions(tokens, item["name"])]


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def named_in_text(bible: dict, text: str) -> dict[str, list[str]]:
    """
    Characters and props that a piece of text names (e.g. an action sentence).

    Location names are removed first, so "lamp room" is not read as the
    brass lamp.
    """
    lowered = text.lower()
    for location in bible.get("locations", []):
        lowered = lowered.replace(location["name"].lower(), " ")

    tokens = _words(lowered)

    return {
        key: [item["id"] for item in bible.get(key, []) if _mentions(tokens, item["name"])]
        for key in ("characters", "props")
    }


_LOCATION_HINT_STOPWORDS = {
    "that", "this", "with", "from", "were", "have", "been", "they", "them", "their",
    "there", "then", "than", "when", "what", "which", "while", "into", "onto", "over",
    "under", "each", "only", "just", "like", "more", "most", "such", "very", "much",
    "many", "some", "even", "also", "because", "about", "after", "before", "again",
    "above", "below", "could", "would", "should", "still", "being", "both", "having",
    "those", "these", "other", "another", "amongst", "through",
}


def location_hint(bible: dict, narration: str) -> str:
    """
    A hint naming the location whose description shares the most words with
    the narration (empty when no location clearly wins).
    """
    narration_words = {
        w for w in _words(narration) if len(w) >= 4 and w not in _LOCATION_HINT_STOPWORDS
    }

    scores = []
    for location in bible.get("locations", []):
        location_words = {
            w
            for w in _words(location["name"] + " " + location["visual"])
            if len(w) >= 4 and w not in _LOCATION_HINT_STOPWORDS
        }
        hits = sorted(
            w for w in narration_words if any(w[:5] == other[:5] for other in location_words)
        )
        scores.append((len(hits), location["id"], hits))

    scores.sort(key=lambda item: -item[0])

    if not scores or scores[0][0] == 0:
        return ""
    if len(scores) > 1 and scores[1][0] == scores[0][0]:
        return ""

    _, location_id, hits = scores[0]
    return (
        f" Hint: the narration words ({', '.join(hits[:3])}) match {location_id}; "
        "use it unless the narration clearly moves elsewhere."
    )


def opening(action: str) -> str:
    """The first three words of an action, lower case (used to avoid repeated openings)."""
    return " ".join(action.lower().split()[:3])


def build_shot_schema(
    bible: dict,
    allowed_characters: list[str] | None = None,
    allowed_props: list[str] | None = None,
    avoid_camera: list[str] | None = None,
    avoid_motion: list[str] | None = None,
) -> dict:
    """
    Build the JSON Schema sent to Ollama for one shot design.

    Location, character and prop ids are restricted to the bible's ids (and
    characters and props to those allowed in this shot), so the model cannot
    invent one. Recently used cameras and motions are left out so they cannot
    repeat. Property order is the order the model writes them in: the action
    comes first so the rest follows from it.
    """
    ids = bible_ids(bible)
    characters_allowed = ids["characters"] if allowed_characters is None else allowed_characters
    props_allowed = ids["props"] if allowed_props is None else allowed_props
    cameras = [c for c in CAMERAS if c not in (avoid_camera or [])]
    motions = [m for m in MOTIONS if m not in (avoid_motion or [])]

    def id_list(values):
        return {"type": "array", "items": {"type": "string", "enum": values}}

    properties = {
        "action": {"type": "string"},
        "location": {"type": "string", "enum": ids["locations"]},
        "characters": id_list(characters_allowed),
        "props": id_list(props_allowed),
        "state_notes": {"type": "string"},
        "lighting": {"type": "string"},
        "camera": {"type": "string", "enum": cameras},
        "motion": {"type": "string", "enum": motions},
    }

    # With nothing allowed the field is left out (a missing list means none),
    # because an empty enum is not valid JSON Schema.
    if not characters_allowed:
        del properties["characters"]
    if not props_allowed:
        del properties["props"]

    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
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


def validate_shot_design(
    data: dict,
    bible: dict,
    allowed_characters: list[str] | None = None,
    allowed_props: list[str] | None = None,
    avoid_camera: list[str] | None = None,
    avoid_motion: list[str] | None = None,
    avoid_openings: list[str] | None = None,
) -> dict:
    """
    Validate one shot design against the bible and return a cleaned copy.

    allowed_characters and allowed_props limit who and what may appear in
    this shot; avoid_camera, avoid_motion and avoid_openings (first three
    words of recent actions) are recently used values, which must not repeat.
    Characters and props named in the action or state notes must be listed
    in the design, and must be allowed in this shot.

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
    if allowed_characters is not None:
        not_allowed = [c for c in characters if c not in allowed_characters]
        if not_allowed:
            problems.append(
                f'"characters" lists {", ".join(not_allowed)}, but the narration of this '
                f'shot does not mention them. Characters allowed in this shot: '
                f'{", ".join(allowed_characters) if allowed_characters else "none"}.'
            )

    props = _clean_id_list(data.get("props"), ids["props"], "props", problems)
    if allowed_props is not None:
        too_early = [p for p in props if p not in allowed_props]
        if too_early:
            problems.append(
                f'"props" lists {", ".join(too_early)}, which the narration of '
                f'this shot does not mention. Props allowed in this shot: '
                f'{", ".join(allowed_props) if allowed_props else "none"}.'
            )

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
    elif camera in (avoid_camera or []):
        problems.append(
            f'"camera" was used in a recent shot ({camera}); choose another: '
            f'{", ".join(c for c in CAMERAS if c not in avoid_camera)}.'
        )

    motion = _enum_value(data.get("motion"))
    if motion not in MOTIONS:
        problems.append(f'"motion" must be one of: {", ".join(MOTIONS)}; got \'{_text(data.get("motion"))}\'.')
    elif motion in (avoid_motion or []):
        problems.append(
            f'"motion" was used in a recent shot ({motion}); choose another: '
            f'{", ".join(m for m in MOTIONS if m not in avoid_motion)}.'
        )

    named = named_in_text(bible, action + " " + state_notes)
    for key, listed, allowed in (
        ("characters", characters, allowed_characters),
        ("props", props, allowed_props),
    ):
        for item_id in named[key]:
            if item_id in listed:
                continue
            if allowed is not None and item_id not in allowed:
                problems.append(
                    f'The action or state_notes name \'{item_id}\', which this shot may not show. '
                    "Remove it from the text."
                )
            else:
                problems.append(
                    f'The action or state_notes name \'{item_id}\' but "{key}" does not list it. '
                    f'Add it to "{key}" or remove it from the text.'
                )

    if avoid_openings and opening(action) in avoid_openings:
        problems.append(
            f'"action" starts with "{opening(action)}" like a recent shot. '
            "Start it differently."
        )

    if _CAMERA_WORDS.search(action):
        problems.append(
            f'"action" mentions the camera or a shot type ("{_CAMERA_WORDS.search(action).group(0)}"). '
            "Describe only what is in the scene; camera and motion are separate fields."
        )

    for label, value in (("action", action), ("state_notes", state_notes), ("lighting", lighting)):
        if _FORBIDDEN_IN_VISUALS.search(value):
            problems.append(
                f'"{label}" must not mention readable text, letters, numbers or dates; '
                "describe writing as illegible marks."
            )

    for label, value in (("action", action), ("state_notes", state_notes), ("lighting", lighting)):
        if _ID_IN_TEXT.search(value):
            problems.append(
                f'"{label}" contains an id with underscores ({_ID_IN_TEXT.search(value).group(0)}); '
                "write plain words instead."
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

            words = _name_words(item["name"])

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