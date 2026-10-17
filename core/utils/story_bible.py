"""
Story Bible Utilities

Parsing, validation and display helpers for the story bible: the fixed list
of characters, locations and props (each with a stable id and one visual
description) that every later stage reuses so the video stays consistent.
"""

import json
import re

from core.exceptions.exceptions import ValidationError

MAX_CHARACTERS = 4
MAX_LOCATIONS = 5
MAX_PROPS = 6
MIN_VISUAL_WORDS = 8
MAX_VISUAL_CHARS = 600
MAX_NAME_CHARS = 60

_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,31}$")

# (key, minimum items, maximum items)
_GROUPS = (
    ("characters", 1, MAX_CHARACTERS),
    ("locations", 1, MAX_LOCATIONS),
    ("props", 0, MAX_PROPS),
)

_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "name": {"type": "string"},
        "visual": {"type": "string"},
    },
    "required": ["id", "name", "visual"],
}

# JSON Schema sent to Ollama so the model must return all three lists
# (plain JSON mode lets small models answer with an empty object).
# Counts and text rules are enforced afterwards by validate_bible().
BIBLE_SCHEMA = {
    "type": "object",
    "properties": {
        "characters": {"type": "array", "items": _ITEM_SCHEMA},
        "locations": {"type": "array", "items": _ITEM_SCHEMA},
        "props": {"type": "array", "items": _ITEM_SCHEMA},
    },
    "required": ["characters", "locations", "props"],
}

_SKIPPED_NAME_WORDS = {"the", "and", "old", "young", "mr", "mrs", "dr"}


def _text(value) -> str:
    """Return a whitespace-normalised string, or "" for non-strings."""
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def _slug(value: str) -> str:
    """Turn any text into a snake_case identifier."""
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _as_item_list(items):
    """
    Accept the shapes small models commonly return instead of a list:
    a single item object, or an object mapping ids to item objects.
    Anything else is returned unchanged so validation can report it.
    """
    if not isinstance(items, dict):
        return items

    if "name" in items or "visual" in items:
        return [items]

    if items and all(isinstance(v, dict) for v in items.values()):
        return [{"id": key, **value} for key, value in items.items()]

    return items


def extract_json(raw: str) -> dict:
    """
    Parse a JSON object from a model reply.

    Tolerates markdown code fences and text around the object.

    Raises:
        ValidationError: if no valid JSON object can be found.
    """
    text = raw.strip()

    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end <= start:
            raise ValidationError("The reply contains no JSON object.")

        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValidationError(
                f"The reply is not valid JSON: {exc.msg} "
                f"(line {exc.lineno}, column {exc.colno})."
            )

    if not isinstance(data, dict):
        raise ValidationError("The top-level JSON value must be an object.")

    return data


def validate_bible(data: dict) -> dict:
    """
    Validate a story bible and return a cleaned copy.

    All problems are collected and reported together so they can be sent
    back to the model in one retry message.

    Raises:
        ValidationError: listing every problem found.
    """
    problems = []
    cleaned = {}
    seen_ids = set()

    for key, minimum, maximum in _GROUPS:
        items = _as_item_list(data.get(key, [] if minimum == 0 else None))

        if not isinstance(items, list):
            problems.append(
                f'"{key}" must be a JSON list of objects, like [{{"id": ..., '
                f'"name": ..., "visual": ...}}].'
            )
            continue

        if not minimum <= len(items) <= maximum:
            problems.append(
                f'"{key}" must contain {minimum} to {maximum} items, '
                f"got {len(items)}."
            )

        cleaned_items = []

        for index, item in enumerate(items, start=1):
            label = f"{key}[{index}]"

            if not isinstance(item, dict):
                problems.append(f"{label} must be an object.")
                continue

            name = _text(item.get("name"))
            visual = _text(item.get("visual"))
            item_id = _slug(_text(item.get("id")) or name)

            # Models often copy a numbered placeholder ("silas_1"): drop the
            # number unless that would clash with another id.
            unnumbered = re.sub(r"_\d+$", "", item_id)
            if unnumbered and unnumbered != item_id and unnumbered not in seen_ids:
                item_id = unnumbered

            if not name:
                problems.append(f'{label} needs a non-empty "name".')
            elif len(name) > MAX_NAME_CHARS:
                problems.append(f'{label} "name" is longer than {MAX_NAME_CHARS} characters.')

            if not _ID_PATTERN.match(item_id):
                problems.append(
                    f'{label} "id" must be lowercase snake_case '
                    f"(letters, digits, underscores, starting with a letter), got '{item_id}'."
                )
            elif item_id in seen_ids:
                problems.append(f"{label} id '{item_id}' is used more than once.")
            seen_ids.add(item_id)

            if len(visual.split()) < MIN_VISUAL_WORDS:
                problems.append(
                    f'{label} "visual" needs at least {MIN_VISUAL_WORDS} words of '
                    "concrete visual description."
                )
            elif len(visual) > MAX_VISUAL_CHARS:
                problems.append(
                    f'{label} "visual" is longer than {MAX_VISUAL_CHARS} characters.'
                )

            cleaned_items.append({"id": item_id, "name": name, "visual": visual})

        cleaned[key] = cleaned_items

    if problems:
        raise ValidationError(
            "Story bible is invalid:\n" + "\n".join(f"- {p}" for p in problems)
        )

    return cleaned


def bible_warnings(bible: dict, story: str) -> list[str]:
    """
    Soft checks that should not block the bible: characters, locations
    or props whose name never appears in the story.
    """
    story_lower = story.lower()
    warnings = []

    for key, label in (
        ("characters", "Character"),
        ("locations", "Location"),
        ("props", "Prop"),
    ):
        for item in bible.get(key, []):
            words = [
                w
                for w in re.findall(r"[a-z]+", item["name"].lower())
                if len(w) >= 3 and w not in _SKIPPED_NAME_WORDS
            ]

            if words and not any(w in story_lower for w in words):
                warnings.append(
                    f"{label} '{item['name']}' is never mentioned in the story."
                )

    return warnings


def format_bible(bible: dict) -> str:
    """Render a bible as readable text for the approval gate."""
    lines = [f"ART STYLE: {bible.get('art_style', '(not set)')}"]

    for key, title in (
        ("characters", "CHARACTERS"),
        ("locations", "LOCATIONS"),
        ("props", "PROPS"),
    ):
        items = bible.get(key, [])
        if not items:
            continue

        lines.append("")
        lines.append(title)

        for item in items:
            lines.append(f"  [{item['id']}] {item['name']}")
            lines.append(f"      {item['visual']}")

    return "\n".join(lines)