"""
Image Prompt Builder

Turns the approved story bible and one designed shot into the text prompt
for an image model. Pure functions: no network, no files.

The most important parts come first, because image models pay most
attention to the start of a prompt and cut off the end of a long one:

    style, camera + action, characters, setting, props, state, lighting,
    composition, closing words

When the prompt is longer than the limit, the descriptions are shortened
step by step instead of cutting the end of the text. Props and the setting
are shortened first (the setting before the props, because a small story
object matters more to the picture than the background). A character keeps
its full description, including clothing, which keeps the person looking the
same from shot to shot, until nothing else can be shortened.
"""

import re

from core.exceptions.exceptions import ValidationError
from core.utils.shot_design import CAMERAS

STYLE_PREFIX = "Anime film illustration,"

COMPOSITION_HINT = (
    "Cinematic widescreen composition, main subject in the middle band of the frame"
)

CLOSING_WORDS = "No text, no watermark"

NO_PEOPLE = "No people in the scene."

CAMERA_PHRASES = {
    "wide": "Wide establishing shot",
    "medium": "Medium shot",
    "close_up": "Close-up shot",
    "extreme_close_up": "Extreme close-up shot",
    "over_the_shoulder": "Over-the-shoulder shot",
    "low_angle": "Low-angle shot looking up",
    "high_angle": "High-angle shot looking down",
    "point_of_view": "First-person point-of-view shot",
}

# How much each step shortens the descriptions, in the order they are tried.
LEVEL_NOTES = (
    "full descriptions",
    "props shortened to one sentence",
    "props and setting shortened to one sentence",
    "setting reduced to its name",
    "props reduced to names",
    "characters shortened to one sentence",
)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def clean(text) -> str:
    """Collapse all whitespace to single spaces."""
    return " ".join(str(text or "").split())


def first_sentence(text) -> str:
    """Return the first sentence of a text."""
    return _SENTENCE_END.split(clean(text), maxsplit=1)[0]


def sentence(text) -> str:
    """Return the text as one sentence ending in a full stop (empty stays empty)."""
    text = clean(text).rstrip(" .;,")
    return text + "." if text else ""


def _lookup(bible: dict, key: str, item_id: str) -> dict:
    """Find a bible entry by id."""
    for item in bible.get(key, []):
        if item.get("id") == item_id:
            return item

    raise ValidationError(f"The shot refers to {key[:-1]} '{item_id}', which is not in the story bible.")


def build_image_prompt(bible: dict, design: dict, max_chars: int) -> dict:
    """
    Build the image prompt for one designed shot.

    Returns a dict with: prompt, length, level, level_note, fits.
    """
    camera_phrase = CAMERA_PHRASES.get(design.get("camera"))

    if camera_phrase is None:
        raise ValidationError(f"Unknown camera '{design.get('camera')}' in the shot design.")

    location = _lookup(bible, "locations", design.get("location"))
    characters = [_lookup(bible, "characters", item) for item in design.get("characters") or []]
    props = [_lookup(bible, "props", item) for item in design.get("props") or []]

    art_style = clean(bible.get("art_style"))
    action = clean(design.get("action"))
    lighting = clean(design.get("lighting"))
    state_notes = clean(design.get("state_notes"))

    def render(level: int) -> str:
        parts = [sentence(f"{STYLE_PREFIX} {art_style}".strip(" ,"))]
        parts.append(sentence(f"{camera_phrase}: {action}"))

        for item in characters:
            visual = first_sentence(item["visual"]) if level >= 5 else clean(item["visual"])
            parts.append(sentence(f"{item['name']}: {visual}"))

        if not characters:
            parts.append(NO_PEOPLE)

        if level >= 3:
            parts.append(sentence(f"Setting: {location['name']}"))
        elif level >= 2:
            parts.append(sentence(f"Setting: {first_sentence(location['visual'])}"))
        else:
            parts.append(sentence(f"Setting: {clean(location['visual'])}"))

        for item in props:
            if level >= 4:
                parts.append(sentence(f"Prop: {item['name']}"))
            elif level >= 1:
                parts.append(sentence(f"{item['name']}: {first_sentence(item['visual'])}"))
            else:
                parts.append(sentence(f"{item['name']}: {clean(item['visual'])}"))

        if state_notes:
            parts.append(sentence(state_notes))

        if lighting:
            parts.append(sentence(f"Lighting: {lighting}"))

        parts.append(sentence(COMPOSITION_HINT))
        parts.append(sentence(CLOSING_WORDS))

        return " ".join(part for part in parts if part)

    level = 0
    prompt = render(level)

    while len(prompt) > max_chars and level < len(LEVEL_NOTES) - 1:
        level += 1
        prompt = render(level)

    return {
        "prompt": prompt,
        "length": len(prompt),
        "level": level,
        "level_note": LEVEL_NOTES[level],
        "fits": len(prompt) <= max_chars,
    }


def build_all_prompts(bible: dict, entries: list[dict], max_chars: int) -> list[dict]:
    """
    Build the image prompt for every designed shot.

    Raises:
        ValidationError: if a shot refers to something missing from the bible,
        or a prompt cannot be made short enough.
    """
    items = []
    too_long = []

    for entry in entries:
        design = entry["design"]
        built = build_image_prompt(bible, design, max_chars)

        if not built["fits"]:
            too_long.append(f"shot {entry['shot_id']} ({built['length']} characters)")

        items.append(
            {
                "shot_id": entry["shot_id"],
                "duration_seconds": entry["duration_seconds"],
                "location": design["location"],
                "characters": list(design.get("characters") or []),
                "props": list(design.get("props") or []),
                "camera": design["camera"],
                "motion": design["motion"],
                **built,
            }
        )

    if too_long:
        raise ValidationError(
            f"These prompts are still longer than {max_chars} characters after shortening: "
            + ", ".join(too_long)
            + ". Raise IMAGE_PROMPT_MAX_CHARS or shorten the bible descriptions."
        )

    return items


def prompts_document(source: str, max_chars: int, items: list[dict]) -> dict:
    """The structure saved to the prompts file."""
    return {
        "source": source,
        "max_chars": max_chars,
        "shot_count": len(items),
        "prompts": items,
    }


def format_prompt_report(items: list[dict], max_chars: int) -> str:
    """A readable report of every prompt for the human review."""
    lines = []
    lines.append("=" * 78)

    for item in items:
        note = "" if item["level"] == 0 else f"  [shortened: {item['level_note']}]"
        lines.append(
            f"Shot {item['shot_id']:>2} | {item['length']:>3}/{max_chars} characters | "
            f"{item['camera']} | {item['location']}{note}"
        )
        lines.append(item["prompt"])
        lines.append("-" * 78)

    shortened = sum(1 for item in items if item["level"] > 0)
    longest = max(item["length"] for item in items)
    lines.append(
        f"{len(items)} prompts, longest {longest} characters, {shortened} shortened to fit {max_chars}."
    )
    lines.append("=" * 78)

    return "\n".join(lines)