"""
Reference Prompt Builder

Builds the text prompt of a shot (or of a place) for an image model that also
receives reference pictures. Pure functions: no network, no files.

The prompt says what each reference picture is, so the model keeps the same
face, clothes and place, and it fixes the colour style, so the dull "dim and
grey" words in a shot design cannot make the picture dark and colourless.
"""

from core.exceptions.exceptions import ValidationError
from core.utils.image_prompt import CAMERA_PHRASES
from core.utils.image_prompt import clean
from core.utils.image_prompt import first_sentence
from core.utils.image_prompt import sentence

COLOUR_NOTE = "Keep the colours rich and saturated, with warm glowing lights against deep teal and navy tones."

CLOSING = "Each person has exactly two arms and two hands. No text, no letters, no watermark, no signature."

PLACE_CLOSING = "No text, no letters, no watermark, no signature."

MAX_PROMPT_CHARS = 2048

LEVEL_NOTES = (
    "full descriptions",
    "props shortened to names",
    "props shortened and lighting dropped",
)


def _find(bible: dict, key: str, item_id: str) -> dict:
    for item in bible.get(key, []):
        if item.get("id") == item_id:
            return item

    raise ValidationError(f"The shot refers to {key[:-1]} '{item_id}', which is not in the story bible.")


def reference_note(index: int, reference: dict) -> str:
    """One sentence that says what reference picture number `index` is."""
    kind = reference["kind"]
    name = reference["name"]

    if kind == "character":
        view = reference["view"].replace("_", " ")
        return (
            f"Image {index} shows the character {name} ({view} view): keep exactly this character's face, hair, "
            "body, clothes and colours. Do not copy its plain background."
        )

    if kind == "face":
        return f"Image {index} is a close-up of {name}'s face: keep exactly this face."

    return (
        f"Image {index} shows the place ({name}): draw the same place with the same buildings, shapes, "
        "materials, weather and colours."
    )


def build_reference_prompt(bible: dict, design: dict, references: list[dict], scene_style: str) -> dict:
    """
    Build the prompt of one designed shot that is generated with reference pictures.

    Returns a dict with: prompt, length, level, level_note, fits.
    """
    camera_phrase = CAMERA_PHRASES.get(design.get("camera"))

    if camera_phrase is None:
        raise ValidationError(f"Unknown camera '{design.get('camera')}' in the shot design.")

    location = _find(bible, "locations", design.get("location"))
    characters = [_find(bible, "characters", item) for item in design.get("characters") or []]
    props = [_find(bible, "props", item) for item in design.get("props") or []]

    with_reference = {item["id"] for item in references if item["kind"] == "character"}
    has_place = any(item["kind"] == "place" for item in references)

    action = clean(design.get("action"))
    lighting = clean(design.get("lighting"))
    state_notes = clean(design.get("state_notes"))

    def render(level: int) -> str:
        parts = [sentence(scene_style)]
        parts += [reference_note(index, item) for index, item in enumerate(references)]
        parts.append(sentence(f"{camera_phrase}: {action}"))

        for item in characters:
            if item["id"] not in with_reference:
                parts.append(sentence(f"{item['name']}: {first_sentence(item['visual'])}"))

        if not characters:
            parts.append("No people in the scene.")

        if has_place:
            parts.append(sentence(f"Setting: {location['name']}"))
        else:
            parts.append(sentence(f"Setting: {first_sentence(location['visual'])}"))

        for item in props:
            if level >= 1:
                parts.append(sentence(f"Prop: {item['name']}"))
            else:
                parts.append(sentence(f"{item['name']}: {first_sentence(item['visual'])}"))

        if state_notes:
            parts.append(sentence(state_notes))

        if lighting and level < 2:
            parts.append(sentence(f"Lighting: {lighting}"))

        parts.append(COLOUR_NOTE)
        parts.append(CLOSING)

        return " ".join(part for part in parts if part)

    level = 0
    prompt = render(level)

    while len(prompt) > MAX_PROMPT_CHARS and level < len(LEVEL_NOTES) - 1:
        level += 1
        prompt = render(level)

    return {
        "prompt": prompt,
        "length": len(prompt),
        "level": level,
        "level_note": LEVEL_NOTES[level],
        "fits": len(prompt) <= MAX_PROMPT_CHARS,
    }


def build_place_prompt(location: dict, scene_style: str, has_style_picture: bool) -> str:
    """
    Build the prompt of a place picture (no people).

    When has_style_picture is True, image 0 is an approved place picture whose
    painting style, colours and weather the new place must match.
    """
    parts = [sentence(scene_style)]

    if has_style_picture:
        parts.append(
            "Image 0 shows the painting style, colour palette and weather to match: paint the new place in exactly "
            "the same style, colours and mood, but draw the new place described below, not the buildings in image 0."
        )

    parts.append(sentence(f"{location['name']}: {clean(location['visual'])[:1200]}"))
    parts.append("Wide view of the place, empty, no people.")
    parts.append(COLOUR_NOTE)
    parts.append(PLACE_CLOSING)

    return " ".join(part for part in parts if part)


def reference_labels(references: list[dict]) -> list[str]:
    """Short readable names of the references, for reports and files."""
    labels = []

    for item in references:
        if item["kind"] == "place":
            labels.append(f"place:{item['id']}")
        elif item["kind"] == "face":
            labels.append(f"face:{item['id']}")
        else:
            labels.append(f"{item['id']}:{item['view']}")

    return labels


def format_reference_report(items: list[dict]) -> str:
    """A readable report of every shot prompt and its reference pictures, for the human review."""
    lines = ["=" * 78]

    for item in items:
        note = "" if item["level"] == 0 else f"  [shortened: {item['level_note']}]"
        lines.append(
            f"Shot {item['shot_id']:>2} | {item['length']:>4} characters | {item['camera']} | {item['location']}{note}"
        )
        lines.append("References: " + (", ".join(item["refs"]) if item["refs"] else "none"))

        if item["missing"]:
            lines.append("MISSING: " + ", ".join(item["missing"]))

        lines.append(item["prompt"])
        lines.append("-" * 78)

    missing_shots = sum(1 for item in items if item["missing"])
    lines.append(f"{len(items)} prompts, {missing_shots} shots with missing reference pictures.")
    lines.append("=" * 78)

    return "\n".join(lines)