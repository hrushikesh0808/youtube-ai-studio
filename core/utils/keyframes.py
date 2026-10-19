"""
Keyframe Bookkeeping

Pure helpers for the keyframe stage: choosing shots, naming files and the
manifest that records every generated candidate (so an interrupted run can
resume and nothing is generated twice).
"""

import hashlib
import json
from pathlib import Path

from core.exceptions.exceptions import ValidationError


def parse_shot_selection(text: str, available: list[int]) -> list[int]:
    """
    Turn "1,2,8", "1-3,8" or "all" into a sorted list of shot ids.

    Raises:
        ValidationError: if the text is malformed or names a shot that does not exist.
    """
    text = (text or "").strip().lower()

    if not text:
        raise ValidationError("No shots given. Use for example --keyframes 1,2,8 or --keyframes all.")

    if text == "all":
        return sorted(available)

    chosen = set()

    for part in text.split(","):
        part = part.strip()

        try:
            if "-" in part:
                start_text, end_text = part.split("-", 1)
                start, end = int(start_text), int(end_text)

                if start > end:
                    raise ValueError
                chosen.update(range(start, end + 1))
            else:
                chosen.add(int(part))
        except ValueError:
            raise ValidationError(
                f"Cannot read '{part}' as a shot number or range. Use for example 1,2,8 or 1-3 or all."
            )

    missing = sorted(chosen - set(available))

    if missing:
        raise ValidationError(
            f"These shots do not exist: {', '.join(str(item) for item in missing)}. "
            f"The story has shots {min(available)}-{max(available)}."
        )

    return sorted(chosen)


def prompt_hash(prompt: str) -> str:
    """Short fingerprint of a prompt, to notice when it changed."""
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]


def keyframe_filename(name: str, shot_id: int, candidate: int, kind: str, extension: str) -> str:
    """
    File name (relative to the output folder) of one candidate picture.

    kind is "full" for the picture as received or "crop" for the video-shaped crop.
    """
    return f"{name}_keyframes/shot_{shot_id:02d}_c{candidate}_{kind}.{extension}"


def new_manifest(source: str, provider: str, model: str, steps: int, aspect: str) -> dict:
    """An empty manifest."""
    return {
        "source": source,
        "provider": provider,
        "model": model,
        "steps": steps,
        "aspect": aspect,
        "shots": {},
    }


def load_manifest(path: Path, source: str, provider: str, model: str, steps: int, aspect: str) -> dict:
    """
    Load the saved manifest, or start an empty one.

    A saved manifest made with a different provider, model, step count or
    shape is not reused: its pictures would not match the current settings.
    """
    fresh = new_manifest(source, provider, model, steps, aspect)
    path = Path(path)

    if not path.is_file():
        return fresh

    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fresh

    if not isinstance(saved, dict) or not isinstance(saved.get("shots"), dict):
        return fresh

    same = all(saved.get(key) == fresh[key] for key in ("source", "provider", "model", "steps", "aspect"))

    return saved if same else fresh


def format_keyframe_report(manifest: dict, shot_ids: list[int]) -> str:
    """A readable summary of the candidates of the chosen shots."""
    lines = ["=" * 78, f"{'shot':>4}  {'cand':>4}  {'sec':>5}  {'size':>9}  {'crop':>9}  file", "=" * 78]
    blanks = 0
    total = 0

    for shot_id in shot_ids:
        entry = manifest["shots"].get(str(shot_id))

        if not entry:
            lines.append(f"{shot_id:>4}  (nothing generated)")
            continue

        for candidate in entry["candidates"]:
            total += 1
            blanks += int(candidate["blank"])
            full = "x".join(str(item) for item in candidate["full_size"])
            crop = "x".join(str(item) for item in candidate["crop_size"])
            flag = "  <-- BLANK" if candidate["blank"] else ""
            lines.append(
                f"{shot_id:>4}  {candidate['number']:>4}  {candidate['seconds']:>5.1f}  "
                f"{full:>9}  {crop:>9}  {candidate['crop']}{flag}"
            )

    lines.append("=" * 78)
    lines.append(f"{total} pictures, {blanks} blank.")
    return "\n".join(lines)