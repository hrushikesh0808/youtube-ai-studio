
"""
Shot Designer Agent

Designs what the camera shows in every shot, one shot at a time, using
only the characters, locations and props of the approved story bible.
"""

import time

from core.config.settings import settings
from core.exceptions.exceptions import AgentError
from core.exceptions.exceptions import ValidationError
from core.logging.logger import logger
from core.utils.shot_design import CAMERAS
from core.utils.shot_design import MOTIONS
from core.utils.shot_design import build_shot_schema
from core.utils.shot_design import design_warnings
from core.utils.shot_design import validate_shot_design
from core.utils.story_bible import extract_json
from prompts.prompt_loader import PromptLoader
from services.llm.ollama_provider import OllamaProvider


def bible_prompt_text(bible: dict) -> str:
    """Render the bible as the compact reference list used in the prompt."""
    lines = []

    for key, title in (
        ("characters", "Characters"),
        ("locations", "Locations"),
        ("props", "Props"),
    ):
        items = bible.get(key, [])
        if not items:
            continue

        lines.append(f"{title}:")
        for item in items:
            lines.append(f"- {item['id']} ({item['name']}): {item['visual']}")

    return "\n".join(lines)


def previous_shot_text(previous: dict | None) -> str:
    """Describe the previous shot's design for continuity."""
    if previous is None:
        return "none, this is the first shot."

    parts = [
        f"location {previous['location']}",
        f"characters {', '.join(previous['characters']) or 'none'}",
        f"props {', '.join(previous['props']) or 'none'}",
        f"camera {previous['camera']}",
        f"motion {previous['motion']}",
    ]
    if previous["state_notes"]:
        parts.append(f"state: {previous['state_notes']}")

    return "; ".join(parts) + f". Action: {previous['action']}"


class ShotDesigner:
    """
    Designs shots one at a time.

    Each answer is checked against the bible; an invalid answer is retried a
    bounded number of times with the problems sent back to the model.
    """

    def __init__(self, provider=None):
        self.provider = provider or OllamaProvider()
        self.prompt_loader = PromptLoader()

    def design_shot(
        self,
        bible: dict,
        shot: dict,
        shot_count: int,
        previous: dict | None,
        next_shot: dict | None,
    ) -> dict:
        """
        Return a validated design for one shot.

        Raises:
            AgentError: if no valid design is produced within the allowed attempts.
        """
        attempts = settings.shot_design_max_attempts
        schema = build_shot_schema(bible)

        base_prompt = self.prompt_loader.load(
            "shot.txt",
            bible_text=bible_prompt_text(bible),
            shot_id=shot["id"],
            shot_count=shot_count,
            narration=shot["narration"],
            previous=previous_shot_text(previous),
            next_narration=next_shot["narration"] if next_shot else "none, this is the last shot.",
            cameras=", ".join(CAMERAS),
            motions=", ".join(MOTIONS),
        )
        prompt = base_prompt
        last_problem = ""

        for attempt in range(1, attempts + 1):
            raw = self.provider.generate(prompt, json_schema=schema)

            try:
                design = validate_shot_design(extract_json(raw), bible)
            except ValidationError as exc:
                last_problem = str(exc)
                logger.warning(
                    "Shot %d attempt %d of %d rejected: %s",
                    shot["id"], attempt, attempts, last_problem,
                )
                prompt = (
                    base_prompt
                    + "\n\nYour previous answer was rejected for these reasons:\n"
                    + last_problem
                    + "\n\nReturn the complete corrected JSON object only."
                )
                continue

            for warning in design_warnings(design, shot["narration"], bible):
                logger.warning("Shot %d: %s", shot["id"], warning)

            return design

        raise AgentError(
            f"Shot {shot['id']} could not be designed after {attempts} attempts. "
            f"Last problem: {last_problem}"
        )

    def design_all(
        self,
        bible: dict,
        shots: list[dict],
        entries: list[dict],
        on_progress=None,
    ) -> list[dict]:
        """
        Design every shot that has no saved design yet and return all entries.

        `entries` are designs from an earlier run (resume). `on_progress` is
        called with the full entry list after each newly designed shot, so
        the caller can save it; a failure on shot N therefore keeps shots 1..N-1.

        Raises:
            AgentError: if a shot cannot be designed.
        """
        by_id = {entry["shot_id"]: entry for entry in entries}
        total = len(shots)

        if by_id:
            logger.info("Resuming: %d of %d shots already designed.", len(by_id), total)

        for index, shot in enumerate(shots):
            if shot["id"] in by_id:
                continue

            previous_entry = by_id.get(shots[index - 1]["id"]) if index > 0 else None
            next_shot = shots[index + 1] if index + 1 < total else None

            started = time.monotonic()
            design = self.design_shot(
                bible,
                shot,
                total,
                previous_entry["design"] if previous_entry else None,
                next_shot,
            )

            by_id[shot["id"]] = {
                "shot_id": shot["id"],
                "narration": shot["narration"],
                "start_seconds": shot["start_seconds"],
                "duration_seconds": shot["duration_seconds"],
                "design": design,
            }

            logger.info(
                "Shot %d/%d designed in %.1f s (%s, %s, %s)",
                shot["id"], total, time.monotonic() - started,
                design["location"], design["camera"], design["motion"],
            )

            if on_progress is not None:
                on_progress([by_id[key] for key in sorted(by_id)])

        return [by_id[key] for key in sorted(by_id)]

