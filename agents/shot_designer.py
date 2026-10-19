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
from core.utils.shot_design import RECENT_SHOTS_AVOIDED
from core.utils.shot_design import location_hint
from core.utils.shot_design import opening
from core.utils.shot_design import shot_allowances
from core.utils.shot_design import unnamed_props
from core.utils.shot_design import validate_shot_design
from core.utils.story_bible import extract_json
from prompts.prompt_loader import PromptLoader
from services.llm.ollama_provider import OllamaProvider


def bible_prompt_text(bible: dict, allowed: dict[str, list[str]] | None = None) -> str:
    """
    Render the reference list used in the prompt for one shot.

    Locations are listed by name only: their descriptions mention objects
    (for example a lamp) that the model would otherwise copy into the action.
    Characters and props are listed only when `allowed` includes them, so the
    model is never shown something this shot may not use. Their full
    appearance is added to the image prompts later.
    """
    lines = []

    for key, title in (
        ("characters", "Characters"),
        ("locations", "Locations"),
        ("props", "Props"),
    ):
        items = bible.get(key, [])
        if key in ("characters", "props") and allowed is not None:
            items = [item for item in items if item["id"] in allowed[key]]

        if not items:
            continue

        lines.append(f"{title}:")
        for item in items:
            lines.append(f"- {item['id']} ({item['name']})")

    return "\n".join(lines)


def previous_shot_text(previous: dict | None) -> str:
    """
    Describe the previous shot for continuity.

    Only where it was and how it was filmed: its characters, props and notes
    are left out because the model copies them into a shot that must not
    show them.
    """
    if previous is None:
        return "none, this is the first shot."

    return (
        f"location {previous['location']}; camera {previous['camera']}; "
        f"motion {previous['motion']}."
    )


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
        recent: list[dict],
        next_shot: dict | None,
    ) -> dict:
        """
        Return a validated design for one shot.

        `recent` holds the designs of the previous shots, nearest first. The
        narration decides which characters and props may appear, and the
        cameras and motions of the last shots may not be reused.

        Raises:
            AgentError: if no valid design is produced within the allowed attempts.
        """
        attempts = settings.shot_design_max_attempts
        previous = recent[0] if recent else None
        allowed = shot_allowances(bible, shot["narration"], previous)
        avoid_camera = [d["camera"] for d in recent[:RECENT_SHOTS_AVOIDED]]
        avoid_motion = [d["motion"] for d in recent[:RECENT_SHOTS_AVOIDED]]
        avoid_openings = list(dict.fromkeys(opening(d["action"]) for d in recent[:RECENT_SHOTS_AVOIDED]))
        schema = build_shot_schema(
            bible, allowed["characters"], allowed["props"], avoid_camera, avoid_motion
        )

        def avoid_clause(values):
            if not values:
                return ""
            return " It must not be " + " or ".join(dict.fromkeys(values)) + " (used in the last shots)."

        base_prompt = self.prompt_loader.load(
            "shot.txt",
            bible_text=bible_prompt_text(bible, allowed),
            shot_id=shot["id"],
            shot_count=shot_count,
            narration=shot["narration"],
            previous=previous_shot_text(previous),
            next_narration=next_shot["narration"] if next_shot else "none, this is the last shot.",
            cameras=", ".join(c for c in CAMERAS if c not in avoid_camera),
            motions=", ".join(m for m in MOTIONS if m not in avoid_motion),
            allowed_characters=", ".join(allowed["characters"]) or "(nobody: leave the list out)",
            allowed_props=", ".join(allowed["props"]) or "(none: leave the list out)",
            avoid_camera=avoid_clause(avoid_camera),
            avoid_motion=avoid_clause(avoid_motion),
            avoid_opening=(
                " Do not start it with: " + "; ".join(f'"{o}"' for o in avoid_openings) + " (recent shots did)."
                if avoid_openings
                else ""
            ),
            location_hint=location_hint(bible, shot["narration"]),
            people_rule=(
                ""
                if allowed["characters"]
                else " Nobody is in this shot: do not mention any person or character name."
            ),
        )
        prompt = base_prompt
        last_problem = ""

        for attempt in range(1, attempts + 1):
            raw = self.provider.generate(prompt, json_schema=schema)

            try:
                design = validate_shot_design(
                    extract_json(raw),
                    bible,
                    allowed["characters"],
                    allowed["props"],
                    avoid_camera,
                    avoid_motion,
                    avoid_openings,
                )
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

        for name in unnamed_props(bible, shots):
            logger.warning(
                "Prop '%s' is never named in the narration, so no shot will show it.", name
            )

        if by_id:
            logger.info("Resuming: %d of %d shots already designed.", len(by_id), total)

        for index, shot in enumerate(shots):
            if shot["id"] in by_id:
                continue

            recent = [
                by_id[shots[index - back]["id"]]["design"]
                for back in (1, 2)
                if index - back >= 0 and shots[index - back]["id"] in by_id
            ]
            next_shot = shots[index + 1] if index + 1 < total else None

            started = time.monotonic()
            design = self.design_shot(bible, shot, total, recent, next_shot)

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