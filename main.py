"""
YouTube AI Studio

Main application entry point.

Usage:
    python main.py
        Generate a story, approve it, then split it into timed shots.

    python main.py --story output/my_story.txt
        Skip generation and the approval gate: split an already approved
        story into timed shots.
"""

import argparse
import json
from pathlib import Path

from core.config.settings import settings
from core.logging.logger import logger
from core.constants.constants import PROJECT_NAME
from core.constants.constants import PROJECT_VERSION
from core.config.validator import validate_configuration
from core.exceptions.exceptions import ValidationError
from core.exceptions.exceptions import YouTubeAIStudioError
from core.utils.file_utils import make_filename
from core.utils.shot_splitter import build_shot_plan
from core.utils.text_metrics import count_words
from core.utils.text_metrics import estimate_narration_seconds
from core.workflow.approval_gate import ApprovalGate

from services.llm.ollama_provider import OllamaProvider
from services.output.file_writer import FileWriter

from agents.story_generator import StoryGenerator


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description=PROJECT_NAME)
    parser.add_argument(
        "--story",
        metavar="PATH",
        help="Path to an already approved story (.txt). "
        "Skips story generation and the approval gate.",
    )
    return parser.parse_args()


def report_length(label: str, text: str) -> None:
    """
    Log word count and estimated narration time against the target range.
    """
    words = count_words(text)
    seconds = estimate_narration_seconds(text, settings.narration_wpm)

    logger.info(
        "%s length: %d words (~%d s narrated, target %d-%d s)",
        label,
        words,
        seconds,
        settings.target_min_seconds,
        settings.target_max_seconds,
    )

    if seconds < settings.target_min_seconds:
        logger.warning(
            "%s is shorter than the target (%d s < %d s).",
            label,
            seconds,
            settings.target_min_seconds,
        )
    elif seconds > settings.target_max_seconds:
        logger.warning(
            "%s is longer than the target (%d s > %d s).",
            label,
            seconds,
            settings.target_max_seconds,
        )


def generate_story(story_generator: StoryGenerator, topic: str) -> str:
    """
    Generate a story and report its length.
    """
    story = story_generator.generate(topic)
    report_length("Story", story)
    return story


def load_story_file(path_text: str) -> tuple[str, str]:
    """
    Load an already approved story. Returns (name, story text).
    """
    path = Path(path_text)

    if not path.is_file():
        raise ValidationError(f"Story file not found: {path}")

    story = path.read_text(encoding="utf-8").strip()

    if not story:
        raise ValidationError(f"Story file is empty: {path}")

    return path.stem, story


def generate_approved_story(writer: FileWriter) -> tuple[str, str] | None:
    """
    Generate a story and run it through approval gate 1.

    Returns (topic, story) once approved and saved, or None if the user
    quit or gave no topic.
    """
    provider = OllamaProvider()

    logger.info("Provider        : %s", provider.provider_name)
    logger.info("Ollama Running  : %s", provider.is_available())
    logger.info("Installed Models:")

    for model in provider.list_models():
        logger.info(" - %s", model)

    logger.info("Using Model     : %s", settings.default_model)

    story_generator = StoryGenerator()
    gate = ApprovalGate("Story")

    topic = input("\nEnter Story Topic:\n> ").strip()

    if not topic:
        logger.error("Topic cannot be empty. Aborting.")
        return None

    story = generate_story(story_generator, topic)

    # Approval gate 1: Story (per automation policy — do not proceed
    # until a human approves the story).
    while True:
        decision = gate.review(story)

        if decision == "approve":
            break

        if decision == "regenerate":
            logger.info("Regenerating story for topic: %s", topic)
            story = generate_story(story_generator, topic)
            continue

        if decision == "edit":
            new_topic = input("\nEnter new Story Topic:\n> ").strip()
            if not new_topic:
                logger.warning("Empty topic entered — keeping previous topic.")
                continue
            topic = new_topic
            logger.info("Generating story for new topic: %s", topic)
            story = generate_story(story_generator, topic)
            continue

        if decision == "quit":
            logger.info("Story stage aborted by user before approval. Nothing saved.")
            return None

    # Only save once the story has passed the approval gate.
    # The approved story is the narration script: later stages read this file.
    story_path = writer.save(make_filename(topic), story)

    logger.info("=" * 60)
    logger.info("✅ Story approved and saved (this is the narration script).")
    logger.info("📄 Saved to:")
    logger.info(story_path)
    logger.info("=" * 60)

    return topic, story


def run_shot_split(name: str, story: str, writer: FileWriter) -> None:
    """
    Split the approved story into timed shots and save the shot plan.
    """
    plan = build_shot_plan(
        source=name,
        text=story,
        words_per_minute=settings.narration_wpm,
        target_shot_seconds=settings.shot_target_seconds,
        min_shots=settings.shot_min_count,
        max_shots=settings.shot_max_count,
    )

    plan_path = writer.save(
        make_filename(name, suffix="_shots", extension=".json"),
        json.dumps(plan, indent=2, ensure_ascii=False),
    )

    durations = [shot["duration_seconds"] for shot in plan["shots"]]

    logger.info(
        "Shot plan: %d shots, %.0f s total (shot length min %.1f / avg %.1f / max %.1f s)",
        plan["shot_count"],
        plan["total_seconds"],
        min(durations),
        sum(durations) / len(durations),
        max(durations),
    )

    print("\n" + "=" * 78)
    print(f"{'#':>3}  {'start':>6}  {'dur':>5}  {'words':>5}  narration")
    print("=" * 78)

    for shot in plan["shots"]:
        preview = shot["narration"]
        if len(preview) > 52:
            preview = preview[:49] + "..."

        print(
            f"{shot['id']:>3}  {shot['start_seconds']:>5.1f}s  "
            f"{shot['duration_seconds']:>4.1f}s  {shot['words']:>5}  {preview}"
        )

    print("=" * 78)

    logger.info("📄 Shot plan saved to:")
    logger.info(plan_path)


def main():
    """Start the application."""
    args = parse_args()

    validate_configuration()

    print("=" * 60)
    print(f"🚀 Welcome to {PROJECT_NAME}")
    print("=" * 60)

    logger.info("Application Started")

    writer = FileWriter()

    try:
        if args.story:
            name, story = load_story_file(args.story)
            logger.info("Using approved story from file: %s", args.story)
            report_length("Story", story)
        else:
            result = generate_approved_story(writer)

            if result is None:
                return

            name, story = result

        run_shot_split(name, story, writer)

    except YouTubeAIStudioError as exc:
        logger.error("%s", exc)
        raise SystemExit(1)

    print(f"Version        : {PROJECT_VERSION}")
    print(f"Ollama Host    : {settings.ollama_host}")
    print(f"Default Model  : {settings.default_model}")
    print(f"Coder Model    : {settings.coder_model}")
    print(f"Log Level      : {settings.log_level}")
    print(f"Output Folder  : {settings.output_dir}")
    print(f"Data Folder    : {settings.data_dir}")
    print(f"Debug Mode     : {settings.debug}")


if __name__ == "__main__":
    main()