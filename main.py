"""
YouTube AI Studio

Main application entry point.

Usage:
    python main.py
        Generate a story, approve it, split it into timed shots, build and
        approve the story bible, then design every shot.

    python main.py --story output/my_story.txt
        Skip story generation and its approval gate: start from an already
        approved story.

    python main.py --story output/my_story.txt --use-saved-bible
        Use the saved (possibly hand-edited) story bible instead of
        generating a new one.

    python main.py --story output/my_story.txt --shots-only
        Only split the story into timed shots (no Ollama needed).

    Add --redesign to ignore saved shot designs and design every shot again.

    python main.py --story output/my_story.txt --use-saved-bible --image-prompts
        After the shot designs, build and save the image prompt of every
        shot for review. No images are generated.

    python main.py --story output/my_story.txt --use-saved-bible --keyframes 1,2,8
        Generate candidate pictures (IMAGE_CANDIDATES per shot) for the chosen
        shots: a list such as 1,2,8, a range such as 1-3, or all. Needs the
        Cloudflare settings in .env. Pictures already made are kept; add
        --regen-keyframes to make them again.
"""

import argparse
import json
from pathlib import Path

from core.config.settings import settings
from core.logging.logger import logger
from core.constants.constants import PROJECT_NAME
from core.constants.constants import PROJECT_VERSION
from core.config.validator import validate_configuration
from core.exceptions.exceptions import ProviderError
from core.exceptions.exceptions import ValidationError
from core.exceptions.exceptions import YouTubeAIStudioError
from core.utils.file_utils import make_filename
from core.utils.file_utils import write_text_atomic
from core.utils.image_prompt import build_all_prompts
from core.utils.image_prompt import format_prompt_report
from core.utils.image_prompt import prompts_document
from core.utils.keyframes import format_keyframe_report
from core.utils.keyframes import load_manifest
from core.utils.keyframes import parse_shot_selection
from core.utils.shot_design import format_design_table
from core.utils.shot_design import inputs_hash
from core.utils.shot_design import load_progress
from core.utils.shot_design import progress_document
from core.utils.shot_splitter import build_shot_plan
from core.utils.story_bible import extract_json
from core.utils.story_bible import format_bible
from core.utils.story_bible import load_bible_file
from core.utils.story_bible import validate_bible
from core.utils.text_metrics import count_words
from core.utils.text_metrics import estimate_narration_seconds
from core.workflow.approval_gate import ApprovalGate

from services.image.image_factory import create_image_provider
from services.llm.ollama_provider import OllamaProvider
from services.output.file_writer import FileWriter

from agents.keyframe_generator import KeyframeGenerator
from agents.shot_designer import ShotDesigner
from agents.story_bible_generator import StoryBibleGenerator
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
    parser.add_argument(
        "--shots-only",
        action="store_true",
        help="Stop after the shot plan (skips the story bible, so Ollama is not needed).",
    )
    parser.add_argument(
        "--use-saved-bible",
        action="store_true",
        help="Use the saved story bible file (validated) instead of generating one.",
    )
    parser.add_argument(
        "--redesign",
        action="store_true",
        help="Ignore saved shot designs and design every shot again.",
    )
    parser.add_argument(
        "--image-prompts",
        action="store_true",
        help="Build and save the image prompt of every shot for review (no images are generated).",
    )
    parser.add_argument(
        "--keyframes",
        metavar="SHOTS",
        help="Generate candidate pictures for these shots: a list (1,2,8), a range (1-3) or all.",
    )
    parser.add_argument(
        "--regen-keyframes",
        action="store_true",
        help="Make the chosen keyframes again even if pictures already exist.",
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


def run_shot_split(name: str, story: str, writer: FileWriter) -> dict:
    """
    Split the approved story into timed shots, save the shot plan and return it.
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

    return plan


def require_ollama() -> OllamaProvider:
    """
    Return an Ollama provider, or raise ProviderError if Ollama is not reachable.
    """
    provider = OllamaProvider()

    if not provider.is_available():
        raise ProviderError(
            f"Ollama is not reachable at {settings.ollama_host}. "
            "Start Ollama, or run with --shots-only."
        )

    return provider


def edit_bible_by_hand(bible: dict, draft_path: Path) -> dict:
    """
    Let the user fix the bible in a text editor.

    The bible is written to a draft file; after the user saves it, it is
    read back and validated. Returns the edited bible, or the original one
    if the user cancels.
    """
    write_text_atomic(draft_path, json.dumps(bible, indent=2, ensure_ascii=False))

    print(f"\nThe bible was written to:\n  {draft_path}")
    print("Open it in your editor, change what you need, and save it.")

    while True:
        answer = input("Press Enter when saved (or type C to cancel)\n> ").strip().lower()

        if answer in ("c", "cancel"):
            logger.info("Edit cancelled; keeping the previous bible.")
            return bible

        try:
            data = extract_json(draft_path.read_text(encoding="utf-8"))
            cleaned = validate_bible(data)
        except ValidationError as exc:
            print(f"\nThe edited file is not valid:\n{exc}")
            print("Fix it and save again.")
            continue
        except OSError as exc:
            print(f"\nCould not read the file: {exc}")
            continue

        art_style = " ".join(str(data.get("art_style") or "").split()) or bible["art_style"]

        logger.info("Edited bible accepted.")
        return {"art_style": art_style, **cleaned}


def run_bible_stage(name: str, story: str, writer: FileWriter) -> dict | None:
    """
    Generate the story bible and run it through approval gate 2.

    Returns the approved bible (saved to disk), or None if the user quit.

    Raises:
        ProviderError: if Ollama is not reachable.
        AgentError: if no valid bible could be produced.
    """
    provider = require_ollama()

    generator = StoryBibleGenerator(provider)
    gate = ApprovalGate("Story Bible (characters, locations, props)")
    draft_path = writer.path(make_filename(name, suffix="_bible_draft", extension=".json"))

    bible = generator.generate(story)

    while True:
        decision = gate.review(format_bible(bible))

        if decision == "approve":
            break

        if decision == "regenerate":
            logger.info("Regenerating story bible.")
            bible = generator.generate(story)
            continue

        if decision == "edit":
            bible = edit_bible_by_hand(bible, draft_path)
            continue

        if decision == "quit":
            logger.info("Story bible stage aborted by user before approval. Nothing saved.")
            return None

    bible_path = writer.save(
        make_filename(name, suffix="_bible", extension=".json"),
        json.dumps(bible, indent=2, ensure_ascii=False),
    )

    if draft_path.exists():
        draft_path.unlink()

    logger.info("=" * 60)
    logger.info("✅ Story bible approved and saved.")
    logger.info("📄 Saved to:")
    logger.info(bible_path)
    logger.info("=" * 60)

    return bible


def load_saved_bible(name: str, writer: FileWriter) -> dict:
    """
    Load and validate the saved story bible for this story.

    Raises:
        ValidationError: if it is missing or invalid.
    """
    path = writer.path(make_filename(name, suffix="_bible", extension=".json"))

    if not path.is_file():
        raise ValidationError(
            f"No saved story bible at {path}. Run once without --use-saved-bible."
        )

    bible = load_bible_file(path)

    logger.info("Using saved story bible: %s", path)
    return bible


def run_shot_design(
    name: str,
    plan: dict,
    bible: dict,
    writer: FileWriter,
    redesign: bool,
) -> list[dict]:
    """
    Design every shot (resuming from saved progress), save the designs and return them.

    Raises:
        ProviderError: if Ollama is not reachable.
        AgentError: if a shot cannot be designed (earlier shots stay saved).
    """
    shots = plan["shots"]
    filename = make_filename(name, suffix="_designs", extension=".json")
    designs_path = writer.path(filename)
    fingerprint = inputs_hash(shots, bible)

    if redesign:
        logger.info("--redesign given: ignoring saved shot designs.")
        entries, complete = [], False
    else:
        entries, complete = load_progress(designs_path, fingerprint, len(shots))

    if complete:
        logger.info("All %d shots are already designed (use --redesign to redo them).", len(shots))
    else:
        designer = ShotDesigner(require_ollama())

        def save_progress(current: list[dict]) -> None:
            writer.save(
                filename,
                json.dumps(
                    progress_document(name, fingerprint, current, complete=False),
                    indent=2,
                    ensure_ascii=False,
                ),
            )

        entries = designer.design_all(bible, shots, entries, on_progress=save_progress)

        writer.save(
            filename,
            json.dumps(
                progress_document(name, fingerprint, entries, complete=True),
                indent=2,
                ensure_ascii=False,
            ),
        )

    print("\n" + format_design_table(entries))

    logger.info("📄 Shot designs saved to:")
    logger.info(designs_path)

    return entries


def run_image_prompts(
    name: str,
    bible: dict,
    entries: list[dict],
    writer: FileWriter,
    show: bool = True,
) -> list[dict]:
    """
    Build the image prompt of every shot, save them, return them (and show them for review if asked).

    Raises:
        ValidationError: if a shot refers to something missing from the bible,
            or a prompt cannot be shortened enough.
    """
    max_chars = settings.image_prompt_max_chars
    items = build_all_prompts(bible, entries, max_chars)

    path = writer.save(
        make_filename(name, suffix="_prompts", extension=".json"),
        json.dumps(prompts_document(name, max_chars, items), indent=2, ensure_ascii=False),
    )

    if show:
        print("\n" + format_prompt_report(items, max_chars))

    logger.info("📄 Image prompts saved to:")
    logger.info(path)

    return items


def run_keyframes(
    name: str,
    items: list[dict],
    writer: FileWriter,
    selection: str,
    regenerate: bool,
) -> None:
    """
    Generate candidate pictures for the chosen shots and save a contact sheet.

    Raises:
        ValidationError: if the shot selection is invalid.
        ConfigurationError: if the image provider is not configured.
        ImageGenerationError: if the service fails (pictures made so far stay saved).
    """
    shot_ids = parse_shot_selection(selection, [item["shot_id"] for item in items])

    provider = create_image_provider()
    provider.initialize()

    aspect = (settings.video_width, settings.video_height)
    manifest_name = make_filename(name, suffix="_keyframes", extension=".json")

    if regenerate:
        logger.info("--regen-keyframes given: making the chosen pictures again.")

    manifest = load_manifest(
        writer.path(manifest_name),
        name,
        provider.provider_name,
        provider.model_name,
        settings.image_steps,
        f"{aspect[0]}:{aspect[1]}",
    )

    logger.info(
        "Keyframes: shots %s, up to %d candidates each (%s, %s, %d steps).",
        ", ".join(str(item) for item in shot_ids),
        settings.image_candidates,
        provider.provider_name,
        provider.model_name,
        settings.image_steps,
    )

    def save_manifest(current: dict) -> None:
        writer.save(manifest_name, json.dumps(current, indent=2, ensure_ascii=False))

    generator = KeyframeGenerator(
        provider, writer, settings.image_steps, aspect, settings.image_crop_bias
    )

    try:
        generator.generate(
            name,
            items,
            shot_ids,
            settings.image_candidates,
            manifest,
            regenerate=regenerate,
            on_progress=save_manifest,
        )
    finally:
        if manifest["shots"]:
            save_manifest(manifest)

    print("\n" + format_keyframe_report(manifest, shot_ids))

    sheet_path = generator.make_contact_sheet(name, manifest, shot_ids)

    logger.info("📄 Keyframe list saved to:")
    logger.info(writer.path(manifest_name))

    if sheet_path:
        logger.info("🖼️ Contact sheet (all candidates on one picture):")
        logger.info(sheet_path)


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

        plan = run_shot_split(name, story, writer)

        if args.shots_only:
            logger.info("--shots-only given: stopping after the shot plan.")
        else:
            if args.use_saved_bible:
                bible = load_saved_bible(name, writer)
            else:
                bible = run_bible_stage(name, story, writer)

            if bible is None:
                return

            entries = run_shot_design(name, plan, bible, writer, args.redesign)

            if args.image_prompts or args.keyframes:
                items = run_image_prompts(name, bible, entries, writer, show=args.image_prompts)

                if args.keyframes:
                    run_keyframes(name, items, writer, args.keyframes, args.regen_keyframes)

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