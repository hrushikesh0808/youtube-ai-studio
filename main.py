"""
YouTube AI Studio

Main application entry point.
"""

from core.config.settings import settings
from core.logging.logger import logger
from core.constants.constants import PROJECT_NAME
from core.constants.constants import PROJECT_VERSION
from core.config.validator import validate_configuration
from core.utils.file_utils import make_filename
from core.utils.text_metrics import count_words
from core.utils.text_metrics import estimate_narration_seconds
from core.workflow.approval_gate import ApprovalGate

from services.llm.ollama_provider import OllamaProvider
from services.output.file_writer import FileWriter

from agents.story_generator import StoryGenerator


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


def main():
    """Start the application."""
    validate_configuration()

    print("=" * 60)
    print(f"🚀 Welcome to {PROJECT_NAME}")
    print("=" * 60)

    logger.info("Application Started")

    provider = OllamaProvider()

    logger.info("Provider        : %s", provider.provider_name)
    logger.info("Ollama Running  : %s", provider.is_available())
    logger.info("Installed Models:")

    for model in provider.list_models():
        logger.info(" - %s", model)

    logger.info("Using Model     : %s", settings.default_model)

    story_generator = StoryGenerator()
    writer = FileWriter()
    gate = ApprovalGate("Story")

    topic = input("\nEnter Story Topic:\n> ").strip()

    if not topic:
        logger.error("Topic cannot be empty. Aborting.")
        return

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
            return

    # Only save once the story has passed the approval gate.
    # The approved story is the narration script: later stages read this file.
    story_path = writer.save(make_filename(topic), story)

    logger.info("=" * 60)
    logger.info("✅ Story approved and saved (this is the narration script).")
    logger.info("📄 Saved to:")
    logger.info(story_path)
    logger.info("=" * 60)

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