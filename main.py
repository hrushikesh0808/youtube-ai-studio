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

from services.llm.ollama_provider import OllamaProvider
from services.output.file_writer import FileWriter

from agents.story_generator import StoryGenerator

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

    topic = input("\nEnter Story Topic:\n> ").strip()

    story = story_generator.generate(topic)

    # Save story to file

    writer = FileWriter()

    filename = make_filename(topic)

    file_path = writer.save(
        filename,
        story,
    )


    logger.info("=" * 60)
    logger.info("✅ Story generated successfully.")
    logger.info("📄 Saved to:")
    logger.info(file_path)
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