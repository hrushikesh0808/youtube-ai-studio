"""
YouTube AI Studio

Main application entry point.
"""

from core.base import provider
from core.config.settings import settings
from core.logging.logger import logger
from core.constants.constants import PROJECT_NAME
from core.constants.constants import PROJECT_VERSION
from core.config.validator import validate_configuration
from services.llm.ollama_provider import OllamaProvider

def main():
    """Start the application."""
    validate_configuration()
    logger.info("Application Started")
    
    provider = OllamaProvider()

    logger.info("Provider        : %s", provider.provider_name)
    logger.info("Ollama Running  : %s", provider.is_available())
    logger.info("Installed Models:")

    for model in provider.list_models():
        logger.info(" - %s", model)

    logger.info("Using Model     : %s", settings.default_model)

    ai_response = provider.generate(
        "Write one motivational sentence for YouTube creators."
    )

    logger.info("AI Response:")
    logger.info(ai_response)

    print("=" * 60)
    print(f"🚀 Welcome to {PROJECT_NAME}")
    print("=" * 60)

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