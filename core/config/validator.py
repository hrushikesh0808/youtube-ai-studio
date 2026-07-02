"""
Configuration Validator

Validates application configuration before startup.
"""

from core.config.settings import settings
from core.exceptions.exceptions import ConfigurationError


def validate_configuration() -> None:
    """
    Validate all application configuration values.

    Raises:
        ConfigurationError:
            If any required configuration is invalid.
    """

    if not settings.app_name.strip():
        raise ConfigurationError(
            "APP_NAME cannot be empty."
        )

    if not settings.app_version.strip():
        raise ConfigurationError(
            "APP_VERSION cannot be empty."
        )

    if not settings.ollama_host.startswith("http"):
        raise ConfigurationError(
            "OLLAMA_HOST must start with http or https."
        )

    if not settings.default_model.strip():
        raise ConfigurationError(
            "DEFAULT_MODEL cannot be empty."
        )

    if not settings.coder_model.strip():
        raise ConfigurationError(
            "CODER_MODEL cannot be empty."
        )

    if not settings.output_dir.strip():
        raise ConfigurationError(
            "OUTPUT_DIR cannot be empty."
        )

    if not settings.data_dir.strip():
        raise ConfigurationError(
            "DATA_DIR cannot be empty."
        )