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

    if settings.narration_wpm <= 0:
        raise ConfigurationError(
            "NARRATION_WPM must be greater than 0."
        )

    if settings.target_min_seconds <= 0:
        raise ConfigurationError(
            "TARGET_MIN_SECONDS must be greater than 0."
        )

    if settings.target_min_seconds > settings.target_max_seconds:
        raise ConfigurationError(
            "TARGET_MIN_SECONDS cannot be greater than TARGET_MAX_SECONDS."
        )

    if settings.shot_target_seconds <= 0:
        raise ConfigurationError(
            "SHOT_TARGET_SECONDS must be greater than 0."
        )

    if settings.shot_min_count <= 0:
        raise ConfigurationError(
            "SHOT_MIN_COUNT must be greater than 0."
        )

    if settings.shot_min_count > settings.shot_max_count:
        raise ConfigurationError(
            "SHOT_MIN_COUNT cannot be greater than SHOT_MAX_COUNT."
        )
        
    if not settings.art_style.strip():
        raise ConfigurationError(
            "ART_STYLE cannot be empty."
        )

    if settings.bible_max_attempts <= 0:
        raise ConfigurationError(
            "BIBLE_MAX_ATTEMPTS must be greater than 0."
        )
        
    
    if settings.shot_design_max_attempts <= 0:
        raise ConfigurationError(
            "SHOT_DESIGN_MAX_ATTEMPTS must be greater than 0."
        )

    if not 200 <= settings.image_prompt_max_chars <= 2048:
        raise ConfigurationError(
            "IMAGE_PROMPT_MAX_CHARS must be between 200 and 2048."
        )

    if settings.image_provider not in ("cloudflare",):
        raise ConfigurationError(
            "IMAGE_PROVIDER must be: cloudflare."
        )

    if not 1 <= settings.image_steps <= 8:
        raise ConfigurationError(
            "IMAGE_STEPS must be between 1 and 8."
        )

    if not 1 <= settings.image_candidates <= 6:
        raise ConfigurationError(
            "IMAGE_CANDIDATES must be between 1 and 6."
        )

    if settings.video_width <= 0 or settings.video_height <= 0:
        raise ConfigurationError(
            "VIDEO_WIDTH and VIDEO_HEIGHT must be greater than 0."
        )

    if not 0.0 <= settings.image_crop_bias <= 1.0:
        raise ConfigurationError(
            "IMAGE_CROP_BIAS must be between 0 and 1 (0 = top, 1 = bottom)."
        )