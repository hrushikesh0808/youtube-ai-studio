"""
Application Configuration Manager

Loads configuration from the .env file.
"""

from pathlib import Path
import os

from dotenv import load_dotenv

from core.exceptions.exceptions import ConfigurationError


def _get_int(name: str, default: int) -> int:
    """
    Read an integer environment variable with a clear error on bad input.
    """
    raw = os.getenv(name, str(default)).strip()

    try:
        return int(raw)
    except ValueError:
        raise ConfigurationError(
            f"{name} must be a whole number, got '{raw}'."
        )


def _get_optional_float(name: str) -> float | None:
    """
    Read an optional float environment variable (blank or missing = None).
    """
    raw = os.getenv(name, "").strip()

    if not raw:
        return None

    try:
        return float(raw)
    except ValueError:
        raise ConfigurationError(
            f"{name} must be a number, got '{raw}'."
        )


class Settings:
    """Application configuration."""

    def __init__(self):
        # Project root directory
        self.project_root = Path(__file__).resolve().parents[2]

        # Load environment variables
        load_dotenv(self.project_root / ".env")

        # Application
        self.app_name: str = os.getenv(
            "APP_NAME",
            "YouTube AI Studio"
        )

        self.app_version: str = os.getenv(
            "APP_VERSION",
            "0.1.0"
        )

        # Ollama
        self.ollama_host: str = os.getenv(
            "OLLAMA_HOST",
            "http://localhost:11434"
        )

        self.default_model: str = os.getenv(
            "DEFAULT_MODEL",
            "gemma3:4b"
        )

        self.coder_model: str = os.getenv(
            "CODER_MODEL",
            "qwen2.5-coder"
        )

        # Optional sampling temperature (blank = use the model's default)
        self.ollama_temperature: float | None = _get_optional_float(
            "OLLAMA_TEMPERATURE"
        )

        # Logging
        self.log_level: str = os.getenv(
            "LOG_LEVEL",
            "INFO"
        )

        # Directories
        self.output_dir: str = os.getenv(
            "OUTPUT_DIR",
            "output"
        )

        self.data_dir: str = os.getenv(
            "DATA_DIR",
            "data"
        )

        # Narration / target video length
        self.narration_wpm: int = _get_int("NARRATION_WPM", 150)
        self.target_min_seconds: int = _get_int("TARGET_MIN_SECONDS", 120)
        self.target_max_seconds: int = _get_int("TARGET_MAX_SECONDS", 180)

        # Shot planning
        self.shot_target_seconds: int = _get_int("SHOT_TARGET_SECONDS", 8)
        self.shot_min_count: int = _get_int("SHOT_MIN_COUNT", 15)
        self.shot_max_count: int = _get_int("SHOT_MAX_COUNT", 24)

        # Debug
        self.debug: bool = (
            os.getenv("DEBUG", "False").lower() == "true"
        )

    @property
    def target_min_words(self) -> int:
        """Minimum story/script length in words for the target duration."""
        return self.target_min_seconds * self.narration_wpm // 60

    @property
    def target_max_words(self) -> int:
        """Maximum story/script length in words for the target duration."""
        return self.target_max_seconds * self.narration_wpm // 60


# Global settings instance
settings = Settings()