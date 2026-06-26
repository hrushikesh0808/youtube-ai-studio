"""
Application Configuration Manager

Loads configuration from the .env file.
"""

from pathlib import Path
import os

from dotenv import load_dotenv


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

        # Debug
        self.debug: bool = (
            os.getenv("DEBUG", "False").lower() == "true"
        )
        
# Global settings instance
settings = Settings()