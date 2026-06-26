"""
YouTube AI Studio

Main application entry point.
"""

from core.config.settings import settings


def main():
    """Start the application."""

    print("=" * 60)
    print(f"🚀 Welcome to {settings.app_name}")
    print("=" * 60)

    print(f"Version        : {settings.app_version}")
    print(f"Ollama Host    : {settings.ollama_host}")
    print(f"Default Model  : {settings.default_model}")
    print(f"Coder Model    : {settings.coder_model}")
    print(f"Log Level      : {settings.log_level}")
    print(f"Output Folder  : {settings.output_dir}")
    print(f"Data Folder    : {settings.data_dir}")
    print(f"Debug Mode     : {settings.debug}")


if __name__ == "__main__":
    main()