"""
Application Constants

Contains project-wide constants used throughout the application.
"""

from pathlib import Path

# ==========================================================
# Project Information
# ==========================================================

PROJECT_NAME = "YouTube AI Studio"
PROJECT_VERSION = "1.0.0"

# ==========================================================
# Directory Names
# ==========================================================

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
LOG_DIR = Path("logs")
ASSETS_DIR = Path("assets")
PROMPTS_DIR = Path("prompts")

# ==========================================================
# Output Directories
# ==========================================================

OUTPUT_SCRIPT_DIR = OUTPUT_DIR / "scripts"
OUTPUT_IMAGE_DIR = OUTPUT_DIR / "images"
OUTPUT_VIDEO_DIR = OUTPUT_DIR / "videos"
OUTPUT_AUDIO_DIR = OUTPUT_DIR / "voice"
OUTPUT_SUBTITLE_DIR = OUTPUT_DIR / "subtitles"
OUTPUT_FINAL_DIR = OUTPUT_DIR / "final"

# ==========================================================
# Asset Directories
# ==========================================================

ASSET_IMAGE_DIR = ASSETS_DIR / "images"
ASSET_VIDEO_DIR = ASSETS_DIR / "video"
ASSET_AUDIO_DIR = ASSETS_DIR / "audio"
ASSET_MUSIC_DIR = ASSETS_DIR / "music"
ASSET_SFX_DIR = ASSETS_DIR / "sfx"

# ==========================================================
# Supported File Extensions
# ==========================================================

IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".webp"
)

VIDEO_EXTENSIONS = (
    ".mp4",
    ".mov",
    ".mkv"
)

AUDIO_EXTENSIONS = (
    ".wav",
    ".mp3",
    ".ogg"
)

TEXT_EXTENSIONS = (
    ".txt",
    ".md",
    ".json"
)