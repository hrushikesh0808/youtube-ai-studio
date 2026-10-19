"""
File Writer

Responsible for saving generated content.
"""

from pathlib import Path

from core.config.settings import settings
from core.utils.file_utils import write_bytes_atomic
from core.utils.file_utils import write_text_atomic


class FileWriter:
    """Save generated files into the configured output folder."""

    def __init__(self, output_dir: str | None = None):
        self.output_dir = Path(output_dir or settings.output_dir)

    def path(self, filename: str) -> Path:
        """Return the full path a file of this name is saved to."""
        return self.output_dir / filename

    def save(self, filename: str, content: str) -> Path:
        """Save a text file atomically and return its path."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        file_path = self.path(filename)

        write_text_atomic(file_path, content)

        return file_path

    def save_bytes(self, filename: str, data: bytes) -> Path:
        """Save a binary file (for example a picture) atomically and return its path."""
        file_path = self.path(filename)

        write_bytes_atomic(file_path, data)

        return file_path
