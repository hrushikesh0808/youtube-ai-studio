"""
File Writer

Responsible for saving generated content.
"""

from pathlib import Path


class FileWriter:
    """Save generated files."""

    def save(self, filename: str, content: str) -> Path:
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        file_path = output_dir / filename

        file_path.write_text(
            content,
            encoding="utf-8",
        )

        return file_path