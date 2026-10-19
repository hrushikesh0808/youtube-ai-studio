import os
import re
from pathlib import Path


def make_filename(text: str, suffix: str = "", extension: str = ".txt") -> str:
    """
    Convert user input into a safe filename.

    An optional suffix (for example "_shots") is added before the
    extension, so one topic can own several related files.
    """

    filename = text.lower()

    filename = re.sub(r"[^a-z0-9]+", "_", filename)

    filename = filename.strip("_")

    return filename + suffix + extension


def write_text_atomic(path: Path, content: str) -> None:
    """
    Write a text file so that it is either fully written or untouched.

    The content goes to a temporary file next to the target and is then
    renamed over it, so a crash mid-write never leaves a half-written file.
    """
    path = Path(path)
    temp_path = path.with_name(path.name + ".tmp")

    try:
        temp_path.write_text(content, encoding="utf-8")
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def write_bytes_atomic(path: Path, data: bytes) -> None:
    """
    Write a binary file (for example a picture) so that it is either fully
    written or untouched. Missing folders are created.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")

    try:
        temp_path.write_bytes(data)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
