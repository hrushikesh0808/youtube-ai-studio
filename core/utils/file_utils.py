import re


def make_filename(text: str) -> str:
    """
    Convert user input into a safe filename.
    """

    filename = text.lower()

    filename = re.sub(r"[^a-z0-9]+", "_", filename)

    filename = filename.strip("_")

    return filename + ".txt"