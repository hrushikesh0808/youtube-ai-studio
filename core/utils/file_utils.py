import re


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