"""
Place Generator

Generates candidate pictures of the story's places (no people) so the human can
approve one per place. An approved picture is copied into the references
folder and then used as a reference in every shot that happens there.

Existing candidate files are kept (a rerun makes only the missing ones).
"""

import time
from pathlib import Path

from core.logging.logger import logger
from core.utils.image_utils import contact_sheet
from core.utils.image_utils import encode_png
from core.utils.image_utils import extension_for
from core.utils.image_utils import looks_blank
from core.utils.image_utils import open_image
from services.output.file_writer import FileWriter

MAX_BLANK_TRIES = 2


class PlaceGenerator:
    """Generate and store candidate pictures of places."""

    def __init__(self, provider, writer: FileWriter, size: tuple[int, int]):
        self.provider = provider
        self.writer = writer
        self.size = size

    def candidate_name(self, name: str, location_id: str, number: int, extension: str) -> str:
        return f"{name}_places/{location_id}_c{number}.{extension}"

    def _existing(self, name: str, location_id: str, number: int) -> Path | None:
        for extension in ("png", "jpg", "webp"):
            path = self.writer.path(self.candidate_name(name, location_id, number, extension))

            if path.is_file():
                return path

        return None

    def generate(self, name: str, jobs: list[dict], candidates: int, regenerate: bool = False) -> dict:
        """
        Make the missing candidates for every job.

        Each job has: location_id, prompt, references (a list of picture bytes).
        Returns {location_id: [file paths]}.

        Raises:
            ImageGenerationError: if the service fails (pictures made so far stay saved).
        """
        result = {}

        for index, job in enumerate(jobs):
            location_id = job["location_id"]
            result[location_id] = []

            for number in range(1, candidates + 1):
                existing = self._existing(name, location_id, number)

                if existing is not None and not regenerate:
                    logger.info("Place %s candidate %d already exists, skipping.", location_id, number)
                    result[location_id].append(existing)
                    continue

                logger.info("Place %s candidate %d/%d ...", location_id, number, candidates)
                result[location_id].append(self._make(name, location_id, number, job, 1000 + index * 10 + number))

        return result

    def _make(self, name: str, location_id: str, number: int, job: dict, seed: int) -> Path:
        for attempt in range(1, MAX_BLANK_TRIES + 1):
            begin = time.monotonic()
            data = self.provider.generate(
                job["prompt"],
                None,
                references=job["references"],
                width=self.size[0],
                height=self.size[1],
                seed=seed + attempt - 1,
            )
            seconds = time.monotonic() - begin

            image = open_image(data)
            blank = looks_blank(image)

            if not blank:
                break

            logger.warning("Place %s candidate %d came back blank (try %d/%d).", location_id, number, attempt, MAX_BLANK_TRIES)

        path = self.writer.save_bytes(self.candidate_name(name, location_id, number, extension_for(image)), data)
        logger.info("Place %s candidate %d saved: %dx%d, %.1f s%s", location_id, number, image.width, image.height, seconds, " (BLANK)" if blank else "")

        return path

    def make_contact_sheet(self, name: str, paths: dict) -> Path | None:
        """Save one picture showing every candidate of every place; return its path."""
        rows = []

        for location_id, files in paths.items():
            row = []

            for number, path in enumerate(files, start=1):
                row.append((f"{location_id}  candidate {number}", open_image(Path(path).read_bytes())))

            if row:
                rows.append(row)

        if not rows:
            return None

        filename = f"{name}_places_sheet.png"
        self.writer.save_bytes(filename, encode_png(contact_sheet(rows, thumb_width=480)))

        return self.writer.path(filename)