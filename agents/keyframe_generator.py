"""
Keyframe Generator

Generates several candidate pictures for each chosen shot from its image
prompt. For every candidate it saves the picture as received and a crop in
the video's shape, and records it in the manifest after each picture, so an
interrupted run resumes where it stopped.
"""

import time
from pathlib import Path

from core.exceptions.exceptions import ImageGenerationError
from core.logging.logger import logger
from core.utils.image_utils import contact_sheet
from core.utils.image_utils import crop_box
from core.utils.image_utils import crop_to_aspect
from core.utils.image_utils import encode_png
from core.utils.image_utils import extension_for
from core.utils.image_utils import looks_blank
from core.utils.image_utils import open_image
from core.utils.keyframes import keyframe_filename
from core.utils.keyframes import prompt_hash
from services.output.file_writer import FileWriter

# A blank picture is asked for again (a new random picture) this many times in total.
MAX_BLANK_TRIES = 2


class KeyframeGenerator:
    """Generate and store candidate pictures for shots."""

    def __init__(
        self,
        provider,
        writer: FileWriter,
        steps: int,
        aspect: tuple[int, int],
        crop_bias: float = 0.5,
    ):
        self.provider = provider
        self.writer = writer
        self.steps = steps
        self.aspect = aspect
        self.crop_bias = crop_bias

    def generate(
        self,
        name: str,
        prompts: list[dict],
        shot_ids: list[int],
        candidates: int,
        manifest: dict,
        regenerate: bool = False,
        on_progress=None,
    ) -> dict:
        """
        Make the missing candidates for the chosen shots.

        Pictures that already exist for an unchanged prompt are kept. The
        manifest is updated in place, and on_progress(manifest) is called
        after every picture.

        Raises:
            ImageGenerationError: if the service fails (pictures made so far stay saved).
        """
        by_id = {item["shot_id"]: item for item in prompts}

        for shot_id in shot_ids:
            prompt = by_id[shot_id]["prompt"]
            fingerprint = prompt_hash(prompt)
            entry = manifest["shots"].get(str(shot_id))

            if regenerate or entry is None or entry["prompt_hash"] != fingerprint:
                entry = {"prompt_hash": fingerprint, "candidates": [], "selected": None}

            entry["candidates"] = [item for item in entry["candidates"] if self._files_exist(item)]
            manifest["shots"][str(shot_id)] = entry

            for item in entry["candidates"]:
                self._recrop_if_needed(item)

            have = {item["number"] for item in entry["candidates"]}

            for number in range(1, candidates + 1):
                if number in have:
                    logger.info("Shot %d candidate %d already exists, skipping.", shot_id, number)
                    continue

                logger.info("Shot %d candidate %d/%d ...", shot_id, number, candidates)
                entry["candidates"].append(self._make_candidate(name, shot_id, number, prompt))
                entry["candidates"].sort(key=lambda item: item["number"])

                if on_progress:
                    on_progress(manifest)

        return manifest

    def make_contact_sheet(self, name: str, manifest: dict, shot_ids: list[int]) -> Path | None:
        """Save one picture showing every candidate of the chosen shots; return its path."""
        rows = []

        for shot_id in shot_ids:
            entry = manifest["shots"].get(str(shot_id))

            if not entry:
                continue

            row = []

            for item in entry["candidates"]:
                try:
                    picture = open_image(self.writer.path(item["full"]).read_bytes())
                except (OSError, ImageGenerationError):
                    continue

                box = crop_box(picture.size, *self.aspect, self.crop_bias)
                row.append((f"shot {shot_id}  candidate {item['number']}", picture, box))

            if row:
                rows.append(row)

        if not rows:
            return None

        filename = f"{name}_keyframes_sheet.png"
        self.writer.save_bytes(filename, encode_png(contact_sheet(rows)))

        return self.writer.path(filename)

    def _recrop_if_needed(self, item: dict) -> None:
        """Cut the crop again from the saved picture when the crop position changed (no new image is requested)."""
        if item.get("crop_bias") == self.crop_bias:
            return

        image = open_image(self.writer.path(item["full"]).read_bytes())
        crop = crop_to_aspect(image, *self.aspect, self.crop_bias)

        self.writer.save_bytes(item["crop"], encode_png(crop))

        item["crop_size"] = [crop.width, crop.height]
        item["crop_bias"] = self.crop_bias

        logger.info("Re-cropped %s (crop position %.2f).", item["crop"], self.crop_bias)

    def _files_exist(self, item: dict) -> bool:
        return self.writer.path(item["full"]).is_file() and self.writer.path(item["crop"]).is_file()

    def _make_candidate(self, name: str, shot_id: int, number: int, prompt: str) -> dict:
        """Generate one picture (asking again if it is blank), save it and describe it."""
        for attempt in range(1, MAX_BLANK_TRIES + 1):
            begin = time.monotonic()
            data = self.provider.generate(prompt, self.steps)
            seconds = time.monotonic() - begin

            image = open_image(data)
            blank = looks_blank(image)

            if not blank:
                break

            logger.warning(
                "Shot %d candidate %d came back blank (try %d/%d).", shot_id, number, attempt, MAX_BLANK_TRIES
            )

        full_name = keyframe_filename(name, shot_id, number, "full", extension_for(image))
        crop_name = keyframe_filename(name, shot_id, number, "crop", "png")
        crop = crop_to_aspect(image, *self.aspect, self.crop_bias)

        self.writer.save_bytes(full_name, data)
        self.writer.save_bytes(crop_name, encode_png(crop))

        logger.info(
            "Shot %d candidate %d saved: %dx%d, crop %dx%d, %.1f s%s",
            shot_id,
            number,
            image.width,
            image.height,
            crop.width,
            crop.height,
            seconds,
            " (BLANK)" if blank else "",
        )

        return {
            "number": number,
            "full": full_name,
            "crop": crop_name,
            "full_size": [image.width, image.height],
            "crop_size": [crop.width, crop.height],
            "crop_bias": self.crop_bias,
            "seconds": round(seconds, 1),
            "blank": blank,
        }