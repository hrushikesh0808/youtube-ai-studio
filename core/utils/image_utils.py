"""
Image Utilities

Small Pillow helpers: decode an image, detect a blank picture, crop to the
video's shape and build a contact sheet for review.
"""

import io

from PIL import Image
from PIL import ImageDraw
from PIL import UnidentifiedImageError

from core.exceptions.exceptions import ImageGenerationError

# A picture is "blank" when almost every pixel is the same dark value,
# or when it is a single flat colour (for example the green or black images
# some GPUs produce when a model fails).
BLANK_MAX_BRIGHTNESS = 8
BLANK_MIN_SPREAD = 2.0

EXTENSIONS = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}


def open_image(data: bytes) -> Image.Image:
    """
    Decode image bytes into an RGB picture.

    Raises:
        ImageGenerationError: if the bytes are not a readable image.
    """
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageGenerationError("The service returned data that is not a readable image.") from exc

    picture = image.convert("RGB")
    picture.info["source_format"] = image.format
    return picture


def extension_for(image: Image.Image) -> str:
    """File extension for the format an image was decoded from."""
    return EXTENSIONS.get(image.info.get("source_format"), "img")


def looks_blank(image: Image.Image) -> bool:
    """Return True when the picture is (almost) a single flat colour."""
    import numpy as np

    pixels = np.asarray(image.convert("RGB"), dtype="float32")

    if not np.isfinite(pixels).all():
        return True

    spread = float(pixels.reshape(-1, 3).std(axis=0).mean())
    brightest = float(pixels.max())

    return brightest <= BLANK_MAX_BRIGHTNESS or spread < BLANK_MIN_SPREAD


def crop_box(size: tuple[int, int], ratio_width: int, ratio_height: int, bias: float = 0.5) -> tuple:
    """
    The (left, top, right, bottom) box of a crop to the given shape (for example 16:9).

    Only one direction needs cutting. bias says where the kept part sits
    along it: 0 = top (or left), 0.5 = centre, 1 = bottom (or right).
    """
    if ratio_width <= 0 or ratio_height <= 0:
        raise ValueError("The aspect ratio must be positive.")

    if not 0.0 <= bias <= 1.0:
        raise ValueError("The crop bias must be between 0 and 1.")

    target = ratio_width / ratio_height
    width, height = size

    if width / height > target:
        new_width = round(height * target)
        left = round((width - new_width) * bias)
        return (left, 0, left + new_width, height)

    new_height = round(width / target)
    top = round((height - new_height) * bias)
    return (0, top, width, top + new_height)


def crop_to_aspect(image: Image.Image, ratio_width: int, ratio_height: int, bias: float = 0.5) -> Image.Image:
    """Crop a picture to the given shape (for example 16:9); see crop_box for bias."""
    return image.crop(crop_box(image.size, ratio_width, ratio_height, bias))


def encode_png(image: Image.Image) -> bytes:
    """Encode a picture as PNG bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def contact_sheet(rows: list[list[tuple]], thumb_width: int = 360) -> Image.Image:
    """
    Build one picture that shows many pictures with labels, a row per shot.

    Each row is a list of (label, picture) or (label, picture, box). When a
    box is given, a frame is drawn on the picture to show the part that is kept.
    """
    if not rows or not any(rows):
        raise ValueError("There is nothing to put on the contact sheet.")

    label_height = 18
    gap = 6
    first = next(cell[1] for row in rows for cell in row)
    thumb_height = round(thumb_width * first.height / first.width)

    columns = max(len(row) for row in rows)
    cell_height = thumb_height + label_height

    sheet = Image.new(
        "RGB",
        (columns * (thumb_width + gap) + gap, len(rows) * (cell_height + gap) + gap),
        (30, 30, 30),
    )
    draw = ImageDraw.Draw(sheet)

    for row_index, row in enumerate(rows):
        for column_index, cell in enumerate(row):
            label, picture = cell[0], cell[1]
            box = cell[2] if len(cell) > 2 else None
            x = gap + column_index * (thumb_width + gap)
            y = gap + row_index * (cell_height + gap)
            draw.text((x + 2, y + 3), label, fill=(235, 235, 235))
            sheet.paste(picture.resize((thumb_width, thumb_height)), (x, y + label_height))

            if box:
                scale = thumb_width / picture.width
                left, top, right, bottom = (round(value * scale) for value in box)
                draw.rectangle(
                    (x + left, y + label_height + top, x + right - 1, y + label_height + bottom - 1),
                    outline=(255, 215, 0),
                    width=2,
                )

    return sheet