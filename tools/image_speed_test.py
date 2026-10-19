"""
Image Speed Test (stand-alone measurement tool, not part of the pipeline)

Generates test images with a Stable Diffusion model on the local GPU and
reports, for one memory mode and precision:

- how long loading and generating take,
- peak GPU memory use,
- whether the picture came out blank (black or green images are a known
  half-precision problem on GTX 16-series cards).

Usage (from the project folder, virtual environment active):

    python tools/image_speed_test.py --mode cuda --dtype fp16

Results are appended to output/benchmarks/results.jsonl and the pictures are
saved next to it.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

DEFAULT_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"

DEFAULT_PROMPT = (
    "painterly anime style, a tall granite lighthouse on a rocky coast under a dark "
    "stormy sky, rain, muted moody colour palette, soft cinematic lighting, detailed "
    "background, no people"
)

DEFAULT_NEGATIVE = "text, letters, watermark, signature, blurry, low quality, deformed"

MODES = ("cuda", "offload", "sequential")

# A picture is "blank" when almost every pixel is the same dark value,
# or when it contains NaN-derived uniform colour.
BLANK_MAX_BRIGHTNESS = 8
BLANK_MIN_SPREAD = 2.0

MIN_FREE_VRAM_GB = 3.0


def looks_blank(image) -> bool:
    """Return True when the picture is (almost) a single flat colour."""
    import numpy as np

    pixels = np.asarray(image.convert("RGB"), dtype="float32")

    if not np.isfinite(pixels).all():
        return True

    # Variation between pixels, per colour channel (a flat green picture has
    # none, even though its channels differ from each other).
    spread = float(pixels.reshape(-1, 3).std(axis=0).mean())
    brightest = float(pixels.max())

    return brightest <= BLANK_MAX_BRIGHTNESS or spread < BLANK_MIN_SPREAD


def save_image_atomic(image, path: Path) -> None:
    """Save a PNG through a temporary file so a crash never leaves a half file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    image.save(temporary, format="PNG")
    os.replace(temporary, path)


def append_result(path: Path, result: dict) -> None:
    """Append one result as a JSON line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False) + "\n")


def verdict(result: dict) -> str:
    """Plain-language conclusion for one result."""
    if result["blank_images"]:
        return (
            "BLANK IMAGE: this precision does not work on this GPU. "
            "Try --dtype fp32 (with --mode sequential) or another model."
        )

    return (
        f"OK: {result['seconds_per_image']:.1f} s per {result['width']}x{result['height']} "
        f"image, peak GPU memory {result['peak_vram_gb']:.2f} GB."
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Measure image generation on the local GPU.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id or local folder.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative", default=DEFAULT_NEGATIVE)
    parser.add_argument("--width", type=int, default=768, help="Multiple of 8.")
    parser.add_argument("--height", type=int, default=432, help="Multiple of 8.")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--runs", type=int, default=3, help="Images to make; the first includes warm-up.")
    parser.add_argument("--mode", choices=MODES, default="cuda")
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument(
        "--no-attention-slicing",
        action="store_true",
        help="cuda mode only: skip attention slicing (faster if the GPU memory allows it).",
    )
    parser.add_argument("--outdir", default="output/benchmarks")
    return parser.parse_args(argv)


def validate_args(args) -> None:
    if args.width % 8 or args.height % 8:
        raise ValueError("Width and height must be multiples of 8.")
    if args.width < 256 or args.height < 256 or args.width > 1024 or args.height > 1024:
        raise ValueError("Width and height must be between 256 and 1024.")
    if args.steps < 1 or args.steps > 100:
        raise ValueError("Steps must be between 1 and 100.")
    if args.runs < 1 or args.runs > 10:
        raise ValueError("Runs must be between 1 and 10.")


def run(args) -> dict:
    import torch
    from diffusers import StableDiffusionPipeline

    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch cannot see the GPU. Install the CUDA build of PyTorch "
            "(see the setup commands) and check nvidia-smi works in WSL."
        )

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb = free_bytes / 1024**3
    print(f"GPU: {torch.cuda.get_device_name(0)} | free {free_gb:.2f} GB of {total_bytes / 1024**3:.2f} GB")

    if free_gb < MIN_FREE_VRAM_GB:
        print(
            f"WARNING: only {free_gb:.2f} GB of GPU memory is free. Stop programs using the GPU "
            "(for example run `ollama stop gemma3:4b`) or use --mode sequential."
        )

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    started = time.monotonic()
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
        use_safetensors=True,
    )

    if args.mode == "cuda":
        pipe = pipe.to("cuda")
        if not args.no_attention_slicing:
            pipe.enable_attention_slicing()
        pipe.vae.enable_slicing()
    elif args.mode == "offload":
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_slicing()
    else:
        pipe.enable_sequential_cpu_offload()
        pipe.vae.enable_slicing()

    load_seconds = time.monotonic() - started
    print(f"Model loaded in {load_seconds:.1f} s (mode={args.mode}, dtype={args.dtype})")

    torch.cuda.reset_peak_memory_stats()

    outdir = Path(args.outdir)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    timings = []
    blanks = 0

    for index in range(args.runs):
        generator = torch.Generator(device="cpu").manual_seed(args.seed + index)

        begin = time.monotonic()
        image = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative,
            width=args.width,
            height=args.height,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator,
        ).images[0]
        elapsed = time.monotonic() - begin

        blank = looks_blank(image)
        blanks += int(blank)
        timings.append(elapsed)

        name = f"{stamp}_{args.mode}_{args.dtype}_{args.width}x{args.height}_run{index + 1}.png"
        save_image_atomic(image, outdir / name)
        print(f"  image {index + 1}/{args.runs}: {elapsed:.1f} s{'  <-- BLANK' if blank else ''}  ({name})")

    steady = timings[1:] if len(timings) > 1 else timings

    return {
        "time": stamp,
        "model": args.model,
        "mode": args.mode,
        "dtype": args.dtype,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "attention_slicing": args.mode == "cuda" and not args.no_attention_slicing,
        "load_seconds": round(load_seconds, 1),
        "first_image_seconds": round(timings[0], 1),
        "seconds_per_image": round(sum(steady) / len(steady), 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "blank_images": blanks,
        "runs": args.runs,
    }


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        validate_args(args)
        result = run(args)
    except Exception as exc:  # report a clear message instead of a traceback
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    append_result(Path(args.outdir) / "results.jsonl", result)
    print(json.dumps(result, indent=2))
    print(verdict(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())