"""Sample PCB images: real files if present, otherwise synthetic."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_images"
EXTS = {".jpg", ".jpeg", ".png"}


def create_sample_pcb_image(width=1024, height=1024, seed=0) -> Image.Image:
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (width, height), (24, 96, 52))
    d = ImageDraw.Draw(img)
    for x in range(0, width, 64):
        d.line((x, 0, x, height), fill=(30, 110, 60), width=2)
    for y in range(0, height, 64):
        d.line((0, y, width, y), fill=(30, 110, 60), width=2)
    for _ in range(40):
        x1, y1 = int(rng.integers(0, width)), int(rng.integers(0, height))
        x2, y2 = int(rng.integers(0, width)), int(rng.integers(0, height))
        d.line((x1, y1, x2, y1), fill=(196, 150, 60), width=6)
        d.line((x2, y1, x2, y2), fill=(196, 150, 60), width=6)
    for _ in range(30):
        cx, cy, r = int(rng.integers(0, width)), int(rng.integers(0, height)), int(rng.integers(8, 28))
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(60, 60, 60), outline=(200, 200, 200), width=2)
    return img


def get_sample_images() -> dict:
    """{display name: PIL image}; real files from sample_images/, else 4 synthetic PCBs."""
    files = sorted(p for p in SAMPLE_DIR.glob("*") if p.suffix.lower() in EXTS) if SAMPLE_DIR.is_dir() else []
    if files:
        return {f"Sample {i + 1}: {p.name}": Image.open(p).convert("RGB") for i, p in enumerate(files)}
    return {f"Sample {i + 1} (synthetic)": create_sample_pcb_image(seed=i) for i in range(4)}
