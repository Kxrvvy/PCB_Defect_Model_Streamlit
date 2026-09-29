"""Inference, detection extraction, and mock predictions."""
import hashlib
import io
import time

import numpy as np
from PIL import Image, ImageDraw

IMGSZ = 1024


def run_inference(model, image: Image.Image, conf: float, iou: float = 0.45):
    """Run YOLO segmentation on a PIL image; returns a list of Results."""
    arr = np.array(image.convert("RGB"))
    return model.predict(source=arr, conf=conf, iou=iou, imgsz=IMGSZ, verbose=False)


def _polygon_area(xy: np.ndarray) -> float:
    x, y = xy[:, 0], xy[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def extract_defects(results, class_names: dict = None) -> list:
    """Flatten a Results list into [{'class','confidence','bbox','area'}, ...]."""
    r = results[0]
    names = class_names or getattr(r, "names", {}) or {}
    if r.boxes is None or len(r.boxes) == 0:
        return []
    polys = r.masks.xy if getattr(r, "masks", None) is not None else None
    out = []
    for i, box in enumerate(r.boxes):
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
        area = (x2 - x1) * (y2 - y1)
        if polys is not None and i < len(polys) and len(polys[i]) >= 3:
            area = _polygon_area(np.asarray(polys[i]))
        cid = int(box.cls[0])
        out.append({
            "class": names.get(cid, f"class_{cid}"),
            "confidence": float(box.conf[0]),
            "bbox": (round(x1), round(y1), round(x2), round(y2)),
            "area": round(area),
        })
    return out


def mock_detections(image: Image.Image, class_names: dict, conf: float):
    """Deterministic fake detections (seeded by image content), filtered by conf."""
    buf = io.BytesIO()
    image.convert("RGB").resize((64, 64)).save(buf, "PNG")
    seed = int(hashlib.md5(buf.getvalue()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    w, h = image.size
    names = list(class_names.values())
    dets = []
    for _ in range(int(rng.integers(2, 7))):
        bw = int(rng.integers(max(8, w // 30), max(16, w // 8)))
        bh = int(rng.integers(max(8, h // 30), max(16, h // 8)))
        x1 = int(rng.integers(0, max(1, w - bw)))
        y1 = int(rng.integers(0, max(1, h - bh)))
        dets.append({
            "class": names[int(rng.integers(len(names)))],
            "confidence": float(rng.uniform(0.30, 0.99)),
            "bbox": (x1, y1, x1 + bw, y1 + bh),
            "area": round(np.pi * bw * bh / 4),
        })
    return [d for d in dets if d["confidence"] >= conf]


def draw_mock(image: Image.Image, detections: list) -> Image.Image:
    """Annotate mock detections with translucent ellipses, boxes, and labels."""
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        d.ellipse((x1, y1, x2, y2), fill=(255, 40, 40, 90))
        d.rectangle((x1, y1, x2, y2), outline=(255, 40, 40, 255), width=3)
        d.text((x1 + 2, max(0, y1 - 12)), f"{det['class']} {det['confidence']:.2f}",
               fill=(255, 255, 255, 255))
    return Image.alpha_composite(base, overlay).convert("RGB")


def analyze(model, image: Image.Image, conf: float, class_names: dict):
    """Run real inference (or mock when model is None).

    Returns (detections, annotated PIL image, elapsed seconds).
    """
    start = time.perf_counter()
    if model is None:
        dets = mock_detections(image, class_names, conf)
        annotated = draw_mock(image, dets)
    else:
        results = run_inference(model, image, conf)
        dets = extract_defects(results, class_names)
        annotated = Image.fromarray(results[0].plot()[..., ::-1])  # BGR -> RGB
    return dets, annotated, time.perf_counter() - start
