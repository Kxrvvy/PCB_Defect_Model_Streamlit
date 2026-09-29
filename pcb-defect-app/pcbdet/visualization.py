"""Tables and metrics for detections."""
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image

COLUMNS = ["#", "Type", "Confidence (%)", "X1", "Y1", "X2", "Y2", "Area (px²)"]


def draw_detections(image: Image.Image, results) -> Image.Image:
    """Annotate using Ultralytics' plot() (BGR array -> PIL RGB)."""
    return Image.fromarray(results[0].plot()[..., ::-1])


def format_results_table(detections: list) -> pd.DataFrame:
    rows = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    data = [
        [i + 1, d["class"], round(d["confidence"] * 100, 1), *d["bbox"], d["area"]]
        for i, d in enumerate(rows)
    ]
    return pd.DataFrame(data, columns=COLUMNS)


def create_metrics(detections: list, inference_time: float = None) -> dict:
    counts = Counter(d["class"] for d in detections)
    return {
        "total": len(detections),
        "avg_conf": float(np.mean([d["confidence"] for d in detections])) if detections else 0.0,
        "top_type": counts.most_common(1)[0][0] if counts else "—",
        "counts": dict(counts),
        "inference_time": inference_time,
    }
