"""Model discovery and loading."""
from pathlib import Path

import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best.pt"


def get_model_path():
    """Return the path to models/best.pt as a string, or None if missing."""
    return str(MODEL_PATH) if MODEL_PATH.is_file() else None


def is_model_available(model_path=None) -> bool:
    """Cheap existence check; does not load the model."""
    return Path(model_path or MODEL_PATH).is_file()


@st.cache_resource(show_spinner="Loading model...")
def load_yolo_model(weight_path: str, mtime: float = 0.0):
    """Load a YOLO checkpoint once. `mtime` busts the cache when best.pt is swapped.

    Returns None on any failure (missing file, corrupt weights, ultralytics absent).
    """
    try:
        from ultralytics import YOLO

        return YOLO(weight_path)
    except Exception:
        return None


def load_default_model():
    """Load models/best.pt if present, else None."""
    path = get_model_path()
    if path is None:
        return None
    return load_yolo_model(path, Path(path).stat().st_mtime)


def get_class_names(model=None) -> dict:
    """Class names from model.names, or placeholders when unavailable."""
    names = getattr(model, "names", None)
    if names:
        return dict(names)
    return {i: f"Defect Type {i + 1}" for i in range(3)}


def get_device() -> str:
    try:
        import torch

        return "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    except Exception:
        return "CPU"
