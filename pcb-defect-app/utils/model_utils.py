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

    Returns (model, None) on success or (None, error_text) on any failure, so the UI
    can tell "file missing" apart from "file present but failed to load".
    """
    try:
        size = Path(weight_path).stat().st_size
        if size < 1_000_000:
            return None, f"best.pt is only {size} bytes (a Git LFS pointer or truncated file?)"
        from ultralytics import YOLO

        return YOLO(weight_path), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def load_default_model():
    """Load models/best.pt. Returns (model, error); model is None if missing or failed."""
    path = get_model_path()
    if path is None:
        return None, "models/best.pt not found"
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
