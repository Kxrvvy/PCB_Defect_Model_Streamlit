"""Production app: real YOLO inference; falls back to simulated predictions if models/best.pt is missing."""
from utils.ui import run_app

run_app(mock_only=False)
