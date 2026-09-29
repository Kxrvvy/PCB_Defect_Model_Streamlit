"""Development app: never loads a model; always simulated predictions."""
from utils.ui import run_app

run_app(mock_only=True)
