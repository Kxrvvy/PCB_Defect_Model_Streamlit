#!/bin/bash
# Downloads models/best.pt when missing. Set MODEL_URL to a direct-download or
# Google Drive share link. Streamlit Cloud does NOT run this automatically;
# see DEPLOYMENT.md for options that work there.
set -e
mkdir -p models
if [ ! -f "models/best.pt" ]; then
  if [ -z "$MODEL_URL" ]; then
    echo "MODEL_URL not set; skipping download (app will run in mock mode)."
    exit 0
  fi
  echo "Downloading model..."
  pip install -q gdown
  gdown --fuzzy "$MODEL_URL" -O models/best.pt || curl -L "$MODEL_URL" -o models/best.pt
fi
