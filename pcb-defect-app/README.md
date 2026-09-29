# PCB Defect Detection

Streamlit demo for YOLO26-seg PCB defect segmentation (IEEE paper demo).

## Run locally (Python 3.11–3.13)

```bash
pip install -r requirements.txt
streamlit run app_dev.py   # simulated predictions, no model needed
streamlit run app.py       # real inference if models/best.pt exists, else simulated
```

Python 3.14 is not supported by the pinned dependency ranges (torch/numpy wheels).
Class names are read from the checkpoint (`model.names`); swap models by replacing `models/best.pt`.

## Features

Upload/analyze, demo samples (real images from `sample_images/`, otherwise synthetic),
confidence slider + presets (25/50/75%), batch (max 5), before/after comparison,
CSV + PDF export, metrics dashboard, model info, session history.

## Layout

`app.py` / `app_dev.py` are thin entry points over `utils/ui.py`. Logic lives in
`utils/{model_utils,inference,visualization,export,sample_images}.py`.

## Tests

```bash
pip install pytest && pytest tests
python tests/real_smoke.py path/to/weights.pt   # manual real-model check
```
