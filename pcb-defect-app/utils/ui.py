"""Shared Streamlit UI used by app.py (real model, mock fallback) and app_dev.py (mock only)."""
import io
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from utils.export import (export_batch_csv, export_comparison_pdf, export_csv,
                          export_pdf, get_pdf_filename)
from utils.inference import analyze
from utils.model_utils import (MODEL_PATH, get_class_names, get_device,
                               load_default_model)
from utils.sample_images import get_sample_images
from utils.visualization import create_metrics, format_results_table

MODES = ["Analyze Upload", "Demo Mode", "Batch", "Comparison"]
PRESETS = {"Low (25%)": 25, "Medium (50%)": 50, "High (75%)": 75}
MAX_BATCH = 5
HISTORY_LEN = 5
MODEL_LABEL = "YOLO26-seg"


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _open(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


@st.cache_data(show_spinner=False)
def _sample_bytes() -> dict:
    return {name: _png_bytes(img) for name, img in get_sample_images().items()}


@st.cache_data(show_spinner=False, max_entries=64)
def _cached_analyze(_model, model_key, data: bytes, conf: float, names: tuple):
    """Cached per (image, threshold, model); `_model` is excluded from the hash."""
    return analyze(_model, _open(data), conf, dict(names))


def _run(ctx, data: bytes, conf_pct: int):
    with st.spinner("Analyzing PCB..."):
        return _cached_analyze(ctx["model"], ctx["model_key"], data, conf_pct / 100,
                               tuple(ctx["class_names"].items()))


def _set_conf(v):
    st.session_state["conf_pct"] = v


def _set_current(name, data):
    st.session_state["current"] = {"name": name, "data": data}


def _remember(name, data):
    hist = [h for h in st.session_state["history"] if h["data"] != data]
    st.session_state["history"] = ([{"name": name, "data": data}] + hist)[:HISTORY_LEN]


def _sidebar(ctx):
    sb = st.sidebar
    sb.title("⚙️ Settings")
    mode = sb.radio("Mode", MODES, key="mode")

    sb.slider("Confidence threshold (%)", 0, 100, key="conf_pct")
    cols = sb.columns(3)
    for col, (label, val) in zip(cols, PRESETS.items()):
        col.button(label, key=f"preset_{val}", on_click=_set_conf, args=(val,))

    with sb.expander("ℹ️ Model Information", expanded=True):
        size = f"{MODEL_PATH.stat().st_size / 1e6:.0f} MB" if MODEL_PATH.is_file() else "~128 MB (expected)"
        last = st.session_state.get("last_time")
        st.markdown(
            f"- **Model:** {MODEL_LABEL}\n- **Input size:** 1024×1024\n"
            f"- **File:** best.pt ({size})\n- **Device:** {get_device()}\n"
            f"- **Status:** {'⚠️ Mock Mode' if ctx['mock'] else '✅ Ready'}\n"
            f"- **Classes:** {', '.join(ctx['class_names'].values())}\n"
            f"- **Inference time:** {f'{last:.2f} sec' if last else '—'}"
        )
        if not ctx["mock_only"]:
            st.caption("CPU inference is typically 8–15 sec per image on the free tier.")

    sb.subheader("📜 Recent Uploads (This Session)")
    hist = st.session_state["history"]
    if not hist:
        sb.caption("Nothing yet.")
    for i, h in enumerate(hist):
        sb.image(_open(h["data"]).resize((120, 120)), caption=h["name"])
        sb.button("Re-analyze", key=f"hist_{i}", on_click=_set_current, args=(h["name"], h["data"]))
    if hist and sb.button("Clear history"):
        st.session_state["history"] = []
        st.session_state["current"] = None
        st.rerun()
    return mode


def _show_results(ctx, name, data):
    conf_pct = st.session_state["conf_pct"]
    image = _open(data)
    dets, annotated, elapsed = _run(ctx, data, conf_pct)
    st.session_state["last_time"] = elapsed
    m = create_metrics(dets, elapsed)

    st.caption(f"**{m['total']}** defects detected at **{conf_pct}%** confidence")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📊 Total Defects", m["total"])
    c2.metric("🎯 Avg Confidence", f"{m['avg_conf'] * 100:.1f}%")
    c3.metric("🔍 Most Common Type", m["top_type"])
    c4.metric("⏱️ Processing Time", f"{elapsed:.2f} sec")

    left, right = st.columns(2)
    left.image(image, caption="Original", use_column_width=True)
    right.image(annotated, caption="Predictions", use_column_width=True)

    if dets:
        if len(m["counts"]) > 1:
            st.bar_chart(pd.Series(m["counts"], name="Defects"))
        st.dataframe(format_results_table(dets), use_container_width=True, hide_index=True)
    else:
        st.info("No defects detected at this threshold. Try lowering it.")

    d1, d2 = st.columns(2)
    d1.download_button("📥 Download Results (CSV)", export_csv(dets, name),
                       file_name=f"{Path(name).stem}_results.csv", mime="text/csv")
    d2.download_button("📄 Download Report (PDF)",
                       export_pdf(image, annotated, dets, name, conf_pct / 100, MODEL_LABEL),
                       file_name=get_pdf_filename(), mime="application/pdf")


def _upload_mode(ctx):
    up = st.file_uploader("Upload PCB image (JPG/PNG, max 10 MB)", type=["jpg", "jpeg", "png"])
    if st.button("🔍 Analyze", disabled=up is None, type="primary"):
        data = up.getvalue()
        _set_current(up.name, data)
        _remember(up.name, data)
    cur = st.session_state["current"]
    if cur:
        _show_results(ctx, cur["name"], cur["data"])
    else:
        st.info("Upload an image and click **Analyze**, or try **Demo Mode**.")


def _demo_mode(ctx):
    samples = _sample_bytes()
    name = st.selectbox("Sample image", list(samples))
    _show_results(ctx, name, samples[name])


def _batch_mode(ctx):
    files = st.file_uploader("Upload 1–5 images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if len(files) > MAX_BATCH:
        st.warning(f"Only the first {MAX_BATCH} images will be processed.")
        files = files[:MAX_BATCH]
    if st.button("🗂️ Process Multiple Images", disabled=not files, type="primary"):
        conf_pct = st.session_state["conf_pct"]
        results, bar = {}, st.progress(0.0)
        for i, f in enumerate(files):
            bar.progress(i / len(files), text=f"Analyzing {f.name} ({i + 1}/{len(files)})")
            dets, _, _ = _run(ctx, f.getvalue(), conf_pct)
            results[f.name] = dets
        bar.progress(1.0, text="Done")
        st.session_state["batch"] = results
    results = st.session_state.get("batch")
    if results:
        summary = pd.DataFrame([{"Image": n, "Defects": len(d),
                                 "Avg Confidence (%)": round(create_metrics(d)["avg_conf"] * 100, 1)}
                                for n, d in results.items()])
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.download_button("📥 Download Results (CSV)", export_batch_csv(results),
                           file_name="batch_results.csv", mime="text/csv")


def _compare_mode(ctx):
    c1, c2 = st.columns(2)
    fa = c1.file_uploader("First image (e.g. before)", type=["jpg", "jpeg", "png"], key="cmp_a")
    fb = c2.file_uploader("Second image (e.g. after)", type=["jpg", "jpeg", "png"], key="cmp_b")
    if not (fa and fb):
        st.info("Upload two images to compare them side by side.")
        return
    conf_pct = st.session_state["conf_pct"]
    out = []
    for f in (fa, fb):
        dets, ann, _ = _run(ctx, f.getvalue(), conf_pct)
        out.append({"name": f.name, "annotated": ann, "detections": dets})
    for col, r in zip((c1, c2), out):
        col.image(r["annotated"], caption=f"{r['name']} — {len(r['detections'])} defects", use_column_width=True)
    a, b = len(out[0]["detections"]), len(out[1]["detections"])
    st.metric("Improvement (defects resolved)", a - b, delta=f"{a} → {b}", delta_color="off")
    st.download_button("📄 Download Comparison (PDF)",
                       export_comparison_pdf(out[0], out[1], conf_pct / 100, MODEL_LABEL),
                       file_name=get_pdf_filename(), mime="application/pdf")


def run_app(mock_only: bool = False):
    st.set_page_config(page_title="PCB Defect Detection", page_icon="🔍", layout="wide")
    st.session_state.setdefault("conf_pct", 25)
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("current", None)

    model = None if mock_only else load_default_model()
    ctx = {
        "model": model,
        "model_key": str(MODEL_PATH.stat().st_mtime) if model is not None else "mock",
        "mock": model is None,
        "mock_only": mock_only,
        "class_names": get_class_names(model),
    }

    st.title("🔍 PCB Defect Detection")
    st.caption("Instance segmentation of printed circuit board defects")
    if mock_only:
        st.info("🧪 Development build: all predictions are simulated (no model is loaded).")
    elif ctx["mock"]:
        st.warning("⚠️ Model not available. Place `best.pt` in `models/` for real inference. "
                   "Showing **simulated** predictions meanwhile.")

    mode = _sidebar(ctx)
    {"Analyze Upload": _upload_mode, "Demo Mode": _demo_mode,
     "Batch": _batch_mode, "Comparison": _compare_mode}[mode](ctx)
