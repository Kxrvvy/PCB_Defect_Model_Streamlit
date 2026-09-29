from streamlit.testing.v1 import AppTest

from utils.export import export_batch_csv, export_comparison_pdf, export_csv, export_pdf
from utils.inference import analyze
from utils.model_utils import get_class_names
from utils.sample_images import get_sample_images


def _run(script):
    at = AppTest.from_file(script, default_timeout=60).run()
    assert not at.exception, at.exception
    return at


def test_dev_app_demo_mode_and_presets():
    at = _run("app_dev.py")
    at.sidebar.radio[0].set_value("Demo Mode").run()
    assert not at.exception
    assert len(at.metric) == 4
    at.sidebar.button[1].click().run()  # Medium (50%)
    assert at.session_state["conf_pct"] == 50 and not at.exception


def test_prod_app_falls_back_to_mock_without_model():
    at = _run("app.py")
    assert any("Model not available" in w.value for w in at.warning)


def test_exports():
    imgs = get_sample_images()
    img = next(iter(imgs.values()))
    dets, ann, _ = analyze(None, img, 0.1, get_class_names(None))
    assert dets
    csv = export_csv(dets, "a.jpg").decode("utf-8-sig")
    assert csv.splitlines()[0].startswith("Image Filename,Defect Type")
    assert len(csv.splitlines()) == len(dets) + 1
    assert export_batch_csv({"a": dets, "b": []})
    assert export_pdf(img, ann, dets, "a.jpg").startswith(b"%PDF")
    assert export_pdf(img, ann, [], "a.jpg").startswith(b"%PDF")
    a = {"name": "a", "annotated": ann, "detections": dets}
    assert export_comparison_pdf(a, {**a, "detections": []}).startswith(b"%PDF")


def test_threshold_filters_mock():
    img = next(iter(get_sample_images().values()))
    n = get_class_names(None)
    lo = analyze(None, img, 0.0, n)[0]
    hi = analyze(None, img, 0.9, n)[0]
    assert len(hi) <= len(lo)
