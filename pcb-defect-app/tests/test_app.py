from streamlit.testing.v1 import AppTest

from pcbdet.export import export_batch_csv, export_comparison_pdf, export_csv, export_pdf
from pcbdet.inference import analyze
from pcbdet.model_utils import get_class_names
from pcbdet.sample_images import get_sample_images


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


def test_prod_app_falls_back_to_mock_without_model(monkeypatch, tmp_path):
    import pcbdet.model_utils as mu

    monkeypatch.setattr(mu, "MODEL_PATH", tmp_path / "missing.pt")
    at = _run("app.py")
    assert any("Model not available" in w.value for w in at.warning)
    assert "not found" in at.code[0].value


def test_prod_app_reports_load_failure(monkeypatch, tmp_path):
    import pcbdet.model_utils as mu

    bad = tmp_path / "best.pt"
    bad.write_bytes(b"x" * 100)
    monkeypatch.setattr(mu, "MODEL_PATH", bad)
    mu.load_yolo_model.clear()
    at = _run("app.py")
    assert "100 bytes" in at.code[0].value


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
