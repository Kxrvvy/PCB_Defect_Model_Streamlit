"""Manual smoke test of the real-model path: python tests/real_smoke.py <weights.pt>"""
import sys
sys.path.insert(0, ".")
from ultralytics import YOLO
from pcbdet.inference import analyze
from pcbdet.model_utils import get_class_names
from pcbdet.sample_images import create_sample_pcb_image

m = YOLO(sys.argv[1])
img = create_sample_pcb_image(seed=1)
dets, ann, t = analyze(m, img, 0.01, get_class_names(m))
print(len(dets), "dets", ann.size, f"{t:.2f}s", dets[:2])
