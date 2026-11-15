ISRO LunarFusion — Trained Lunar Crater Models
==================================================

1. MODELS INCLUDED:
-------------------
• yolo11_lunar_crater.pt (5.2 MB)
  - Fine-tuned Ultralytics YOLO11 Nano model.
  - Detects lunar impact craters across multi-mission imagery (Chandrayaan-1/2, OHRC, TMC-2).
  - Class: 'crater' (id: 0)

• teacher_yolov5l_ohrc.pt (89 MB)
  - Teacher YOLOv5-Large model trained on high-resolution Chandrayaan-2 OHRC imagery.


2. QUICK START (PYTHON):
------------------------
pip install ultralytics

from ultralytics import YOLO

# Load model:
model = YOLO("yolo11_lunar_crater.pt")

# Run inference on an image:
results = model.predict(source="path/to/lunar_image.png", conf=0.25)

# View detections:
for r in results:
    r.show()
