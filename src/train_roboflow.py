import os
from ultralytics import YOLO
# Disable MIOpen Find-DB (avoids buggy cached convolution algorithms)
os.environ["MIOPEN_DISABLE_CACHE"] = "1"

# Train on pond dataset
model = YOLO("yolov8n.pt")
model.train(
    data="dirs/data_storage/roboflow/data.yaml",
    epochs=50,
    imgsz=256,
    batch=16,
    project="runs/roboflow_training",
    name="yolov8n_roboflow",
    amp=False,
    exist_ok=True,
    optimizer='SGD',
    lr0=1e-4
)
