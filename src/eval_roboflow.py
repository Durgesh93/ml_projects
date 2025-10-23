import os
from ultralytics import YOLO
import pandas as pd 

# Disable MIOpen Find-DB (avoids buggy cached convolution algorithms)
os.environ["MIOPEN_DISABLE_CACHE"] = "1"

# Paths to dataset configs (make sure YAMLs do NOT contain "detect:")
roboflow_yaml = "dirs/data_storage/roboflow/data.yaml"
pond_yaml = "dirs/data_storage/pond/data.yaml"

# Load best Roboflow-trained model
best_model_path = "runs/roboflow_training/yolov8n_roboflow/weights/best.pt"
model = YOLO(best_model_path)

# -------------------- Evaluate --------------------
print("🔎 Evaluating on Roboflow dataset...")
metrics_roboflow = model.val(data=roboflow_yaml, split="test")

print("🔎 Evaluating on Pond dataset...")
metrics_pond = model.val(data=pond_yaml, split="test")

# Collect results into DataFrame
data = {
    "Dataset": ["Roboflow", "Pond"],
    "Precision": [metrics_roboflow.box.mp, metrics_pond.box.mp],
    "Recall": [metrics_roboflow.box.mr, metrics_pond.box.mr],
    "mAP50": [metrics_roboflow.box.map50, metrics_pond.box.map50],
    "mAP50-95": [metrics_roboflow.box.map, metrics_pond.box.map],
}
df = pd.DataFrame(data)
print("\n📊 Comparison of Results:\n")
print(df)
df.to_csv("evaluation_comparison.csv", index=False)

# -------------------- Predictions --------------------
print("🔎 Saving prediction grids...")

# Predict on Roboflow test images (save annotated images, overwrite if exists)
model.predict(
    source="dirs/data_storage/roboflow/Test/images", 
    save=True, 
    project="runs/predictions",
    name="roboflow_model_on_roboflow",
    max_det=50,
    exist_ok=True
)

# Predict on Pond test images (overwrite if exists)
model.predict(
    source="dirs/data_storage/pond/Test/images", 
    save=True, 
    project="runs/predictions",
    name="roboflow_model_on_pond",
    max_det=50,
    exist_ok=True
)

print("✅ Predictions saved under runs/predictions/roboflow_test_preds/ and runs/predictions/pond_test_preds/")
