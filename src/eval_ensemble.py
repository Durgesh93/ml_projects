import os
import pandas as pd 
from ultralytics import YOLO

# Disable MIOpen Find-DB (avoids buggy cached convolution algorithms on ROCm/AMD GPUs)
os.environ["MIOPEN_DISABLE_CACHE"] = "1"

# Dataset configs
roboflow_yaml = "dirs/data_storage/roboflow/data.yaml"
pond_yaml = "dirs/data_storage/pond/data.yaml"

# Best weights
roboflow_best = "runs/roboflow_training/yolov8n_roboflow/weights/best.pt"
pond_best     = "runs/pond_training/yolov8n_pond/weights/best.pt"

# Load models
roboflow_model = YOLO(roboflow_best)
pond_model     = YOLO(pond_best)

# Evaluate both models on Roboflow dataset
m_rf_rf = roboflow_model.val(data=roboflow_yaml, split="test")
m_pd_rf = pond_model.val(data=roboflow_yaml, split="test")

# Evaluate both models on Pond dataset
m_rf_pd = roboflow_model.val(data=pond_yaml, split="test")
m_pd_pd = pond_model.val(data=pond_yaml, split="test")

# Helper: average two sets of metrics
def avg_metrics(m1, m2):
    return {
        "Precision": (m1.box.mp + m2.box.mp) / 2,
        "Recall": (m1.box.mr + m2.box.mr) / 2,
        "mAP50": (m1.box.map50 + m2.box.map50) / 2,
        "mAP50-95": (m1.box.map + m2.box.map) / 2,
    }

# Ensemble metrics
ensemble_rf = avg_metrics(m_rf_rf, m_pd_rf)
ensemble_pd = avg_metrics(m_rf_pd, m_pd_pd)

# Put into DataFrame
df = pd.DataFrame([
    {"Dataset": "Roboflow", **ensemble_rf},
    {"Dataset": "Pond", **ensemble_pd},
])

print("\n📊 Ensemble Evaluation Results:\n")
print(df)
df.to_csv("evaluation_comparison_ensemble.csv", index=False)