import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch
from ultralytics import YOLO

# Disable MIOpen Find-DB cache (AMD ROCm workaround)
os.environ["MIOPEN_DISABLE_CACHE"] = "1"

# Paths
pond_test_images     = "dirs/data_storage/pond/Test/images"
roboflow_test_images = "dirs/data_storage/roboflow/Test/images"

pond_best     = "runs/pond_training/yolov8n_pond/weights/best.pt"
roboflow_best = "runs/roboflow_training/yolov8n_roboflow/weights/best.pt"

# Load models
pond_model     = YOLO(pond_best)
roboflow_model = YOLO(roboflow_best)

# Pick one random image from each test set
pond_img     = os.path.join(pond_test_images, random.choice(os.listdir(pond_test_images)))
roboflow_img = os.path.join(roboflow_test_images, random.choice(os.listdir(roboflow_test_images)))

print(f"🔎 Selected Pond image: {pond_img}")
print(f"🔎 Selected Roboflow image: {roboflow_img}")

# Run predictions
pred_pd_on_pond = pond_model(pond_img, verbose=False)[0]
pred_rf_on_pond = roboflow_model(pond_img, verbose=False)[0]

pred_pd_on_rf   = pond_model(roboflow_img, verbose=False)[0]
pred_rf_on_rf   = roboflow_model(roboflow_img, verbose=False)[0]

# ---- IoU function ----
def box_iou(box1, box2):
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    xa = max(box1[0], box2[0])
    ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2])
    yb = min(box1[3], box2[3])

    inter = max(0, xb - xa) * max(0, yb - ya)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

# ---- Ensemble averaging ----
def average_boxes(pred1, pred2, iou_thresh=0.5):
    boxes1 = pred1.boxes.xyxy.cpu().numpy()
    conf1  = pred1.boxes.conf.cpu().numpy()
    boxes2 = pred2.boxes.xyxy.cpu().numpy()
    conf2  = pred2.boxes.conf.cpu().numpy()

    used2 = set()
    averaged = []

    for i, b1 in enumerate(boxes1):
        for j, b2 in enumerate(boxes2):
            if j in used2:
                continue
            if box_iou(b1, b2) > iou_thresh:
                # Average coordinates + confidence
                avg_box = (b1 + b2) / 2.0
                avg_conf = (conf1[i] + conf2[j]) / 2.0
                averaged.append((avg_box, avg_conf))
                used2.add(j)
                break

    return averaged

# ---- Draw ensemble boxes ----
def draw_ensemble(ax, image_path, preds1, preds2, title):
    img = Image.open(image_path)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title)

    avg_preds = average_boxes(preds1, preds2, iou_thresh=0.5)

    for box, conf in avg_preds:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f"{conf:.2f}", color="yellow", fontsize=8,
                bbox=dict(facecolor="black", alpha=0.5))

# ---- Plot Pond + Roboflow images in 1x2 grid ----
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

draw_ensemble(axes[0], pond_img, pred_pd_on_pond, pred_rf_on_pond,
              "Averaged Ensemble on Pond image")
draw_ensemble(axes[1], roboflow_img, pred_pd_on_rf, pred_rf_on_rf,
              "Averaged Ensemble on Roboflow image")
# Save to file
os.makedirs("figures", exist_ok=True)
save_path = "figures/ensemble_predictions.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"✅ Figure saved to {save_path}")
