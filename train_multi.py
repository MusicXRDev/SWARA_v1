from ultralytics import YOLO
from datetime import datetime
import torch
import os


DATA = "/home/Unified_Yolo/data.yaml"


MODELS = [
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolo11l.pt",
    "yolo11x.pt"
]


# OUTPUT ROOT

PROJECT_ROOT = "/home/Unified_Yolo/model_comparison"
os.makedirs(PROJECT_ROOT, exist_ok=True)


TRAIN_ARGS = dict(
    data=DATA,
    imgsz=1280,
    epochs=50,
    batch=4,
    workers=8,
    device=0,

    optimizer="AdamW",
    lr0=0.0008,
    lrf=0.01,
    weight_decay=0.0005,
    patience=40,

    box=12.0,
    cls=0.7,
    dfl=1.5,

    max_det=3000,
    iou=0.5,

    mosaic=0.0,
    scale=0.4,
    translate=0.05,
    degrees=2,

    hsv_h=0,
    hsv_s=0,
    hsv_v=0,
    flipud=0,
    fliplr=0,

    plots=True,
    save=True,
    verbose=True
)


# GPU INFO
# ======================================================
print("====================================")
print("Torch:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU Memory:",
          round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), "GB")
print("====================================")


for model_name in MODELS:

    print("\n=======================================")
    print("TRAINING:", model_name)
    print("=======================================")

    model = YOLO(model_name)

    short_name = model_name.replace(".pt", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = f"{short_name}_Unified_{timestamp}"

   
    model.train(
        **TRAIN_ARGS,
        project=PROJECT_ROOT,
        name=run_name,
        exist_ok=False
    )

    
    best_weights = os.path.join(
        PROJECT_ROOT,
        run_name,
        "weights",
        "best.pt"
    )

    if not os.path.exists(best_weights):
        raise FileNotFoundError(f"Best model not found at {best_weights}")

    print("Validating best model:", best_weights)

    best_model = YOLO(best_weights)

   
    best_model.val(
        data=DATA,
        imgsz=1152,
        conf=0.001,
        iou=0.5,
        max_det=3000,
        plots=True,
        save_json=True,
        project=PROJECT_ROOT,
        name=run_name + "_val"
    )

print("\n ALL MODEL TRAINING FINISHED")
