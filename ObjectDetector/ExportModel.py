from pathlib import Path
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load an official model

# Train a model
input_width = 1280 # 640 for 16:9
input_height = 736 # 384 for 16:9

data_path = Path(__file__).parent / "datasets" / "kitti" / "kitti.yaml"

results = model.train(
  data=data_path, epochs=300, imgsz=640, batch=16, profile=True, save_period=10, patience=50, 
  lr0=0.0054, 
  optimizer="MuSGD", 
  lrf=0.0495,
  momentum=0.947, 
  weight_decay=0.00064, 
  warmup_epochs=0.98,
  cls=0.56,
  box=5.63,
  dfl=9.04,
  mosaic=0.909,
  mixup=	0.012,
  copy_paste=0.075,
  scale=0.562,
  fliplr=0.606,
  degrees=1.11,
  shear=1.46,
  translate=0.071,
  hsv_h=0.014,
  hsv_s=0.645,
  hsv_v=0.566,
  bgr=0.106
  )

# Export's parameters setting
half = True
device = 0 # If use GPU (=0), auto set when use TensorRT; CPU (device=cpu)

# Export the model
"""
format selection is based on what platform you use:
1. engine: Using NVIDIA GPU
2. onnx
"""

path = model.export(format="engine", imgsz=640, half=half, device=device, workspace=4)
print(f"Model successfully exported to: {path}")