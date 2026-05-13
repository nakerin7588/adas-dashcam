from ultralytics import YOLO
from pathlib import Path

# Load an official or custom model
path = "/home/nakarin/adas-dashcam/runs/detect/train-7/weights/best.engine"
model = YOLO(model=path, task="detect")

# int8 = True
# optimize = False # True if use mobile or constrained environments
# dynamic = True # Enable dyanamic input size, auto set when use TensorRT
# device = 0 # If use GPU (=0), auto set when use TensorRT; CPU (device=cpu)

# end2end = None # Set it to false to enable traditional way of post process

input_width = 1280 # 640 for 16:9
input_height = 736 # 384 for 16:9
# path = model.export(format="ncnn", imgsz=[input_height,input_width], half=True)
# print(f"Model successfully exported to: {path}")

# Perform tracking with the model
# results = model.track("https://youtu.be/LNwODJXcvt4", show=True)  # Tracking with default tracker
results = model.predict(
    "/home/nakarin/adas-dashcam/RawVideos/Resize/ลงทางด่วนไปซีคอน.mp4",
    show=True,             # Set to True to see the video window 
    stream=True,           # Keep True for memory efficiency
    save=True,
    # persist=True,
    half=True,
    max_det=10,
    #tracker="bytetrack.yaml",
    agnostic_nms=False,
    iou=0.7
)
print("Inference started...")
for r in results:
    # กระบวนการจัดการผลลัพธ์ของคุณ
    pass

print("Finished processing!")