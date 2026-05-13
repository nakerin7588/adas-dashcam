import cv2
import yaml
import numpy as np

from ultralytics import YOLO
from ObjectDetector.DistanceCalculator import DistanceCalculator
from Overlay.Overlay import Overlay

class ObjectDetector():
  def __init__(self, yaml_path: str):
    # Read yaml file.
    with open(yaml_path, 'r') as f:
      self.config = yaml.safe_load(f)
      
    # Extrack the configurations.
    model_cfg = self.config['obj_model_settings']
    self.path = model_cfg['path']
    self.stream = model_cfg['stream']
    self.persist = model_cfg['persist']
    self.half = model_cfg['half']
    self.max_det = model_cfg['max_det']
    self.iou = model_cfg['iou']
    self.tracker = model_cfg['tracker']
    self.agnostic_nms = model_cfg['agnostic_nms']
    self.save = model_cfg['save']
    self.use_tracker = model_cfg['use_tracker']
    self.verbose = model_cfg['verbose']
    self.classes = model_cfg['classes']
    self.colors = model_cfg['colors']
    
    self.model = YOLO(self.path, task='detect')
    self.distance_cal = DistanceCalculator(yaml_path)
    
    self.overlay = Overlay()
    
#     self.zone_points = np.array([
#     [100, 720],  # Bottom-Left
#     [500, 450],  # Top-Left
#     [780, 450],  # Top-Right
#     [1180, 720]  # Bottom-Right
# ], np.int32)
    
    # Add logging
    
  def DetectFrame(self, frame: cv2):
    if (self.use_tracker):
      return self.model.track(
        frame, 
        stream=self.stream, 
        persist=self.persist, 
        half=self.half,
        max_det=self.max_det,
        classes=self.classes,
        iou=self.iou,
        tracker=self.tracker,
        verbose=self.verbose,
        agnostic_nms=self.agnostic_nms,
        save=self.save
      )
    return self.model.predict(
        frame, 
        stream=self.stream, 
        half=self.half,
        max_det=self.max_det,
        verbose=self.verbose,
        classes=self.classes,
        iou=self.iou,
        agnostic_nms=self.agnostic_nms,
        save=self.save
      )
  
  def Draw(self, frame, results):
    """Draws basic bounding boxes and returns a list of detected vehicle distances."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thick = 1
    detections = []

    for result in results:
        names = result.names
        for box in result.boxes:
            coords = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
            class_name = names[int(box.cls[0])]
            
            # Calculate distance
            distance = self.distance_cal.CalculateDistance(bbox_width=x2-x1, cls=class_name)
            bottom_center = (int((x1 + x2) / 2), y2)
            
            det_info = {
                "class": class_name,
                "dist": distance,
                "pos": (int((x1 + x2) / 2), y2) # Bottom center
            }
            detections.append(det_info) # <--- Add to list

            # Draw basic box (White/Default)
            # bgr_color = tuple(self.colors.get(class_name, [255, 255, 255]))
            # cv2.rectangle(frame, (x1, y1), (x2, y2), bgr_color, 2)
            
            # label_text = f"{class_name} {distance:.1f}m"
            # (tw, th), _ = cv2.getTextSize(label_text, font, scale, thick)
            # tx, ty = x1, y1 - 5 if y1 - 5 > th else y1 + th + 5
            # cv2.rectangle(frame, (tx, ty - th - 2), (tx + tw + 4, ty + 2), bgr_color, -1)
            # cv2.putText(frame, label_text, (tx + 2, ty), font, scale, (0, 0, 0), thick, cv2.LINE_AA)
            
    return detections