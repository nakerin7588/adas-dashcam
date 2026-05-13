import math
import yaml

class DistanceCalculator():
  def __init__(self, yaml_path: str):
    with open(yaml_path, 'r') as f:
      self.config = yaml.safe_load(f)
      
    camera_cfg = self.config['camera_settings']
    self.original_width = camera_cfg['width']
    self.original_height = camera_cfg['height']
    self.original_fov = camera_cfg['fov']
    
    self.ref_size = {
      "car" : (1.8, 1.5), 
      "van" : (2.0, 2.2),
      "truck" : (2.5, 3.2),
      "pedestrian" : (0.6, 1.7),
      "cyclist" : (0.8, 1.8), 
    }
    
  def CalculateDistance(self, bbox_width, cls, yolo_input_width=640):
    diagonal_pixels = math.sqrt(self.original_width**2 + self.original_height**2)
    focal_length_px = (diagonal_pixels / 2) / math.tan(math.radians(self.original_fov / 2))
    
    # Scale focal length if you resized the image for YOLO
    scale_factor = yolo_input_width / self.original_width
    adjusted_focal_length = focal_length_px * scale_factor
    
    if bbox_width <= 0:
        print("Error: Bounding box width must be greater than zero.")
        return 0.0
        
    # Calculate final distance
    distance = (self.ref_size[cls][0] * adjusted_focal_length) / bbox_width
    return distance
  
  def check_forward_collision(self, distance):
    if (distance <= 2):
      return 1
    return 0