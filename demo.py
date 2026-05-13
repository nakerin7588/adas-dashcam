import cv2
import os
import time
import yaml
import threading
import numpy as np

from ObjectDetector.ObjectDetector import ObjectDetector
from ObjectDetector.ObjectDetector import DistanceCalculator

from LaneDetector.ufldDetector.ultrafastLaneDetector import UltrafastLaneDetector
from LaneDetector.ufldDetector.ultrafastLaneDetectorV2 import UltrafastLaneDetectorV2
from LaneDetector.ufldDetector.utils import LaneModelType, OffsetType, CurvatureType

from Overlay.Overlay import Overlay

video_path = "/home/nakarin/adas-dashcam/RawVideos/Resize/กลับกรุงเทพ.mp4"
yaml_path = "./cfg/config.yaml"

with open(yaml_path, 'r') as f:
  config = yaml.safe_load(f)

lane_config = {
	# "model_path": "./models/lane/ufld_fp16.engine",
	# "model_type" : LaneModelType.UFLD_TUSIMPLE,
	"model_path": "./models/lane/ufldv2_fp16.engine",
	"model_type" : LaneModelType.UFLDV2_TUSIMPLE,
}

target_fps = 25
frame_time_ms = 1000 / target_fps

class AsyncLaneDetector:
    def __init__(self, detector):
        self.detector = detector
        self.frame_to_process = None
        self.lock = threading.Lock()
        self.running = True
        # daemon=True means this thread will auto-kill when the main script ends
        self.thread = threading.Thread(target=self._run_inference, daemon=True)
        self.thread.start()

    def update(self, frame):
        # Give the thread a fresh frame to work on without freezing the main loop
        with self.lock:
            self.frame_to_process = frame.copy()

    def draw(self, frame_show):
        # We lock the thread while drawing so the background process 
        # doesn't change the lane data in the middle of a draw call!
        with self.lock:
            self.detector.DrawDetectedOnFrame(frame_show)
            self.detector.DrawAreaOnFrame(frame_show)

    def stop(self):
        self.running = False
        self.thread.join()

    def _run_inference(self):
        while self.running:
          frame = None
          # Safely grab the latest frame
          with self.lock:
            if self.frame_to_process is not None:
                frame = self.frame_to_process
                self.frame_to_process = None # Consume it

          if frame is not None:
              # Lock the thread while detecting to prevent crashes with draw()
              with self.lock:
                  self.detector.DetectFrame(frame)
          else:
              # Chill out for 5ms if there's no new frame to save CPU
              time.sleep(0.005)
    def get_ego_lane_polygon(self):
      """Returns a polygon representing the area between the two ego lanes."""
      with self.lock:
          # According to your __process_output:
          # index 1 is left-ego, index 2 is right-ego
          lanes = self.detector.lane_info.lanes_points
          
          if len(lanes) > 2:
              left_lane = lanes[1]
              right_lane = lanes[2]
              
              # Check if we have enough points to form a zone
              if len(left_lane) > 1 and len(right_lane) > 1:
                  # Construct polygon: Left lane (top-to-bottom) then Right lane (bottom-to-top)
                  # This ensures the polygon doesn't 'self-intersect'
                  poly_points = np.array(left_lane + right_lane[::-1], dtype=np.int32)
                  return poly_points
      return None
                
class AsyncOverlay:
  def __init__(self, overlay):
      self.overlay = overlay
      self.frame = None
      self.val1 = 0
      self.val2 = 0
      self.is_drawing = False
      self.lock = threading.Lock()
      self.running = True
      self.thread = threading.Thread(target=self._run, daemon=True)
      self.thread.start()

  def trigger_draw(self, frame_show, val1, val2):
      # Tell the background thread to start drawing on this frame
      with self.lock:
        self.frame = frame_show
        self.val1 = val1
        self.val2 = val2
        self.is_drawing = True

  def wait_until_done(self):
      # The main thread will pause here for a fraction of a millisecond
      # to ensure the overlay is finished before showing the video.
      while self.is_drawing:
        time.sleep(0.001)

  def _run(self):
    while self.running:
      if self.is_drawing:
        # Draw directly onto the frame reference
        self.overlay.Draw(self.frame, self.val1, self.val2)
        with self.lock:
          self.is_drawing = False
      else:
        time.sleep(0.002) # Rest to save CPU
              
  def stop(self):
    self.running = False
    self.thread.join()
  
def check_lane_departure(async_lane, width):
    """
    Consolidated LDW logic without direction.
    Returns: (text, color)
    """
    # Adjust these based on testing:
    # 0.03 = ~3% off center, 0.06 = ~6% off center
    warn_thresh = 0.07
    alert_thresh = 0.12
    
    with async_lane.lock:
        lanes = async_lane.detector.lane_info.lanes_points
        # Check if we actually see the lines
        if len(lanes) < 3 or len(lanes[1]) < 2 or len(lanes[2]) < 2:
            return "LDW: SEARCHING", (150, 150, 150)
        
        # Get bottom X coordinates of left and right ego lanes
        left_x = lanes[1][-1][0]
        right_x = lanes[2][-1][0]
        
        lane_center = (left_x + right_x) / 2
        car_center = width / 2
        
        # Calculate how far we are from the middle
        abs_offset = abs(car_center - lane_center) / width

    # Status determination (Same style as FCW)
    if abs_offset > alert_thresh:
        return "LDW: DEPARTURE", (0, 0, 255)    # Red Alert
    elif abs_offset > warn_thresh:
        return "LDW: UNSTABLE", (0, 165, 255)   # Orange Warning
    else:
        return "LDW: STABLE", (100, 220, 100)    # Green Clear
  
def check_forward_collision(obj_results, lane_polygon, distance_cal):
    car_is_near = 0
    # Safety threshold: Alert if car is in lane and closer than 5 meters
    danger_threshold = 5.0 
    
    if lane_polygon is None:
        return 0

    for result in obj_results:
        for box in result.boxes:
            # Get coordinates and class
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_name = result.names[int(box.cls[0])]
            
            # 1. Filter for vehicles only
            if class_name in ['car', 'truck', 'bus', 'motorcycle']:
                # 2. Get the bottom-center of the vehicle (where it touches the road)
                bottom_center = (int((x1 + x2) / 2), int(y2))
                
                # 3. Use pointPolygonTest: 1=inside, 0=on edge, -1=outside
                is_in_lane = cv2.pointPolygonTest(lane_polygon, bottom_center, False) >= 0
                
                if is_in_lane:
                    # 4. Check distance only for cars in our lane
                    dist = distance_cal.CalculateDistance(bbox_width=x2-x1, cls=class_name)
                    if dist < danger_threshold:
                        car_is_near = 1
                        return car_is_near # Trigger immediately
                        
    return car_is_near

def main():
  # Initialize read and save video
  cap = cv2.VideoCapture(video_path)
  if (not cap.isOpened()) :
    raise Exception("video path is error. please check it.")

  width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  
  # Handle the directory
  output_dir = "demo"
  os.makedirs(output_dir, exist_ok=True)

  # Get the base filename (excluding original directory path)
  filename = os.path.basename(video_path)
  base_name = os.path.splitext(filename)[0]
  extension = ".mp4"
  counter = 1

  # Create the initial output path inside the demo folder
  output_path = os.path.join(output_dir, f"{base_name}_{counter}{extension}")

  # Check for existing files to avoid overwriting
  while os.path.exists(output_path):
      counter += 1
      output_path = os.path.join(output_dir, f"{base_name}_{counter}{extension}")

  # Initialize VideoWriter
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  vout = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

  cv2.namedWindow("ADAS Simulation", cv2.WINDOW_AUTOSIZE)
  
  # Initialize Object Detector model
  objDetector = ObjectDetector(yaml_path)
  
  # Initialize Lane Detector model
  if ( "UFLDV2" in lane_config["model_type"].name) :
    UltrafastLaneDetectorV2.set_defaults(lane_config)
    laneDetector = UltrafastLaneDetectorV2()
  else :
    UltrafastLaneDetector.set_defaults(lane_config)
    laneDetector = UltrafastLaneDetector()
    
  async_lane = AsyncLaneDetector(laneDetector)
  
  # Initialize Overlay
  overlay = Overlay()
  # async_overlay = AsyncOverlay(overlay)
  stats = {
    "t_prepare": [], 
    "t_predict": [], 
    "t_ui": [], 
    "t_total": [], 
    "fps": []
    }
  
  while cap.isOpened():
    loop_start = time.perf_counter() 
    
    # 1. PREPARE: Video Reading + Cropping
    t_prep_start = time.perf_counter()
    ret, frame = cap.read()
    if not ret: break
    
    top_limit = int(height * 0.45)
    bottom_limit = int(height * 0.23)
    crop_frame = frame.copy()
    crop_frame[0:top_limit, 0:width] = 0
    crop_frame[height - bottom_limit : height, 0:width] = 0
    t_prepare = (time.perf_counter() - t_prep_start) * 1000

    # 2. PREDICT: Object Detection + Lane Trigger
    t_pred_start = time.perf_counter()
    # Trigger lane (Async) and run Object Detection (Sync)
    async_lane.update(frame)
    obj_results = objDetector.DetectFrame(crop_frame) 
    t_predict = (time.perf_counter() - t_pred_start) * 1000 
    
    # 3. UI & LOGIC: Drawing, LDW/FCW Logic, and Panel Rendering
    t_ui_start = time.perf_counter()
    frame_show = crop_frame.copy()
    
    # Drawing detections and lanes
    detections = objDetector.Draw(frame_show, obj_results)
    # async_lane.draw(frame_show)
    
    # # ADAS Logic
    ego_lane_poly = async_lane.get_ego_lane_polygon()
    min_dist = float('inf')
    if ego_lane_poly is not None:
        for d in detections:
            if cv2.pointPolygonTest(ego_lane_poly, d["pos"], False) >= 0:
                if d["dist"] < min_dist:
                    min_dist = d["dist"]

    ldw_text, ldw_color = check_lane_departure(async_lane, width)

    # # Panel Rendering
    if min_dist <= 2.0:
        fcws_text, fcws_color = "FCWS: CAR AHEAD", (0, 0, 255)
    elif min_dist <= 5.0:
        fcws_text, fcws_color = "FCWS: WARNING", (0, 165, 255)
    else:
        fcws_text, fcws_color = "FCWS: CLEAR", (100, 220, 100)

    t_size, _ = cv2.getTextSize(fcws_text, overlay.font, overlay.font_scale, overlay.thickness)
    fcws_x = width - t_size[0] - 50 
    overlay._draw_transparent_panel(frame_show, fcws_text, (255, 255, 255), fcws_color, fcws_x, 20)
    overlay._draw_transparent_panel(frame_show, ldw_text, (255, 255, 255), ldw_color, 20, 20)
    t_ui = (time.perf_counter() - t_ui_start) * 1000

    # 4. TOTAL & FPS
    t_total = (time.perf_counter() - loop_start) * 1000
    fps = 1000 / t_total if t_total > 0 else 0
    
    # Store metrics
    stats["t_prepare"].append(t_prepare)
    stats["t_predict"].append(t_predict)
    stats["t_ui"].append(t_ui)
    # t_ui = 0
    stats["t_total"].append(t_total)
    stats["fps"].append(fps)
    
    # Terminal Dashboard
    print(f"Prepare: {t_prepare:4.1f}ms | Predict: {t_predict:4.1f}ms | "
          f"UI: {t_ui:4.1f}ms | Total: {t_total:4.1f}ms | FPS: {fps:4.1f}", end='\r')

    # Display FPS on frame
    cv2.putText(frame_show, f"FPS: {fps:.1f}", (width // 2 - 50, height - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.imshow("ADAS Simulation", frame_show)
    vout.write(frame_show)
    
    if cv2.waitKey(1) == ord('q'): 
        break
            
  # Clean up and final report
  async_lane.stop()
  
  print("\n\n" + "="*60)
  print(f"{'METRIC':<20} | {'AVG (ms)':<10} | {'MIN (ms)':<10} | {'MAX (ms)':<10}")
  print("-"*60)
  
  for key, values in stats.items():
      if not values: continue
      avg_val, min_val, max_val = np.mean(values), np.min(values), np.max(values)
      unit = "" if key == "fps" else " ms"
      print(f"{key:<20} | {avg_val:<10.2f} | {min_val:<10.2f} | {max_val:<10.2f}{unit}")

  print("="*60 + "\n")
  return 0


if __name__ == "__main__":
    main()