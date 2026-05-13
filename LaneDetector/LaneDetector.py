import cv2
import yaml
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit 
from scipy.special import softmax

class LaneDetector():
    def __init__(self, yaml_path: str):
        with open(yaml_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        lane_cfg = self.config['lane_model_settings']
        self.engine_path = lane_cfg['path']
        
        # Load Engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # --- AUTO-DETECT ENGINE REQUIREMENTS ---
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        
        # Get shapes (e.g., [1, 3, 640, 800])
        self.input_shape = self.engine.get_tensor_shape(self.input_name)
        self.output_shape = self.engine.get_tensor_shape(self.output_name)
        
        # Extract dimensions for resizing later
        # TensorRT shapes are usually [Batch, Channels, Height, Width]
        self.exp_h = self.input_shape[2]
        self.exp_w = self.input_shape[3]
        self.exp_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        
        print(f"[LaneDetector] Engine loaded. Expects: {self.exp_w}x{self.exp_h} ({self.exp_dtype})")

        # Config parameters for drawing
        self.num_lanes = lane_cfg.get('num_lanes', 4)
        self.griding_num = lane_cfg.get('griding_num', 200)
        self.row_anchors = np.array(lane_cfg.get('row_anchors', []))
        self.cls_num_per_lane = len(self.row_anchors)
        self.draw_color = lane_cfg.get('color', [0, 255, 0])
        self.draw_thickness = lane_cfg.get('thickness', 5)
        
        # Allocate buffers
        self.inputs, self.outputs, self.stream = self._allocate_buffers()

    def _allocate_buffers(self):
        inputs, outputs = [], []
        stream = cuda.Stream()
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.context.set_tensor_address(name, int(device_mem))
            
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
                
        return inputs, outputs, stream

    def DetectFrame(self, frame: np.ndarray):
        original_h, original_w = frame.shape[:2]
        
        # 1. Pre-processing (Now using self.exp_w and self.exp_h from engine)
        resized = cv2.resize(frame, (self.exp_w, self.exp_h))
        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize and cast to the EXACT dtype the engine wants (float32 or float16)
        img_rgb = img_rgb.astype(self.exp_dtype) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=self.exp_dtype)
        std = np.array([0.229, 0.224, 0.225], dtype=self.exp_dtype)
        img_rgb = (img_rgb - mean) / std
        
        img_chw = np.transpose(img_rgb, (2, 0, 1)).ravel()
        
        # 2. Inference
        np.copyto(self.inputs[0]['host'], img_chw) # Error fixed: shapes now match
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()

        # 3. Post-processing
        out = self.outputs[0]['host'].reshape(self.output_shape)[0]
        
        prob = softmax(out[:-1, :, :], axis=0)
        idx = np.arange(self.griding_num - 1) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        
        detected_lanes = []
        for i in range(self.num_lanes):
            lane_points = []
            for j in range(self.cls_num_per_lane):
                if out[-1, j, i] < np.max(out[:-1, j, i]): 
                    x = int(loc[j, i] / self.griding_num * original_w)
                    # Use actual engine expected height for normalization
                    y = int(self.row_anchors[j] / self.exp_h * original_h)
                    lane_points.append((x, y))
            if lane_points: detected_lanes.append(lane_points)

        return detected_lanes

    def Draw(self, frame, lane_results):
        bgr_color = tuple(self.draw_color)
        for lane_points in lane_results:
            for i in range(len(lane_points) - 1):
                cv2.line(frame, lane_points[i], lane_points[i+1], bgr_color, self.draw_thickness)
        return frame