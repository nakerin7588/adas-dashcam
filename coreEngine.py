import abc, os
import numpy as np
import onnxruntime
import tensorrt as trt
import pycuda.driver as cuda

# --- 1. GLOBAL CONTEXT MANAGER ---
# This fixes the "pop warning" and LogicErrors by ensuring all models
# share the same CUDA context tray.
_global_cuda_context = None

def get_cuda_context():
    global _global_cuda_context
    if _global_cuda_context is None:
        cuda.init()
        device = cuda.Device(0)
        # Create context; make_context automatically pushes it
        _global_cuda_context = device.make_context()
        # Pop it immediately so the stack starts clean
        _global_cuda_context.pop()
    return _global_cuda_context

class EngineBase(abc.ABC):
    '''
    Currently supports Onnx/TensorRT framework (.onnx, .trt, .engine)
    '''
    def __init__(self, model_path):
        if not os.path.isfile(model_path):
            raise Exception("The model path [%s] can't be found!" % model_path)
        
        valid_ext = ('.onnx', '.trt', '.engine')
        assert model_path.endswith(valid_ext), f"Model must be one of {valid_ext}"
        self._framework_type = None

    @property
    def framework_type(self):
        if self._framework_type is None:
            raise Exception("Framework type can't be None")
        return self._framework_type
    
    @framework_type.setter
    def framework_type(self, value):
        self._framework_type = value
    
    @abc.abstractmethod
    def get_engine_input_shape(self): pass
    
    @abc.abstractmethod
    def get_engine_output_shape(self): pass
    
    @abc.abstractmethod
    def engine_inference(self, input_tensor): pass

class TensorRTBase():
    def __init__(self, engine_file_path):
        self.providers = 'CUDAExecutionProvider'
        self.framework_type = "trt"
        
        # Use shared context
        self.cuda_driver_context = get_cuda_context()

        # Push to perform setup
        self.cuda_driver_context.push()
        try:
            self.stream = cuda.Stream()
            TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
            runtime = trt.Runtime(TRT_LOGGER)
            
            with open(engine_file_path, "rb") as f:
                self.engine = runtime.deserialize_cuda_engine(f.read())

            self.context = self.engine.create_execution_context()
            
            # Get dtype from the first tensor name (TRT 10 requirement)
            first_name = self.engine.get_tensor_name(0)
            self.dtype = trt.nptype(self.engine.get_tensor_dtype(first_name))
            
            # Allocate buffers
            self.host_inputs, self.cuda_inputs, self.host_outputs, self.cuda_outputs, self.bindings = self._allocate_buffers()
        finally:
            # Pop to keep stack balanced
            self.cuda_driver_context.pop()

    def _allocate_buffers(self):
        host_inputs, cuda_inputs, host_outputs, cuda_outputs, bindings = [], [], [], [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(cuda_mem))
            
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)
        return host_inputs, cuda_inputs, host_outputs, cuda_outputs, bindings

    def inference(self, input_tensor):
        self.cuda_driver_context.push()
        try:
            # 1. Copy to GPU
            np.copyto(self.host_inputs[0], input_tensor.ravel())
            cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)

            # 2. Set Tensor Addresses (TRT 10+)
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                self.context.set_tensor_address(name, self.bindings[i])

            # 3. Execute V3
            self.context.execute_async_v3(stream_handle=self.stream.handle)

            # 4. Copy to Host
            for h_out, c_out in zip(self.host_outputs, self.cuda_outputs):
                cuda.memcpy_dtoh_async(h_out, c_out, self.stream)
            
            self.stream.synchronize()
            return self.host_outputs
        finally:
            # GUARANTEED POP: Fixes the pop warning/crash
            self.cuda_driver_context.pop()

class TensorRTEngine(EngineBase, TensorRTBase):
    def __init__(self, engine_file_path):
        EngineBase.__init__(self, engine_file_path)
        TensorRTBase.__init__(self, engine_file_path)
        
        # Attribute required by LaneDetector
        self.engine_dtype = self.dtype 
        self.__load_engine_interface()

    def __load_engine_interface(self):
        self.__input_shape, self.__input_names = [], []
        self.__output_shapes, self.__output_names = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.__input_shape.append(shape)
                self.__input_names.append(name)
            else:
                self.__output_names.append(name)
                self.__output_shapes.append(shape)

    def get_engine_input_shape(self): return self.__input_shape[0]
    def get_engine_output_shape(self): return self.__output_shapes, self.__output_names

    def engine_inference(self, input_tensor):
        host_outputs = self.inference(input_tensor)
        trt_outputs = {}
        for i, (name, shape) in enumerate(zip(self.__output_names, self.__output_shapes)):
            actual_shape = list(shape)
            # Handle dynamic batch
            if len(actual_shape) > 0 and actual_shape[0] <= 0:
                actual_shape[0] = input_tensor.shape[0]
            trt_outputs[name] = host_outputs[i].reshape(actual_shape)
        return trt_outputs

class OnnxEngine(EngineBase):
    def __init__(self, onnx_file_path):
        EngineBase.__init__(self, onnx_file_path)
        prov = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if onnxruntime.get_device() == 'GPU' else ['CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(onnx_file_path, providers=prov)
        
        # Attribute required by LaneDetector
        in_type = self.session.get_inputs()[0].type
        self.engine_dtype = np.float16 if 'float16' in in_type else np.float32
        
        self.framework_type = "onnx"
        self.__load_engine_interface()

    def __load_engine_interface(self):
        self.__input_names = [x.name for x in self.session.get_inputs()]
        self.__output_names = [x.name for x in self.session.get_outputs()]
        self.__output_shapes = [x.shape for x in self.session.get_outputs()]

    def get_engine_input_shape(self): return self.session.get_inputs()[0].shape
    def get_engine_output_shape(self): return self.__output_shapes, self.__output_names
  
    def engine_inference(self, input_tensor):
        onnx_res = self.session.run(self.__output_names, {self.__input_names[0]: input_tensor})
        return {name: arr for name, arr in zip(self.__output_names, onnx_res)}