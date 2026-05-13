import tensorrt as trt

onnx_file = "/home/nakarin/Ultra-Fast-Lane-Detection/ufld_v1_final.onnx"
engine_file = "models/lane/ufld_v1_fp16.engine"

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
config = builder.create_builder_config()

# Optimization for your RTX 5060: Use FP16 for speed
if builder.platform_has_fast_fp16:
    config.set_flag(trt.BuilderFlag.FP16)

# Standard UFLDv2 export usually uses Explicit Batch
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

with open(onnx_file, 'rb') as model:
    if not parser.parse(model.read()):
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        exit()

# Build the engine
print("Building engine... this might take a minute.")
plan = builder.build_serialized_network(network, config)
with open(engine_file, 'wb') as f:
    f.write(plan)

print(f"Done! Engine saved to {engine_file}")