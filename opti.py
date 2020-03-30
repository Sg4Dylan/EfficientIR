import onnx
from onnx import optimizer

# 简化模型
onnxfile = 'imagenet-b2'
onnx_model = onnx.load(f'{onnxfile}.onnx')
passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
optimized_model = optimizer.optimize(onnx_model, passes)
onnx.save(optimized_model, f'{onnxfile}-opti.onnx')
