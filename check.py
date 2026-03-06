import torch
import onnx
import onnxruntime as ort
import tvm
from tvm import relay

print(f"PyTorch version: {torch.__version__}")
print(f"ONNX version: {onnx.__version__}")
print(f"ONNX Runtime: {ort.get_device()}")
print(f"TVM version: {tvm.__version__}")

# 检查是否支持 LLVM (CPU 编译必选)
if tvm.runtime.enabled("llvm"):
    print("✅ TVM LLVM Backend is ready!")
else:
    print("❌ TVM LLVM Backend is missing. You may need to rebuild TVM with USE_LLVM=ON.")