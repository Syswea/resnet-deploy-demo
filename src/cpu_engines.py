import torch
import onnxruntime as ort
import numpy as np
import tvm
from tvm.contrib import graph_executor
import time

class BaseEngine:
    """所有推理引擎的基类，确保接口统一"""
    def __init__(self):
        self.model = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

# 1. PyTorch CPU Engine (基准测试)
class PyTorchCPUEngine(BaseEngine):
    def __init__(self, model_ptr):
        super().__init__()
        self.model = model_ptr
        self.model.eval()
        # 面试点：限制线程数以获得稳定的 benchmark 环境
        torch.set_num_threads(1) 

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor_in = torch.from_numpy(inputs)
            output = self.model(tensor_in)
        return output.numpy()

# 2. ONNX Runtime CPU Engine
class ORTCPUEngine(BaseEngine):
    def __init__(self, onnx_path):
        super().__init__()
        # 设置线程属性，优化 CPU 利用率
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        self.session = ort.InferenceSession(
            onnx_path, 
            sess_options, 
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: inputs})[0]

# 3. TVM CPU Engine
class TVMCPUEngine(BaseEngine):
    def __init__(self, lib_path):
        super().__init__()
        # 加载编译后的动态库 (.so 或 .tar)
        loaded_lib = tvm.runtime.load_module(lib_path)
        self.module = graph_executor.GraphModule(loaded_lib["default"](tvm.cpu()))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # TVM 需要显式设置输入数据
        self.module.set_input("input", tvm.nd.array(inputs))
        self.module.run()
        # 获取第0个输出
        return self.module.get_output(0).asnumpy()