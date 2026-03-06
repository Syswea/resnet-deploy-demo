import torch
import onnxruntime as ort
import tvm
from tvm.contrib import graph_executor
import os
import numpy as np

class ResNetCPUBase:
    def __init__(self, threads=1):
        self.input_shape = (1, 3, 224, 224)
        self.threads = threads

class PyTorchEngine(ResNetCPUBase):
    def __init__(self, threads=1):
        super().__init__(threads)
        import torchvision.models as models
        # 使用权重枚举确保兼容新旧版本 torchvision
        weights = models.ResNet18_Weights.DEFAULT
        self.model = models.resnet18(weights=weights)
        self.model.eval()
        torch.set_num_threads(self.threads)

    def run(self, input_data):
        with torch.no_grad():
            output = self.model(torch.from_numpy(input_data))
            return output.numpy()

class ORTEngine(ResNetCPUBase):
    def __init__(self, model_path="models/resnet18_sim.onnx", threads=1):
        super().__init__(threads)
        # 修正：使用正确的 SessionOptions 类名
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = self.threads
        sess_options.inter_op_num_threads = self.threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            model_path, 
            sess_options=sess_options, 
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name

    def run(self, input_data):
        return self.session.run(None, {self.input_name: input_data})[0]

class TVMEngine(ResNetCPUBase):
    def __init__(self, lib_path="models/resnet18_tvm_cpu.so", threads=1):
        super().__init__(threads)
        # 显式设置环境变量，控制 TVM 运行时线程
        os.environ["TVM_NUM_THREADS"] = str(self.threads)
        
        # 加载二进制编译库
        self.lib = tvm.runtime.load_module(lib_path)
        self.dev = tvm.cpu(0)
        self.module = graph_executor.GraphModule(self.lib["default"](self.dev))

    def run(self, input_data):
        self.module.set_input("input", tvm.nd.array(input_data))
        self.module.run()
        return self.module.get_output(0).asnumpy()