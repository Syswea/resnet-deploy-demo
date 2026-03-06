import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from typing import Optional, Any
import onnxruntime as ort


class ResNet18Simple(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=False)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc = resnet.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class PyTorchCPUEngine:
    def __init__(self, num_threads: int = 1):
        self.num_threads = num_threads
        torch.set_num_threads(num_threads)
        self.model = ResNet18Simple()
        self.model.eval()
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.from_numpy(input_data)
            output = self.model(tensor)
            return output.numpy()
    
    def get_name(self) -> str:
        return f"PyTorch-CPU(threads={self.num_threads})"


class ONNXRuntimeCPUEngine:
    def __init__(self, model_path: str, intra_op_num_threads: int = 1):
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = intra_op_num_threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        output = self.session.run([self.output_name], {self.input_name: input_data})[0]
        return output
    
    def get_name(self) -> str:
        return f"ONNX-Runtime-CPU(threads={self.session.get_session_options().intra_op_num_threads})"


class TVMCPUEngine:
    def __init__(self, lib_path: str, ctx: str = 'cpu'):
        import tvm
        from tvm import relay
        from tvm.contrib import graph_executor
        
        self.lib = tvm.runtime.load_module(lib_path)
        self.dev = tvm.device(ctx, 0)
        self.module = graph_executor.GraphModule(self.lib["default"](self.dev))
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        import tvm
        input_tensor = tvm.nd.array(input_data, device=self.dev)
        self.module.set_input('input', input_tensor)
        self.module.run()
        output = self.module.get_output(0).numpy()
        return output
    
    def get_name(self) -> str:
        return "TVM-CPU(llvm)"


def load_pytorch_engine(num_threads: int = 1) -> PyTorchCPUEngine:
    return PyTorchCPUEngine(num_threads)


def load_onnx_runtime_engine(model_path: str, intra_op_num_threads: int = 1) -> ONNXRuntimeCPUEngine:
    return ONNXRuntimeCPUEngine(model_path, intra_op_num_threads)


def load_tvm_engine(lib_path: str) -> TVMCPUEngine:
    return TVMCPUEngine(lib_path)
