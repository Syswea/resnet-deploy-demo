import torch
import torch.nn as nn
import torchvision.models as models
import os
import onnx
from onnxsim import simplify


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


def export_resnet18_to_onnx(output_path: str, simplify_onnx: bool = True):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    model = ResNet18Simple()
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    
    onnx_path = output_path
    if simplify_onnx:
        onnx_path = output_path.replace('.onnx', '_raw.onnx')
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Exported ONNX to {onnx_path}")
    
    if simplify_onnx:
        onnx_model = onnx.load(onnx_path)
        model_simp, check = simplify(onnx_model)
        
        if check:
            onnx.save(model_simp, output_path)
            print(f"Simplified ONNX saved to {output_path}")
        else:
            print("Simplification failed, using original ONNX")
            import shutil
            shutil.copy(onnx_path, output_path)
    
    return output_path


def generate_test_data(output_path: str, num_samples: int = 10):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    test_data = []
    for _ in range(num_samples):
        sample = torch.randn(1, 3, 224, 224).numpy()
        test_data.append(sample)
    
    import numpy as np
    np.save(output_path, test_data)
    print(f"Generated {num_samples} test samples at {output_path}")
    
    return output_path


if __name__ == '__main__':
    export_resnet18_to_onnx('models/resnet18.onnx')
    generate_test_data('data/test_image.npy')
