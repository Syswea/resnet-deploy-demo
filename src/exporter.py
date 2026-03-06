import torch
import torchvision.models as models
import onnx
from onnxsim import simplify
import os

def export_resnet18(output_dir="models"):
    # 1. 创建保存目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    onnx_path = os.path.join(output_dir, "resnet18.onnx")
    sim_path = os.path.join(output_dir, "resnet18_sim.onnx")

    # 2. 加载预训练模型 (使用 ResNet-18)
    # weights=ResNet18_Weights.DEFAULT 是新版 torchvision 的写法
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()

    # 3. 定义伪输入 (Dummy Input)
    # ResNet 的输入通常是 [Batch, Channel, Height, Width] -> [1, 3, 224, 224]
    dummy_input = torch.randn(1, 3, 224, 224)

    # 4. 导出为原始 ONNX
    print(f"[*] Exporting model to {onnx_path}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True, 
        opset_version=12,      # 建议使用 12，兼容性较好
        do_constant_folding=True, 
        input_names=['input'], 
        output_names=['output']
    )

    # 5. 使用 onnx-simplifier 进行简化 (关键优化步骤)
    print(f"[*] Simplifying ONNX model...")
    model_onnx = onnx.load(onnx_path)
    model_simp, check = simplify(model_onnx)
    
    if check:
        onnx.save(model_simp, sim_path)
        print(f"[+] Simplified model saved to {sim_path}")
    else:
        print("[!] Simplified model check failed!")

if __name__ == "__main__":
    export_resnet18()