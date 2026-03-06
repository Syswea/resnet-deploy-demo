import torch
import torchvision.models as models
import onnx
from onnxsim import simplify

def export_and_optimize():
    # 1. 加载预训练模型
    print("[Step 1] Loading Pretrained ResNet-18...")
    model = models.resnet18(pretrained=True)
    model.eval() # 必须设置为 eval 模式，否则 BN 层会继续更新均值/方差

    # 2. 准备虚拟输入 (Dummy Input)
    # 形状为 [Batch, Channels, Height, Width]
    dummy_input = torch.randn(1, 3, 224, 224)

    # 3. 导出为标准 ONNX
    raw_onnx_path = "models/resnet18.onnx"
    print(f"[Step 2] Exporting to {raw_onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        raw_onnx_path,
        export_params=True,        # 存储训练好的权重参数
        opset_version=13,          # 建议使用 13，对现代 CPU 指令集支持较好
        do_constant_folding=True,  # 基础的常量折叠
        input_names=['input'],     # 设定输入节点名称，方便后续 TVM 调用
        output_names=['output'],   # 设定输出节点名称
    )

    # 4. 使用 ONNX-Simplifier 进行深度图优化
    # 这是面试中的亮点：手动进行图简化，消除冗余算子
    print("[Step 3] Simplifying ONNX Graph...")
    model_onnx = onnx.load(raw_onnx_path)
    model_simp, check = simplify(model_onnx)
    
    assert check, "Simplified ONNX model could not be validated"
    
    sim_onnx_path = "models/resnet18_sim.onnx"
    onnx.save(model_simp, sim_onnx_path)
    print(f"[Success] Simplified model saved to {sim_onnx_path}")

if __name__ == "__main__":
    import os
    if not os.path.exists("models"):
        os.makedirs("models")
    export_and_optimize()