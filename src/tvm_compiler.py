import onnx
import tvm
import tvm.relay as relay
import os

def compile_model(onnx_path="models/resnet18_sim.onnx", output_path="models/resnet18_tvm_cpu.so"):
    # 1. 检查输入文件
    if not os.path.exists(onnx_path):
        print(f"[-] Error: {onnx_path} not found. Please run exporter.py first.")
        return

    # 2. 加载 ONNX 模型并定义输入尺寸
    onnx_model = onnx.load(onnx_path)
    input_name = "input"  # 需与 exporter.py 中的 input_names 一致
    shape_dict = {input_name: (1, 3, 224, 224)}

    # 3. 将 ONNX 转换为 TVM Relay IR
    # Relay 是 TVM 的高级函数式中间表示
    print("[*] Converting ONNX to Relay IR...")
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    # 4. 设定 Target (CPU 优化的灵魂)
    # 'llvm' 表示使用 LLVM 编译器后端
    # '-mcpu=native' 会指示 TVM 探测当前 CPU 的指令集（如 AVX2/AVX-512/NEON）
    target = "llvm -mcpu=skylake"
    
    # 5. 执行编译优化
    # opt_level=3 开启包括算子融合、布局转换在内的全部优化 Pass
    print(f"[*] Compiling with {target} (opt_level=3)...")
    with tvm.transform.PassContext(opt_level=3):
        factory = relay.build(mod, target=target, params=params)

    # 6. 导出编译后的二进制库 (.so 文件)
    # 部署时只需要加载这个 .so 文件和轻量级的 TVM runtime
    factory.export_library(output_path)
    print(f"[+] Success! TVM compiled library saved to: {output_path}")

if __name__ == "__main__":
    compile_model()