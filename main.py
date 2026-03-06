import os
import json
import numpy as np
import matplotlib.pyplot as plt
from src.exporter import export_resnet18
from src.tvm_compiler import compile_model
import src.benchmark as benchmark

def run_visualization():
    """读取 report.json 并生成性能对比柱状图"""
    report_path = "results/final_report.json"
    if not os.path.exists(report_path):
        print("[-] Error: Report not found, cannot visualize.")
        return

    with open(report_path, "r") as f:
        data = json.load(f)

    # 提取单线程和4线程的数据
    threads = ['1 Thread', '4 Threads']
    backends = ['PyTorch', 'ONNX RT', 'TVM']
    
    # 提取 Latency 数据
    t1_data = [data['threads_1']['pytorch'], data['threads_1']['onnx'], data['threads_1']['tvm']]
    t4_data = [data['threads_4']['pytorch'], data['threads_4']['onnx'], data['threads_4']['tvm']]

    x = np.arange(len(backends))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, t1_data, width, label='1 Thread', color='#3498db')
    rects2 = ax.bar(x + width/2, t4_data, width, label='4 Threads', color='#2ecc71')

    ax.set_ylabel('Latency (ms) - Lower is Better')
    ax.set_title('ResNet-18 CPU Inference Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(backends)
    ax.legend()

    # 标注数值
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.savefig("results/perf_bench.png")
    print("[+] Success! Visualization saved to results/perf_bench.png")

def main():
    print("=== ResNet-18 CPU Deployment Toolchain ===\n")

    # 1. 模型准备
    if not os.path.exists("models/resnet18_sim.onnx"):
        print("[*] Step 1: Exporting ONNX model...")
        export_resnet18()
    else:
        print("[*] Step 1: Skip export (model exists).")

    # 2. TVM 编译
    if not os.path.exists("models/resnet18_tvm_cpu.so"):
        print("[*] Step 2: Compiling with TVM...")
        compile_model()
    else:
        print("[*] Step 2: Skip TVM compilation (library exists).")

    # 3. 运行 Benchmark
    print("[*] Step 3: Running Benchmark (this may take a minute)...")
    benchmark.main()

    # 4. 可视化
    print("[*] Step 4: Generating performance charts...")
    run_visualization()

    print("\n[!] All steps completed! Check 'results/' folder for detailed report and charts.")

if __name__ == "__main__":
    main()