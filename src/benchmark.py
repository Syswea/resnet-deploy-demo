import numpy as np
import time
import json
import os
import sys
try:
    # 适配从 main.py 调用
    from src.cpu_engines import PyTorchEngine, ORTEngine, TVMEngine
except ImportError:
    # 适配直接在 src 目录下运行 python benchmark.py
    from cpu_engines import PyTorchEngine, ORTEngine, TVMEngine

def cosine_similarity(a, b):
    a = a.flatten()
    b = b.flatten()
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b) if (norm_a * norm_b) != 0 else 0.0

def run_test(engine_class, input_data, threads, label):
    try:
        engine = engine_class(threads=threads)
        # Warm-up
        for _ in range(5):
            _ = engine.run(input_data)
        
        latencies = []
        for _ in range(50): # 运行50次取平均
            t0 = time.perf_counter()
            output = engine.run(input_data)
            latencies.append((time.perf_counter() - t0) * 1000)
        
        avg_lat = np.mean(latencies)
        fps = 1000.0 / avg_lat
        return avg_lat, fps, output
    except Exception as e:
        print(f"\n[!] Error running {label}: {e}")
        return None, None, None

def main():
    # 创建结果目录
    if not os.path.exists("results"): os.makedirs("results")
    
    # 准备标准输入
    input_data = np.random.uniform(-1, 1, (1, 3, 224, 224)).astype(np.float32)
    all_results = {}

    for t in [1, 4]:
        print(f"\n" + "="*60)
        print(f" TESTING CONFIGURATION: {t} THREAD(S)")
        print("="*60)
        print(f"{'Backend':<15} | {'Latency (ms)':<15} | {'FPS':<10} | {'Cos_Sim':<10}")
        print("-" * 60)

        # 1. PyTorch (Baseline)
        pt_lat, pt_fps, pt_out = run_test(PyTorchEngine, input_data, t, "PyTorch")
        if pt_lat:
            print(f"{'PyTorch':<15} | {pt_lat:>12.2f} | {pt_fps:>10.2f} | {'1.0000':<10}")

        # 2. ONNX Runtime
        ort_lat, ort_fps, ort_out = run_test(ORTEngine, input_data, t, "ONNX RT")
        if ort_lat:
            sim = cosine_similarity(pt_out, ort_out)
            print(f"{'ONNX RT':<15} | {ort_lat:>12.2f} | {ort_fps:>10.2f} | {sim:>10.4f}")

        # 3. TVM (Target)
        tvm_lat, tvm_fps, tvm_out = run_test(TVMEngine, input_data, t, "TVM")
        if tvm_lat:
            sim = cosine_similarity(pt_out, tvm_out)
            print(f"{'TVM':<15} | {tvm_lat:>12.2f} | {tvm_fps:>10.2f} | {sim:>10.4f}")

        all_results[f"threads_{t}"] = {
            "pytorch": pt_lat, "onnx": ort_lat, "tvm": tvm_lat
        }

    with open("results/final_report.json", "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\n[+] All tests done. Report: results/final_report.json")

if __name__ == "__main__":
    main()