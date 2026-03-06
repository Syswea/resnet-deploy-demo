import os
import sys
import argparse

sys.path.insert(0, 'src')

from exporter import export_resnet18_to_onnx, generate_test_data
from tvm_compiler import compile_model_to_llvm
from cpu_engines import load_pytorch_engine, load_onnx_runtime_engine, load_tvm_engine
from benchmark import run_benchmark, print_benchmark_summary
import numpy as np


def setup_project():
    print("=" * 60)
    print("STEP 1: Exporting PyTorch model to ONNX")
    print("=" * 60)
    
    onnx_path = 'models/resnet18.onnx'
    export_resnet18_to_onnx(onnx_path, simplify_onnx=True)
    
    print("\n" + "=" * 60)
    print("STEP 2: Generating test data")
    print("=" * 60)
    
    generate_test_data('data/test_image.npy', num_samples=10)
    
    return onnx_path


def compile_with_tvm(onnx_path: str):
    print("\n" + "=" * 60)
    print("STEP 3: Compiling with TVM (LLVM)")
    print("=" * 60)
    
    tvm_lib_path = 'models/resnet18_tvm_cpu.so'
    
    if os.path.exists(onnx_path):
        compile_model_to_llvm(onnx_path, tvm_lib_path, target="llvm -mcpu=native", opt_level=3)
    else:
        print(f"ONNX model not found at {onnx_path}")
    
    return tvm_lib_path


def run_benchmarks():
    print("\n" + "=" * 60)
    print("STEP 4: Running benchmarks")
    print("=" * 60)
    
    test_data = np.load('data/test_image.npy')
    
    engines = []
    
    print("Loading PyTorch CPU engine (1 thread)...")
    engines.append(load_pytorch_engine(num_threads=1))
    
    print("Loading PyTorch CPU engine (4 threads)...")
    engines.append(load_pytorch_engine(num_threads=4))
    
    onnx_sim_path = 'models/resnet18_sim.onnx'
    if os.path.exists(onnx_sim_path):
        print("Loading ONNX Runtime CPU engine...")
        engines.append(load_onnx_runtime_engine(onnx_sim_path, intra_op_num_threads=4))
    
    tvm_lib_path = 'models/resnet18_tvm_cpu.so'
    if os.path.exists(tvm_lib_path):
        print("Loading TVM CPU engine...")
        engines.append(load_tvm_engine(tvm_lib_path))
    
    results = run_benchmark(engines, test_data, 'results/cpu_report.json')
    print_benchmark_summary(results)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='ResNet-18 CPU Deployment Benchmark')
    parser.add_argument('--skip-export', action='store_true', help='Skip model export')
    parser.add_argument('--skip-compile', action='store_true', help='Skip TVM compilation')
    parser.add_argument('--benchmark-only', action='store_true', help='Only run benchmark')
    
    args = parser.parse_args()
    
    if args.benchmark_only:
        run_benchmarks()
        return
    
    if not args.skip_export:
        onnx_path = setup_project()
    
    if not args.skip_compile:
        onnx_path = 'models/resnet18_sim.onnx'
        if os.path.exists(onnx_path):
            compile_with_tvm(onnx_path)
        else:
            print(f"Simplified ONNX not found at {onnx_path}")
    
    run_benchmarks()
    
    print("\n" + "=" * 60)
    print("All tasks completed!")
    print("Results saved to: results/cpu_report.json")
    print("=" * 60)


if __name__ == '__main__':
    main()
