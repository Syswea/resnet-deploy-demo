import time
import numpy as np
import json
import os
from typing import List, Dict, Any
from scipy.spatial.distance import cosine


def warmup(engine, test_data: np.ndarray, warmup_iterations: int = 10):
    for _ in range(warmup_iterations):
        _ = engine.predict(test_data)


def measure_latency(engine, test_data: np.ndarray, num_iterations: int = 100) -> Dict[str, float]:
    latencies = []
    
    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = engine.predict(test_data)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    
    latencies.sort()
    
    return {
        'mean_ms': float(np.mean(latencies)),
        'median_ms': float(np.median(latencies)),
        'min_ms': float(np.min(latencies)),
        'max_ms': float(np.max(latencies)),
        'p95_ms': float(np.percentile(latencies, 95)),
        'p99_ms': float(np.percentile(latencies, 99)),
    }


def measure_throughput(engine, test_data: np.ndarray, duration_seconds: int = 5) -> Dict[str, float]:
    num_inferences = 0
    start_time = time.perf_counter()
    
    while time.perf_counter() - start_time < duration_seconds:
        _ = engine.predict(test_data)
        num_inferences += 1
    
    elapsed = time.perf_counter() - start_time
    throughput = num_inferences / elapsed
    
    return {
        'inferences': num_inferences,
        'elapsed_seconds': float(elapsed),
        'fps': float(throughput),
    }


def calculate_accuracy(output1: np.ndarray, output2: np.ndarray) -> float:
    return float(1 - cosine(output1.flatten(), output2.flatten()))


def run_benchmark(engines: List[Any], test_data: List[np.ndarray], output_path: str):
    results = {}
    
    for engine in engines:
        engine_name = engine.get_name()
        print(f"\nBenchmarking {engine_name}...")
        
        warmup_data = test_data[0] if isinstance(test_data, list) else test_data
        warmup(engine, warmup_data, warmup_iterations=10)
        
        latency_results = measure_latency(engine, warmup_data, num_iterations=100)
        
        throughput_results = measure_throughput(engine, warmup_data, duration_seconds=5)
        
        raw_output = engine.predict(warmup_data)
        
        results[engine_name] = {
            'latency_ms': latency_results,
            'throughput_fps': throughput_results,
            'output_shape': list(raw_output.shape),
        }
    
    engine_names = [e.get_name() for e in engines]
    for i in range(len(engine_names)):
        for j in range(i + 1, len(engine_names)):
            name1, name2 = engine_names[i], engine_names[j]
            output1 = engines[i].predict(test_data[0])
            output2 = engines[j].predict(test_data[0])
            accuracy = calculate_accuracy(output1, output2)
            
            key = f"cosine_similarity_{name1}_vs_{name2}"
            results[key] = accuracy
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    return results


def print_benchmark_summary(results: Dict[str, Any]):
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    for engine_name, metrics in results.items():
        if 'latency_ms' in metrics:
            print(f"\n{engine_name}:")
            print(f"  Latency (mean): {metrics['latency_ms']['mean_ms']:.2f} ms")
            print(f"  Latency (median): {metrics['latency_ms']['median_ms']:.2f} ms")
            print(f"  Latency (p95): {metrics['latency_ms']['p95_ms']:.2f} ms")
            print(f"  Throughput: {metrics['throughput_fps']['fps']:.2f} fps")
        elif engine_name.startswith('cosine_similarity'):
            print(f"  {engine_name}: {metrics:.6f}")
    
    print("=" * 60)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, 'src')
    from cpu_engines import load_pytorch_engine, load_onnx_runtime_engine, load_tvm_engine
    
    import numpy as np
    
    test_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    print("Loading engines...")
    engines = [
        load_pytorch_engine(num_threads=1),
        load_pytorch_engine(num_threads=4),
    ]
    
    if os.path.exists('models/resnet18_sim.onnx'):
        engines.append(load_onnx_runtime_engine('models/resnet18_sim.onnx', intra_op_num_threads=4))
    
    if os.path.exists('models/resnet18_tvm_cpu.so'):
        engines.append(load_tvm_engine('models/resnet18_tvm_cpu.so'))
    
    results = run_benchmark(engines, [test_data], 'results/cpu_report.json')
    print_benchmark_summary(results)
