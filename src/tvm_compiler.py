import onnx
import tvm
import tvm.relay as relay
import tvm.auto_scheduler as auto_scheduler
import os
import numpy as np

AUTOTUNING_RECORDS = "tuning_records.json"


def compile_model(onnx_path="models/resnet18_sim.onnx", output_path="models/resnet18_tvm_cpu.so", auto_tune=True, tuning_trials=1000):
    # 1. 检查输入文件
    if not os.path.exists(onnx_path):
        print(f"[-] Error: {onnx_path} not found. Please run exporter.py first.")
        return

    # 2. 加载 ONNX 模型并定义输入尺寸
    onnx_model = onnx.load(onnx_path)
    input_name = "input"
    shape_dict = {input_name: (1, 3, 224, 224)}

    # 3. 将 ONNX 转换为 TVM Relay IR
    print("[*] Converting ONNX to Relay IR...")
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    # 4. 设定 Target
    target = tvm.target.Target("llvm -mcpu=skylake")

    # 5. 自动调优 (AutoScheduler)
    if auto_tune:
        print("[*] Starting Auto-tuning with AutoScheduler...")
        
        # 创建计算任务
        tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
        print(f"[*] Found {len(tasks)} tuning tasks")
        
        # 加载之前的 tuning records (如果存在)
        if os.path.exists(AUTOTUNING_RECORDS):
            print(f"[*] Loading previous tuning records from {AUTOTUNING_RECORDS}")
            with open(AUTOTUNING_RECORDS, "r") as f:
                old_records = auto_scheduler.load_records(AUTOTUNING_RECORDS)
        else:
            old_records = None
        
        # 创建 tuning options
        tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=tuning_trials,
            num_measures_per_round=64,
            early_stopping=100,
            runner=auto_scheduler.LocalRunner(timeout=30),
            measure_callbacks=[auto_scheduler.RecordToFile(AUTOTUNING_RECORDS)],
        )
        
        # 执行自动调优
        print(f"[*] Running {tuning_trials} tuning trials (this may take a while)...")
        tuner.tune(tune_option)
        
        print(f"[+] Tuning completed! Records saved to {AUTOTUNING_RECORDS}")
    else:
        print("[*] Skipping auto-tuning (using default schedule)")

    # 6. 应用优化并编译
    print(f"[*] Compiling with {target} (opt_level=3 + auto-scheduler)...")
    
    with auto_scheduler.ApplyHistoryBest(AUTOTUNING_RECORDS if auto_tune and os.path.exists(AUTOTUNING_RECORDS) else None):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            factory = relay.build(mod, target=target, params=params)

    # 7. 导出编译后的二进制库
    factory.export_library(output_path)
    print(f"[+] Success! TVM compiled library saved to: {output_path}")


if __name__ == "__main__":
    import sys
    auto_tune = "--no-tune" not in sys.argv
    compile_model(auto_tune=auto_tune)