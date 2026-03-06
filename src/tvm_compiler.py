import os
import tvm
from tvm import relay
import numpy as np


def compile_model_to_llvm(onnx_model_path: str, output_lib_path: str, target: str = "llvm -mcpu=native", opt_level: int = 3):
    onnx_model = tvm.frontend.from_onnx(onnx_model_path)
    
    with tvm.transform.PassContext(opt_level=opt_level):
        lib = relay.build(onnx_model, target=target)
    
    os.makedirs(os.path.dirname(output_lib_path), exist_ok=True)
    lib.export_library(output_lib_path)
    
    print(f"TVM compiled model saved to {output_lib_path}")
    return output_lib_path


def compile_with_autotune(onnx_model_path: str, output_lib_path: str, target: str = "llvm -mcpu=native"):
    from tvm import autotvm
    
    onnx_model = tvm.frontend.from_onnx(onnx_model_path)
    
    tasks = autotvm.task.extract_from_program(onnx_model, target=target, ops=[relay.op.get("conv2d"), relay.op.get("dense")])
    
    tuning_option = {
        'log_filename': 'tuning_log.txt',
        'tuner': 'xgb',
        'n_trial': 10,
        'early_stopping': 5,
        'measure_option': autotvm.measure_option(
            builder=autotvm.builder.Builder('llvm', timeout=10),
            runner=autotvm.runner.Runner(timeout=10)
        )
    }
    
    for i, task in enumerate(tasks):
        print(f"Tuning task {i+1}/{len(tasks)}: {task.name}")
        tuner = autotvm.tuner.XGBTuner(task)
        tuner.tune(n_trial=min(10, len(task.config_space)), early_stopping=5, measure_option=tuning_option['measure_option'])
    
    with autotvm.apply_history_best("tuning_log.txt"):
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(onnx_model, target=target)
    
    os.makedirs(os.path.dirname(output_lib_path), exist_ok=True)
    lib.export_library(output_lib_path)
    
    print(f"TVM compiled model with auto-tuning saved to {output_lib_path}")
    return output_lib_path


if __name__ == '__main__':
    onnx_path = 'models/resnet18_sim.onnx'
    output_path = 'models/resnet18_tvm_cpu.so'
    
    if os.path.exists(onnx_path):
        compile_model_to_llvm(onnx_path, output_path)
    else:
        print(f"ONNX model not found at {onnx_path}")
