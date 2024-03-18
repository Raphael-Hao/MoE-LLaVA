# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# --------
# Licensed under the MIT license.
# --------
# Author: Weihao Cui


import torch

from torch.utils.benchmark import Timer


from brt.jit import make_jit_kernel

from brt.jit.tvm import TVMTuner

from brt.jit.codegen import ModuleKernel

from pynvml import (
    nvmlInit,
    nvmlDeviceGetComputeRunningProcesses,
    nvmlDeviceGetHandleByIndex
)

_NVM_initialized = False

def check_if_exclusive(device_id: int):
    global _NVM_initialized
    if not _NVM_initialized:
        nvmlInit()
        _NVM_initialized = True
    handle = nvmlDeviceGetHandleByIndex(device_id)
    process = nvmlDeviceGetComputeRunningProcesses(handle)
    return len(process) == 0

torch.set_default_device("cuda:0")
if not check_if_exclusive(0):
    raise RuntimeError("device is not exclusive!")

all_bs = [
    # 1,
    # 2,
    # 4,
    # 8,
    # 16,
    # 32,
    # 64,
    # 128,
    # 256,
    # 512,
    620
]


in_out_features = [
    # [768, 3072],
    # [3072, 768]
    [2560,10240],
    [10240,2560],
]


for bs in all_bs:

    for in_features, out_features in in_out_features:
        input_infos = {"input_0": (bs, in_features)}
        output_infos = {"output_0": (bs, out_features)}
        parameters = {
            "in_features": in_features,
            "out_features": out_features,
        }

        kernel_name = f"LinearBias_{bs}_{in_features}_{out_features}"
        linear = torch.nn.Linear(in_features, out_features, bias=True, dtype=torch.float16, device="cuda").eval()
        tvm_tuner = TVMTuner(dtype=torch.float16)

        tvm_tuner.import_pt_netlet(
            "LinearBias",
            "forward",
            linear,
            input_infos,
            output_infos,
            parameters,
            # log_fname,
        )

        print(f"#### # LinearBias {bs} {in_features} {out_features}")

        if tvm_tuner.tune_log_file.exists():
            print(tvm_tuner.tune_log_file)
            with open(str(tvm_tuner.tune_log_file)) as f:
                num_trials = len(f.readlines())
            if num_trials < 64:
                print("#### Find incomplete record, continue")
                tvm_tuner.task_scheduler.load_log_file = str(tvm_tuner.tune_log_file)
            else:
                print("#### Find tuned kernel, pass")
        else:

            print("#### Start tuning kernel")
            tvm_tuner.tune_netlet()

        # tvm_tuner.task_scheduler.load_log_file = str(tvm_tuner.tune_log_file)

        tvm_tuner.tune_netlet()
        tvm_tuner.insert_netlet_to_storage()

