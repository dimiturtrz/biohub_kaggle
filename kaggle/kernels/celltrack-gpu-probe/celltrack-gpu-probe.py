"""Report the accelerator a kernel actually gets — which machineShape yields two T4s, measured not guessed.

A wrong `machineShape` does not fail loudly: Kaggle falls back to its default accelerator (a P100, whose
sm_60 the base image's torch has no kernels for), and the failure only surfaces hours later as
`cudaErrorNoKernelImageForDevice` deep inside the first forward. This probe costs a minute and states the
answer, so the multi-GPU dispatcher can be pointed at a shape known to provision two devices.
"""

import torch

print("device_count:", torch.cuda.device_count())
for ordinal in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(ordinal)
    major, minor = torch.cuda.get_device_capability(ordinal)
    total = torch.cuda.get_device_properties(ordinal).total_memory / 2**30
    print(f"  cuda:{ordinal} = {name} sm_{major}{minor} {total:.1f}GB")
print("torch:", torch.__version__, "| arch list:", torch.cuda.get_arch_list())
