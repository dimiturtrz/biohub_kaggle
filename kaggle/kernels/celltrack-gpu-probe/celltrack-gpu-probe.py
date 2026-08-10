"""Report what the Kaggle runtime actually provides — accelerator and packages — measured, not assumed.

Two failures this file exists to prevent, both of which cost real runs. A wrong `machineShape` does not fail
loudly: Kaggle substitutes its default P100, whose sm_60 the base image's torch has no kernels for, and the
only symptom is `cudaErrorNoKernelImageForDevice` thrown hours later from the first forward. And a linker
whose import is missing from both the base image and the kit's offline wheels dies the same way — late, deep,
and expensive. Kernel *pushes* are unlimited while *submissions* are five a day, so an answer that costs a
minute here is always cheaper than discovering it in a submission.
"""

import importlib.util
import logging

import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

logger.info("device_count: %d", torch.cuda.device_count())
for ordinal in range(torch.cuda.device_count()):
    major, minor = torch.cuda.get_device_capability(ordinal)
    total = torch.cuda.get_device_properties(ordinal).total_memory / 2**30
    logger.info("  cuda:%d = %s sm_%d%d %.1fGB", ordinal, torch.cuda.get_device_name(ordinal), major, minor, total)
logger.info("torch: %s | arch list: %s", torch.__version__, torch.cuda.get_arch_list())

# networkx backs the global min-cost-flow linker; the rest are the ILP path the 0.915 bundle uses.
for package in ("networkx", "scipy", "psutil", "pyscipopt", "ilpy", "motile", "tracksdata", "rustworkx"):
    spec = importlib.util.find_spec(package)
    if spec is None:
        logger.info("  %s: MISSING", package)
        continue
    module = importlib.import_module(package)
    logger.info("  %s: %s", package, getattr(module, "__version__", "present"))
