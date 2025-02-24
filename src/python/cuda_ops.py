import torch
from torch.utils.cpp_extension import load

cuda_ops = load(
    name="cuda_ops",
    sources=[
        "src/cuda/attention.cu",
        "src/cuda/reduction.cu",
        "src/cuda/memory_optimization.cu",
    ],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=True,
)

attention = cuda_ops.attention
reduction = cuda_ops.reduction
memory_optimization = cuda_ops.memory_optimization