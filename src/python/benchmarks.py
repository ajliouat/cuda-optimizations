import torch
import time
from src.python.cuda_ops import attention, reduction, memory_optimization

def benchmark_attention():
    batch_size, seq_len, num_heads, head_dim = 32, 128, 8, 64
    query = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    start = time.time()
    output = attention(query, key, value)
    torch.cuda.synchronize()
    end = time.time()

    print(f"Attention Time: {end - start:.4f}s")

def benchmark_reduction():
    input_tensor = torch.randn(1024 * 1024, device="cuda")

    start = time.time()
    result = reduction(input_tensor)
    torch.cuda.synchronize()
    end = time.time()

    print(f"Reduction Time: {end - start:.4f}s")

def benchmark_memory_optimization():
    input_tensor = torch.randn(1024 * 1024, device="cuda")

    start = time.time()
    output = memory_optimization(input_tensor)
    torch.cuda.synchronize()
    end = time.time()

    print(f"Memory Optimization Time: {end - start:.4f}s")

if __name__ == "__main__":
    benchmark_attention()
    benchmark_reduction()
    benchmark_memory_optimization()