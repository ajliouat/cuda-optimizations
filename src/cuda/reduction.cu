#include <torch/extension.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"

__global__ void reduction_kernel(const float* input, float* output, int size) {
    // Custom CUDA kernel for parallel reduction
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        shared[tid] = input[idx];
    } else {
        shared[tid] = 0.0f;
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = shared[0];
    }
}

torch::Tensor reduction(torch::Tensor input) {
    auto output = torch::zeros({1}, input.options());
    int size = input.numel();
    int threads = 1024;
    int blocks = (size + threads - 1) / threads;

    reduction_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reduction", &reduction, "Custom CUDA reduction kernel");
}