#include <cuda_runtime.h>
#include <iostream>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <limits>


__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

void vector_add_out(torch::Tensor A, torch::Tensor B, torch::Tensor out) {
    TORCH_CHECK(A.is_cuda() && A.device() == B.device() && 
                A.device() == out.device(),
                "a, b, and out must be on the same CUDA device");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 &&
                B.scalar_type() == torch::kFloat32 &&
                out.scalar_type() == torch::kFloat32,
                "a, b, and out must have dtype float32");
    TORCH_CHECK(A.dim() == 1 && A.sizes() == B.sizes() && 
                A.sizes() == out.sizes() && A.is_contiguous() && 
                B.is_contiguous() && out.is_contiguous(),
                "a, b, and out must be contiguous vectors of the same length");
    TORCH_CHECK(A.numel() <= std::numeric_limits<int>::max(),
                "vector is too large for the kernel's int index");
    
    const int n = static_cast<int>(A.numel());
    if (n == 0){
        return;
    }
    const c10::cuda::CUDAGuard device_guard(A.device());
    const auto stream = c10::cuda::getCurrentCUDAStream(A.get_device());
    constexpr int threads = 256;
    const int blocks = (n - 1) / threads + 1;
    vectorAdd<<<blocks, threads, 0, stream.stream()>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), out.data_ptr<float>(), n
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add_out", &vector_add_out);
}