#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>
#include <vector>

#include <torch/extension.h>
#include <torch/types.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <limits>

#define WARP_SIZE 32

__global__ void scale_kernel(const float *x, float *y, float scale, int numElements) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
        y[idx] = x[idx] * scale;
    }

}

void scale_cuda(torch::Tensor x, torch::Tensor y, float scale){
    int n = static_cast<int>(x.numel());

    TORCH_CHECK(x.is_cuda() && x.device() == y.device(), 
                "device check"
                );
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 && 
                y.scalar_type() == torch::kFloat32,
                "scalar type check"
                );
    TORCH_CHECK(x.dim() == 1 && x.sizes() == y.sizes() && 
                x.is_contiguous() && y.is_contiguous(),
                "dim && is_contiguous"
                )
    TORCH_CHECK(x.numel() <= std::numeric_limits<int>::max(),
                "numel check"                
                );
    
    constexpr int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    // const auto stream = 
    
    scale_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), scale, n
    );

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scale_cuda", &scale_cuda);
}

