#include <iostream>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <limits.h>

// y[b, h] = x[b, h] + bias[h]
__global__ void bias_add_f32_kernel(const float *x, const float *b, float *y, int B, int H){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < B*H){
        int h = idx % H;
        y[idx] = x[idx] + b[h];
    }
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)  \
    if (((T).options().dtype() != (th_type))) { \
        std::cout << "Tensor Info:" << (T).options() << std::endl;  \
        throw std::runtime_error("values must be " #th_type);   \
    }   \

#define TORCH_BINDING_BIAS_ADD(packed_type, th_type, element_type) \
    void bias_add_##packed_type(torch::Tensor x, torch::Tensor b, torch::Tensor y) { \
        CHECK_TORCH_TENSOR_DTYPE(x, (th_type)); \
        CHECK_TORCH_TENSOR_DTYPE(b, (th_type)); \
        CHECK_TORCH_TENSOR_DTYPE(y, (th_type)); \
        TORCH_CHECK(x.dim()==2, "x must have shape [B, H]"); \
        const int B = static_cast<int>(x.size(0)); \
        const int H = static_cast<int>(x.size(1)); \
        const int N = static_cast<int>(x.numel()); \
        dim3 block(256); \
        dim3 grid((N + 256 -1) / 256); \
        bias_add_##packed_type_kernel<<<grid, block>>>( \
            reinterpret_cast<element_type *> (x.data_ptr()), \
            reinterpret_cast<element_type *> (b.data_ptr()), \
            reinterpret_cast<element_type *> (y.data_ptr()), \
            B, H \
        ); \
    }

TORCH_BINDING_BIAS_ADD(f32, torch::kFloat32, float)

PYBIND11_MODULE(TORCH_EXTENTION_NAME, m){
    m.def("bias_add_f32", &bias_add_f32);
}