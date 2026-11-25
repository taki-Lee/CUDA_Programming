#include <cuda_runtime.h>
#include <iostream>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)    \
    do {                    \
        cudaError_t status = (call);    \
        if (status != cudaSuccess) {    \
            std::cerr << "CUDA error: " << cudaGetErrorString(status) << " at line " << __LINE__ << std::endl;  \
            exit(EXIT_FAILURE); \
        }   \
    } while (0)



__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

bool test_vector_add(int n){
    std::vector<float> a(n), b(n), c(n);
    for (int i=0;i<n;i++){
        a[i] = 1.0f + static_cast<float>(i % 101);
        b[i] = 2.0f + 0.5f * static_cast<float>(i % 97);
    }
    const size_t bytes = a.size() * sizeof(float);

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_c, 0, bytes));

    constexpr int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    constexpr int warmup = 20;
    constexpr int repeats = 100;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; ++i) {
        vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, n);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) {
        vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, n);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    const float avg_us = total_ms * 1000.0f / repeats;
    std::printf("N=%d, block=%d, average=%.3f us\n", n, threads, avg_us);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    for (int i=0;i<n;i++){
        const float expected = a[i] + b[i];
        if (c[i] != expected){
            std::cerr << "Error: c[" << i << "] = " << c[i] << ", expected " << expected << std::endl;
            return false;
        }
    }

    std::printf("PASS n=%d\n", n);
    return true;
}



int main(){
    const int sizes[] = {1, 31, 32, 33, 255, 256, 257, 1'000'003};
    for (int n : sizes){
        if (!test_vector_add(n)){
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}