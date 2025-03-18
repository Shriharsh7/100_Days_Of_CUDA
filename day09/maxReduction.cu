%%writefile maxReduction.cu

// max_reduction.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <float.h>  // for FLT_MIN and FLT_MAX

#define N 1024
#define BLOCK_SIZE 256

__global__ void maxReductionKernel(float *d_in, float *d_out, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    float max_val = -FLT_MAX;
    
    // Load data into shared memory, taking two elements per thread if available.
    if (i < n) {
        max_val = d_in[i];
        if (i + blockDim.x < n) {
            float temp = d_in[i + blockDim.x];
            if (temp > max_val) {
                max_val = temp;
            }
        }
    }
    sdata[tid] = max_val;
    __syncthreads();
    
    // Parallel reduction in shared memory to compute block maximum.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) {
                sdata[tid] = sdata[tid + s];
            }
        }
        __syncthreads();
    }
    
    // Write the block's maximum value to global memory.
    if (tid == 0) {
        d_out[blockIdx.x] = sdata[0];
    }
}

int main() {
    size_t size = N * sizeof(float);
    float *h_data = (float*)malloc(size);
    
    // Initialize the array with random values.
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)rand() / RAND_MAX * 100.0f; // Random values between 0 and 100
    }
    
    float *d_data, *d_intermediate;
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    
    // Determine grid size: each block processes 2 * BLOCK_SIZE elements.
    int gridSize = (N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
    cudaMalloc(&d_intermediate, gridSize * sizeof(float));
    
    // Launch the max reduction kernel.
    maxReductionKernel<<<gridSize, BLOCK_SIZE>>>(d_data, d_intermediate, N);
    
    // Copy the partial maximum values back to the host.
    float *h_intermediate = (float*)malloc(gridSize * sizeof(float));
    cudaMemcpy(h_intermediate, d_intermediate, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Final reduction on the CPU.
    float max_val = h_intermediate[0];
    for (int i = 1; i < gridSize; i++) {
        if (h_intermediate[i] > max_val) {
            max_val = h_intermediate[i];
        }
    }
    
    printf("Maximum value in the array: %f\n", max_val);
    
    // Cleanup.
    free(h_data);
    free(h_intermediate);
    cudaFree(d_data);
    cudaFree(d_intermediate);
    
    return 0;
}

!nvcc -gencode=arch=compute_75,code=sm_75 -allow-unsupported-compiler maxReduction.cu -o max_reduction
!./max_reduction

Maximum value in the array: 99.999359
