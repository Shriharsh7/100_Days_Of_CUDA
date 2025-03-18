%%writefile softmax.cu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <float.h>

// Naive softmax kernel: every thread recomputes normalization factors by iterating over the entire array.
__global__ void softmaxNaive(const float *d_input, float *d_output, int N) {
    float maxVal = -FLT_MAX;
    // Compute maximum value over the entire array (inefficient but simple)
    for (int i = 0; i < N; i++) {
        maxVal = fmaxf(maxVal, d_input[i]);
    }
    
    float expSum = 0.0f;
    // Compute the sum of the exponentials
    for (int i = 0; i < N; i++) {
        expSum += expf(d_input[i] - maxVal);
    }
    
    // Each thread processes one element
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        d_output[idx] = expf(d_input[idx] - maxVal) / expSum;
    }
}

// Optimized softmax kernel using shared memory reduction (assumes one block per softmax call)
__global__ void softmaxShared(const float *d_input, float *d_output, int N) {
    extern __shared__ float sdata[];  // dynamically allocated shared memory
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Load data into shared memory; if idx is out-of-bounds, use -FLT_MAX for max reduction
    sdata[tid] = (idx < N) ? d_input[idx] : -FLT_MAX;
    __syncthreads();
    
    // Reduction: compute the maximum value in the block
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }
    float localMax = sdata[0];
    __syncthreads();
    
    // Compute exponential values using the block's maximum for numerical stability
    float expVal = (idx < N) ? expf(d_input[idx] - localMax) : 0.0f;
    sdata[tid] = expVal;
    __syncthreads();
    
    // Reduction: compute the sum of exponentials in the block
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    float sumExp = sdata[0];
    __syncthreads();
    
    // Write the normalized softmax result
    if (idx < N) {
        d_output[idx] = expVal / sumExp;
    }
}

int main() {
    // Use a larger vector size for measurable timings.
    int N = 1 << 20;  // 1,048,576 elements
    size_t size = N * sizeof(float);
    
    // Allocate host memory
    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);
    
    // Initialize input with random values
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)rand() / RAND_MAX;
    }
    
    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // --- Measure Naive Softmax ---
    cudaEventRecord(start);
    softmaxNaive<<<blocks, threadsPerBlock>>>(d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float timeNaive;
    cudaEventElapsedTime(&timeNaive, start, stop);
    
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    printf("Naive Softmax Time: %f ms\n", timeNaive);
    
    // --- Measure Shared Memory Softmax ---
    // Reset device input (if necessary)
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    cudaEventRecord(start);
    softmaxShared<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float timeShared;
    cudaEventElapsedTime(&timeShared, start, stop);
    
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    printf("Shared Memory Softmax Time: %f ms\n", timeShared);
    
    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    
    return 0;
}

!nvcc -gencode=arch=compute_75,code=sm_75 -allow-unsupported-compiler softmax.cu -o softmax
!./softmax

Naive Softmax Time: 3633.003662 ms
Shared Memory Softmax Time: 0.095392 ms
