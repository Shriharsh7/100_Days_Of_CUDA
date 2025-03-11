%%writefile matrix_mutiplication.cu

#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel for matrix multiplication
__global__ void matrixMulKernel(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    const int N = 1024;          // Matrix dimension
    const int BLOCK_SIZE = 16;   // Threads per block dimension
    
    float *h_A, *h_B, *h_C; // Host matrices
    float *d_A, *d_B, *d_C; // Device matrices
    size_t size = N * N * sizeof(float);
    
    // Allocate host memory
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);
    
    // Initialize host matrices
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }
    
    // Allocate device memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // Configure grid/block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1)/BLOCK_SIZE, 
                (N + BLOCK_SIZE - 1)/BLOCK_SIZE);
    
    // Launch kernel
    matrixMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    
    // Copy result back
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Verify result
    bool correct = true;
    for (int i = 0; i < N * N; i++) {
        if (fabs(h_C[i] - 2*N) > 1e-5) {
            correct = false;
            break;
        }
    }
    printf(correct ? "Success!\n" : "Failure!\n");
    
    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}

/* 
!nvcc -gencode=arch=compute_75,code=sm_75 -allow-unsupported-compiler matrix_mutiplication.cu -o matrix_mutiplication
!./matrix_mutiplication

Success!

*/
