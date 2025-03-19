%%writefile largestEigenValue.cu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N 1024          // Dimension of the matrix (N x N)
#define NUM_ITER 100    // Run the power method for 100 iterations

// CUDA kernel for matrix-vector multiplication: y = A * x
__global__ void matVecMultiply(const float *A, const float *x, float *y, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += A[row * n + j] * x[j];
        }
        y[row] = sum;
    }
}

int main() {
    // Allocate host memory
    float *h_A = (float*)malloc(N * N * sizeof(float));
    float *h_x = (float*)malloc(N * sizeof(float));
    float *h_y = (float*)malloc(N * sizeof(float));

    // Initialize a diverse symmetric matrix A using random numbers.
    // We set a fixed seed for reproducibility.
    srand(0);
    for (int i = 0; i < N; i++) {
        for (int j = i; j < N; j++) {
            float value = (float)rand() / RAND_MAX; // random float in [0, 1]
            h_A[i * N + j] = value;
            h_A[j * N + i] = value; // Ensure symmetry.
        }
    }

    // Initialize the starting vector x with 1.0 values.
    for (int i = 0; i < N; i++) {
        h_x[i] = 1.0f;
    }

    // Normalize the initial vector x.
    float norm = 0.0f;
    for (int i = 0; i < N; i++) {
        norm += h_x[i] * h_x[i];
    }
    norm = sqrtf(norm);
    for (int i = 0; i < N; i++) {
        h_x[i] /= norm;
    }

    // Allocate device memory.
    float *d_A, *d_x, *d_y;
    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_x, N * sizeof(float));
    cudaMalloc((void**)&d_y, N * sizeof(float));

    // Copy matrix A and initial vector x to device memory.
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

    // Set up kernel launch parameters.
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    float eigenvalue = 0.0f;

    // Compute initial eigenvalue estimate.
    matVecMultiply<<<gridSize, blockSize>>>(d_A, d_x, d_y, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    float lambda = 0.0f;
    for (int i = 0; i < N; i++) {
        lambda += h_x[i] * h_y[i];
    }
    eigenvalue = lambda;
    printf("Iteration 0: Eigenvalue estimate = %f\n", eigenvalue);

    // Power method iterations.
    for (int iter = 1; iter < NUM_ITER; iter++) {
        // Compute y = A * x on the device.
        matVecMultiply<<<gridSize, blockSize>>>(d_A, d_x, d_y, N);
        cudaDeviceSynchronize();
        cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

        // Estimate the eigenvalue using the Rayleigh quotient.
        float lambda = 0.0f;
        for (int i = 0; i < N; i++) {
            lambda += h_x[i] * h_y[i];
        }

        // Normalize y to form the next iterate for x.
        float norm_y = 0.0f;
        for (int i = 0; i < N; i++) {
            norm_y += h_y[i] * h_y[i];
        }
        norm_y = sqrtf(norm_y);
        for (int i = 0; i < N; i++) {
            h_y[i] /= norm_y;
        }

        // Update x with the normalized y.
        for (int i = 0; i < N; i++) {
            h_x[i] = h_y[i];
        }
        // Copy the updated x back to the device.
        cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

        eigenvalue = lambda;
        // Print every 10 iterations.
        if (iter % 10 == 0) {
            printf("Iteration %d: Eigenvalue estimate = %f\n", iter, eigenvalue);
        }
    }

    // Final eigenvalue estimate.
    printf("Final eigenvalue estimate: %f\n", eigenvalue);

    // Free device and host memory.
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
    free(h_A);
    free(h_x);
    free(h_y);

    return 0;
}


!nvcc -gencode=arch=compute_75,code=sm_75 -allow-unsupported-compiler largestEigenValue.cu -o largestEigenValue
!./largestEigenValue

Iteration 0: Eigenvalue estimate = 512.135925
Iteration 10: Eigenvalue estimate = 512.300110
Iteration 20: Eigenvalue estimate = 512.300110
Iteration 30: Eigenvalue estimate = 512.300110
Iteration 40: Eigenvalue estimate = 512.300110
Iteration 50: Eigenvalue estimate = 512.300171
Iteration 60: Eigenvalue estimate = 512.300171
Iteration 70: Eigenvalue estimate = 512.300110
Iteration 80: Eigenvalue estimate = 512.300110
Iteration 90: Eigenvalue estimate = 512.300110
Final eigenvalue estimate: 512.300110
