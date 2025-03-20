%%writefile jacobi.cu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N 1000         // Dimension of the matrix (N x N)
#define NUM_ITER 1000  // Total number of Jacobi iterations

// CUDA kernel for one Jacobi iteration.
// Each thread computes one element of x_new.
__global__ void jacobiKernel(const float *A, const float *x_old, float *x_new, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sigma = 0.0f;
        // Compute the sum of A[i,j]*x_old[j] for all j except i.
        for (int j = 0; j < n; j++) {
            if (j != i) {
                sigma += A[i * n + j] * x_old[j];
            }
        }
        // Jacobi update: x_new[i] = (b[i] - sigma) / A[i,i]
        x_new[i] = (b[i] - sigma) / A[i * n + i];
    }
}

int main() {
    // Allocate host memory.
    float *h_A = (float*)malloc(N * N * sizeof(float));
    float *h_b = (float*)malloc(N * sizeof(float));
    float *h_x = (float*)malloc(N * sizeof(float));

    // Create a diagonally dominant matrix A and vector b.
    // For each row i, off-diagonals are random numbers in [0,1],
    // and the diagonal is set to be greater than the sum of off-diagonals.
    srand(0);
    for (int i = 0; i < N; i++) {
        float rowSum = 0.0f;
        for (int j = 0; j < N; j++) {
            if (i == j) {
                // We'll set the diagonal later.
                h_A[i * N + j] = 0.0f;
            } else {
                float val = (float)rand() / RAND_MAX;
                h_A[i * N + j] = val;
                rowSum += fabs(val);
            }
        }
        // Make the diagonal entry larger than the sum of the off-diagonals.
        h_A[i * N + i] = rowSum + 1.0f;
        // Also, set b[i] to a random value in [0,1]
        h_b[i] = (float)rand() / RAND_MAX;
    }

    // Initialize x (starting guess) to zeros.
    for (int i = 0; i < N; i++) {
        h_x[i] = 0.0f;
    }

    // Allocate device memory.
    float *d_A, *d_b, *d_x_old, *d_x_new;
    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_b, N * sizeof(float));
    cudaMalloc((void**)&d_x_old, N * sizeof(float));
    cudaMalloc((void**)&d_x_new, N * sizeof(float));

    // Copy matrix A, vector b, and initial guess x to device.
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_old, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

    // Set up CUDA kernel launch parameters.
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // For printing, allocate a host array to retrieve the solution.
    float *x_sol = (float*)malloc(N * sizeof(float));

    // Print initial guess (Iteration 0).
    cudaMemcpy(x_sol, d_x_old, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Iteration 0: x[0..4] = [%f, %f, %f, %f, %f]\n",
           x_sol[0], x_sol[1], x_sol[2], x_sol[3], x_sol[4]);

    // Perform Jacobi iterations.
    for (int iter = 1; iter <= NUM_ITER; iter++) {
        // Launch kernel: compute x_new = f(x_old)
        jacobiKernel<<<gridSize, blockSize>>>(d_A, d_x_old, d_x_new, d_b, N);
        cudaDeviceSynchronize();

        // Swap pointers: new becomes old for next iteration.
        float *temp = d_x_old;
        d_x_old = d_x_new;
        d_x_new = temp;

        // Print the solution at selected iterations.
        if ((iter <= 100 && iter % 10 == 0) || (iter >= 900 && iter % 10 == 0)) {
            cudaMemcpy(x_sol, d_x_old, N * sizeof(float), cudaMemcpyDeviceToHost);
            printf("Iteration %d: x[0..4] = [%f, %f, %f, %f, %f]\n",
                   iter, x_sol[0], x_sol[1], x_sol[2], x_sol[3], x_sol[4]);
        }
    }

    // Final solution printout.
    cudaMemcpy(x_sol, d_x_old, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Final solution (first 5 entries): [%f, %f, %f, %f, %f]\n",
           x_sol[0], x_sol[1], x_sol[2], x_sol[3], x_sol[4]);

    // Free device and host memory.
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_x_old);
    cudaFree(d_x_new);
    free(h_A);
    free(h_b);
    free(h_x);
    free(x_sol);

    return 0;
}

!nvcc -gencode=arch=compute_75,code=sm_75 -allow-unsupported-compiler jacobi.cu -o jacobi
!./jacobi

Iteration 0: x[0..4] = [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
Iteration 10: x[0..4] = [0.000053, 0.000354, 0.000224, 0.000885, -0.000481]
Iteration 20: x[0..4] = [0.000063, 0.000364, 0.000234, 0.000895, -0.000471]
Iteration 30: x[0..4] = [0.000072, 0.000373, 0.000244, 0.000904, -0.000462]
Iteration 40: x[0..4] = [0.000082, 0.000383, 0.000253, 0.000914, -0.000452]
Iteration 50: x[0..4] = [0.000091, 0.000392, 0.000263, 0.000923, -0.000443]
Iteration 60: x[0..4] = [0.000100, 0.000401, 0.000272, 0.000932, -0.000434]
Iteration 70: x[0..4] = [0.000109, 0.000410, 0.000281, 0.000941, -0.000425]
Iteration 80: x[0..4] = [0.000118, 0.000419, 0.000289, 0.000950, -0.000416]
Iteration 90: x[0..4] = [0.000126, 0.000428, 0.000298, 0.000959, -0.000407]
Iteration 100: x[0..4] = [0.000135, 0.000436, 0.000306, 0.000967, -0.000399]
Iteration 900: x[0..4] = [0.000467, 0.000769, 0.000639, 0.001300, -0.000066]
Iteration 910: x[0..4] = [0.000469, 0.000770, 0.000641, 0.001301, -0.000065]
Iteration 920: x[0..4] = [0.000471, 0.000772, 0.000642, 0.001303, -0.000063]
Iteration 930: x[0..4] = [0.000472, 0.000774, 0.000644, 0.001304, -0.000062]
Iteration 940: x[0..4] = [0.000474, 0.000775, 0.000645, 0.001306, -0.000060]
Iteration 950: x[0..4] = [0.000475, 0.000777, 0.000647, 0.001308, -0.000058]
Iteration 960: x[0..4] = [0.000477, 0.000778, 0.000648, 0.001309, -0.000057]
Iteration 970: x[0..4] = [0.000478, 0.000780, 0.000650, 0.001311, -0.000055]
Iteration 980: x[0..4] = [0.000480, 0.000781, 0.000651, 0.001312, -0.000054]
Iteration 990: x[0..4] = [0.000481, 0.000783, 0.000653, 0.001313, -0.000053]
Iteration 1000: x[0..4] = [0.000483, 0.000784, 0.000654, 0.001315, -0.000051]
Final solution (first 5 entries): [0.000483, 0.000784, 0.000654, 0.001315, -0.000051]
