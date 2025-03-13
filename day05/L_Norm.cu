%%writefile L_Norm.cu
#include <cuda_runtime.h>
#include <stdio.h>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

// Kernel to compute mean and variance per sample
__global__ void computeMeanVariance(float *input, float *mean, float *variance, 
                                   int N, int D, float epsilon) {
    int idx = blockIdx.x; // One block per sample
    if (idx >= N) return;

    extern __shared__ float sdata[];

    float *sum = sdata;             // First half of shared memory for sum
    float *sum_sq = sdata + blockDim.x; // Second half for sum of squares

    // Initialize thread sums
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float val = input[idx * D + i];
        local_sum += val;
        local_sum_sq += val * val;
    }
    sum[threadIdx.x] = local_sum;
    sum_sq[threadIdx.x] = local_sum_sq;
    __syncthreads();

    // Parallel reduction for sum and sum of squares
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            sum[threadIdx.x] += sum[threadIdx.x + stride];
            sum_sq[threadIdx.x] += sum_sq[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Store mean and variance for this sample
    if (threadIdx.x == 0) {
        float mu = sum[0] / D;
        mean[idx] = mu;
        variance[idx] = (sum_sq[0] / D - mu * mu) + epsilon;
    }
}

// Kernel to apply normalization
__global__ void normalize(float *input, float *output, float *mean, float *variance,
                         float *gamma, float *beta, int N, int D) {
    int idx = blockIdx.x; // One block per sample
    int j = threadIdx.x;  // One thread per feature
    if (idx >= N || j >= D) return;

    float mu = mean[idx];
    float sigma = sqrtf(variance[idx]);
    float x = input[idx * D + j];
    output[idx * D + j] = gamma[j] * (x - mu) / sigma + beta[j];
}

int main() {
    int N = 4;  // Batch size
    int D = 256; // Feature dimension
    float epsilon = 1e-5f;

    // Host arrays
    float *h_input = new float[N * D];
    float *h_output = new float[N * D];
    float *h_gamma = new float[D];
    float *h_beta = new float[D];

    // Initialize input and parameters
    for (int i = 0; i < N * D; i++) h_input[i] = (float)(i % 10); // Dummy data
    for (int i = 0; i < D; i++) {
        h_gamma[i] = 1.0f; // Scale = 1
        h_beta[i] = 0.0f;  // Shift = 0
    }

    // Device arrays
    float *d_input, *d_output, *d_mean, *d_variance, *d_gamma, *d_beta;
    CUDA_CHECK(cudaMalloc(&d_input, N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_variance, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta, D * sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma, D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta, D * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernels
    int threadsPerBlock = 256;
    int blocks = N;
    size_t sharedMemSize = 2 * threadsPerBlock * sizeof(float); // For sum and sum_sq

    computeMeanVariance<<<blocks, threadsPerBlock, sharedMemSize>>>(d_input, d_mean, d_variance, N, D, epsilon);
    CUDA_CHECK(cudaDeviceSynchronize());

    normalize<<<blocks, threadsPerBlock>>>(d_input, d_output, d_mean, d_variance, d_gamma, d_beta, N, D);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy output back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, N * D * sizeof(float), cudaMemcpyDeviceToHost));

    // Print a sample result
    printf("Sample output (first sample):\n");
    for (int j = 0; j < 10; j++) {
        printf("%f ", h_output[j]);
    }
    printf("\n");

    // Cleanup
    delete[] h_input;
    delete[] h_output;
    delete[] h_gamma;
    delete[] h_beta;
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_mean));
    CUDA_CHECK(cudaFree(d_variance));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));

    return 0;
}


!nvcc -gencode=arch=compute_75,code=sm_75 -allow-unsupported-compiler L_Norm.cu -o L_Norm
!./L_Norm

Sample output (first sample):
-1.553531 -1.204668 -0.855805 -0.506942 -0.158079 0.190784 0.539647 0.888511 1.237373 1.586236 
