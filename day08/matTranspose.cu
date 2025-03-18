%%writefile matTranspose.cu

// matrix_transpose.cu
#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8

// Kernel to transpose a matrix using shared memory
__global__ void matrixTranspose(float *odata, const float *idata, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 avoids bank conflicts

    // Calculate the global indices for the input matrix
    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Load the matrix tile from global memory into shared memory
    if (xIndex < width && yIndex < height) {
        tile[threadIdx.y][threadIdx.x] = idata[yIndex * width + xIndex];
    }
    
    __syncthreads();
    
    // Calculate the transposed indices (swap blockIdx.x and blockIdx.y)
    int transposedX = blockIdx.y * TILE_DIM + threadIdx.x;
    int transposedY = blockIdx.x * TILE_DIM + threadIdx.y;
    
    // Write the transposed tile to global memory
    if (transposedX < height && transposedY < width) {
        odata[transposedY * height + transposedX] = tile[threadIdx.x][threadIdx.y];
    }
}

int main(void) {
    // Matrix dimensions (assumed square for simplicity)
    int width = 1024;
    int height = 1024;
    size_t size = width * height * sizeof(float);

    // Allocate host memory
    float *h_in = (float*)malloc(size);
    float *h_out = (float*)malloc(size);

    // Initialize the input matrix with sequential values
    for (int i = 0; i < width * height; i++) {
        h_in[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float *d_in, *d_out;
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);

    // Copy the input matrix from host to device
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // Setup execution configuration
    dim3 blockDim(TILE_DIM, BLOCK_ROWS);
    dim3 gridDim((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);

    // Launch the transpose kernel
    matrixTranspose<<<gridDim, blockDim>>>(d_out, d_in, width, height);

    // Copy the transposed matrix back to host
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    // Print a small sample of the transposed matrix for verification
    printf("Transposed matrix sample:\n");
    for (int i = 0; i < 5; i++) {
        printf("%0.1f ", h_out[i]);
    }
    printf("\n");

    // Free device and host memory
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}

!nvcc -gencode=arch=compute_75,code=sm_75 -allow-unsupported-compiler matTranspose.cu -o matTranspose
!./matTranspose

Transposed matrix sample:
0.0 1024.0 2048.0 3072.0 4096.0 

