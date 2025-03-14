%%writefile conv_2d.cu

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define FILTER_WIDTH 3
#define FILTER_HEIGHT 3
#define TILE_WIDTH 16

// CUDA kernel to perform 2D convolution
__global__ void convolutionKernel(const float* input, float* output, const float* filter, int width, int height)
{
    // Compute the global x and y coordinates of the pixel
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int filterHalfW = FILTER_WIDTH / 2;
    int filterHalfH = FILTER_HEIGHT / 2;
    
    float sum = 0.0f;
    
    // Proceed only if within image bounds
    if (x < width && y < height) {
        // Loop over filter elements
        for (int fy = -filterHalfH; fy <= filterHalfH; ++fy) {
            for (int fx = -filterHalfW; fx <= filterHalfW; ++fx) {
                int ix = x + fx;
                int iy = y + fy;
                // Use zero-padding for boundaries
                if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                    float imageValue = input[iy * width + ix];
                    float filterValue = filter[(fy + filterHalfH) * FILTER_WIDTH + (fx + filterHalfW)];
                    sum += imageValue * filterValue;
                }
            }
        }
        output[y * width + x] = sum;
    }
}

int main()
{
    // Image dimensions
    const int width = 512;
    const int height = 512;
    const int imageSize = width * height * sizeof(float);

    // Define a simple 3x3 convolution filter (edge detection filter)
    float h_filter[FILTER_WIDTH * FILTER_HEIGHT] = {
         -1, -1, -1,
         -1,  8, -1,
         -1, -1, -1
    };

    // Allocate host memory for input and output images
    float* h_input  = (float*)malloc(imageSize);
    float* h_output = (float*)malloc(imageSize);

    // Initialize the input image (for example, fill with ones)
    for (int i = 0; i < width * height; i++) {
        h_input[i] = 1.0f;
    }

    // Allocate device memory
    float *d_input, *d_output, *d_filter;
    cudaMalloc((void**)&d_input, imageSize);
    cudaMalloc((void**)&d_output, imageSize);
    cudaMalloc((void**)&d_filter, FILTER_WIDTH * FILTER_HEIGHT * sizeof(float));

    // Copy input image and filter from host to device memory
    cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, FILTER_WIDTH * FILTER_HEIGHT * sizeof(float), cudaMemcpyHostToDevice);

    // Define CUDA grid and block dimensions
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((width + TILE_WIDTH - 1) / TILE_WIDTH, (height + TILE_WIDTH - 1) / TILE_WIDTH);

    // Launch the convolution kernel
    convolutionKernel<<<gridDim, blockDim>>>(d_input, d_output, d_filter, width, height);
    cudaDeviceSynchronize();

    // Copy the result from device to host
    cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost);

    // Print the first 10 output values as a simple check
    printf("First 10 output values:\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", h_output[i]);
    }
    printf("\n");

    // Free device and host memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_filter);
    free(h_input);
    free(h_output);

    return 0;
}

!nvcc -gencode=arch=compute_75,code=sm_75 -allow-unsupported-compiler conv_2d.cu -o conv_2d
!./conv_2d

First 10 output values:
5.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 
