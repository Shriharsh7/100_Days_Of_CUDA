%%writefile conv_2d_gaussianBlur.cu

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define GAUSSIAN_FILTER_WIDTH 5
#define GAUSSIAN_FILTER_HEIGHT 5
#define TILE_WIDTH 16

// CUDA kernel for Gaussian blur convolution
__global__ void gaussianBlurKernel(const float* input, float* output, const float* filter, int width, int height)
{
    // Compute global x and y coordinates for this thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int filterHalfW = GAUSSIAN_FILTER_WIDTH / 2;
    int filterHalfH = GAUSSIAN_FILTER_HEIGHT / 2;
    
    float sum = 0.0f;

    // Ensure we are inside the image bounds
    if (x < width && y < height) {
        // Loop over the filter window
        for (int fy = -filterHalfH; fy <= filterHalfH; ++fy) {
            for (int fx = -filterHalfW; fx <= filterHalfW; ++fx) {
                int ix = x + fx;
                int iy = y + fy;
                // Use zero-padding when the filter goes out-of-bound
                if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                    float imageValue = input[iy * width + ix];
                    float filterValue = filter[(fy + filterHalfH) * GAUSSIAN_FILTER_WIDTH + (fx + filterHalfW)];
                    sum += imageValue * filterValue;
                }
            }
        }
        // Write the computed value to the output image
        output[y * width + x] = sum;
    }
}

int main()
{
    // Image dimensions and memory size
    const int width = 512;
    const int height = 512;
    const int imageSize = width * height * sizeof(float);

    // Define a 5x5 Gaussian blur filter (normalized with sum=1)
    float h_filter[GAUSSIAN_FILTER_WIDTH * GAUSSIAN_FILTER_HEIGHT] = {
        1/273.0f,  4/273.0f,  7/273.0f,  4/273.0f,  1/273.0f,
        4/273.0f, 16/273.0f, 26/273.0f, 16/273.0f,  4/273.0f,
        7/273.0f, 26/273.0f, 41/273.0f, 26/273.0f,  7/273.0f,
        4/273.0f, 16/273.0f, 26/273.0f, 16/273.0f,  4/273.0f,
        1/273.0f,  4/273.0f,  7/273.0f,  4/273.0f,  1/273.0f
    };

    // Allocate host memory for the input and output images
    float* h_input  = (float*)malloc(imageSize);
    float* h_output = (float*)malloc(imageSize);

    // Initialize the input image (for example, all pixels set to 1.0f)
    for (int i = 0; i < width * height; i++) {
        h_input[i] = 1.0f;
    }

    // Allocate device memory for image and filter data
    float *d_input, *d_output, *d_filter;
    cudaMalloc((void**)&d_input, imageSize);
    cudaMalloc((void**)&d_output, imageSize);
    cudaMalloc((void**)&d_filter, GAUSSIAN_FILTER_WIDTH * GAUSSIAN_FILTER_HEIGHT * sizeof(float));

    // Copy the input image and filter from host to device memory
    cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, GAUSSIAN_FILTER_WIDTH * GAUSSIAN_FILTER_HEIGHT * sizeof(float), cudaMemcpyHostToDevice);

    // Define CUDA grid and block dimensions
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((width + TILE_WIDTH - 1) / TILE_WIDTH, (height + TILE_WIDTH - 1) / TILE_WIDTH);

    // Launch the Gaussian blur convolution kernel
    gaussianBlurKernel<<<gridDim, blockDim>>>(d_input, d_output, d_filter, width, height);
    cudaDeviceSynchronize();

    // Copy the output image back from device to host
    cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost);

    // Print first 10 output values for a quick verification
    printf("First 10 output values from Gaussian blur:\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", h_output[i]);
    }
    printf("\n");

    // Clean up: free device and host memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_filter);
    free(h_input);
    free(h_output);

    return 0;
}

!nvcc -gencode=arch=compute_75,code=sm_75 -allow-unsupported-compiler conv_2d_gaussianBlur.cu -o conv_2d_gaussianBlur
!./conv_2d_gaussianBlur

First 10 output values from Gaussian blur:
0.483516 0.652015 0.695971 0.695971 0.695971 0.695971 0.695971 0.695971 0.695971 0.695971 
