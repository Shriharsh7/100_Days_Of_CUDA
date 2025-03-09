%%writefile vector_add.cu

#include <iostream>
#include <cuda_runtime.h>

__global__ void addVectors(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        c[i] = a[i] + b[i];
    }
}

int main(){
    const int n = 1000000;  // Vector size
    int *ha, *hb, *hc;      // Host pointers (CPU)
    int *da, *db, *dc;      // Device pointers (GPU)

    // Allocate memory on host (CPU)
    ha = new int[n];
    hb = new int[n];
    hc = new int[n];

    // Initialize vectors on host

    for (int i = 0; i < n; i++) {
        ha[i] = i;
        hb[i] = 2 * i;
    }

    // Allocate memory on device (GPU)

    if (cudaMalloc(&da, n * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&db, n * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&dc, n * sizeof(int)) != cudaSuccess) {
        std::cerr << "Error allocating device memory." << std::endl;
        return 1;
    }

    // Copy data from host to device

    cudaMemcpy(da, ha, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, n * sizeof(int), cudaMemcpyHostToDevice);

    // Set up grid and block sizes

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel

    addVectors<<<blocksPerGrid, threadsPerBlock>>>(da, db, dc, n);

    // Check for errors in kernel launch and execution

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel execution failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Copy the result back from device to host

    cudaMemcpy(hc, dc, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Print some results to verify
    
    for (int i = 0; i < 10; i++) {
        std::cout << hc[i] << " ";
    }
    std::cout << std::endl;

    // Free memory

    delete[] ha;
    delete[] hb;
    delete[] hc;
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    return 0;
}
/*

!nvcc -gencode=arch=compute_75,code=sm_75 -allow-unsupported-compiler vector_add.cu -o vector_add
!./vector_add

*/
