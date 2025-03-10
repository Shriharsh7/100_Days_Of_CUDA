%%writefile matrix_add.cu

#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__ void addMatrix(int *device_A, int *device_B, int *device_C, int m, int n ){

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if(row < m && col < n){
    int index = row * n + col;    // Flattened the 2d matrix into 1d as row major and corresponding index.
    device_C[index] = device_A[index] + device_B[index];
  }

}

int main(){
  int m = 1000, n = 1000;
  int size = m * n * sizeof(int);

  // Allocate memory on host

  int *host_A = new int[m * n];
  int *host_B = new int[m * n];
  int *host_C = new int[m * n];

  // Allocate memory on device

  int *device_A, *device_B, *device_C;
  cudaMalloc(&device_A, size);
  cudaMalloc(&device_B, size);
  cudaMalloc(&device_C, size);

  // Matrix initialization

  for (int i = 0; i < m * n; i++) {
        host_A[i] = 1;
        host_B[i] = 68;
    }

  // Copy host matrices to device

  cudaMemcpy(device_A, host_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(device_B, host_B, size, cudaMemcpyHostToDevice);

  // Define block and grid dimensions (adjust block size if needed)

  dim3 blockDim(16, 16);
  dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);    // Confusing part, keep digging into this one.

  // Launch the kernel

  addMatrix<<<gridDim, blockDim>>>(device_A, device_B, device_C, m, n);

  cudaError_t err = cudaGetLastError();

  if (err != cudaSuccess) {
        cerr << "Kernel launch failed: " << cudaGetErrorString(err) << endl;
        return 1;
    }

  // Synchronize to check for any kernel execution errors

  cudaDeviceSynchronize();

  // device to host copy of the result.

  cudaMemcpy(host_C, device_C, size, cudaMemcpyDeviceToHost);

  // Testing if it worked with some sample outputs
  
  cout << "First 10 elements of the result matrix:" << endl;
  for (int i = 0; i < 10; i++){
      cout << host_C[i] << " ";
  }
  cout << endl;

  // Free device and host memory
    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);
    delete[] host_A;
    delete[] host_B;
    delete[] host_C;

    return 69;

}

/*

!nvcc -gencode=arch=compute_75,code=sm_75 -allow-unsupported-compiler matrix_add.cu -o matrix_add
!./matrix_add

*/
