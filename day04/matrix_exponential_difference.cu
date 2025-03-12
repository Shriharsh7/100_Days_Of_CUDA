%%writefile matrix_exponential_difference.cu

#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__device__ float randomFunction(float x, float y)
{
    return expf(x - y); // New transformation: e^(x - y)
}

__global__ void matrixFunction(const float *A, const float *B, float *C, const int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < size && j < size)
    {
        C[i + size * j] = randomFunction(A[i + size * j], B[i + size * j]);
    }
}

int main()
{
    int N = 8;
    int BLOCK_SIZE = 2;
    dim3 blockDim(BLOCK_SIZE * BLOCK_SIZE);
    dim3 gridDim(N + BLOCK_SIZE - 1 / BLOCK_SIZE, N + BLOCK_SIZE - 1 / BLOCK_SIZE);
    int size = sizeof(float) * N * N;

    float *A,*B,*C;
    float *dA,*dB,*dC;
    A = new float[N*N];
    B = new float[N*N];
    C = new float[N*N];

    cudaMalloc((void**)&dA,size);
    cudaMalloc((void**)&dB,size);
    cudaMalloc((void**)&dC,size);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i + N * j] = 0.69 * i; 
            B[i + N * j] = 0.69 * j;
        }
    }
    
    cudaMemcpy(dA,A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(dB,B,size,cudaMemcpyHostToDevice);

    // now we have everything set up
    matrixFunction<<<gridDim,blockDim>>>(dA,dB,dC,N);
    cudaDeviceSynchronize();

    cudaMemcpy(C,dC,size,cudaMemcpyDeviceToHost);

    for (int i = 0; i < N*N; i++) {
        cout << C[i] << " ";
        if ((i + 1) % N == 0)
          cout<<endl;
    }

    return 69;
}

/*

!nvcc -gencode=arch=compute_75,code=sm_75 -allow-unsupported-compiler matrix_exponential_difference.cu -o matrix_exponential_difference
!./matrix_exponential_difference

1 1.99372 3.9749 7.92482 15.7998 31.5004 62.8028 125.211 
0.501576 1 1.99372 3.9749 7.92482 15.7998 31.5004 62.8028 
0.251579 0.501576 1 1.99372 3.9749 7.92482 15.7998 31.5004 
0.126186 0.251579 0.501576 1 1.99372 3.9749 7.92482 15.7998 
0.0632918 0.126186 0.251579 0.501576 1 1.99372 3.9749 7.92482 
0.0317456 0.0632918 0.126186 0.251579 0.501576 1 1.99372 3.9749 
0.0159229 0.0317456 0.0632918 0.126186 0.251579 0.501576 1 1.99372 
0.00798652 0.0159229 0.0317456 0.0632918 0.126186 0.251579 0.501576 1

 */
