# 100_Days_Of_CUDA Each day key notes.

# Day 1

- On GPUs, operations like vector addition benefit from uniform execution across all threads. Since every thread performs the same task (e.g., adding two numbers from corresponding array positions), the GPU can schedule and execute these operations in perfect harmony.
- This uniformity eliminates inefficiencies and maximizes throughput, a fundamental reason why simple parallel tasks like vector addition are so fast on GPUs. This ties into the broader concept of how GPU architectures are designed for data-parallel workloads.

# Day 2

- A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]] is accessed in a row major order as [1,2,3,4,5,6,7,8,9] and element A[i][j] is accessed as A[i * N + j] (where N = 3).
- Block Size: threadsPerBlock(2, 2) means each block has 2 threads in the x-direction (blockDim.x = 2) and 2 in the y-direction (blockDim.y = 2).
- Grid Size: For a 3x3 matrix, blocksPerGrid((N + 1) / 2, (M + 1) / 2) computes to (2, 2), so we have a 2x2 grid of blocks

# Day 3

- Each element of the resulting matrix C is computed by a unique thread, leveraging a 2D grid of thread blocks for parallel execution.
- Threads are organized into blocks of size BLOCK_SIZE x BLOCK_SIZE (e.g., 16x16), defined using dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE).
- The grid size is calculated as gridDim((N + BLOCK_SIZE - 1)/BLOCK_SIZE, (N + BLOCK_SIZE - 1)/BLOCK_SIZE) to ensure all matrix elements are covered, even if N isn’t a multiple of BLOCK_SIZE.
- The matrixMulKernel computes each element of C by iterating over a row of A and a column of B, summing their products.
- The result is verified on the host by checking if each element of C equals 2*N (since A[i] = 1.0 and B[i] = 2.0), confirming the multiplication is correct.

# Day 4

- **Exponential Difference Calculation**: The code performs an element-wise computation, C[i][j] = exp(A[i][j] - B[i][j]), to calculate the exponential of the difference between two matrices, A and B, executed in       parallel on the GPU.
- **Device Function Usage**: A device function, randomFunction, encapsulates the core transformation logic, enhancing code modularity and readability for GPU execution.
- **Parallel Grid and Block Setup**: It employs a 2D grid and block configuration (dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE) and dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE)),          ensuring efficient and complete processing of the matrix elements.
# Day 5

- **Sample-Parallel Processing:** Each thread block independently normalizes one sample's features.
- **Shared Memory Reduction:** Efficiently computes mean and variance using parallel reduction in shared memory.
- **Modular Kernel Design:** Separates statistics computation and normalization into two distinct kernels.
- **Memory Optimization:** Minimizes global memory access by leveraging shared memory for intermediate results.

# Day 6

- Allocates host and device memory for a 512×512 grayscale image and a 3×3 convolution filter.
- Initializes the input image (with a constant value) and copies both image and filter data to the GPU.
- Launches a CUDA kernel that performs 2D convolution with zero-padding for boundary conditions.
- Copies the processed result back to the host and prints the first 10 output values for verification.

# Day 7

- Allocates host and device memory for a 512×512 image and a 5×5 Gaussian filter.
- Transfers the initialized image and filter from CPU memory to the GPU’s global memory.
- Launches a CUDA kernel where each thread computes one pixel’s convolution using a 5×5 window with zero-padding at boundaries.
- Copies the resulting blurred image back to the host and prints a few output values for verification.

# Day 8

- **Shared Memory Tiling & Bank Conflict Avoidance:** The kernel uses a shared memory tile with an extra column (TILE_DIM+1) to avoid bank conflicts, which enhances performance by ensuring efficient memory access.
- **Optimized Memory Access Patterns:** Global memory accesses are coalesced by loading a tile from the input matrix and writing it back in a transposed order, minimizing latency and maximizing throughput.
- **Scalable Grid & Block Configuration:** The execution configuration is computed based on matrix dimensions, allowing the kernel to flexibly handle various sizes of matrices while maintaining high performance.

# Day 9

- **Parallel Maximum Reduction:** Uses CUDA shared memory to perform a parallel reduction that efficiently computes the maximum value from a large array.
- **Hybrid CPU-GPU Approach:** Completes the final reduction on the CPU to minimize kernel overhead and efficiently handle the small set of results from each block.
- **Foundation for Softmax Stability:** The computed maximum value is key for numerically stable softmax implementations by enabling max subtraction to prevent overflow.

# Day 10

- **Dual Kernel Comparison:** Implements both a naive and a shared-memory optimized softmax kernel, allowing you to compare a simple approach against one that leverages CUDA’s shared memory for faster reductions.
- **CUDA Event Timing:** Uses CUDA events to measure kernel execution times, providing clear insights into the performance improvements achieved through optimization.
- **Numerical Stability & Efficiency:** Both implementations subtract the maximum value to ensure numerical stability during exponentiation, with the optimized version reducing redundant work through parallel reduction.





