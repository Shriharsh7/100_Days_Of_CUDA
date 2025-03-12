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
