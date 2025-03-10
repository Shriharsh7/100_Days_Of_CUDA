# 100_Days_Of_CUDA Each day key notes.

# Day 1

On GPUs, operations like vector addition benefit from uniform execution across all threads. Since every thread performs the same task (e.g., adding two numbers from corresponding array positions), the GPU can schedule and execute these operations in perfect harmony. This uniformity eliminates inefficiencies and maximizes throughput, a fundamental reason why simple parallel tasks like vector addition are so fast on GPUs. This ties into the broader concept of how GPU architectures are designed for data-parallel workloads.

# Day 2

A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]] is accessed in a row major order as [1,2,3,4,5,6,7,8,9] and element A[i][j] is accessed as A[i * N + j] (where N = 3).
     
Block Size: threadsPerBlock(2, 2) means each block has 2 threads in the x-direction (blockDim.x = 2) and 2 in the y-direction (blockDim.y = 2).
Grid Size: For a 3x3 matrix, blocksPerGrid((N + 1) / 2, (M + 1) / 2) computes to (2, 2), so we have a 2x2 grid of blocks.

