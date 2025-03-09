# 100_Days_Of_CUDA

# Day 1

On GPUs, operations like vector addition benefit from uniform execution across all threads. Since every thread performs the same task (e.g., adding two numbers from corresponding array positions), the GPU can schedule and execute these operations in perfect harmony. This uniformity eliminates inefficiencies and maximizes throughput, a fundamental reason why simple parallel tasks like vector addition are so fast on GPUs. This ties into the broader concept of how GPU architectures are designed for data-parallel workloads.
