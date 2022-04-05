#ifndef CUDA_CONFIG_H
#define CUDA_CONFIG_H

// Max kernel size should reflect the largest possible size of the kernels
#define MAX_KERNEL_WIDTH 20
#define MAX_KERNEL_HEIGHT 20
#define NUM_KERNELS 1
#define KERNEL_1 "kernels/square.kernel"

// These two generally should not change unless you magically have > 1024 SM per block
#define BLOCK_WIDTH 32
#define BLOCK_HEIGHT 32

#endif  // CUDA_CONFIG_H