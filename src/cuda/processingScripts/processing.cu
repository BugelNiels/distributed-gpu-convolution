#include "../processing.cuh"

// Constant Memory Reservation: Add other kernels here if you need them
__constant__ int constkernel1[MAX_KERNEL_WIDTH * MAX_KERNEL_HEIGHT];

/**
 * @brief Does custom image processing on the provided image.
 *
 * @tparam pixel Pixel type. Can be uint8_t, uint16_t or int (8-bit, 16-bit and 32-bit respectively).
 * @param dim_grid Grid dimensions used for kernel execution.
 * @param dim_block Block dimensions used for kernel execution.
 * @param input Input pixels (on the device memory).
 * @param output Output pixels (on the device memory).
 * @param kernels Array of kernels.
 * @param width Width of the output image.
 * @param height Height of the output image.
 */
template <typename pixel>
void processImage(cudaStream_t stream, const dim3 dim_grid, const dim3 dim_block, pixel *input, pixel *output,
                  const Kernel *kernels, int width, int height) {
  convolution2D<pixel, DILATION><<<dim_grid, dim_block, 0, stream>>>(input, output, kernels[0], width, height, false);
  // Executing: Add other kernels here if you need them
}

/**
 * @brief Loads a specific kernel to constant memory.
 *
 * @param kernelPath The path to the .kernel file.
 * @param symbol Name of the constant memory location.
 * @return Kernel Loaded kernel with as its elements a pointer to constant memory.
 */
Kernel loadKernelToDevice(const char *kernelPath, const void *symbol) {
  cudaError_t cuda_ret;
  Kernel kernel = loadKernel(kernelPath);
  if (kernel.width * kernel.height > CONST_MEMORY_SIZE) {
    FATAL("Kernels of size %dx%d are not supported.\n", kernel.width, kernel.height);
  }
  cuda_ret = cudaMemcpyToSymbol(symbol, kernel.elems, kernel.width * kernel.height * sizeof(int));
  cudaErrCheck(cuda_ret, "Unable to allocate constant memory");
  int *kernelPtr;
  cuda_ret = cudaGetSymbolAddress((void **)&kernelPtr, symbol);
  cudaErrCheck(cuda_ret, "Unable to retrieve constant memory location");
  cuda_ret = cudaDeviceSynchronize();
  cudaErrCheck(cuda_ret, "Unable to sync after copy");
  free(kernel.elems);
  kernel.elems = kernelPtr;
  return kernel;
}

/**
 * @brief Reads all the kernel files specified in the setUpKernels() function. The elements of the kernel are device
 * pointers.
 *
 * @return Kernel* Array containing all the kernels
 */
Kernel *setUpKernels() {
  Kernel *kernels = (Kernel *)malloc(NUM_KERNELS * sizeof(Kernel));

  Kernel kernel1 = loadKernelToDevice(KERNEL_1, constkernel1);
  kernels[0] = kernel1;

  // Loading: Load other kernels to constant memory here if you need them ..

  return kernels;
}

/**
 * @brief Frees all the kernels
 *
 * @param kernels The array of kernels to free
 */
void freeKernels(Kernel *kernels) { free(kernels); }

// Explicit template declerations to prevent having to do stuff in the header file
template void processImage<uint8_t>(cudaStream_t stream, const dim3 dim_grid, const dim3 dim_block, uint8_t *input,
                                    uint8_t *output, const Kernel *kernels, int width, int height);
template void processImage<uint16_t>(cudaStream_t stream, const dim3 dim_grid, const dim3 dim_block, uint16_t *input,
                                     uint16_t *output, const Kernel *kernels, int width, int height);
template void processImage<int>(cudaStream_t stream, const dim3 dim_grid, const dim3 dim_block, int *input, int *output,
                                const Kernel *kernels, int width, int height);