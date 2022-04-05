#ifndef PROCESSING_H
#define PROCESSING_H

#include <stdio.h>
#include <stdlib.h>
extern "C" {
#include "../imageFormats/kernelformat.h"
#include "../util/util.h"
#include "cudaConfig.h"
}
#include "kernels/convKernels.cuh"
#include "processing.cuh"
#include "util/cudaUtils.cuh"

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
void processImage(cudaStream_t stream, const dim3 dim_grid, const dim3 dim_block, const pixel *input, pixel *output,
                  const Kernel *kernels, int width, int height);

/**
 * @brief Reads all the kernel files specified in the setUpKernels() function. The elements of the kernel are device
 * pointers.
 *
 * @return Kernel* Array containing all the kernels
 */
Kernel *setUpKernels();

/**
 * @brief Frees all the kernels
 *
 * @param kernels The array of kernels to free
 */
void freeKernels(Kernel *kernels);

#endif  // PROCESSING_H