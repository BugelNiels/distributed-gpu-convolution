#ifndef CONV_KERNELS_H
#define CONV_KERNELS_H

#include <stdint.h>
extern "C" {
#include "../imageFormats/kernelformat.h"
}

/**
 * @brief Enum to determine the "type" of convolution. Used in the kernel to ensure that the kernel can be fully
 * optimised at compile-time.
 *
 */
typedef enum BinOp { CONVOLUTION, EROSION, DILATION } BinOp;

/**
 * @brief Convolution kernel.
 *
 * @tparam pixel Pixel type. Can be uint8_t, uint16_t or int (8-bit, 16-bit and 32-bit respectively).
 * @tparam op Operator type. Can be CONVOLUTION, EROSION or DILATION.
 * @param input Input device buffer.
 * @param output Output device buffer.
 * @param kernel Kernel containing a pointer to device memory.
 * @param width Width of the output image.
 * @param height Height of the output image.
 * @param padded Whether the image is padded or not.
 * @param divisor Whether the final value for every pixel should be divided or not. Useful for e.g. Gaussian kernels.
 */
template <typename pixel, BinOp op>
__global__ void convolution2D(const pixel *__restrict__ input, pixel *output, const Kernel kernel, int width,
                              int height, const bool padded = false, const double divisor = 0.0);

#endif  // CONV_KERNELS_H