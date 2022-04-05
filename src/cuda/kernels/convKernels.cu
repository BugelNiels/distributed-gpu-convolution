
#include <stdio.h>
#include <stdlib.h>

#include "../util/cudaUtils.cuh"
#include "convKernels.cuh"
#include "cudaConfig.h"

#define MAX(A, B) ((A) > (B) ? (A) : (B))
#define MIN(A, B) ((A) < (B) ? (A) : (B))

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
                              int height, const bool padded, const double divisor) {
  __shared__ int paddedTile[(BLOCK_HEIGHT + MAX_KERNEL_HEIGHT - 1) * (BLOCK_WIDTH + MAX_KERNEL_WIDTH - 1)];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int toY = blockIdx.y * BLOCK_HEIGHT + ty;
  int toX = blockIdx.x * BLOCK_WIDTH + tx;
  int kernelWidth = kernel.width;
  int kernelHeight = kernel.height;

  int padTileWidth = BLOCK_WIDTH + kernelWidth - 1;

  if (padded) {
    // padding is assumed to be kernelWidth - 1
    int pixelWidth = width + kernelWidth - 1;
    int fx, fy = toY;
    for (int y = ty; y < BLOCK_HEIGHT + kernelHeight - 1; y += BLOCK_HEIGHT) {
      fx = toX;
      for (int x = tx; x < BLOCK_WIDTH + kernelWidth - 1; x += BLOCK_WIDTH) {
        paddedTile[y * padTileWidth + x] = input[fy * pixelWidth + fx];
        fx += BLOCK_WIDTH;
      }
      fy += BLOCK_HEIGHT;
    }
  } else {
    int a = kernelWidth / 2;
    int b = kernelHeight / 2;
    int fy = toY - b;
    for (int y = ty; y < BLOCK_HEIGHT + kernelHeight - 1; y += BLOCK_HEIGHT) {
      int fx = toX - a;
      for (int x = tx; x < BLOCK_WIDTH + kernelWidth - 1; x += BLOCK_WIDTH) {
        int inputVal;
        if (fx < 0 || fx >= width || fy < 0 || fy >= height) {
          inputVal = 0;
        } else {
          inputVal = input[fy * width + fx];
        }
        paddedTile[y * padTileWidth + x] = inputVal;
        fx += BLOCK_WIDTH;
      }
      fy += BLOCK_HEIGHT;
    }
  }

  __syncthreads();

  int val = 0;
  int kernelIdx = 0;
  for (int ky = 0; ky < kernelHeight; ky++) {
    int tileIdxY = (ky + ty) * padTileWidth;
    for (int kx = 0; kx < kernelWidth; kx++) {
      // Using templating for the "kind" of convolution allows the compiler to remove this if-statement at compile time
      int tileIdx = tileIdxY + kx + tx;
      if (op == CONVOLUTION) {
        val += paddedTile[tileIdx] * kernel.elems[kernelIdx++];
      } else if (op == DILATION) {
        val = MAX(val, paddedTile[tileIdx] + kernel.elems[kernelIdx++]);
      } else if (op == EROSION) {
        val = MIN(val, paddedTile[tileIdx] - kernel.elems[kernelIdx++]);
      }
    }
  }

  if (toX < width && toY < height) {
    if (divisor == 0.0) {
      output[toY * width + toX] = val;
    } else {
      output[toY * width + toX] = val / divisor + 0.5;
    }
  }
}

// explicit template function instantiation
// quite ugly, but better than copy pasting the whole thing
// due to performance reasons, function pointers are not used
template __global__ void convolution2D<uint8_t, CONVOLUTION>(const uint8_t *__restrict__ input, uint8_t *output,
                                                             const Kernel kernel, int width, int height,
                                                             const bool padded, const double divisor);
template __global__ void convolution2D<uint16_t, CONVOLUTION>(const uint16_t *__restrict__ input, uint16_t *output,
                                                              const Kernel kernel, int width, int height,
                                                              const bool padded, const double divisor);
template __global__ void convolution2D<int, CONVOLUTION>(const int *__restrict__ input, int *output,
                                                         const Kernel kernel, int width, int height, const bool padded,
                                                         const double divisor);

template __global__ void convolution2D<uint8_t, DILATION>(const uint8_t *__restrict__ input, uint8_t *output,
                                                          const Kernel kernel, int width, int height, const bool padded,
                                                          const double divisor);
template __global__ void convolution2D<uint16_t, DILATION>(const uint16_t *__restrict__ input, uint16_t *output,
                                                           const Kernel kernel, int width, int height,
                                                           const bool padded, const double divisor);
template __global__ void convolution2D<int, DILATION>(const int *__restrict__ input, int *output, const Kernel kernel,
                                                      int width, int height, const bool padded, const double divisor);

template __global__ void convolution2D<uint8_t, EROSION>(const uint8_t *__restrict__ input, uint8_t *output,
                                                         const Kernel, int width, int height, const bool padded,
                                                         const double divisor);
template __global__ void convolution2D<uint16_t, EROSION>(const uint16_t *__restrict__ input, uint16_t *output,
                                                          const Kernel kernel, int width, int height, const bool padded,
                                                          const double divisor);
template __global__ void convolution2D<int, EROSION>(const int *__restrict__ input, int *output, const Kernel kernel,
                                                     int width, int height, const bool padded, const double divisor);
