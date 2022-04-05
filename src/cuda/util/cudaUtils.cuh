#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <stdio.h>
#include <stdlib.h>

// Constant memory size of CUDA devices. Generally roughly 64KB, but technically it's slightly less, so we set the limit
// at 60k.
#define CONST_MEMORY_SIZE 60000

// Inspired by https://stackoverflow.com/a/14038590
#define cudaErrCheck(ans, message) \
  { cudaAssert((ans), (message), __FILE__, __LINE__); }

/**
 * @brief Checks for CUDA error codes. If the provided code is not cudaSuccess, it will output the error and exit the
 * program
 *
 * @param code The CUDA error code
 * @param message A custom error message provided by the developer
 * @param file The file in which the error occurred
 * @param line The line at which the error occurred
 */
inline void cudaAssert(cudaError_t code, const char *message, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "\n\nError: %s\n", message);
    // line -1, since checks are always done on the next line
    fprintf(stderr, "Cuda Error Message: %s %s %d\n\n", cudaGetErrorString(code), file, line);
    exit(EXIT_FAILURE);
  }
}

/**
 * @brief Calculate the CUDA thread block dimensions.
 *
 * @return dim3 Block dimensions
 */
dim3 getBlockDims(int blockWidth, int blockHeight);

/**
 * @brief Calculate the CUDA grid dimensions.
 *
 * @param imgWidth Width of the (output) image.
 * @param imgHeight Height of the (output) image.
 * @param dimBlock CUDA thread block dimensions.
 * @return dim3 Grid dimensions
 */
dim3 getGridDims(int imgWidth, int imgHeight, dim3 dimBlock);

void generateOutputLoc(char *outputDir, char *inputLoc, char *buffer);

#endif  // CUDA_UTILS_H