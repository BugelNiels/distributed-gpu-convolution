#include "cudaUtils.cuh"

extern "C" {
#include "../../util/util.h"
}

/**
 * @brief Calculate the CUDA thread block dimensions.
 *
 * @return dim3 Block dimensions
 */
dim3 getBlockDims(int blockWidth, int blockHeight) {
  dim3 dimBlock;
  dimBlock.x = blockWidth;
  dimBlock.y = blockHeight;
  dimBlock.z = 1;
  return dimBlock;
}

/**
 * @brief Calculate the CUDA grid dimensions.
 *
 * @param imgWidth Width of the (output) image.
 * @param imgHeight Height of the (output) image.
 * @param dimBlock CUDA thread block dimensions.
 * @return dim3 Grid dimensions
 */
dim3 getGridDims(int imgWidth, int imgHeight, dim3 dimBlock) {
  dim3 dimGrid;
  dimGrid.x = (imgWidth + dimBlock.x - 1) / dimBlock.x;
  dimGrid.y = (imgHeight + dimBlock.y - 1) / dimBlock.y;
  dimGrid.z = 1;
  return dimGrid;
}

/**
 * @brief Generate an output path of the output image. Resulting path will be <outputDir>/<filename>
 *
 * @param outputDir Path of the output directory.
 * @param inputLoc Path of the input image.
 * @param buffer Buffer to save the output path to.
 */
void generateOutputLoc(char *outputDir, char *inputLoc, char *buffer) {
  char *filename = getFileName(inputLoc);
  if (filename == NULL) {
    FATAL("Invalid input path %s.\n", inputLoc);
  }
  sprintf(buffer, "%s/%s", outputDir, filename);
}