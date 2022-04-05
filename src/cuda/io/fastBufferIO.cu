#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {
#include "../../util/util.h"
}
#include "fastpgm.cuh"

/**
 * @brief Loads an image from a path to the provided buffer. The dynamic range of the image should match the pixel type.
 * Only supports .pgm images for now.
 *
 * @tparam pixel Pixel type. Can be uint8_t, uint16_t or int (8-bit, 16-bit and 32-bit respectively).
 * @param path The path to load the image from.
 * @param buffer Pre-allocated buffer that will be filled with the pixel data from the image.
 * @param width Width of the image will be put here.
 * @param height Height of the image will be put here.
 */
template <typename pixel>
void loadImageToBuffer(const char *path, pixel *buffer, int *width, int *height) {
  char *extension = getFileNameExtension(path);
  if (extension == NULL) {
    FATAL("Input image: '%s' has no extension.\n", path);
  }
  if (strcmp("pgm", extension) == 0) {
    loadPgmImageFast<pixel>(path, buffer, width, height);
  } else {
    FATAL("Extension: %s of path %s not supported.\n", extension, path);
  }
}

/**
 * @brief Saves the image stored in the buffer. Only supports .pgm images for now.
 *
 * @tparam pixel Pixel type. Can be uint8_t, uint16_t or int (8-bit, 16-bit and 32-bit respectively).
 * @param path The path to save the image at.
 * @param buffer Filled buffer with pixel values to save.
 * @param width Width of the image.
 * @param height Height of the image.
 */
template <typename pixel>
void saveImage(const char *path, pixel *buffer, int width, int height) {
  char *extension = getFileNameExtension(path);
  if (extension == NULL) {
    FATAL("Output image: '%s' has no extension.\n", path);
  }
  if (strcmp("pgm", extension) == 0) {
    savePgmImageFast<pixel>(path, buffer, width, height);
  } else {
    FATAL("Extension: %s not supported.\n", extension);
  }
}

template void loadImageToBuffer<uint8_t>(const char *path, uint8_t *buffer, int *width, int *height);
template void loadImageToBuffer<uint16_t>(const char *path, uint16_t *buffer, int *width, int *height);
template void loadImageToBuffer<int>(const char *path, int *buffer, int *width, int *height);

template void saveImage<uint8_t>(const char *path, uint8_t *buffer, int width, int height);
template void saveImage<uint16_t>(const char *path, uint16_t *buffer, int width, int height);
template void saveImage<int>(const char *path, int *buffer, int width, int height);