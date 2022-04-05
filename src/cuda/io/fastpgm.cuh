#ifndef PGM_FORMAT_H
#define PGM_FORMAT_H

#include <stdint.h>

/**
 * @brief Loads a pgm image from a path to the provided buffer. The dynamic range of the image should match the pixel
 * type. Note that comments are not explicitly skipped, so make sure the pgm images do not contain comments.
 *
 * @tparam pixel Pixel type. Can be uint8_t, uint16_t or int (8-bit, 16-bit and 32-bit respectively).
 * @param path The path to load the image from.
 * @param buffer Pre-allocated buffer that will be filled with the pixel data from the image.
 * @param width Width of the image will be put here.
 * @param height Height of the image will be put here.
 */
template <typename pixel>
void loadPgmImageFast(const char *path, pixel *buffer, int *width, int *height);

/**
 * @brief Saves the image stored in the buffer as a pgm image.
 *
 * @tparam pixel Pixel type. Can be uint8_t, uint16_t or int (8-bit, 16-bit and 32-bit respectively).
 * @param path The path to save the image at.
 * @param buffer Filled buffer with pixel values to save.
 * @param width Width of the image.
 * @param height Height of the image.
 */
template <typename pixel>
void savePgmImageFast(const char *path, pixel *buffer, int width, int height);

#endif  // PGM_FORMAT_H