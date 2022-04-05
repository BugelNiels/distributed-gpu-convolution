#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

extern "C" {
#include "../../util/util.h"
}
#include "fastpgm.cuh"

/**
 * @brief Loads a pgm image from a path to the provided buffer. The dynamic range of the image should match the pixel
 * type.
 *
 * @tparam pixel Pixel type. Can be uint8_t, uint16_t or int (8-bit, 16-bit and 32-bit respectively).
 * @param path The path to load the image from.
 * @param buffer Pre-allocated buffer that will be filled with the pixel data from the image.
 * @param width Width of the image will be put here.
 * @param height Height of the image will be put here.
 */
template <typename pixel>
void loadPgmImageFast(const char *path, pixel *buffer, int *width, int *height) {
  FILE *imgFile = openFile(path, "r");
  int itemsRead = fscanf(imgFile, "P5\n%d %d\n%*d\n", width, height);
  if (itemsRead == 1) {
    // photoshop inserts a newline between width and height
    fscanf(imgFile, "\n%d\n%*d\n", height);
  }
  fread(buffer, sizeof(pixel), (*width) * (*height), imgFile);
  fclose(imgFile);
}

// mapped memory. Not very useful, since mmaps are expensive to do for every image, but I'll leave it here for now,
// since it's pretty cool.
template <typename pixel>
pixel *loadPgmImageMapped(const char *path, int *width, int *height, size_t *fileSize) {
  int fd = open(path, O_RDONLY);
  struct stat s;
  int status = fstat(fd, &s);
  *fileSize = s.st_size;
  pixel *mappedFile = (pixel *)mmap(0, *fileSize, PROT_READ, MAP_PRIVATE | MAP_LOCKED, fd, 0);
  FILE *imgFile = fdopen(fd, "r");
  int itemsRead = fscanf(imgFile, "P5\n%d %d", width, height);
  if (itemsRead == 1) {
    // photoshop inserts a newline between width and height
    fscanf(imgFile, "\n%d", height);
  }
  fclose(imgFile);
  return mappedFile;
}

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
void savePgmImageFast(const char *path, pixel *buffer, int width, int height) {
  int npixels = width * height;
  FILE *pgmFile = openFile(path, "w");
  fprintf(pgmFile, "P5\n%d %d\n%d\n", width, height, 255);
  fwrite(buffer, sizeof(pixel), npixels, pgmFile);
  fclose(pgmFile);
}

template void loadPgmImageFast<uint8_t>(const char *path, uint8_t *buffer, int *width, int *height);
template void loadPgmImageFast<uint16_t>(const char *path, uint16_t *buffer, int *width, int *height);
template void loadPgmImageFast<int>(const char *path, int *buffer, int *width, int *height);

template uint8_t *loadPgmImageMapped<uint8_t>(const char *path, int *width, int *height, size_t *fileSize);
template uint16_t *loadPgmImageMapped<uint16_t>(const char *path, int *width, int *height, size_t *fileSize);
template int *loadPgmImageMapped<int>(const char *path, int *width, int *height, size_t *fileSize);

template void savePgmImageFast<uint8_t>(const char *path, uint8_t *buffer, int width, int height);
template void savePgmImageFast<uint16_t>(const char *path, uint16_t *buffer, int width, int height);
template void savePgmImageFast<int>(const char *path, int *buffer, int width, int height);