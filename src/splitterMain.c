#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "imageFormats/pgmformat.h"
#include "imageFormats/tiffformat.h"
#include "util/util.h"

int *getSubImage(int *pixels, int imWidth, int imHeight, int startX, int startY, int subWidth, int subHeight,
                 int vertPadSize, int horPadSize) {
  int *subRaster = calloc(subWidth * subHeight, sizeof(int));
  for (int y = 0; y < subHeight; y++) {
    for (int x = 0; x < subWidth; x++) {
      int pixX = startX + x - horPadSize;
      int pixY = startY + y - vertPadSize;
      if (pixX >= 0 && pixX < imWidth && pixY >= 0 && pixY < imHeight) {
        subRaster[y * subWidth + x] = pixels[pixY * imWidth + pixX];
      }
    }
  }
  return subRaster;
}

void splitAndSave(int *pixels, int imWidth, int imHeight, int blockWidth, int blockHeight, int horPadSize,
                  int vertPadSize, const char *dir, int numBits) {
  int nVertImages = (imHeight + (blockHeight - 1)) / blockHeight;
  int nHorImages = (imWidth + (blockWidth - 1)) / blockWidth;
  char pathBuffer[256];

  int subWidth = blockWidth + 2 * horPadSize;
  int subHeight = blockHeight + 2 * vertPadSize;
#pragma omp parallel for private(pathBuffer)
  for (int y = 0; y < nVertImages; y++) {
    for (int x = 0; x < nHorImages; x++) {
      int *subImage = getSubImage(pixels, imWidth, imHeight, x * blockWidth, y * blockHeight, subWidth, subHeight,
                                  vertPadSize, horPadSize);
      sprintf(pathBuffer, "%s/block_x%d_y%d.pgm", dir, x, y);
      saveAsPgmImage(pathBuffer, subImage, subWidth, subHeight, numBits);
      free(subImage);
    }
  }
}

// TODO: move to other file
int *loadImage(const char *path, int *width, int *height) {
  char *extension = getFileNameExtension(path);
  if (extension == NULL) {
    FATAL("Input image: '%s' has no extension.\n", path);
  }
  if (strcmp("pgm", extension) == 0) {
    return loadPgmImage(path, width, height);
  } else if (strcmp("tiff", extension) == 0 || strcmp("tif", extension) == 0) {
    return loadTiffImage(path, width, height);
  } else {
    FATAL("Input image: extension '%s' not supported\n", extension);
  }
}

int main(int argc, char *argv[]) {
  if (argc != 8) {
    printf(
        "Usage: ./combiner kernelWidth kernelHeight tileWidth tileHeight numBits inputFile tileDirectory\nExample:\n\t "
        "./combiner 3 3 64 64 8 images/lena.pgm inputTiles\n");
    return 0;
  }

  int kernelWidth = atoi(argv[1]);
  int kernelHeight = atoi(argv[2]);
  int blockWidth = atoi(argv[3]);
  int blockHeight = atoi(argv[4]);
  int numBits = atoi(argv[5]);
  const char *path = argv[6];
  const char *tileDir = argv[7];
  int width, height;
  int *pixels = loadImage(path, &width, &height);
  mkdir(tileDir, 0777);
  splitAndSave(pixels, width, height, blockWidth, blockHeight, kernelWidth / 2, kernelHeight / 2, tileDir, numBits);
  free(pixels);
  return 0;
}