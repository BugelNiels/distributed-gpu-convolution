
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "imageFormats/pgmformat.h"
#include "job/job.h"
#include "util/util.h"

void insertTileIntoRaster(int *raster, int rasterWidth, int rasterHeight, int *tilePixels, int blockWidth,
                          int blockHeight, int blockX, int blockY) {
  int startX = blockX * blockWidth;
  int startY = blockY * blockHeight;
  for (int y = 0; y < blockHeight; y++) {
    for (int x = 0; x < blockWidth; x++) {
      int pixX = startX + x;
      int pixY = startY + y;
      if (pixX >= 0 && pixX < rasterWidth && pixY >= 0 && pixY < rasterHeight) {
        raster[pixY * rasterWidth + pixX] = tilePixels[y * blockWidth + x];
      }
    }
  }
}

int *combineTiles(int width, int height, int blockWidth, int blockHeight, const char *tileDir) {
  int nVertImages = (height + (blockHeight - 1)) / blockHeight;
  int nHorImages = (width + (blockWidth - 1)) / blockWidth;
  int *raster = calloc(width * height, sizeof(int));
#pragma omp parallel
  {
    char buffer[256];
    int *tilePixels = malloc(blockWidth * blockHeight * sizeof(int));
#pragma omp for
    for (int y = 0; y < nVertImages; y++) {
      for (int x = 0; x < nHorImages; x++) {
        sprintf(buffer, "%s/block_x%d_y%d.pgm", tileDir, x, y);
        loadPgmImageToMem(buffer, tilePixels);
        insertTileIntoRaster(raster, width, height, tilePixels, blockWidth, blockHeight, x, y);
      }
    }
    free(tilePixels);
  }

  return raster;
}

int main(int argc, char *argv[]) {
  if (argc != 8) {
    printf(
        "Usage: ./combiner finalImgWidth finalImgHeight tileWidth tileHeight numBits outputFile "
        "inputTileDirectory\nExample:\n\t ./combiner 256 256 64 64 8 result.pgm outputTiles\n");
    return 0;
  }

  int finalWidth = atoi(argv[1]);
  int finalHeight = atoi(argv[2]);
  int blockWidth = atoi(argv[3]);
  int blockHeight = atoi(argv[4]);
  int numBits = atoi(argv[5]);
  const char *path = argv[6];
  const char *tileDir = argv[7];

  // make output tile directory. Should fail if it already exists
  int *pixels = combineTiles(finalWidth, finalHeight, blockWidth, blockHeight, tileDir);
  saveAsPgmImage(path, pixels, finalWidth, finalHeight, numBits);
  printf("Result saved to %s\n", path);
  free(pixels);
  return 0;
}