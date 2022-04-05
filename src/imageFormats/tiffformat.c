#include "tiffformat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tiffio.h>

#include "../util/util.h"

void loadTiffImageToMem(const char *path, int *pixels) {
  TIFF *inImg = TIFFOpen(path, "r");
  if (inImg == NULL) {
    FATAL("Cannot open file %s\n", path);
  }
  int width, height;
  TIFFGetField(inImg, TIFFTAG_IMAGEWIDTH, &width);
  TIFFGetField(inImg, TIFFTAG_IMAGELENGTH, &height);

  size_t nPixels = width * height;
  uint32 *raster = (uint32 *)_TIFFmalloc(nPixels * sizeof(uint32));
  if (raster != NULL) {
    if (TIFFReadRGBAImage(inImg, width, height, raster, 0)) {
      for (int i = 0; i < nPixels; i++) {
        pixels[i] = (unsigned char)raster[nPixels - i - 1];
      }
    }
    _TIFFfree(raster);
  }
  TIFFClose(inImg);
}

int *loadTiffImage(const char *path, int *width, int *height) {
  TIFF *inImg = TIFFOpen(path, "r");
  if (inImg == NULL) {
    FATAL("Cannot open file %s\n", path);
  }
  TIFFGetField(inImg, TIFFTAG_IMAGEWIDTH, width);
  TIFFGetField(inImg, TIFFTAG_IMAGELENGTH, height);

  size_t nPixels = *width * *height;
  int *pixels = (int *)malloc(nPixels * sizeof(int));

  uint32 *raster = (uint32 *)_TIFFmalloc(nPixels * sizeof(uint32));
  if (raster != NULL) {
    if (TIFFReadRGBAImage(inImg, *width, *height, raster, 0)) {
      for (int i = 0; i < nPixels; i++) {
        pixels[i] = (unsigned char)raster[nPixels - i - 1];
      }
    }
    _TIFFfree(raster);
  }
  TIFFClose(inImg);
  printf("%s loaded TIFF image - size: %dx%d pixels\n", path, *height, *width);
  return pixels;
}

void saveAsTiffImage(const char *path, int *pixels, int width, int height, int numBits) {
  TIFF *outImg = TIFFOpen(path, "w");
  if (outImg == NULL) {
    FATAL("Cannot open file %s\n", path);
  }
  // number of channels; 1 for grayscale, 3 for rgb, 4 for rgba
  int samplesPerPixel = 1;
  TIFFSetField(outImg, TIFFTAG_IMAGEWIDTH, width);
  TIFFSetField(outImg, TIFFTAG_IMAGELENGTH, height);
  TIFFSetField(outImg, TIFFTAG_SAMPLESPERPIXEL, samplesPerPixel);
  TIFFSetField(outImg, TIFFTAG_BITSPERSAMPLE, numBits);
  TIFFSetField(outImg, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
  TIFFSetField(outImg, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
  TIFFSetField(outImg, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
  TIFFSetField(outImg, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
  TIFFSetField(outImg, TIFFTAG_ROWSPERSTRIP, 1);
  unsigned char *buffer = (unsigned char *)malloc(width * height * sizeof(unsigned char));
  for (size_t i = 0; i < width * height; i++) {
    buffer[i] = pixels[i];
  }
  for (int y = 0; y < height; y++) {
    TIFFWriteEncodedStrip(outImg, y, &buffer[y * width], width * sizeof(unsigned char));
  }
  TIFFClose(outImg);
  free(buffer);
}
