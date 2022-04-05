#include "pgmformat.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../util/util.h"

void verifyPgmImage(const char *path) {
  char *extension = getFileNameExtension(path);
  if (extension == NULL) {
    FATAL("Filename '%s' has no extension.\n", path);
  }
  if (strcmp("pgm", extension) != 0) {
    FATAL("Attempting to open %s as pgm image. Extension is not .pgm.\n", path);
  }
}

// unsafe but fast versions
void load16bitPgmImageToMem(const char *path, uint16_t *buffer) {
  FILE *imgFile = openFile(path, "r");
  int width, height;
  fscanf(imgFile, "P5\n%d %d\n %*d\n", &width, &height);
  int npixels = width * height;
  if (fread(buffer, 2, npixels, imgFile) != npixels) {
    debug_print("loadImagePGM: corrupt PGM %s, file is truncated.\n", path);
  }
  fclose(imgFile);
}

void load8bitPgmImageToMem(const char *path, uint8_t *buffer) {
  FILE *imgFile = openFile(path, "r");
  int width, height;
  fscanf(imgFile, "P5\n%d %d\n %*d\n", &width, &height);
  int npixels = width * height;
  if (fread(buffer, 1, npixels, imgFile) != npixels) {
    debug_print("loadImagePGM: corrupt PGM %s, file is truncated.\n", path);
  }
  fclose(imgFile);
}

// safe version
void loadPgmImageToMem(const char *path, int *pixels) {
  verifyPgmImage(path);
  int magicNumber;
  FILE *imgFile = openFile(path, "r");
  if (fscanf(imgFile, "P%d\n", &magicNumber) != 1) {
    fclose(imgFile);
    FATAL("loadPgmImage: corrupt PGM: no magic number found.\n");
  }
  if (magicNumber != 5) {
    fclose(imgFile);
    FATAL("Illegal magic number P%d found. Only P2 and P5 are valid PGM files.\n", magicNumber);
  }
  char c;
  while ((c = fgetc(imgFile)) == '#') {
    do {
      c = fgetc(imgFile);
    } while ((c != EOF) && (c != '\n'));
    if (c == EOF) {
      FATAL("loadImagePGM: corrupt PGM file.\n");
    }
  }
  ungetc(c, imgFile);
  int maxVal, width, height;
  // read width and height
  if (fscanf(imgFile, "%d %d", &width, &height) != 2) {
    FATAL("loadImagePGM: corrupt PGM: no file dimensions found.\n");
  }
  if (fscanf(imgFile, "%d\n", &maxVal) != 1) {
    FATAL("loadImagePGM: corrupt PGM file: no maximal grey value.\n");
  }
  if ((maxVal < 0) || (maxVal > 65535)) {
    FATAL("loadImagePGM: corrupt PGM: maximum grey value found is %d (must be in range [0..65535]).\n", maxVal);
  }

  int npixels = width * height;
  // magicnumber == 5
  if (maxVal > 255) {  // 2 bytes per pixel, i.e. short
    unsigned short *buffer = malloc(npixels * sizeof(unsigned short));
    if (fread(buffer, 2, npixels, imgFile) != npixels) {
      debug_print("loadImagePGM: corrupt PGM %s, file is truncated.\n", path);
    }
    for (int i = 0; i < npixels; i++) {
      pixels[i] = buffer[i];
    }
    free(buffer);
  } else {  // 1 byte per pixel
    unsigned char *buffer = malloc(npixels * sizeof(unsigned char));
    if (fread(buffer, 1, npixels, imgFile) != npixels) {
      debug_print("loadImagePGM: corrupt PGM %s, file is truncated.\n", path);
    }
    for (int i = 0; i < npixels; i++) {
      pixels[i] = buffer[i];
    }
    free(buffer);
  }
  fclose(imgFile);
}

int *loadPgmImage(const char *path, int *width, int *height) {
  verifyPgmImage(path);
  int magicNumber;
  FILE *imgFile = openFile(path, "r");
  if (fscanf(imgFile, "P%d\n", &magicNumber) != 1) {
    fclose(imgFile);
    FATAL("loadPgmImage: corrupt PGM: no magic number found.\n");
  }

  if (magicNumber != 5) {
    fclose(imgFile);
    FATAL("Illegal magic number P%d found. Only P2 and P5 are valid PGM files.\n", magicNumber);
  }
  // skip comment lines
  char c;
  while ((c = fgetc(imgFile)) == '#') {
    do {
      c = fgetc(imgFile);
    } while ((c != EOF) && (c != '\n'));
    if (c == EOF) {
      FATAL("loadImagePGM: corrupt PGM file.\n");
    }
  }
  ungetc(c, imgFile);
  int maxVal;
  // read width and height
  if (fscanf(imgFile, "%d %d", width, height) != 2) {
    FATAL("loadImagePGM: corrupt PGM: no file dimensions found.\n");
  }
  if (fscanf(imgFile, "%d\n", &maxVal) != 1) {
    FATAL("loadImagePGM: corrupt PGM file: no maximal grey value.\n");
  }
  if ((maxVal < 0) || (maxVal > 65535)) {
    FATAL("loadImagePGM: corrupt PGM: maximum grey value found is %d (must be in range [0..65535]).\n", maxVal);
  }

  int npixels = (*width) * (*height);
  int *pixels = malloc(npixels * sizeof(int));
  // magicnumber == 5
  if (maxVal > 255) {  // 2 bytes per pixel, i.e. short
    unsigned short *buffer = malloc(npixels * sizeof(unsigned short));
    if (fread(buffer, 2, npixels, imgFile) != npixels) {
      FATAL("loadImagePGM: corrupt PGM, file is truncated.\n");
    }
    for (int i = 0; i < npixels; i++) {
      pixels[i] = buffer[i];
    }
    free(buffer);
  } else {  // 1 byte per pixel
    unsigned char *buffer = malloc(npixels * sizeof(unsigned char));
    if (fread(buffer, 1, npixels, imgFile) != npixels) {
      FATAL("loadImagePGM: corrupt PGM, file is truncated.\n");
    }
    for (int i = 0; i < npixels; i++) {
      pixels[i] = buffer[i];
    }
    free(buffer);
  }
  fclose(imgFile);
  return pixels;
}

void saveAsPgmImage(const char *path, int *pixels, int width, int height, int numBits) {
  verifyPgmImage(path);
  FILE *pgmFile = openFile(path, "w");
  if (pgmFile == NULL) {
    FATAL("saveImagePGMasP5: failed to open file '%s'.\n", path);
  }
  fprintf(pgmFile, "P5\n%d %d\n%d\n", width, height, (int)(pow(2, numBits) - 1));

  int npixels = (width) * (height);

  if (numBits == 16) {
    unsigned short *buffer = malloc(npixels * sizeof(unsigned short));
    for (int i = npixels; i < npixels; i++) {
      // TODO: clamp
      buffer[i] = pixels[i];
    }
    fwrite(buffer, 2, npixels, pgmFile);
    free(buffer);
  } else if (numBits == 8) {
    unsigned char *buffer = malloc(npixels * sizeof(unsigned char));
    for (int i = 0; i < npixels; i++) {
      buffer[i] = pixels[i];
    }
    fwrite(buffer, 1, npixels, pgmFile);
    free(buffer);
  } else {
    FATAL("%d-bit images are not supported.\n", numBits);
  }
  fclose(pgmFile);
}

void saveAsPgmImage8Bit(const char *path, uint8_t *buffer, int width, int height) {
  int npixels = width * height;
  FILE *pgmFile = openFile(path, "w");
  fprintf(pgmFile, "P5\n%d %d\n%d\n", width, height, 255);
  fwrite(buffer, 1, npixels, pgmFile);
  fclose(pgmFile);
}

void saveAsPgmImage16Bit(const char *path, uint16_t *buffer, int width, int height) {
  int npixels = width * height;
  FILE *pgmFile = openFile(path, "w");
  fprintf(pgmFile, "P5\n%d %d\n%d\n", width, height, 65535);
  fwrite(buffer, 2, npixels, pgmFile);
  fclose(pgmFile);
}
