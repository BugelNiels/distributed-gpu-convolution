#ifndef PGM_FORMAT_H
#define PGM_FORMAT_H

#include <stdint.h>

void load8bitPgmImageToMem(const char *path, uint8_t *buffer);
void load16bitPgmImageToMem(const char *path, uint16_t *buffer);

void saveAsPgmImage8Bit(const char *path, uint8_t *buffer, int width, int height);
void saveAsPgmImage16Bit(const char *path, uint16_t *buffer, int width, int height);

void loadPgmImageToMem(const char *path, int *pixels);
int *loadPgmImage(const char *path, int *width, int *height);
void saveAsPgmImage(const char *path, int *pixels, int width, int height, int numBits);

#endif  // PGM_FORMAT_H