#ifndef TIFF_FORMAT_H
#define TIFF_FORMAT_H

void loadTiffImageToMem(const char *path, int *pixels);
int *loadTiffImage(const char *path, int *width, int *height);
void saveAsTiffImage(const char *path, int *pixels, int width, int height, int numBits);

#endif  // TIFF_FORMAT_H