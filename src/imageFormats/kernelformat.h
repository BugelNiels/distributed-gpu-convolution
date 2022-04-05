#ifndef KERNEL_FORMAT_H
#define KERNEL_FORMAT_H

typedef struct Kernel {
  int width;
  int height;
  int *elems;
} Kernel;

Kernel loadKernel(const char *path);

#endif  // KERNEL_FORMAT_H