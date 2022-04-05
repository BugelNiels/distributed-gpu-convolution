#include "kernelformat.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../util/util.h"

void verifyKernelImage(const char *path) {
  char *extension = getFileNameExtension(path);
  if (extension == NULL) {
    FATAL("Filename '%s' has no extension.\n", path);
  }
  if (strcmp("kernel", extension) != 0) {
    FATAL("Attempting to open %s as kernel. Extension is not .kernel.\n", path);
  }
}

Kernel loadKernel(const char *path) {
  verifyKernelImage(path);
  FILE *kernelFile = openFile(path, "r");
  if (kernelFile == NULL) {
    FATAL("Failed to open file %s.\n", path);
  }
  Kernel kernel;
  fscanf(kernelFile, "%d %d\n", &(kernel.width), &(kernel.height));
  int *elems = (int *)safeMalloc(kernel.width * kernel.height * sizeof(int));
  int i = 0;
  for (int y = 0; y < kernel.height; y++) {
    for (int x = 0; x < kernel.width - 1; x++) {
      fscanf(kernelFile, "%d ", &elems[i++]);
    }
    fscanf(kernelFile, "%d\n", &elems[i++]);
  }
  kernel.elems = elems;
  return kernel;
}