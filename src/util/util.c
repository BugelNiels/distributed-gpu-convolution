#include "util.h"

#include "string.h"

void *safeMalloc(int sz) {
  void *p = malloc(sz);
  if (p == NULL) {
    FATAL("SafeMalloc(%d) failed.\n", sz);
  }
  return p;
}

void *safeCalloc(int sz) {
  void *p = calloc(sz, 1);
  if (p == NULL) {
    FATAL("SafeCalloc(%d) failed.\n", sz);
  }
  return p;
}

static int **arrangeIntPointers(int **matrix, int width, int height) {
  int *p = (int *)(matrix + height);
  for (int y = 0; y < height; y++) {
    matrix[y] = p + width * y;
  }
  return matrix;
}

int **allocIntMatrix(int width, int height) {
  int **matrix = safeMalloc(height * sizeof(int *) + width * height * sizeof(int));
  return arrangeIntPointers(matrix, width, height);
}

int **callocIntMatrix(int width, int height) {
  int **matrix = safeCalloc(height * sizeof(int *) + width * height * sizeof(int));
  return arrangeIntPointers(matrix, width, height);
}

void swapInt(int *a, int *b) {
  int c = *a;
  *a = *b;
  *b = c;
}

char *getFileNameExtension(const char *path) {
  char *extension = strrchr(path, '.');
  return (extension == NULL ? NULL : extension + 1);
}

char *getFileName(char *path) {
  char *extension = strrchr(path, '/');
  return (extension == NULL ? path : extension + 1);
}

void startTime(Timer *timer) { gettimeofday(&(timer->startTime), NULL); }

void stopTime(Timer *timer) { gettimeofday(&(timer->endTime), NULL); }

float elapsedTime(Timer timer) {
  return ((float)((timer.endTime.tv_sec - timer.startTime.tv_sec) +
                  (timer.endTime.tv_usec - timer.startTime.tv_usec) / 1.0e6));
}

FILE *openFile(const char *path, const char *mode) {
  FILE *file;
  file = fopen(path, mode);
  if (file == NULL) {
    FATAL("File %s cannot be opened. Make sure the file/directory exists.\n", path);
  }
  return file;
}