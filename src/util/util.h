#ifndef UTIL_H
#define UTIL_H

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#ifdef RELEASE
#define debug_print(fmt, ...) \
  do {                        \
    continue;                 \
  } while (0)
#else
#define debug_print(fmt, ...)                                                       \
  do {                                                                              \
    fprintf(stderr, "%s:%d:%s(): " fmt, __FILE__, __LINE__, __func__, __VA_ARGS__); \
  } while (0)
#endif

#define FATAL(fmt, ...)                                                                                          \
  do {                                                                                                           \
    fprintf(stderr, "Fatal error: %s in %s() on line %d:\n\t" fmt, __FILE__, __func__, __LINE__, ##__VA_ARGS__); \
    fflush(stderr);                                                                                              \
    exit(EXIT_FAILURE);                                                                                          \
  } while (0)

typedef struct {
  struct timeval startTime;
  struct timeval endTime;
} Timer;

void *safeMalloc(int sz);
void *safeCalloc(int sz);
int **allocIntMatrix(int width, int height);
int **callocIntMatrix(int width, int height);

void swapInt(int *a, int *b);

char *getFileNameExtension(const char *path);
char *getFileName(char *path);
FILE *openFile(const char *path, const char *mode);

void startTime(Timer *timer);
void stopTime(Timer *timer);
float elapsedTime(Timer timer);

#endif  // UTIL_H