#ifndef ASYNC_IO_H
#define ASYNC_IO_H

#include <stdint.h>

#include "../util/stack.cuh"

extern "C" {
#include "../../job/job.h"
#include "../../mpi/worker.h"
}
#include "../util/cudaUtils.cuh"
#include "fastBufferIO.cuh"

/**
 * @brief Simple structure to contain some basic infomrmation about an image.
 *
 * @tparam pixel Type of the pixel. Used for differences in dynamic range.
 */
template <typename pixel>
struct Image {
  pixel *buffer;
  // inputWidth != width if and only if the input image was padded (as specific in the job file)
  int inputWidth;
  int inputHeight;
  int width;
  int height;
  char *outputLoc;
};

/**
 * @brief Structure that contains all the stacks necessary to have seperate save/load threads.
 *
 * @tparam pixel Type of the pixel. Used for differences in dynamic range.
 */
template <typename pixel>
struct AsyncIO {
  Job job;
  pthread_mutex_t bufferLock;
  pthread_mutex_t loadLock;
  pthread_mutex_t saveLock;

  Stack<pixel *> bufferStack;
  Stack<Image<pixel> *> loadStack;
  Stack<Image<pixel> *> saveStack;
  Stack<char *> pathStack;
  int imgLeft;
};

/**
 * @brief Initialises the structure used by the saving/loading threads.
 *
 * @tparam pixel Type of the pixel. Used for differences in dynamic range.
 * @param job The job file.
 * @param numThreads Number of threads that will be used for saving/loading.
 * @param numBuffers Number of page-locked host buffers that should be allocated.
 * @return AsyncIO<pixel> AIO structure containing all the stacks and locks.
 */
template <typename pixel>
AsyncIO<pixel> initAsyncIO(Job job, int numThreads, int numBuffers) {
  AsyncIO<pixel> aio;
  aio.bufferStack = allocateStack<pixel *>(numBuffers);
  aio.loadStack = allocateStack<Image<pixel> *>(job.info.numImages);
  aio.saveStack = allocateStack<Image<pixel> *>(job.info.numImages);
  aio.pathStack = allocateStack<char *>(job.info.numImages);

  pthread_mutex_init(&aio.bufferLock, NULL);
  pthread_mutex_init(&aio.loadLock, NULL);
  pthread_mutex_init(&aio.saveLock, NULL);
  aio.imgLeft = job.info.numImages;
  aio.job = job;
  size_t allocSize = job.info.maxImgWidth * job.info.maxImgHeight * sizeof(pixel);

  pthread_mutex_lock(&aio.bufferLock);
  for (int i = 0; i < numBuffers; i++) {
    pixel *buffer;
    cudaError_t cuda_ret = cudaHostAlloc((void **)&buffer, allocSize, cudaHostAllocDefault);
    cudaErrCheck(cuda_ret, "Unable to allocate page-locked buffer host memory");
    push(&(aio.bufferStack), buffer);
  }

  for (int i = 0; i < job.info.numImages; i++) {
    push(&(aio.pathStack), job.imgPaths[i]);
  }

  pthread_mutex_unlock(&aio.bufferLock);
  return aio;
}

/**
 * @brief Tries to pop a loaded image from the load stack. If not available, it will return NULL.
 *
 * @tparam pixel Type of the pixel. Used for differences in dynamic range.
 * @param aio AIO struct.
 * @return Image<pixel>* An image with a loaded buffer.
 */
template <typename pixel>
Image<pixel> *loadImageAsync(AsyncIO<pixel> *aio) {
  pthread_mutex_lock(&aio->loadLock);
  if (isEmpty(&aio->loadStack)) {
    pthread_mutex_unlock(&aio->loadLock);
    return NULL;
  }
  Image<pixel> *item = pop(&aio->loadStack);
  pthread_mutex_unlock(&aio->loadLock);
  return item;
}

/**
 * @brief Pushes a save request onto the save stack.
 *
 * @tparam pixel Type of the pixel. Used for differences in dynamic range.
 * @param aio AIO struct.
 * @param image The image to save.
 */
template <typename pixel>
void saveImageAsync(AsyncIO<pixel> *aio, Image<pixel> *image) {
  pthread_mutex_lock(&aio->saveLock);
  push(&aio->saveStack, image);
  pthread_mutex_unlock(&aio->saveLock);
}

/**
 * @brief Determines whether there are still images left to process.
 *
 * @tparam pixel Type of the pixel. Used for differences in dynamic range.
 * @param aio AIO struct.
 * @return true If there are still images left to process.
 * @return false If all images have been processed.
 */
template <typename pixel>
bool imagesLeft(AsyncIO<pixel> *aio) {
  return aio->imgLeft > 0;
}

/**
 * @brief Frees all the buffers used by the AIO struct.
 *
 * @tparam pixel Type of the pixel. Used for differences in dynamic range.
 * @param aio AIO to free the memory of. Does not free the struct itself.
 */
template <typename pixel>
void freeBuffers(AsyncIO<pixel> *aio) {
  while (!isEmpty(&aio->bufferStack)) {
    pixel *buf = pop(&aio->bufferStack);
    cudaFreeHost(buf);
  }
  freeStack(&aio->bufferStack);
  freeStack(&aio->saveStack);
  freeStack(&aio->loadStack);
  freeStack(&aio->pathStack);
}

/**
 * @brief Let's a read/write thread attempt to pop from the save queue, save the image and put the buffer back onto the
 * buffer stack.
 *
 * @tparam pixel Type of the pixel. Used for differences in dynamic range.
 * @param aio AIO struct.
 */
template <typename pixel>
void saveAction(AsyncIO<pixel> *aio) {
  // save from save stack and put back into buffer stack
  pthread_mutex_lock(&aio->saveLock);
  if (isEmpty(&aio->saveStack)) {
    pthread_mutex_unlock(&aio->saveLock);
    return;
  }
  Image<pixel> *img = pop(&aio->saveStack);
  pthread_mutex_unlock(&aio->saveLock);
  saveImage<pixel>(img->outputLoc, img->buffer, img->width, img->height);

  pthread_mutex_lock(&aio->bufferLock);
  aio->imgLeft--;
  push(&aio->bufferStack, img->buffer);
  pthread_mutex_unlock(&aio->bufferLock);
  workerSignalDone(0);
}

/**
 * @brief Let's a read/write thread attempt to retrieve an available buffer, an available path and load the image at
 * said path into the buffer.
 *
 * @tparam pixel Type of the pixel. Used for differences in dynamic range.
 * @param aio AIO struct.
 */
template <typename pixel>
void loadAction(AsyncIO<pixel> *aio) {
  pthread_mutex_lock(&aio->bufferLock);
  if (isEmpty(&aio->bufferStack) || isEmpty(&aio->pathStack)) {
    pthread_mutex_unlock(&aio->bufferLock);
    return;
  }
  char *inputLoc = pop(&aio->pathStack);
  pixel *buf = pop(&aio->bufferStack);
  pthread_mutex_unlock(&aio->bufferLock);

  // load image into buffer from buffer stack and put in load stack
  int imgWidth, imgHeight;
  loadImageToBuffer<pixel>(inputLoc, buf, &imgWidth, &imgHeight);
  Image<pixel> *img = (Image<pixel> *)malloc(sizeof(Image<pixel>));
  img->buffer = buf;
  img->inputWidth = imgWidth;
  img->inputHeight = imgHeight;
  img->width = imgWidth - 2 * aio->job.info.padX;
  img->height = imgHeight - 2 * aio->job.info.padY;
  char *outputLoc = (char *)malloc(150 * sizeof(char));
  generateOutputLoc(aio->job.outputDir, inputLoc, outputLoc);
  img->outputLoc = outputLoc;

  pthread_mutex_lock(&aio->loadLock);
  push(&aio->loadStack, img);
  pthread_mutex_unlock(&aio->loadLock);
}

/**
 * @brief Let's the read/write thread do a save/load action depending on the state of the stacks. If there are more
 * images to save than there are buffers available, then it will do a save action.
 *
 * @tparam pixel
 * @param aio
 */
template <typename pixel>
void doThreadAction(AsyncIO<pixel> *aio) {
  if (getSize(&aio->saveStack) > getSize(&aio->bufferStack) || isEmpty(&aio->pathStack)) {
    saveAction(aio);
  } else {
    loadAction(aio);
  }
}

#endif  // ASYNC_IO_H