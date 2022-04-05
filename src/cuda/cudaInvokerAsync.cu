#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

extern "C" {
#include "../imageFormats/kernelformat.h"
#include "../mpi/worker.h"
#include "../util/util.h"
#include "cudaConfig.h"
}
#include "cudaInvokerAsync.cuh"
#include "io/asyncIO.cuh"
#include "io/fastBufferIO.cuh"
#include "processing.cuh"
#include "util/cudaUtils.cuh"

template <typename pixel>
void allocateDeviceBuffers(pixel **input_d, pixel **output_d, int maxImgWidth, int maxImgHeight) {
  size_t allocSize = maxImgWidth * maxImgHeight * sizeof(pixel);
  cudaError_t cuda_ret;
  cuda_ret = cudaMalloc((void **)input_d, allocSize);
  cudaErrCheck(cuda_ret, "Unable to allocate input device memory");

  cuda_ret = cudaMalloc((void **)output_d, allocSize);
  cudaErrCheck(cuda_ret, "Unable to allocate output device memory");
}

template <typename T>
void *ioThreadFunc(void *arg) {
  AsyncIO<T> *io = (AsyncIO<T> *)arg;
  while (imagesLeft(io)) {
    doThreadAction(io);
  }
  return 0;
}

template <typename pixel>
struct CallbackArgs {
  AsyncIO<pixel> *io;
  Image<pixel> *image;
};

template <typename pixel>
void CUDART_CB saveCallback(void *args) {
  CallbackArgs<pixel> *cargs = (CallbackArgs<pixel> *)args;
  saveImageAsync(cargs->io, cargs->image);
  free(cargs);
}

template <typename pixel>
void processImagesAsync2(Job job, int numThreads, int numBuffers, int numStreams) {
  AsyncIO<pixel> asyncHandler = initAsyncIO<pixel>(job, numThreads, numBuffers);
  pthread_t *streamThreads = (pthread_t *)malloc(numThreads * sizeof(pthread_t));
  for (int i = 0; i < numThreads; i++) {
    if (pthread_create(&streamThreads[i], NULL, ioThreadFunc<pixel>, &asyncHandler)) {
      FATAL("Error creating thread.\n");
    }
    pthread_detach(streamThreads[i]);
  }

  pixel **inputDeviceBuffers = (pixel **)malloc(numStreams * sizeof(pixel *));
  pixel **outputDeviceBuffers = (pixel **)malloc(numStreams * sizeof(pixel *));
  cudaStream_t *streams = (cudaStream_t *)malloc(numStreams * sizeof(cudaStream_t));

  for (size_t i = 0; i < numStreams; i++) {
    pixel *input_d, *output_d;
    allocateDeviceBuffers(&input_d, &output_d, job.info.maxImgWidth, job.info.maxImgHeight);
    inputDeviceBuffers[i] = input_d;
    outputDeviceBuffers[i] = output_d;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    streams[i] = stream;
  }
  Kernel *kernels = setUpKernels();
  dim3 dimBlock = getBlockDims(BLOCK_WIDTH, BLOCK_HEIGHT);

  cudaDeviceSynchronize();

  int sIdx = 0;
  while (imagesLeft(&asyncHandler)) {
    Image<pixel> *img = loadImageAsync(&asyncHandler);
    if (img == NULL) {
      continue;
    }
    cudaStream_t stream = streams[sIdx];
    size_t input_size = img->inputWidth * img->inputHeight * sizeof(pixel);
    cudaError_t cuda_ret =
        cudaMemcpyAsync(inputDeviceBuffers[sIdx], img->buffer, input_size, cudaMemcpyHostToDevice, stream);
    cudaErrCheck(cuda_ret, "Something went wrong with copy Host->Device");
    dim3 dimGrid = getGridDims(img->width, img->height, dimBlock);

    processImage<pixel>(stream, dimGrid, dimBlock, inputDeviceBuffers[sIdx], outputDeviceBuffers[sIdx], kernels,
                        img->width, img->height);

    size_t output_size = img->width * img->height * sizeof(pixel);
    cuda_ret = cudaMemcpyAsync(img->buffer, outputDeviceBuffers[sIdx], output_size, cudaMemcpyDeviceToHost, stream);
    cudaErrCheck(cuda_ret, "Something went wrong with copy Device->Host");

    CallbackArgs<pixel> *args = (CallbackArgs<pixel> *)malloc(sizeof(CallbackArgs<pixel>));
    args->io = &asyncHandler;
    args->image = img;
    cudaLaunchHostFunc(stream, saveCallback<pixel>, args);

    sIdx = (sIdx + 1) % numStreams;
  }
  for (size_t i = 0; i < numStreams; i++) {
    cudaFree(inputDeviceBuffers[i]);
    cudaFree(outputDeviceBuffers[i]);
  }
  freeKernels(kernels);
  freeBuffers(&asyncHandler);
  free(inputDeviceBuffers);
  free(outputDeviceBuffers);
}

/**
 * @brief Processes the images specified in the job on the GPU.
 *
 * @param job The job containing information on which images to process.
 * @param numThreads Number of threads the CUDA invoker is allowed to spawn.
 */
void processImagesCudaAsync(Job job, int numThreads, int numBuffers, int numStreams) {
  if (job.info.numImages < 1) {
    return;
  }

  const int numBits = job.info.numBits;
  switch (numBits) {
    case 8:
      processImagesAsync2<uint8_t>(job, numThreads, numBuffers, numStreams);
      break;
    case 16:
      processImagesAsync2<uint16_t>(job, numThreads, numBuffers, numStreams);
      break;
    case 32:
      processImagesAsync2<int>(job, numThreads, numBuffers, numStreams);
      break;
    default:
      FATAL("%d-bit convolution operations are not supported.\n", numBits);
  }

  cudaDeviceReset();
}