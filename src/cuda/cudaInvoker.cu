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
#include "cudaInvoker.cuh"
#include "io/fastBufferIO.cuh"
#include "processing.cuh"
#include "util/cudaUtils.cuh"

/**
 * @brief Length of the buffer for the output path of an image. If long names are used, be sure to increase this.
 *
 */
#define BUFFER_LEN 120

/**
 * @brief Arguments for the threaded cuda streams.
 *
 */
typedef struct CudaStreamArgs {
  Job job;
  Kernel *kernels;
  int startIdx;
  int offset;
} CudaStreamArgs;

/**
 * @brief Create arguments for the threaded cuda streams.
 *
 * @param job The job
 * @param kernels The kernels
 * @param startIdx The start index of the thread in the job list.
 * @param offset Offset between each index in the image path array.
 * @return CudaStreamArgs* Thread arguments
 */
CudaStreamArgs *createArgs(Job job, Kernel *kernels, int startIdx, int offset) {
  CudaStreamArgs *args = (CudaStreamArgs *)malloc(sizeof(CudaStreamArgs));
  args->job = job;
  args->kernels = kernels;
  args->startIdx = startIdx;
  args->offset = offset;
  return args;
}

/**
 * @brief Allocates 3 1D buffers of size width*height: 1 input buffer on the device, 1 output buffer on the device and 1
 * page-locked input/output buffer on the host.
 *
 * @tparam pixel Type of pixels stored in the buffer. Can be uint8_t, uint16_t or int (8-bit, 16-bit and 32-bit
 * respectively).
 * @param input_d Device input buffer.
 * @param output_d Device output buffer.
 * @param buffer_h Host input/output buffer.
 * @param maxImgWidth Maximum width an image can have. Used for determining the buffer size.
 * @param maxImgHeight Maximum height an image can have. Used for determining the buffer size.
 */
template <typename pixel>
void allocateBuffers(pixel **input_d, pixel **output_d, pixel **buffer_h, int maxImgWidth, int maxImgHeight) {
  size_t allocSize = maxImgWidth * maxImgHeight * sizeof(pixel);
  cudaError_t cuda_ret;
  cuda_ret = cudaMalloc((void **)input_d, allocSize);
  cudaErrCheck(cuda_ret, "Unable to allocate input device memory");

  cuda_ret = cudaMalloc((void **)output_d, allocSize);
  cudaErrCheck(cuda_ret, "Unable to allocate output device memory");

  cuda_ret = cudaHostAlloc((void **)buffer_h, allocSize, cudaHostAllocDefault);
  cudaErrCheck(cuda_ret, "Unable to allocate page-locked buffer host memory");
}

/**
 * @brief Processes a number of images within a stream. Note that this function does not explicitly create the streams.
 * We compile with the "--default-stream per-thread" flag, which automatically creates a stream for every thread. If no
 * threads are used, then the default stream is simply used.
 *
 * @tparam pixel Pixel type. Can be uint8_t, uint16_t or int (8-bit, 16-bit and 32-bit respectively).
 * @param job The job
 * @param kernels The kernels
 * @param streamIndex The index of the stream. Used to determine which images from the job this particular stream should
 * process.
 * @param offset Offset between each index in the image path array.
 */
template <typename pixel>
void processImagesInStream(Job job, Kernel *kernels, int streamIndex, int offset) {
  cudaSetDevice(0);
  cudaError_t cuda_ret;
  cudaEvent_t start, stop, avgStart, avgStop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventCreate(&avgStart);
  cudaEventCreate(&avgStop);
  float avgMil;
  int cnt = 0;

  dim3 dimBlock = getBlockDims(BLOCK_WIDTH, BLOCK_HEIGHT);

  pixel *input_d, *output_d, *buffer_h;
  allocateBuffers<pixel>(&input_d, &output_d, &buffer_h, job.info.maxImgWidth, job.info.maxImgHeight);

  char outputLoc[BUFFER_LEN];

  cudaEventRecord(start);
  // The function is quite long, but its difficult to factor out parts without passing tons of parameters. Since this
  // function theoretically never has to change, we leave it like this.
  for (int imgIdx = streamIndex; imgIdx < job.info.numImages; imgIdx += offset) {
    char *inputLoc = job.imgPaths[imgIdx];

    int imgWidth, imgHeight;
    loadImageToBuffer<pixel>(inputLoc, buffer_h, &imgWidth, &imgHeight);
    size_t input_size = imgWidth * imgHeight * sizeof(pixel);

    cuda_ret = cudaMemcpyAsync(input_d, buffer_h, input_size, cudaMemcpyHostToDevice);
    cudaErrCheck(cuda_ret, "Something went wrong with copy Host->Device");

    imgWidth = imgWidth - 2 * job.info.padX;
    imgHeight = imgHeight - 2 * job.info.padY;
    dim3 dimGrid = getGridDims(imgWidth, imgHeight, dimBlock);

    cudaEventRecord(avgStart);
    processImage<pixel>(0, dimGrid, dimBlock, input_d, output_d, kernels, imgWidth, imgHeight);
    cudaEventRecord(avgStop);

    size_t output_size = imgWidth * imgHeight * sizeof(pixel);

    cuda_ret = cudaMemcpyAsync(buffer_h, output_d, output_size, cudaMemcpyDeviceToHost);
    cudaErrCheck(cuda_ret, "Something went wrong with copy Device->Host");

    generateOutputLoc(job.outputDir, inputLoc, outputLoc);

    cuda_ret = cudaStreamSynchronize(0);
    cudaErrCheck(cuda_ret, "Failed to synchronize");
    float kernMils;
    cudaEventElapsedTime(&kernMils, avgStart, avgStop);
    avgMil += kernMils;
    cnt++;

    saveImage<pixel>(outputLoc, buffer_h, imgWidth, imgHeight);
    workerSignalDone(job.startIdx + imgIdx);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("\t Stream %d -> All %d images took: %lf msec - Total kernel time: %lf -  Average kernel time: %lf msec\n",
         streamIndex, cnt, milliseconds, avgMil, avgMil / (float)cnt);

  cudaFree(input_d);
  cudaFree(output_d);
  cudaFreeHost(buffer_h);
}

/**
 * @brief Processes the given images by calling the processImagesInStream function with the correct pixel type.
 *
 * @tparam pixel Pixel type. Can be uint8_t, uint16_t or int (8-bit, 16-bit and 32-bit respectively).
 * @param job The job
 * @param kernels The kernels
 * @param streamIndex The index of the stream. Used to determine which images from the job a particular stream should
 * process.
 * @param offset Offset between each index in the image path array.
 */
void processGivenImages(Job job, Kernel *kernels, int streamIndex, int offset) {
  const int numBits = job.info.numBits;
  switch (numBits) {
    case 8:
      processImagesInStream<uint8_t>(job, kernels, streamIndex, offset);
      break;
    case 16:
      processImagesInStream<uint16_t>(job, kernels, streamIndex, offset);
      break;
    case 32:
      processImagesInStream<int>(job, kernels, streamIndex, offset);
      break;
    default:
      FATAL("%d-bit convolution operations are not supported.\n", numBits);
  }
}

/**
 * @brief Thread function for processGivenImages
 *
 * @param vArgs Arguments
 * @return void* Exit code
 */
void *processImagesAsync(void *vArgs) {
  CudaStreamArgs *args = (CudaStreamArgs *)vArgs;
  processGivenImages(args->job, args->kernels, args->startIdx, args->offset);
  free(args);
  return 0;
}

/**
 * @brief Processes the images specified in the job on the GPU.
 *
 * @param job The job containing information on which images to process.
 * @param numThreads Number of threads the CUDA invoker is allowed to spawn.
 */
void processImagesCuda(Job job, int numThreads) {
  if (job.info.numImages < 1) {
    return;
  }
  cudaSetDevice(0);
  Kernel *kernels = setUpKernels();
  if (numThreads == 1) {
    processGivenImages(job, kernels, 0, 1);
  } else {
    pthread_t *streamThreads = (pthread_t *)malloc(numThreads * sizeof(pthread_t));
    for (int i = 0; i < numThreads; i++) {
      CudaStreamArgs *args = createArgs(job, kernels, i, numThreads);
      if (pthread_create(&streamThreads[i], NULL, processImagesAsync, args)) {
        FATAL("Error creating thread.\n");
      }
    }

    for (int i = 0; i < numThreads; i++) {
      if (pthread_join(streamThreads[i], NULL)) {
        FATAL("Error joining thread.\n");
      }
    }
    free(streamThreads);
  }

  freeKernels(kernels);
  cudaDeviceReset();
}