#ifndef CUDA_INVOKER_ASYNC_H
#define CUDA_INVOKER_ASYNC_H

extern "C" {
#include "../job/job.h"
}

/**
 * @brief Processes the images specified in the job on the GPU.
 *
 * @param job The job containing information on which images to process.
 * @param numThreads Number of threads the CUDA invoker is allowed to spawn.
 *
 */
extern "C" void processImagesCudaAsync(Job job, int numThreads, int numBuffers, int numStreams);

#endif  // CUDA_INVOKER_ASYNC_H