#include "worker.h"

#include <assert.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "../util/util.h"

// forward declaration for CUDA function
void processImagesCuda(Job job, int numThreads);
void processImagesCudaAsync(Job job, int numThreads, int numBuffers, int numStreams);

/**
 * @brief Send to the master it has finished a certain image
 *
 * @param imageIdx the index of the image it has finished
 */
void workerSignalDone(int imageIdx) { MPI_Send(&imageIdx, 1, MPI_INT, 0, 0, MPI_COMM_WORLD); }

/**
 * @brief Processes the images specififed in the job file located at jobPath. Only processes the images in the path list
 * between [startIdx, startIdx + numImages). The results are saved in the provided output directory.
 *
 * @param jobPath Path to the job file.
 * @param outputDir Directory where the results should be saved.
 * @param numImages Number of images this worker will process.
 * @param rank Index of this worker.
 * @param startIdx Start index in the job's image list.
 * @param failureRecovery Whether failure recovery should be enabled or not.
 */
void workerProcess(const char *jobPath, char *outputDir, int numImages, int rank, int startIdx, int failureRecovery) {
  Timer timer;
  startTime(&timer);
  printf("- Worker node %d starting image processing\n", rank);

	int numThreads = omp_get_num_threads();
  Job job = getPartialJob(jobPath, startIdx, numImages);
  job.outputDir = outputDir;
  // processImagesCuda(job, numThreads);
  // processImagesCuda(job, 2);
	processImagesCudaAsync(job, 2, 1, 1);

  stopTime(&timer);
  printf("- Worker node %d completed all images: %lf sec.\n", rank, elapsedTime(timer));
  freeJob(job);

  if (failureRecovery) {
    // signal you have completed all your work.
    workerSignalDone(-1);
    int buf[1];
    while (1) {
      MPI_Recv(&buf, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if (buf[0] == -1) {
        break;
      }
      Job singleImgJob = getPartialJob(jobPath, buf[0], 1);
      singleImgJob.outputDir = outputDir;
      processImagesCuda(singleImgJob, numThreads);
      workerSignalDone(-1);
      freeJob(singleImgJob);
    }
  }
}