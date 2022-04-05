#include "jobExecution.h"

#include <mpi.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#include "../util/util.h"
#include "master.h"
#include "worker.h"

/**
 * @brief Arguments to pass to the master thread function.
 *
 */
typedef struct MasterArgs {
  int totalNumImages;
  int numWorkers;
  int failureRecovery;
} MasterArgs;

/**
 * @brief Create a the arguments for the master thread function.
 *
 * @param totalNumImages Total number of images to process.
 * @param numWorkers Total number of workers.
 * @param failureRecovery Whether the system should run with failure recovery or not.
 * @return MasterArgs* Master arguments.
 */
MasterArgs *createMasterArgs(int totalNumImages, int numWorkers, int failureRecovery) {
  MasterArgs *args = (MasterArgs *)malloc(sizeof(MasterArgs));
  args->totalNumImages = totalNumImages;
  args->numWorkers = numWorkers;
  args->failureRecovery = failureRecovery;
  return args;
}

/**
 * @brief Calculates how many images each worker should process and what their start index in the job path list is.
 *
 * @param numImagesPerWorker Empty array that will eventually contain for each worker the number of images to process.
 * @param startIndices Empty array that will eventually contain for each worker the start index in the job path list.
 * @param numWorkers The total number of workers.
 * @param numImages The total number of images.
 */
void calculateJobImgIndices(int *numImagesPerWorker, int *startIndices, int numWorkers, int numImages) {
  int rem = (numImages) % (numWorkers);
  int sum = 0;

  for (int i = 0; i < numWorkers; i++) {
    numImagesPerWorker[i] = (numImages) / (numWorkers);
    if (rem > 0) {
      numImagesPerWorker[i]++;
      rem--;
    }
    startIndices[i] = sum;
    sum += numImagesPerWorker[i];
  }
}

/**
 * @brief Does the master process asynchronously.
 *
 * @param vArgs the arguments for the master function.
 * @return void* Exit code of thread function.
 */
void *masterProcessAsync(void *vArgs) {
  MasterArgs *args = (MasterArgs *)vArgs;
  masterProcess(args->totalNumImages, args->numWorkers, args->failureRecovery);
  free(args);
  return 0;
}

/**
 * @brief Processes the images as indicated by the job file located at jobPath and puts the resulting images in the
 * outputDir directory.
 *
 * @param jobPath Path to the job file
 * @param outputDir Directory to put the output images in
 * @param failureRecovery If the system should run with failure recovery. Should be 0 for no, 1 for yes.
 */
void executeJobs(char *jobPath, char *outputDir, int failureRecovery) {
  int rank, numWorkers;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numWorkers);

  int totalNumImages = getTotalNumImagesJob(jobPath);

  // calculate which jobs each worker has to do
  int *numImagesPerWorker = malloc(numWorkers * sizeof(int));
  int *startIndices = malloc(numWorkers * sizeof(int));
  calculateJobImgIndices(numImagesPerWorker, startIndices, numWorkers, totalNumImages);

  if (rank == 0) {
    pthread_t masterThread;
    MasterArgs *mArgs = createMasterArgs(totalNumImages, numWorkers, failureRecovery);
    if (pthread_create(&masterThread, NULL, masterProcessAsync, mArgs)) {
      FATAL("Error creating thread.\n");
    }
    workerProcess(jobPath, outputDir, numImagesPerWorker[rank], rank, startIndices[rank], failureRecovery);
    if (pthread_join(masterThread, NULL)) {
      FATAL("Error joining thread.\n");
    }
  } else {
    workerProcess(jobPath, outputDir, numImagesPerWorker[rank], rank, startIndices[rank], failureRecovery);
  }

  free(numImagesPerWorker);
  free(startIndices);
}