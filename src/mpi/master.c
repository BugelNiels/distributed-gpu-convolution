#include "master.h"

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "../util/util.h"

/**
 * @brief Waits for image completions and then redistributes any images that are left.
 *
 * @param totalNumImages Total number of images to be processed.
 * @param numWorkers Total number of workers.
 */
void handleImageCompletions(int totalNumImages, int numWorkers) {
  int *imgsDone = calloc(totalNumImages, sizeof(int));
  int numImgsLeft = totalNumImages - 1;
  int numWorkersLeft = numWorkers;
  MPI_Status status;
  while (numWorkersLeft > 0) {
    int buf[1];
    MPI_Recv(buf, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    if (buf[0] == -1) {
      // the worker is done and can be send more work
      // get the last unfinished image
      while (numImgsLeft >= 0) {
        if (imgsDone[numImgsLeft] == 0) {
          MPI_Send(&numImgsLeft, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
          numImgsLeft--;
          break;
        }
        numImgsLeft--;
      }

      // no more images should be send
      if (numImgsLeft == -1) {
        numWorkersLeft--;
        buf[0] = -1;
        MPI_Send(buf, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
      }
    } else {
      int isImgDone = imgsDone[buf[0]];
      if (!isImgDone) {
        imgsDone[buf[0]] = 1;
      }
    }
  }
  free(imgsDone);
}

/**
 * @brief Waits for each worker to send a signal it completed an image.
 *
 * @param totalNumImages Total number of images to be processed.
 */
void awaitImageCompletions(int totalNumImages) {
  int numImgsLeft = totalNumImages;
  int buf[1];
  while (numImgsLeft > 0) {
    MPI_Recv(buf, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    numImgsLeft--;
  }
}

/**
 * @brief Starts up the master. Awaits for image completions and if enabled redistributes some of the uncompleted images
 * to other workes.
 *
 * @param totalNumImages Total number of images to process.
 * @param numWorkers Total number of workers.
 * @param failureRecovery Whether failure recovery should be enabled or not.
 */
void masterProcess(int totalNumImages, int numWorkers, int failureRecovery) {
  Timer timer;
  startTime(&timer);
  printf("Master process listening for image completions..\n");

  if (failureRecovery) {
    handleImageCompletions(totalNumImages, numWorkers);
  } else {
    awaitImageCompletions(totalNumImages);
  }

  stopTime(&timer);
  printf("All images completed. Took %lf sec.\n", elapsedTime(timer));
}