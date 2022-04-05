#ifndef WORKER_H
#define WORKER_H

#include "../job/job.h"

/**
 * @brief Send to the master it has finished a certain image
 *
 * @param imageIdx the index of the image it has finished
 */
void workerSignalDone(int imageIdx);

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
void workerProcess(const char *jobPath, char *outputDir, int numImages, int rank, int startIdx, int failureRecovery);

#endif  // WORKER_H