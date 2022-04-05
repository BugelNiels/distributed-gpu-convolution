#ifndef MASTER_H
#define MASTER_H

#include "../job/job.h"

/**
 * @brief Starts up the master. Awaits for image completions and if enabled redistributes some of the uncompleted images
 * to other workes.
 *
 * @param totalNumImages Total number of images to process.
 * @param numWorkers Total number of workers.
 * @param failureRecovery Whether failure recovery should be enabled or not.
 */
void masterProcess(int totalNumImages, int numWorkers, int failureRecovery);

#endif  // MASTER_H