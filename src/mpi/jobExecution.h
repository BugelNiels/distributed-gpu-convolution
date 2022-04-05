#ifndef JOB_EXECUTION_H
#define JOB_EXECUTION_H

#include "../job/job.h"

/**
 * @brief Processes the images as indicated by the job file located at jobPath and puts the resulting images in the
 * outputDir directory.
 *
 * @param jobPath Path to the job file
 * @param outputDir Directory to put the output images in
 * @param failureRecovery If the system should run with failure recovery. Should be 0 for no, 1 for yes.
 */
void executeJobs(char *jobPath, char *outputDir, int failureRecovery);

#endif  // JOB_EXECUTION_H