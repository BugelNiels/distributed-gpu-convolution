#ifndef JOB_H
#define JOB_H

/**
 * @brief Structure containing some metadata about the job.
 *
 */
typedef struct JobInfo {
  int numImages;
  int numBits;
  int maxImgWidth;
  int maxImgHeight;
  int padX;
  int padY;
} JobInfo;

/**
 * @brief Structure containing the images to process for a specific job.
 *
 */
typedef struct Job {
  JobInfo info;
  char *outputDir;
  char **imgPaths;
  int startIdx;
} Job;

/**
 * @brief Get the total number of images in the job file located at the provided path.
 *
 * @param path Location of the job file.
 * @return int Total number of images in the job file.
 */
int getTotalNumImagesJob(const char *path);

/**
 * @brief Get the job info from the job file located at the provided path.
 *
 * @param path Location of the job file.
 * @return JobInfo Info about the job.
 */
JobInfo getJobInfo(const char *path);

/**
 * @brief Get the complete job from the job file located at the provided path.
 *
 * @param path Location of the job file.
 * @return Job The complete job.
 */
Job getJob(const char *path);

/**
 * @brief Get the partial job from the job file located at the provided path.
 * Only includes the paths in the range [startIdx, startIdx + numImgs).
 *
 * @param path Location of the job file.
 * @param startIdx Start index in the list of input images.
 * @param numImgs Number of images to retrieve from the list, starting at startIdx.
 * @return Job The partial job.
 */
Job getPartialJob(const char *path, int startIdx, int numImgs);

/**
 * @brief Free memory used by the job.
 *
 * @param job The job to free.
 */
void freeJob(Job job);

#endif  // JOB_H