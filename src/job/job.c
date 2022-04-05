#include "job.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../util/util.h"

/**
 * @brief Get the total number of images in the job file located at the provided path.
 *
 * @param path Location of the job file.
 * @return int Total number of images in the job file.
 */
int getTotalNumImagesJob(const char *path) {
  int numImages;
  FILE *fp = openFile(path, "r");
  fscanf(fp, "%d\n", &numImages);
  fclose(fp);
  return numImages;
}

/**
 * @brief Get the job info from the job file located at the provided file pointer.
 *
 * @param fp File pointer to the job file.
 * @return JobInfo Info about the job.
 */
JobInfo getJobInfoFromFile(FILE *fp) {
  JobInfo info;
  fscanf(fp, "%d\n", &info.numImages);
  fscanf(fp, "%d\n", &info.numBits);
  fscanf(fp, "%d %d\n", &info.maxImgWidth, &info.maxImgHeight);
  fscanf(fp, "%d %d\n", &info.padX, &info.padY);
  return info;
}

/**
 * @brief Get the job info from the job file located at the provided path.
 *
 * @param path Location of the job file.
 * @return JobInfo Info about the job.
 */
JobInfo getJobInfo(const char *path) {
  FILE *fp = openFile(path, "r");
  JobInfo info = getJobInfoFromFile(fp);
  fclose(fp);
  return info;
}

/**
 * @brief Get the partial job from the job file located at the provided path.
 * Only includes the paths in the range [startIdx, startIdx + numImgs).
 *
 * @param path Location of the job file.
 * @param startIdx Start index in the list of input images.
 * @param numImgs Number of images to retrieve from the list, starting at startIdx.
 * @return Job The partial job.
 */
Job getPartialJob(const char *path, int startIdx, int numImages) {
  FILE *fp = openFile(path, "r");
  char **imgPaths = safeMalloc(numImages * sizeof(char *));

  JobInfo info = getJobInfoFromFile(fp);

  Job job;
  job.info = info;
  job.startIdx = startIdx;
  if (numImages <= 0) {
    job.info.numImages = 0;
    return job;
  }

  // have two indexes, one to keep track of the lines in the file, another for the current job
  int lineIdx = 0;
  int jobIdx = 0;
  char *line = NULL;
  size_t len = 0;
  int read;
  while ((read = getline(&line, &len, fp)) != -1 && lineIdx < startIdx + numImages) {
    lineIdx++;
    if (lineIdx <= startIdx) {
      continue;
    }
    char *path = (char *)malloc(read * sizeof(char));
    memcpy(path, line, read * sizeof(char));
    if (path[read - 1] == '\n') {
      path[read - 1] = '\0';
    }
    imgPaths[jobIdx] = path;
    jobIdx++;
  }
  fclose(fp);
  job.info.numImages = jobIdx;
  job.imgPaths = imgPaths;
  return job;
}

/**
 * @brief Get the complete job from the job file located at the provided path.
 *
 * @param path Location of the job file.
 * @return Job The complete job.
 */
Job getJob(const char *path) { return getPartialJob(path, 0, getTotalNumImagesJob(path)); }

/**
 * @brief Free memory used by the job.
 *
 * @param job The job to free.
 */
void freeJob(Job job) {
  for (size_t i = 0; i < job.info.numImages; i++) {
    free(job.imgPaths[i]);
  }
  free(job.imgPaths);
}