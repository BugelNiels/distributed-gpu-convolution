#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "mpi/jobExecution.h"

/**
 * @brief Starts up the convolution program.
 *
 * @param argc Argument count.
 * @param argv Arguments: <jobfile> <outputDir> [failureRecovery]
 * @return int Exit code.
 */
int main(int argc, char *argv[]) {
  if (argc < 3) {
    printf("Usage: ./conv jobFile outputDir [failureRecovery]\nExample\n\t./conv joblist.txt outputImgs\n");
    return 0;
  }
  int failureRecovery = 0;
  if (argc == 4) {
    failureRecovery = atoi(argv[3]);
  }
  MPI_Init(NULL, NULL);
  executeJobs(argv[1], argv[2], failureRecovery);
  MPI_Finalize();
  return 0;
}
