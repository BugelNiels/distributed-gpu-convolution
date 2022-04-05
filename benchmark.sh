#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:02:00
#SBATCH --job-name=dilation4
#SBATCH --output=result4_2.out
#

# Compile the program
make RELEASE=1

# Specify the number of threads that OpenMP applications can use.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load foss/2020a
module load CUDA/11.1.1
srun ./conv benchmark.txt /data/s3405583/outputTiles4

make clean