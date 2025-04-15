#!/bin/bash

#SBATCH --job-name=TedJoin-brute  #the name of your job

#change to your NAU ID below
#SBATCH --output=/scratch/bc2497/TedJoin-brute-%j.out
#SBATCH --error=/scratch/bc2497/TedJoin-brute-%j.out

#SBATCH --time=48:00:00
#SBATCH --mem=100000         #memory requested in MiB
#SBATCH -G 1 #resource requirement (1 GPU)
#SBATCH -C a100 #GPU Model: k80, p100, v100, a100
#SBATCH --exclusive=user

# Gowanlock Partition
#SBATCH --account=gowanlock_condo
#SBATCH --partition=gowanlock

# Main partition
##SBATCH --account=gowanlock

set -e

# Code will not compile if we don't load the module
module load cuda/11.4

# A sanity check to make sure I get the same results as the paper
#make clean
#make monsoon INPUT_DATA_DIM=18 COMPUTE_DIM=24
#echo "susy running"
#srun --unbuffered ./main /scratch/bc2497/datasets/SUSY_normalize_0_1.txt 0.021 4
#echo "susy finished"

# Get a plot of tedjoin flops for various dimensionality datasets to compare to my routine.
# Targeting a selectivity of 64 for these tests
# Do brute force search
make clean
make monsoon INPUT_DATA_DIM=64 COMPUTE_DIM=64
echo "64D running"
srun --unbuffered ./main /scratch/bc2497/datasets/tedjoin_expo_data/dataset_fixed_len_pts_expo_NDIM_64_pts_100000.txt 0.1754 4
srun --unbuffered ./main /scratch/bc2497/datasets/tedjoin_expo_data/dataset_fixed_len_pts_expo_NDIM_64_pts_100000.txt 0.1754 4
srun --unbuffered ./main /scratch/bc2497/datasets/tedjoin_expo_data/dataset_fixed_len_pts_expo_NDIM_64_pts_100000.txt 0.1754 4


make clean
make monsoon INPUT_DATA_DIM=128 COMPUTE_DIM=128
echo "128D running"
srun --unbuffered ./main /scratch/bc2497/datasets/tedjoin_expo_data/dataset_fixed_len_pts_expo_NDIM_128_pts_100000.txt 0.2873 4
srun --unbuffered ./main /scratch/bc2497/datasets/tedjoin_expo_data/dataset_fixed_len_pts_expo_NDIM_128_pts_100000.txt 0.2873 4
srun --unbuffered ./main /scratch/bc2497/datasets/tedjoin_expo_data/dataset_fixed_len_pts_expo_NDIM_128_pts_100000.txt 0.2873 4


make clean
make monsoon INPUT_DATA_DIM=256 COMPUTE_DIM=256
echo "256D running"
srun --unbuffered ./main /scratch/bc2497/datasets/tedjoin_expo_data/dataset_fixed_len_pts_expo_NDIM_256_pts_100000.txt 0.449 4
srun --unbuffered ./main /scratch/bc2497/datasets/tedjoin_expo_data/dataset_fixed_len_pts_expo_NDIM_256_pts_100000.txt 0.449 4
srun --unbuffered ./main /scratch/bc2497/datasets/tedjoin_expo_data/dataset_fixed_len_pts_expo_NDIM_256_pts_100000.txt 0.449 4


make clean
make monsoon INPUT_DATA_DIM=384 COMPUTE_DIM=384
echo "384D running"
srun --unbuffered ./main /scratch/bc2497/datasets/tedjoin_expo_data/dataset_fixed_len_pts_expo_NDIM_384_pts_100000.txt 0.4452 4
srun --unbuffered ./main /scratch/bc2497/datasets/tedjoin_expo_data/dataset_fixed_len_pts_expo_NDIM_384_pts_100000.txt 0.4452 4
srun --unbuffered ./main /scratch/bc2497/datasets/tedjoin_expo_data/dataset_fixed_len_pts_expo_NDIM_384_pts_100000.txt 0.4452 4


#make clean
#make monsoon INPUT_DATA_DIM=128 COMPUTE_DIM=128
#echo "sift running"
#srun --unbuffered ./main /scratch/bc2497/datasets/sift10m_unscaled.txt 122.5 4
#srun --unbuffered ./main /scratch/bc2497/datasets/sift10m_unscaled.txt 122.5 4
#srun --unbuffered ./main /scratch/bc2497/datasets/sift10m_unscaled.txt 122.5 4
#srun --unbuffered ./main /scratch/bc2497/datasets/sift10m_unscaled.txt 136.5 4
#srun --unbuffered ./main /scratch/bc2497/datasets/sift10m_unscaled.txt 136.5 4
#srun --unbuffered ./main /scratch/bc2497/datasets/sift10m_unscaled.txt 136.5 4
#srun --unbuffered ./main /scratch/bc2497/datasets/sift10m_unscaled.txt 152.5 4
#srun --unbuffered ./main /scratch/bc2497/datasets/sift10m_unscaled.txt 152.5 4
#srun --unbuffered ./main /scratch/bc2497/datasets/sift10m_unscaled.txt 152.5 4
#echo "sift finished"
#
#make clean
#make monsoon INPUT_DATA_DIM=384 COMPUTE_DIM=384
#echo "tiny running"
#srun --unbuffered ./main /scratch/bc2497/datasets/tiny5m_unscaled.txt 0.18310546875 4
#srun --unbuffered ./main /scratch/bc2497/datasets/tiny5m_unscaled.txt 0.18310546875 4
#srun --unbuffered ./main /scratch/bc2497/datasets/tiny5m_unscaled.txt 0.18310546875 4
#srun --unbuffered ./main /scratch/bc2497/datasets/tiny5m_unscaled.txt 0.20458984375 4
#srun --unbuffered ./main /scratch/bc2497/datasets/tiny5m_unscaled.txt 0.20458984375 4
#srun --unbuffered ./main /scratch/bc2497/datasets/tiny5m_unscaled.txt 0.20458984375 4
#srun --unbuffered ./main /scratch/bc2497/datasets/tiny5m_unscaled.txt 0.2275390625 4
#srun --unbuffered ./main /scratch/bc2497/datasets/tiny5m_unscaled.txt 0.2275390625 4
#srun --unbuffered ./main /scratch/bc2497/datasets/tiny5m_unscaled.txt 0.2275390625 4
#echo "tiny finished"

#make clean
#make monsoon INPUT_DATA_DIM=512 COMPUTE_DIM=512
#echo "cifar running"
#srun --unbuffered ./main /scratch/bc2497/datasets/cifar60k_unscaled.txt 0.62890625 4
#echo "cifar finished"
#
#make clean
#make monsoon INPUT_DATA_DIM=960 COMPUTE_DIM=960
#echo "gist running"
#srun --unbuffered ./main /scratch/bc2497/datasets/gist_unscaled.txt 0.4736328125 4
#echo "gist finished"


#compute-sanitizer --tool=memcheck ./main "/scratch/bc2497/datasets/bigcross.txt" 0.03
#compute-sanitizer --tool=racecheck ./main "/scratch/bc2497/datasets/bigcross.txt" 0.001
# -f overwrite profile if it exists
# --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Tables
#srun ncu -f -o "tedjoin_profile_%i" --import-source yes --source-folder . --clock-control=none --set full ./main /scratch/bc2497/datasets/bigcross.txt 0.1
#srun nsys profile ./main


echo "----------------- JOB FINISHED -------------"
