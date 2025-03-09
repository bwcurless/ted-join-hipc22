#!/bin/bash

#SBATCH --job-name=TedJoin  #the name of your job

#change to your NAU ID below
#SBATCH --output=/scratch/bc2497/TedJoin-%j.out
#SBATCH --error=/scratch/bc2497/TedJoin-%j.out

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
make clean
make monsoon OUTPUT_NEIGHBORS=false INPUT_DATA_DIM=18 COMPUTE_DIM=24
echo "susy running"
srun --unbuffered ./main /scratch/bc2497/datasets/SUSY_normalize_0_1.txt 0.021 21
echo "susy finished"

make clean
make monsoon OUTPUT_NEIGHBORS=false INPUT_DATA_DIM=128 COMPUTE_DIM=128
echo "sift running"
srun --unbuffered ./main /scratch/bc2497/datasets/sift10m_unscaled.txt 122.5 21
srun --unbuffered ./main /scratch/bc2497/datasets/sift10m_unscaled.txt 122.5 21
srun --unbuffered ./main /scratch/bc2497/datasets/sift10m_unscaled.txt 122.5 21
srun --unbuffered ./main /scratch/bc2497/datasets/sift10m_unscaled.txt 136.5 21
srun --unbuffered ./main /scratch/bc2497/datasets/sift10m_unscaled.txt 136.5 21
srun --unbuffered ./main /scratch/bc2497/datasets/sift10m_unscaled.txt 136.5 21
srun --unbuffered ./main /scratch/bc2497/datasets/sift10m_unscaled.txt 152.5 21
srun --unbuffered ./main /scratch/bc2497/datasets/sift10m_unscaled.txt 152.5 21
srun --unbuffered ./main /scratch/bc2497/datasets/sift10m_unscaled.txt 152.5 21
echo "sift finished"

make clean
make monsoon OUTPUT_NEIGHBORS=false INPUT_DATA_DIM=384 COMPUTE_DIM=384
echo "tiny running"
srun --unbuffered ./main /scratch/bc2497/datasets/tiny5m_unscaled.txt 0.18310546875 21
srun --unbuffered ./main /scratch/bc2497/datasets/tiny5m_unscaled.txt 0.18310546875 21
srun --unbuffered ./main /scratch/bc2497/datasets/tiny5m_unscaled.txt 0.18310546875 21
srun --unbuffered ./main /scratch/bc2497/datasets/tiny5m_unscaled.txt 0.20458984375 21
srun --unbuffered ./main /scratch/bc2497/datasets/tiny5m_unscaled.txt 0.20458984375 21
srun --unbuffered ./main /scratch/bc2497/datasets/tiny5m_unscaled.txt 0.20458984375 21
srun --unbuffered ./main /scratch/bc2497/datasets/tiny5m_unscaled.txt 0.2275390625 21
srun --unbuffered ./main /scratch/bc2497/datasets/tiny5m_unscaled.txt 0.2275390625 21
srun --unbuffered ./main /scratch/bc2497/datasets/tiny5m_unscaled.txt 0.2275390625 21
echo "tiny finished"

#make clean
#make monsoon OUTPUT_NEIGHBORS=false INPUT_DATA_DIM=512 COMPUTE_DIM=512
#echo "cifar running"
#srun --unbuffered ./main /scratch/bc2497/datasets/cifar60k_unscaled.txt 0.62890625 21
#echo "cifar finished"
#
#make clean
#make monsoon OUTPUT_NEIGHBORS=false INPUT_DATA_DIM=960 COMPUTE_DIM=960
#echo "gist running"
#srun --unbuffered ./main /scratch/bc2497/datasets/gist_unscaled.txt 0.4736328125 21
#echo "gist finished"


#compute-sanitizer --tool=memcheck ./main "/scratch/bc2497/datasets/bigcross.txt" 0.03
#compute-sanitizer --tool=racecheck ./main "/scratch/bc2497/datasets/bigcross.txt" 0.001
# -f overwrite profile if it exists
# --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Tables
#srun ncu -f -o "tedjoin_profile_%i" --import-source yes --source-folder . --clock-control=none --set full ./main /scratch/bc2497/datasets/bigcross.txt 0.003
#srun nsys profile ./main


echo "----------------- JOB FINISHED -------------"
