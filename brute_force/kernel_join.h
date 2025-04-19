#ifndef KERNEL_JOIN_H
#define KERNEL_JOIN_H

#include <cuda_fp16.h>

#include "params.h"

__global__ void printMatrix(double *matrix, unsigned int nbElements);
__global__ void printMatrixTranspose(double *matrix, unsigned int size,
                                     unsigned int nbElements);
__global__ void printMatrixResult(double *matrix, unsigned int size,
                                  unsigned int nbElements);

__global__ void convertDataset(INPUT_DATA_TYPE *in, COMPUTE_TYPE *out,
                               unsigned int nbPoints);

__global__ void preComputedSquaredCoordinates(COMPUTE_TYPE *dataset,
                                              ACCUM_TYPE *preComputeCoordinates,
                                              unsigned int nbQueryPoints);

__global__ void
preComputedSquaredCoordinatesComplete(COMPUTE_TYPE *dataset,
                                      ACCUM_TYPE *preComputeCoordinates,
                                      unsigned int nbQueryPoints);

__global__ void transposeDataset(COMPUTE_TYPE *inputDataset,
                                 COMPUTE_TYPE *outputDataset,
                                 unsigned int nbQueryPoints);

__global__ void fillResultMatrix(ACCUM_TYPE *preComputedSquaredCoordinates,
                                 ACCUM_TYPE *resultMatrix,
                                 unsigned int nbQueryPoints);

__global__ void finishResultMatrix(ACCUM_TYPE *preComputedSquaredCoordinates,
                                   ACCUM_TYPE *resultMatrix,
                                   unsigned int nbQueryPoints,
                                   unsigned long long *cnt,
                                   ACCUM_TYPE *epsilon);

//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\


__global__ void distanceCalculationBruteForceCuda(unsigned int *nbQueryPoints,
                                                  COMPUTE_TYPE *dataset,
                                                  ACCUM_TYPE *epsilon,
                                                  unsigned long long *cnt);

__global__ void
distanceCalculationBruteForceCudaAlt(unsigned int *nbQueryPoints,
                                     COMPUTE_TYPE *dataset, ACCUM_TYPE *epsilon,
                                     unsigned long long *cnt);

//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\


__global__ void distanceCalculationBruteForceTensorBasic(
    unsigned int *nbQueryPoints, COMPUTE_TYPE *dataset, ACCUM_TYPE *epsilon,
    COMPUTE_TYPE *identityMatrix, unsigned long long *cnt);

__global__ void distanceCalculationBruteForceTensorHalfOpti(
    unsigned int *nbQueryPoints, COMPUTE_TYPE *dataset, ACCUM_TYPE *epsilon,
    unsigned long long *cnt, ACCUM_TYPE *preComputedSquaredCoordinates);

//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\


__global__ void distanceCalculationBruteForceTensorDoubleOpti(
    unsigned int *nbQueryPoints, double *dataset, double *epsilon,
    unsigned long long *cnt, double *preComputedSquaredCoordinates);

struct distanceCalcsSharedMemAllocations {
public:
  static constexpr size_t sharedArrayQueryPoints_size =
      WARP_PER_BLOCK * 8 * COMPUTE_DIM;
  static constexpr size_t sharedArrayTmp8x4_size = WARP_PER_BLOCK * 8 * 4;
  static constexpr size_t sharedArraySquaredQueries_size =
      WARP_PER_BLOCK * 8 * (COMPUTE_DIM / 4);
  static constexpr size_t sharedArraySquaredCandidates_size =
      WARP_PER_BLOCK * 8;
  static constexpr size_t sharedArrayResultTmp_size = WARP_PER_BLOCK * 8 * 8;
  static constexpr size_t sharedArrayResult_size = WARP_PER_BLOCK * 8 * 8;

  static constexpr __device__ __host__ size_t getTotalSize() {
    return 8 *
           (sharedArrayQueryPoints_size + sharedArrayTmp8x4_size +
            sharedArraySquaredQueries_size + sharedArraySquaredCandidates_size +
            sharedArrayResultTmp_size + sharedArrayResult_size);
  }
};
#endif
