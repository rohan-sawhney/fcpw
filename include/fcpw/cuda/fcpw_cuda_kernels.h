#pragma once

#include <cuda_runtime.h>

#include <fcpw/cuda/bvh_interop_structures.h>

namespace fcpw {

// Launcher API used by CUDAScene. Each function dispatches one kernel family
// specialized by BVH payload type (line/triangle, with/without silhouette data).
// Inputs are type-erased buffers produced by CUDABvhBuffers.

void launchCudaRayIntersection(CUDABvhType type,
                               const void *nodes,
                               const void *primitives,
                               const void *silhouettes,
                               const CUDARay *rays,
                               CUDAInteraction *interactions,
                               uint32_t nQueries,
                               uint32_t checkForOcclusion,
                               uint32_t nThreadsPerBlock,
                               cudaStream_t stream);

void launchCudaSphereIntersection(CUDABvhType type,
                                  const void *nodes,
                                  const void *primitives,
                                  const void *silhouettes,
                                  const CUDABoundingSphere *spheres,
                                  const float3 *randNums,
                                  CUDAInteraction *interactions,
                                  uint32_t nQueries,
                                  uint32_t nThreadsPerBlock,
                                  cudaStream_t stream);

void launchCudaClosestPoint(CUDABvhType type,
                            const void *nodes,
                            const void *primitives,
                            const void *silhouettes,
                            const CUDABoundingSphere *spheres,
                            CUDAInteraction *interactions,
                            uint32_t nQueries,
                            uint32_t recordNormals,
                            uint32_t nThreadsPerBlock,
                            cudaStream_t stream);

void launchCudaClosestSilhouettePoint(CUDABvhType type,
                                      const void *nodes,
                                      const void *primitives,
                                      const void *silhouettes,
                                      const CUDABoundingSphere *spheres,
                                      const uint32_t *flipNormalOrientation,
                                      CUDAInteraction *interactions,
                                      uint32_t nQueries,
                                      float squaredMinRadius,
                                      float precision,
                                      uint32_t nThreadsPerBlock,
                                      cudaStream_t stream);

void launchCudaRefit(CUDABvhType type,
                     void *nodes,
                     const void *primitives,
                     const void *silhouettes,
                     const uint32_t *nodeIndices,
                     uint32_t firstNodeOffset,
                     uint32_t nodeCount,
                     uint32_t nThreadsPerBlock,
                     cudaStream_t stream);

} // namespace fcpw
