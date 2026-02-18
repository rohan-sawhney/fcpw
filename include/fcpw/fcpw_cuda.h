#pragma once

#include <fcpw/cuda/bvh_interop_structures.h>
#include <fcpw/cuda/fcpw_cuda_kernels.h>
#include <fcpw/fcpw.h>

namespace fcpw {

// CUDA backend API that mirrors the public shape of GPUScene.
// This is an independent code path from the Slang backend.

template<size_t DIM>
class CUDAScene {
public:
    // fcpwDirectoryPath is unused in CUDA backend (kept for API parity).
    CUDAScene(const std::string& fcpwDirectoryPath_=std::string(), bool printLogs_=false);
    ~CUDAScene();

    // Upload CPU BVH data into CUDA buffers.
    void transferToGPU(Scene<DIM>& scene);
    // Refit the uploaded BVH after geometry updates.
    void refit(Scene<DIM>& scene, bool updateGeometry=true);

    // Ray intersection queries.
    void intersect(const Eigen::MatrixXf& rayOrigins,
                   const Eigen::MatrixXf& rayDirections,
                   const Eigen::VectorXf& rayDistanceBounds,
                   std::vector<CUDAInteraction>& interactions,
                   bool checkForOcclusion=false);
    void intersect(const std::vector<CUDARay>& rays,
                   std::vector<CUDAInteraction>& interactions,
                   bool checkForOcclusion=false);

    // Sphere intersection / random primitive sampling queries.
    void intersect(const Eigen::MatrixXf& sphereCenters,
                   const Eigen::VectorXf& sphereSquaredRadii,
                   const Eigen::MatrixXf& randNums,
                   std::vector<CUDAInteraction>& interactions);
    void intersect(const std::vector<CUDABoundingSphere>& boundingSpheres,
                   const std::vector<float3>& randNums,
                   std::vector<CUDAInteraction>& interactions);

    // Closest-point queries.
    void findClosestPoints(const Eigen::MatrixXf& queryPoints,
                           const Eigen::VectorXf& squaredMaxRadii,
                           std::vector<CUDAInteraction>& interactions,
                           bool recordNormals=false);
    void findClosestPoints(const std::vector<CUDABoundingSphere>& boundingSpheres,
                           std::vector<CUDAInteraction>& interactions,
                           bool recordNormals=false);

    // Closest silhouette-point queries.
    void findClosestSilhouettePoints(const Eigen::MatrixXf& queryPoints,
                                     const Eigen::VectorXf& squaredMaxRadii,
                                     const Eigen::VectorXi& flipNormalOrientation,
                                     std::vector<CUDAInteraction>& interactions,
                                     float squaredMinRadius=0.0f,
                                     float precision=1e-3f);
    void findClosestSilhouettePoints(const std::vector<CUDABoundingSphere>& boundingSpheres,
                                     const std::vector<uint32_t>& flipNormalOrientation,
                                     std::vector<CUDAInteraction>& interactions,
                                     float squaredMinRadius=0.0f,
                                     float precision=1e-3f);

private:
    // Persistent CUDA BVH storage + execution stream.
    CUDABvhBuffers bvhBuffers;
    cudaStream_t stream;
    uint32_t nThreadsPerBlock;
    bool printLogs;
};

} // namespace fcpw

#include "fcpw_cuda.inl"
