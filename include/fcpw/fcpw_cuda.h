#pragma once

#include <fcpw/fcpw.h>
#include <fcpw/cuda/cuda_interop_structures.h>

namespace fcpw {

template<size_t DIM>
class CUDAScene {
public:
    // constructor
    CUDAScene(bool printLogs_=false);

    // destructor
    ~CUDAScene();

    /////////////////////////////////////////////////////////////////////////////////////////////
    // API to transfer scene to the GPU, and to refit it if needed

    // transfers a binary (non-vectorized) BVH aggregate, constructed on the CPU using
    // the 'build' function in the Scene class, to the GPU. NOTE: Currently only supports
    // scenes with a single object, i.e., no CSG trees, instanced or transformed aggregates,
    // or nested hierarchies of aggregates. When using 'build', set 'vectorize' to false.
    void transferToGPU(Scene<DIM>& scene);

    // refits the BVH on the GPU after updating the geometry, either via calls to
    // updateObjectVertex in the Scene class, or directly in CUDA code in user kernels
    // (set updateGeometry to false if the geometry is updated directly on the GPU).
    // NOTE: Before calling this function, the BVH must already have been transferred to the GPU.
    void refit(Scene<DIM>& scene, bool updateGeometry=true);

    /////////////////////////////////////////////////////////////////////////////////////////////
    // API for GPU queries; NOTE: GPU queries are not thread-safe!

    // intersects the scene with the given rays, returning the closest interaction if it exists.
    void intersect(const Eigen::MatrixXf& rayOrigins,
                   const Eigen::MatrixXf& rayDirections,
                   const Eigen::VectorXf& rayDistanceBounds,
                   std::vector<CUDAInteraction>& interactions,
                   bool checkForOcclusion=false);
    void intersect(const std::vector<CUDARay>& rays,
                   std::vector<CUDAInteraction>& interactions,
                   bool checkForOcclusion=false);

    // intersects the scene with the given spheres, randomly selecting one geometric primitive
    // contained inside each sphere and sampling a random point on that primitive
    void intersect(const Eigen::MatrixXf& sphereCenters,
                   const Eigen::VectorXf& sphereSquaredRadii,
                   const Eigen::MatrixXf& randNums,
                   std::vector<CUDAInteraction>& interactions);
    void intersect(const std::vector<CUDABoundingSphere>& boundingSpheres,
                   const std::vector<CUDAFloat3>& randNums,
                   std::vector<CUDAInteraction>& interactions);

    // finds the closest points in the scene to the given query points.
    void findClosestPoints(const Eigen::MatrixXf& queryPoints,
                           const Eigen::VectorXf& squaredMaxRadii,
                           std::vector<CUDAInteraction>& interactions,
                           bool recordNormals=false);
    void findClosestPoints(const std::vector<CUDABoundingSphere>& boundingSpheres,
                           std::vector<CUDAInteraction>& interactions,
                           bool recordNormals=false);

    // finds the closest points on the visibility silhouette in the scene to the given query points.
    void findClosestSilhouettePoints(const Eigen::MatrixXf& queryPoints,
                                     const Eigen::VectorXf& squaredMaxRadii,
                                     const Eigen::VectorXi& flipNormalOrientation,
                                     std::vector<CUDAInteraction>& interactions,
                                     float squaredMinRadius=0.0f, float precision=1e-3f);
    void findClosestSilhouettePoints(const std::vector<CUDABoundingSphere>& boundingSpheres,
                                     const std::vector<uint32_t>& flipNormalOrientation,
                                     std::vector<CUDAInteraction>& interactions,
                                     float squaredMinRadius=0.0f, float precision=1e-3f);

private:
    // members
    CUDABvhBuffers bvhBuffers;
    void* stream;  // cudaStream_t stored as void* to avoid cuda_runtime.h in header
    uint32_t nThreadsPerGroup;
    bool printLogs;

    // helper to free device memory
    void freeDeviceBuffers();

    // allocate BVH data on GPU
    template<typename NodeType, typename PrimitiveType, typename SilhouetteType,
             typename CUDANodeType, typename CUDAPrimitiveType, typename CUDASilhouetteType>
    void allocateBuffers(const SceneData<DIM> *cpuSceneData,
                         bool allocatePrimitiveData,
                         bool allocateNodeData, bool allocateRefitData);
};

} // namespace fcpw

#include "fcpw_cuda.inl"
