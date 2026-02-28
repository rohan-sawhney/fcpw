#pragma once

#include <fcpw/fcpw.h>
#include <fcpw/gpu/bvh_interop_structures.h>

namespace fcpw {

template<size_t DIM>
class GPUScene {
public:
    // constructor
    GPUScene(const std::string& fcpwDirectoryPath_, bool printLogs_=false);

    /////////////////////////////////////////////////////////////////////////////////////////////
    // API to transfer scene to the GPU, and to refit it if needed

    // transfers a binary (non-vectorized) BVH aggregate, constructed on the CPU using 
    // the 'build' function in the Scene class, to the GPU. NOTE: Currently only supports
    // scenes with a single object, i.e., no CSG trees, instanced or transformed aggregates,
    // or nested hierarchies of aggregates. When using 'build', set 'vectorize' to false.
    void transferToGPU(Scene<DIM>& scene, const std::string& deviceBackend="default");

    // refits the BVH on the GPU after updating the geometry, either via calls to
    // updateObjectVertex in the Scene class, or directly in GPU code in the user's slang
    // shaders (set updateGeometry to false if the geometry is updated directly on the GPU).
    // NOTE: Before calling this function, the BVH must already have been transferred to the GPU.
    void refit(Scene<DIM>& scene, bool updateGeometry=true);

    /////////////////////////////////////////////////////////////////////////////////////////////
    // API for GPU queries; NOTE: GPU queries are not thread-safe!

    // intersects the scene with the given rays, returning the closest interaction if it exists.
    void intersect(const Eigen::MatrixXf& rayOrigins,
                   const Eigen::MatrixXf& rayDirections,
                   const Eigen::VectorXf& rayDistanceBounds,
                   std::vector<GPUInteraction>& interactions,
                   bool checkForOcclusion=false);
    void intersect(const std::vector<GPURay>& rays,
                   std::vector<GPUInteraction>& interactions,
                   bool checkForOcclusion=false);

    // intersects the scene with the given spheres, randomly selecting one geometric primitive
    // contained inside each sphere and sampling a random point on that primitive (written to 
    // GPUInteraction.p) using the random numbers randNums[3] (float3.z is ignored for DIM = 2);
    // the selection pdf value is written to GPUInteraction.d along with the primitive index
    void intersect(const Eigen::MatrixXf& sphereCenters,
                   const Eigen::VectorXf& sphereSquaredRadii,
                   const Eigen::MatrixXf& randNums,
                   std::vector<GPUInteraction>& interactions);
    void intersect(const std::vector<GPUBoundingSphere>& boundingSpheres,
                   const std::vector<float3>& randNums,
                   std::vector<GPUInteraction>& interactions);

    // finds the closest points in the scene to the given query points. The max radius specifies
    // a conservative radius guess around the query point inside which the search is performed.
    void findClosestPoints(const Eigen::MatrixXf& queryPoints,
                           const Eigen::VectorXf& squaredMaxRadii,
                           std::vector<GPUInteraction>& interactions,
                           bool recordNormals=false);
    void findClosestPoints(const std::vector<GPUBoundingSphere>& boundingSpheres,
                           std::vector<GPUInteraction>& interactions,
                           bool recordNormals=false);

    // finds the closest points on the visibility silhouette in the scene to the given query points.
    // The max radius specifies a conservative radius guess around the query point inside which the
    // search is performed. Optionally specify a minimum radius to stop the closest silhouette
    // search, as well as a precision parameter to help classify silhouettes.
    void findClosestSilhouettePoints(const Eigen::MatrixXf& queryPoints,
                                     const Eigen::VectorXf& squaredMaxRadii,
                                     const Eigen::VectorXi& flipNormalOrientation,
                                     std::vector<GPUInteraction>& interactions,
                                     float squaredMinRadius=0.0f, float precision=1e-3f);
    void findClosestSilhouettePoints(const std::vector<GPUBoundingSphere>& boundingSpheres,
                                     const std::vector<uint32_t>& flipNormalOrientation,
                                     std::vector<GPUInteraction>& interactions,
                                     float squaredMinRadius=0.0f, float precision=1e-3f);

private:
    // members
    std::string fcpwGpuDirectoryPath;
    GPUContext context;
    GPUBvhBuffers bvhBuffers;
    ComputeShader refitShader;
    ComputeShader rayIntersectionShader;
    ComputeShader sphereIntersectionShader;
    ComputeShader closestPointShader;
    ComputeShader closestSilhouettePointShader;
    std::function<void(const ComputeShader&, const ShaderCursor&)> bindBvhResources;
    uint32_t nThreadsPerGroup;
    bool printLogs;
};

} // namespace fcpw

#include "fcpw_gpu.inl"
