#pragma once

#include <fcpw/fcpw.h>
#include <fcpw/gpu/bvh_commands.h>

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
    void transferToGPU(Scene<DIM>& scene);

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
                   GPUInteractions& interactions,
                   bool checkForOcclusion=false);
    void intersect(GPURays& rays,
                   GPUInteractions& interactions,
                   bool checkForOcclusion=false);

    // intersects the scene with the given spheres, randomly selecting one geometric primitive
    // contained inside each sphere and sampling a random point on that primitive (written to 
    // GPUInteraction.p) using the random numbers randNums[3] (float3.z is ignored for DIM = 2);
    // the selection pdf value is written to GPUInteraction.d along with the primitive index
    void intersect(const Eigen::MatrixXf& sphereCenters,
                   const Eigen::VectorXf& sphereSquaredRadii,
                   const Eigen::MatrixXf& randNums,
                   GPUInteractions& interactions);
    void intersect(GPUBoundingSphere& boundingSpheres,
                   GPUFloat3List& randNums,
                   GPUInteractions& interactions);

    // finds the closest points in the scene to the given query points. The max radius specifies
    // a conservative radius guess around the query point inside which the search is performed.
    void findClosestPoints(const Eigen::MatrixXf& queryPoints,
                           const Eigen::VectorXf& squaredMaxRadii,
                           GPUInteractions& interactions,
                           bool recordNormals=false);
    void findClosestPoints(GPUBoundingSphere& boundingSpheres,
                           GPUInteractions& interactions,
                           bool recordNormals=false);

    // finds the closest points on the visibility silhouette in the scene to the given query points.
    // The max radius specifies a conservative radius guess around the query point inside which the
    // search is performed. Optionally specify a minimum radius to stop the closest silhouette
    // search, as well as a precision parameter to help classify silhouettes.
    void findClosestSilhouettePoints(const Eigen::MatrixXf& queryPoints,
                                     const Eigen::VectorXf& squaredMaxRadii,
                                     const Eigen::VectorXi& flipNormalOrientation,
                                     GPUInteractions& interactions,
                                     float squaredMinRadius=0.0f, float precision=1e-3f);
    void findClosestSilhouettePoints(GPUBoundingSphere& boundingSpheres,
                                     GPUUintList& flipNormalOrientation,
                                     GPUInteractions& interactions,
                                     float squaredMinRadius=0.0f, float precision=1e-3f);

private:
    // members
    GPUContext gpuContext;
    GPUBvhBuffers gpuBvhBuffers;
    std::string fcpwModule;
    std::string refitShaderModule;
    std::string traversalShaderModule;
    Shader refitShader;
    Shader rayIntersectionShader;
    Shader sphereIntersectionShader;
    Shader closestPointShader;
    Shader closestSilhouettePointShader;
    uint32_t nThreadsPerGroup;
    bool printLogs;
};

} // namespace fcpw

#include "fcpw_gpu.inl"