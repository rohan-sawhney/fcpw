#include <fcpw/gpu/cuda/cuda_bvh_kernels.cuh>
#include <fcpw/gpu/cuda/cuda_utils.h>

namespace fcpw {
namespace cuda {

/////////////////////////////////////////////////////////////////////////////////////////////
// Explicit template instantiations for all 4 BVH types

// Type 1: Line Segment BVH (no silhouettes)
template __global__ void rayIntersectionKernel<GPUBvhNode, GPULineSegment>(
    const GPUBvhNode*, const GPULineSegment*, const GPURay*,
    GPUInteraction*, uint32_t, uint32_t);

template __global__ void sphereIntersectionKernel<GPUBvhNode, GPULineSegment>(
    const GPUBvhNode*, const GPULineSegment*, const GPUBoundingSphere*,
    const float3*, GPUInteraction*, uint32_t);

template __global__ void closestPointKernel<GPUBvhNode, GPULineSegment>(
    const GPUBvhNode*, const GPULineSegment*, const GPUBoundingSphere*,
    GPUInteraction*, uint32_t, uint32_t);

template __global__ void refitKernel<GPUBvhNode, GPULineSegment, GPUNoSilhouette>(
    GPUBvhNode*, const GPULineSegment*, const GPUNoSilhouette*,
    const uint32_t*, uint32_t, uint32_t);

// Type 2: Triangle BVH (no silhouettes)
template __global__ void rayIntersectionKernel<GPUBvhNode, GPUTriangle>(
    const GPUBvhNode*, const GPUTriangle*, const GPURay*,
    GPUInteraction*, uint32_t, uint32_t);

template __global__ void sphereIntersectionKernel<GPUBvhNode, GPUTriangle>(
    const GPUBvhNode*, const GPUTriangle*, const GPUBoundingSphere*,
    const float3*, GPUInteraction*, uint32_t);

template __global__ void closestPointKernel<GPUBvhNode, GPUTriangle>(
    const GPUBvhNode*, const GPUTriangle*, const GPUBoundingSphere*,
    GPUInteraction*, uint32_t, uint32_t);

template __global__ void refitKernel<GPUBvhNode, GPUTriangle, GPUNoSilhouette>(
    GPUBvhNode*, const GPUTriangle*, const GPUNoSilhouette*,
    const uint32_t*, uint32_t, uint32_t);

// Type 3: Line Segment SNCH (with silhouettes - vertices)
template __global__ void rayIntersectionKernel<GPUSnchNode, GPULineSegment>(
    const GPUSnchNode*, const GPULineSegment*, const GPURay*,
    GPUInteraction*, uint32_t, uint32_t);

template __global__ void sphereIntersectionKernel<GPUSnchNode, GPULineSegment>(
    const GPUSnchNode*, const GPULineSegment*, const GPUBoundingSphere*,
    const float3*, GPUInteraction*, uint32_t);

template __global__ void closestPointKernel<GPUSnchNode, GPULineSegment>(
    const GPUSnchNode*, const GPULineSegment*, const GPUBoundingSphere*,
    GPUInteraction*, uint32_t, uint32_t);

template __global__ void closestSilhouettePointKernel<GPUSnchNode, GPULineSegment, GPUVertex>(
    const GPUSnchNode*, const GPULineSegment*, const GPUVertex*,
    const GPUBoundingSphere*, const uint32_t*, GPUInteraction*,
    float, float, uint32_t);

template __global__ void refitKernel<GPUSnchNode, GPULineSegment, GPUVertex>(
    GPUSnchNode*, const GPULineSegment*, const GPUVertex*,
    const uint32_t*, uint32_t, uint32_t);

// Type 4: Triangle SNCH (with silhouettes - edges)
template __global__ void rayIntersectionKernel<GPUSnchNode, GPUTriangle>(
    const GPUSnchNode*, const GPUTriangle*, const GPURay*,
    GPUInteraction*, uint32_t, uint32_t);

template __global__ void sphereIntersectionKernel<GPUSnchNode, GPUTriangle>(
    const GPUSnchNode*, const GPUTriangle*, const GPUBoundingSphere*,
    const float3*, GPUInteraction*, uint32_t);

template __global__ void closestPointKernel<GPUSnchNode, GPUTriangle>(
    const GPUSnchNode*, const GPUTriangle*, const GPUBoundingSphere*,
    GPUInteraction*, uint32_t, uint32_t);

template __global__ void closestSilhouettePointKernel<GPUSnchNode, GPUTriangle, GPUEdge>(
    const GPUSnchNode*, const GPUTriangle*, const GPUEdge*,
    const GPUBoundingSphere*, const uint32_t*, GPUInteraction*,
    float, float, uint32_t);

template __global__ void refitKernel<GPUSnchNode, GPUTriangle, GPUEdge>(
    GPUSnchNode*, const GPUTriangle*, const GPUEdge*,
    const uint32_t*, uint32_t, uint32_t);

/////////////////////////////////////////////////////////////////////////////////////////////
// Kernel launcher functions for runtime dispatch

// Helper macro to define launcher functions
#define DEFINE_KERNEL_LAUNCHERS(NodeType, PrimitiveType, SilhouetteType, suffix) \
    void launchRayIntersection##suffix( \
        const NodeType* nodes, const PrimitiveType* primitives, \
        const GPURay* rays, GPUInteraction* interactions, \
        uint32_t checkForOcclusion, uint32_t nQueries, \
        const KernelLaunchConfig& config) \
    { \
        rayIntersectionKernel<NodeType, PrimitiveType><<<config.gridDim, config.blockDim>>>( \
            nodes, primitives, rays, interactions, checkForOcclusion, nQueries); \
        CUDA_CHECK(cudaGetLastError()); \
    } \
    \
    void launchSphereIntersection##suffix( \
        const NodeType* nodes, const PrimitiveType* primitives, \
        const GPUBoundingSphere* spheres, const float3* randNums, \
        GPUInteraction* interactions, uint32_t nQueries, \
        const KernelLaunchConfig& config) \
    { \
        sphereIntersectionKernel<NodeType, PrimitiveType><<<config.gridDim, config.blockDim>>>( \
            nodes, primitives, spheres, randNums, interactions, nQueries); \
        CUDA_CHECK(cudaGetLastError()); \
    } \
    \
    void launchClosestPoint##suffix( \
        const NodeType* nodes, const PrimitiveType* primitives, \
        const GPUBoundingSphere* spheres, GPUInteraction* interactions, \
        uint32_t recordNormals, uint32_t nQueries, \
        const KernelLaunchConfig& config) \
    { \
        closestPointKernel<NodeType, PrimitiveType><<<config.gridDim, config.blockDim>>>( \
            nodes, primitives, spheres, interactions, recordNormals, nQueries); \
        CUDA_CHECK(cudaGetLastError()); \
    } \
    \
    void launchRefit##suffix( \
        NodeType* nodes, const PrimitiveType* primitives, const SilhouetteType* silhouettes, \
        const uint32_t* nodeIndices, uint32_t firstNodeOffset, uint32_t nodeCount, \
        const KernelLaunchConfig& config) \
    { \
        refitKernel<NodeType, PrimitiveType, SilhouetteType><<<config.gridDim, config.blockDim>>>( \
            nodes, primitives, silhouettes, nodeIndices, firstNodeOffset, nodeCount); \
        CUDA_CHECK(cudaGetLastError()); \
    }

// Type 1: Line Segment BVH
DEFINE_KERNEL_LAUNCHERS(GPUBvhNode, GPULineSegment, GPUNoSilhouette, LineSegmentBvh)

// Type 2: Triangle BVH
DEFINE_KERNEL_LAUNCHERS(GPUBvhNode, GPUTriangle, GPUNoSilhouette, TriangleBvh)

// Type 3: Line Segment SNCH
DEFINE_KERNEL_LAUNCHERS(GPUSnchNode, GPULineSegment, GPUVertex, LineSegmentSnch)

void launchClosestSilhouettePointLineSegmentSnch(
    const GPUSnchNode* nodes, const GPULineSegment* primitives, const GPUVertex* silhouettes,
    const GPUBoundingSphere* spheres, const uint32_t* flipNormalOrientation,
    GPUInteraction* interactions, float squaredMinRadius, float precision,
    uint32_t nQueries, const KernelLaunchConfig& config)
{
    closestSilhouettePointKernel<GPUSnchNode, GPULineSegment, GPUVertex>
        <<<config.gridDim, config.blockDim>>>(
            nodes, primitives, silhouettes, spheres, flipNormalOrientation,
            interactions, squaredMinRadius, precision, nQueries);
    CUDA_CHECK(cudaGetLastError());
}

// Type 4: Triangle SNCH
DEFINE_KERNEL_LAUNCHERS(GPUSnchNode, GPUTriangle, GPUEdge, TriangleSnch)

void launchClosestSilhouettePointTriangleSnch(
    const GPUSnchNode* nodes, const GPUTriangle* primitives, const GPUEdge* silhouettes,
    const GPUBoundingSphere* spheres, const uint32_t* flipNormalOrientation,
    GPUInteraction* interactions, float squaredMinRadius, float precision,
    uint32_t nQueries, const KernelLaunchConfig& config)
{
    closestSilhouettePointKernel<GPUSnchNode, GPUTriangle, GPUEdge>
        <<<config.gridDim, config.blockDim>>>(
            nodes, primitives, silhouettes, spheres, flipNormalOrientation,
            interactions, squaredMinRadius, precision, nQueries);
    CUDA_CHECK(cudaGetLastError());
}

#undef DEFINE_KERNEL_LAUNCHERS

} // namespace cuda
} // namespace fcpw
