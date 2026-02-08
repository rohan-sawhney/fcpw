#pragma once

#include <fcpw/gpu/cuda/cuda_bvh_traversal.cuh>

namespace fcpw {
namespace cuda {

/////////////////////////////////////////////////////////////////////////////////////////////
// Kernel 1: Ray intersection

template<typename NodeType, typename PrimitiveType>
__global__ void rayIntersectionKernel(
    const NodeType* nodes,
    const PrimitiveType* primitives,
    const GPURay* rays,
    GPUInteraction* interactions,
    uint32_t checkForOcclusion,
    uint32_t nQueries)
{
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nQueries) {
        return;
    }

    GPURay ray = rays[index];
    GPUInteraction interaction;
    interaction.index = FCPW_GPU_UINT_MAX;

    bool hit = traverseBvhRayIntersection<NodeType, PrimitiveType>(
        nodes, primitives, ray, checkForOcclusion != 0, interaction);

    if (hit) {
        interactions[index] = interaction;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////
// Kernel 2: Sphere intersection (probabilistic sampling)

template<typename NodeType, typename PrimitiveType>
__global__ void sphereIntersectionKernel(
    const NodeType* nodes,
    const PrimitiveType* primitives,
    const GPUBoundingSphere* boundingSpheres,
    const float3* randNums,
    GPUInteraction* interactions,
    uint32_t nQueries)
{
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nQueries) {
        return;
    }

    GPUBoundingSphere sphere = boundingSpheres[index];
    float3 randNum = randNums[index];
    GPUInteraction interaction;
    interaction.index = FCPW_GPU_UINT_MAX;

    bool hit = traverseBvhSphereIntersection<NodeType, PrimitiveType>(
        nodes, primitives, sphere, randNum, interaction);

    if (hit) {
        interactions[index] = interaction;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////
// Kernel 3: Closest point

template<typename NodeType, typename PrimitiveType>
__global__ void closestPointKernel(
    const NodeType* nodes,
    const PrimitiveType* primitives,
    const GPUBoundingSphere* boundingSpheres,
    GPUInteraction* interactions,
    uint32_t recordNormals,
    uint32_t nQueries)
{
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nQueries) {
        return;
    }

    GPUBoundingSphere sphere = boundingSpheres[index];
    GPUInteraction interaction;
    interaction.index = FCPW_GPU_UINT_MAX;

    bool found = traverseBvhClosestPoint<NodeType, PrimitiveType>(
        nodes, primitives, sphere, recordNormals != 0, interaction);

    if (found) {
        interactions[index] = interaction;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////
// Kernel 4: Closest silhouette point

template<typename NodeType, typename PrimitiveType, typename SilhouetteType>
__global__ void closestSilhouettePointKernel(
    const NodeType* nodes,
    const PrimitiveType* primitives,
    const SilhouetteType* silhouettes,
    const GPUBoundingSphere* boundingSpheres,
    const uint32_t* flipNormalOrientation,
    GPUInteraction* interactions,
    float squaredMinRadius,
    float precision,
    uint32_t nQueries)
{
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nQueries) {
        return;
    }

    GPUBoundingSphere sphere = boundingSpheres[index];
    bool flipNormal = flipNormalOrientation[index] != 0;
    GPUInteraction interaction;
    interaction.index = FCPW_GPU_UINT_MAX;

    bool found = traverseBvhClosestSilhouettePoint<NodeType, SilhouetteType>(
        nodes, silhouettes, sphere, flipNormal,
        squaredMinRadius, precision, interaction);

    if (found) {
        interactions[index] = interaction;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////
// Kernel 5: BVH refit

template<typename NodeType, typename PrimitiveType, typename SilhouetteType>
__global__ void refitKernel(
    NodeType* nodes,
    const PrimitiveType* primitives,
    const SilhouetteType* silhouettes,
    const uint32_t* nodeIndices,
    uint32_t firstNodeOffset,
    uint32_t nodeCount)
{
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nodeCount) {
        return;
    }

    uint32_t nodeIndex = nodeIndices[firstNodeOffset + index];

    // Dispatch based on whether we have silhouettes
    if constexpr (sizeof(SilhouetteType) == sizeof(GPUNoSilhouette)) {
        // BVH without silhouettes
        refitBvhNode<NodeType, PrimitiveType>(nodes, primitives, nodeIndex);
    } else {
        // SNCH with silhouettes
        refitSnchNode<NodeType, PrimitiveType, SilhouetteType>(
            nodes, primitives, silhouettes, nodeIndex);
    }
}

} // namespace cuda
} // namespace fcpw
