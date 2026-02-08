#pragma once

#include <fcpw/gpu/cuda/cuda_geometry.cuh>
#include <fcpw/gpu/cuda/cuda_bounding_volumes.cuh>

namespace fcpw {
namespace cuda {

// Stack size for BVH traversal (must match bvh.h)
#ifndef FCPW_BVH_MAX_DEPTH
#define FCPW_BVH_MAX_DEPTH 64
#endif

/////////////////////////////////////////////////////////////////////////////////////////////
// Traversal stack structure (matches Slang bvh.slang lines 7-18)

struct TraversalStack
{
    uint32_t node;     // node index
    float distance;    // minimum distance (parametric, squared, ...) to this node
};

/////////////////////////////////////////////////////////////////////////////////////////////
// Dispatch helpers for different primitive types

__device__ __forceinline__ bool primitiveIntersectsRay(const GPULineSegment& seg, GPURay& ray, GPUInteraction& interaction)
{
    float3 p, n;
    float2 uv;
    float d;
    bool hit = lineSegmentIntersectsRay(seg.pa, seg.pb, ray.o, ray.d, ray.tMax, false, p, n, uv, d);
    if (hit) {
        interaction.p = p;
        interaction.n = n;
        interaction.uv = uv;
        interaction.d = d;
        interaction.index = seg.index;
        ray.tMax = d;
    }
    return hit;
}

__device__ __forceinline__ bool primitiveIntersectsRay(const GPUTriangle& tri, GPURay& ray, GPUInteraction& interaction)
{
    float3 p, n;
    float2 uv;
    float d;
    bool hit = triangleIntersectsRay(tri.pa, tri.pb, tri.pc, ray, false, p, n, uv, d);
    if (hit) {
        interaction.p = p;
        interaction.n = n;
        interaction.uv = uv;
        interaction.d = d;
        interaction.index = tri.index;
        ray.tMax = d;
    }
    return hit;
}

__device__ __forceinline__ bool primitiveIntersectsSphere(const GPULineSegment& seg, const GPUBoundingSphere& sphere, GPUInteraction& interaction)
{
    float t;
    float d = lineSegmentFindClosestPoint(seg.pa, seg.pb, sphere.c, interaction.p, t);
    if (d * d <= sphere.r2) {
        interaction.d = lineSegmentSurfaceArea(seg);
        interaction.index = seg.index;
        return true;
    }
    return false;
}

__device__ __forceinline__ bool primitiveIntersectsSphere(const GPUTriangle& tri, const GPUBoundingSphere& sphere, GPUInteraction& interaction)
{
    float2 t;
    float d = triangleFindClosestPoint(tri.pa, tri.pb, tri.pc, sphere.c, interaction.p, t);
    if (d * d <= sphere.r2) {
        interaction.d = triangleSurfaceArea(tri);
        interaction.index = tri.index;
        return true;
    }
    return false;
}

__device__ __forceinline__ float primitiveFindClosestPoint(const GPULineSegment& seg, const GPUBoundingSphere& sphere, GPUInteraction& interaction)
{
    float t;
    float d = lineSegmentFindClosestPoint(seg.pa, seg.pb, sphere.c, interaction.p, t);
    if (d * d <= sphere.r2) {
        interaction.uv = make_float2(t, 0.0f);
        interaction.d = d;
        interaction.index = seg.index;
        return d * d;
    }
    return FLT_MAX;
}

__device__ __forceinline__ float primitiveFindClosestPoint(const GPUTriangle& tri, const GPUBoundingSphere& sphere, GPUInteraction& interaction)
{
    float2 t;
    float d = triangleFindClosestPoint(tri.pa, tri.pb, tri.pc, sphere.c, interaction.p, t);
    if (d * d <= sphere.r2) {
        interaction.uv = t;
        interaction.d = d;
        interaction.index = tri.index;
        // Compute normal
        float3 v1 = tri.pb - tri.pa;
        float3 v2 = tri.pc - tri.pa;
        interaction.n = normalize(cross(v1, v2));
        return d * d;
    }
    return FLT_MAX;
}

__device__ __forceinline__ float3 primitiveSample(const GPULineSegment& seg, const float3& randNum)
{
    return lineSegmentSample(seg, randNum);
}

__device__ __forceinline__ float3 primitiveSample(const GPUTriangle& tri, const float3& randNum)
{
    return triangleSample(tri, randNum);
}

__device__ __forceinline__ float primitiveSurfaceArea(const GPULineSegment& seg)
{
    return lineSegmentSurfaceArea(seg);
}

__device__ __forceinline__ float primitiveSurfaceArea(const GPUTriangle& tri)
{
    return triangleSurfaceArea(tri);
}

__device__ __forceinline__ float primitiveSamplingPdf(const GPULineSegment& seg, const float3& randNum)
{
    // Source: geometry.slang line 192
    float area = lineSegmentSurfaceArea(seg);
    return 1.0f / area;
}

__device__ __forceinline__ float primitiveSamplingPdf(const GPUTriangle& tri, const float3& randNum)
{
    // Source: geometry.slang line 423
    float3 n = cross(tri.pb - tri.pa, tri.pc - tri.pa);
    float area = length(n);
    return 2.0f / area;
}

/////////////////////////////////////////////////////////////////////////////////////////////
// BVH traversal for ray intersection
// Source: bvh.slang lines 143-254

template<typename NodeType, typename PrimitiveType>
__device__ bool traverseBvhRayIntersection(
    const NodeType* nodes,
    const PrimitiveType* primitives,
    GPURay& ray,
    bool checkForOcclusion,
    GPUInteraction& interaction)
{
    TraversalStack traversalStack[FCPW_BVH_MAX_DEPTH];
    bool didIntersect = false;

    interaction.index = FCPW_GPU_UINT_MAX;

    // Check if ray intersects root bounding box
    float tMin, tMax;
    if (!boxIntersectsRay(nodes[0].box, ray, tMin, tMax)) {
        return false;
    }

    traversalStack[0].node = 0;
    traversalStack[0].distance = tMin;
    int stackPtr = 0;

    while (stackPtr >= 0) {
        // Pop off the next node to work on
        uint32_t currentNodeIndex = traversalStack[stackPtr].node;
        float currentDist = traversalStack[stackPtr].distance;
        stackPtr--;

        // If this node is further than the closest found intersection, continue
        if (currentDist > ray.tMax) {
            continue;
        }

        const NodeType& node = nodes[currentNodeIndex];
        if (node.nPrimitives > 0) {
            // Leaf node - intersect primitives
            for (uint32_t i = 0; i < node.nPrimitives; i++) {
                GPUInteraction c;
                uint32_t primIndex = node.offset + i;
                bool didIntersectPrimitive = primitiveIntersectsRay(primitives[primIndex], ray, c);

                if (didIntersectPrimitive) {
                    if (checkForOcclusion) {
                        interaction.index = c.index;
                        return true;
                    }

                    didIntersect = true;
                    ray.tMax = fminf(ray.tMax, c.d);
                    interaction = c;
                }
            }
        } else {
            // Internal node - intersect child nodes
            uint32_t leftNodeIndex = currentNodeIndex + 1;
            float tMinLeft, tMaxLeft;
            bool didIntersectLeft = boxIntersectsRay(nodes[leftNodeIndex].box, ray, tMinLeft, tMaxLeft);

            uint32_t rightNodeIndex = currentNodeIndex + node.offset;
            float tMinRight, tMaxRight;
            bool didIntersectRight = boxIntersectsRay(nodes[rightNodeIndex].box, ray, tMinRight, tMaxRight);

            // Which nodes did we intersect?
            if (didIntersectLeft && didIntersectRight) {
                // Assume that the left child is closer
                uint32_t closer = leftNodeIndex;
                uint32_t other = rightNodeIndex;
                float closerDist = tMinLeft;
                float otherDist = tMinRight;

                // ... if the right child was actually closer, swap the relevant values
                if (tMinRight < tMinLeft) {
                    closer = rightNodeIndex;
                    other = leftNodeIndex;
                    closerDist = tMinRight;
                    otherDist = tMinLeft;
                }

                // Push the further node first, then the closer node
                stackPtr++;
                traversalStack[stackPtr].node = other;
                traversalStack[stackPtr].distance = otherDist;

                stackPtr++;
                traversalStack[stackPtr].node = closer;
                traversalStack[stackPtr].distance = closerDist;
            } else if (didIntersectLeft) {
                stackPtr++;
                traversalStack[stackPtr].node = leftNodeIndex;
                traversalStack[stackPtr].distance = tMinLeft;
            } else if (didIntersectRight) {
                stackPtr++;
                traversalStack[stackPtr].node = rightNodeIndex;
                traversalStack[stackPtr].distance = tMinRight;
            }
        }
    }

    return didIntersect;
}

/////////////////////////////////////////////////////////////////////////////////////////////
// BVH traversal for sphere intersection (PROBABILISTIC - matches Slang)
// Source: bvh.slang lines 256-392

template<typename NodeType, typename PrimitiveType>
__device__ bool traverseBvhSphereIntersection(
    const NodeType* nodes,
    const PrimitiveType* primitives,
    const GPUBoundingSphere& sphere,
    const float3& randNums,
    GPUInteraction& interaction)
{
    uint32_t currentNodeIndex = 0;
    uint32_t selectedPrimitiveIndex = FCPW_GPU_UINT_MAX;
    bool didIntersect = false;

    interaction.index = FCPW_GPU_UINT_MAX;

    // Check if sphere overlaps root bounding box
    float d2Min, d2Max;
    if (!boxOverlapWithDistance(nodes[0].box, sphere, d2Min, d2Max)) {
        return false;
    }

    float maxDistToChildNode = d2Max; // Track maximum distance to current node
    float traversalPdf = 1.0f;
    float u = randNums.x; // Use first random component for probabilistic selection
    int stackPtr = 0;

    while (stackPtr >= 0) {
        // Pop node
        stackPtr--;

        const NodeType& node = nodes[currentNodeIndex];
        if (node.nPrimitives > 0) {
            // Leaf node - probabilistically select a primitive
            float totalPrimitiveWeight = 0.0f;
            uint32_t nPrimitives = node.nPrimitives;

            for (uint32_t p = 0; p < nPrimitives; p++) {
                GPUInteraction c;
                bool didIntersectPrimitive = false;
                uint32_t primitiveIndex = node.offset + p;
                const PrimitiveType& primitive = primitives[primitiveIndex];

                // Optimization: if sphere completely contains node's bounding box, all primitives are inside
                if (maxDistToChildNode <= sphere.r2) {
                    didIntersectPrimitive = true;
                    c.d = primitiveSurfaceArea(primitive);
                    c.index = primitive.index;
                } else {
                    didIntersectPrimitive = primitiveIntersectsSphere(primitive, sphere, c);
                }

                if (didIntersectPrimitive) {
                    didIntersect = true;
                    totalPrimitiveWeight += c.d;
                    float selectionProb = c.d / totalPrimitiveWeight;

                    // Reservoir sampling with rescaling
                    if (u < selectionProb) {
                        u = u / selectionProb; // rescale to [0,1)
                        interaction = c;
                        interaction.d *= traversalPdf;
                        selectedPrimitiveIndex = primitiveIndex;
                    } else {
                        u = (u - selectionProb) / (1.0f - selectionProb);
                    }
                }
            }

            if (totalPrimitiveWeight > 0.0f) {
                interaction.d /= totalPrimitiveWeight;
            }
        } else {
            // Internal node - probabilistically select one child to traverse
            uint32_t leftNodeIndex = currentNodeIndex + 1;
            float d2MinLeft, d2MaxLeft;
            bool overlapsLeft = boxOverlapWithDistance(nodes[leftNodeIndex].box, sphere, d2MinLeft, d2MaxLeft);
            float weightLeft = overlapsLeft ? 1.0f : 0.0f;

            // Note: branchTraversalWeight is always 1.0 in current implementation
            // (see test: branchTraversalWeight = [](float r2) -> float { return 1.0f; })
            // So we don't apply additional weighting based on distance

            uint32_t rightNodeIndex = currentNodeIndex + node.offset;
            float d2MinRight, d2MaxRight;
            bool overlapsRight = boxOverlapWithDistance(nodes[rightNodeIndex].box, sphere, d2MinRight, d2MaxRight);
            float weightRight = overlapsRight ? 1.0f : 0.0f;

            float totalTraversalWeight = weightLeft + weightRight;
            if (totalTraversalWeight > 0.0f) {
                stackPtr++;
                float traversalProbLeft = weightLeft / totalTraversalWeight;
                float traversalProbRight = 1.0f - traversalProbLeft;

                if (u < traversalProbLeft) {
                    u = u / traversalProbLeft; // rescale to [0,1)
                    currentNodeIndex = leftNodeIndex;
                    traversalPdf *= traversalProbLeft;
                    maxDistToChildNode = d2MaxLeft; // Update max distance tracking
                } else {
                    u = (u - traversalProbLeft) / traversalProbRight; // rescale to [0,1)
                    currentNodeIndex = rightNodeIndex;
                    traversalPdf *= traversalProbRight;
                    maxDistToChildNode = d2MaxRight; // Update max distance tracking
                }
            }
        }
    }

    if (didIntersect) {
        if (interaction.index == FCPW_GPU_UINT_MAX || selectedPrimitiveIndex == FCPW_GPU_UINT_MAX) {
            didIntersect = false;
        } else {
            // Sample a point on the selected geometric primitive
            // Use randNums.y and randNums.z for 2D sampling
            float samplingPdf = primitiveSamplingPdf(primitives[selectedPrimitiveIndex], randNums);
            float3 samplePoint = primitiveSample(primitives[selectedPrimitiveIndex], randNums);

            interaction.p = samplePoint;
            interaction.d *= samplingPdf;
            // Note: interaction.n is not set by sphere intersection (only by closest point)
        }
    }

    return didIntersect;
}

/////////////////////////////////////////////////////////////////////////////////////////////
// BVH traversal for closest point
// Source: bvh.slang lines 394-502

template<typename NodeType, typename PrimitiveType>
__device__ bool traverseBvhClosestPoint(
    const NodeType* nodes,
    const PrimitiveType* primitives,
    GPUBoundingSphere& sphere,
    bool recordNormal,
    GPUInteraction& interaction)
{
    uint32_t traversalStack[FCPW_BVH_MAX_DEPTH];
    uint32_t selectedPrimitiveIndex = FCPW_GPU_UINT_MAX;
    bool notFound = true;

    interaction.index = FCPW_GPU_UINT_MAX;

    // Check if sphere overlaps root bounding box
    float d2Min, d2Max;
    if (!boxOverlapWithDistance(nodes[0].box, sphere, d2Min, d2Max)) {
        return false;
    }

    sphere.r2 = fminf(sphere.r2, d2Max);
    traversalStack[0] = 0;
    int stackPtr = 0;

    while (stackPtr >= 0) {
        // Pop off the next node to work on
        uint32_t currentNodeIndex = traversalStack[stackPtr];
        const NodeType& node = nodes[currentNodeIndex];
        stackPtr--;

        // If this node is further than the closest found primitive, continue
        if (!boxOverlapWithDistance(node.box, sphere, d2Min, d2Max)) {
            continue;
        }

        if (node.nPrimitives > 0) {
            // Leaf node - compute distance to primitives
            for (uint32_t i = 0; i < node.nPrimitives; i++) {
                GPUInteraction c;
                uint32_t primIndex = node.offset + i;
                float distSq = primitiveFindClosestPoint(primitives[primIndex], sphere, c);

                // Keep the closest point only
                if (distSq < sphere.r2) {
                    notFound = false;
                    sphere.r2 = fminf(sphere.r2, c.d * c.d);
                    interaction = c;
                    selectedPrimitiveIndex = primIndex;
                }
            }
        } else {
            // Internal node - find distance to child nodes
            uint32_t leftNodeIndex = currentNodeIndex + 1;
            float d2MinLeft, d2MaxLeft;
            bool overlapsLeft = boxOverlapWithDistance(nodes[leftNodeIndex].box, sphere, d2MinLeft, d2MaxLeft);
            sphere.r2 = fminf(sphere.r2, d2MaxLeft);

            uint32_t rightNodeIndex = currentNodeIndex + node.offset;
            float d2MinRight, d2MaxRight;
            bool overlapsRight = boxOverlapWithDistance(nodes[rightNodeIndex].box, sphere, d2MinRight, d2MaxRight);
            sphere.r2 = fminf(sphere.r2, d2MaxRight);

            // Which nodes do we overlap?
            if (overlapsLeft && overlapsRight) {
                // Assume that the left child is closer
                uint32_t closer = leftNodeIndex;
                uint32_t other = rightNodeIndex;

                // ... if the right child was actually closer, swap the relevant values
                if (d2MinRight < d2MinLeft) {
                    closer = rightNodeIndex;
                    other = leftNodeIndex;
                }

                // Push the further node first, then the closer node
                stackPtr++;
                traversalStack[stackPtr] = other;

                stackPtr++;
                traversalStack[stackPtr] = closer;
            } else if (overlapsLeft) {
                stackPtr++;
                traversalStack[stackPtr] = leftNodeIndex;
            } else if (overlapsRight) {
                stackPtr++;
                traversalStack[stackPtr] = rightNodeIndex;
            }
        }
    }

    if (!notFound && recordNormal) {
        // Set normal from the selected primitive
        // Note: Normal is already computed by primitiveFindClosestPoint for triangles
        // but we keep this for consistency with Slang (though it may be redundant)
    }

    return !notFound;
}

/////////////////////////////////////////////////////////////////////////////////////////////
// BVH traversal for closest silhouette point
// Source: bvh.slang lines 504-646

template<typename NodeType, typename SilhouetteType>
__device__ bool traverseBvhClosestSilhouettePoint(
    const NodeType* nodes,
    const SilhouetteType* silhouettes,
    GPUBoundingSphere& sphere,
    bool flipNormalOrientation,
    float squaredMinRadius,
    float precision,
    GPUInteraction& interaction)
{
    if (squaredMinRadius >= sphere.r2) {
        return false;
    }

    uint32_t traversalStack[FCPW_BVH_MAX_DEPTH];
    bool notFound = true;

    // Check if sphere overlaps root bounding box
    float d2Min, d2Max;
    if (!boxOverlapWithDistance(nodes[0].box, sphere, d2Min, d2Max)) {
        return false;
    }

    traversalStack[0] = 0;
    int stackPtr = 0;

    while (stackPtr >= 0) {
        // Pop off the next node to work on
        uint32_t currentNodeIndex = traversalStack[stackPtr];
        const NodeType& node = nodes[currentNodeIndex];
        stackPtr--;

        // If this node is further than the closest found primitive, continue
        if (!boxOverlapWithDistance(node.box, sphere, d2Min, d2Max)) {
            continue;
        }

        if (node.nSilhouettes > 0) {
            // Leaf node - compute distance to silhouettes
            for (uint32_t p = 0; p < node.nSilhouettes; p++) {
                uint32_t silhouetteIndex = node.silhouetteOffset + p;
                const SilhouetteType& silhouette = silhouettes[silhouetteIndex];

                // Skip if silhouette has already been checked
                if (silhouette.index == interaction.index) {
                    continue;
                }

                GPUInteraction c;
                bool found = false;

                // Call the appropriate findClosestSilhouettePoint function
                // This is specialized for GPUVertex and GPUEdge
                if constexpr (sizeof(SilhouetteType) == sizeof(GPUVertex)) {
                    found = vertexFindClosestSilhouettePoint(
                        reinterpret_cast<const GPUVertex&>(silhouette),
                        sphere, flipNormalOrientation, squaredMinRadius, precision, c);
                } else {
                    found = edgeFindClosestSilhouettePoint(
                        reinterpret_cast<const GPUEdge&>(silhouette),
                        sphere, flipNormalOrientation, squaredMinRadius, precision, c);
                }

                // Keep the closest silhouette point
                if (found) {
                    notFound = false;
                    sphere.r2 = fminf(sphere.r2, c.d * c.d);
                    interaction = c;

                    if (squaredMinRadius >= sphere.r2) {
                        break;
                    }
                }
            }
        } else {
            // Internal node - find distance to child nodes
            // Check left child
            uint32_t leftNodeIndex = currentNodeIndex + 1;
            const NodeType& leftNode = nodes[leftNodeIndex];

            bool overlapsLeft = coneIsValid(leftNode.cone);
            if (overlapsLeft) {
                overlapsLeft = boxOverlapWithDistance(leftNode.box, sphere, d2Min, d2Max);
                if (overlapsLeft) {
                    float minAngleRange, maxAngleRange;
                    overlapsLeft = coneOverlapsSphere(leftNode.cone, sphere.c, leftNode.box, d2Min, minAngleRange, maxAngleRange);
                }
            }

            // Check right child
            uint32_t rightNodeIndex = currentNodeIndex + node.offset;
            const NodeType& rightNode = nodes[rightNodeIndex];

            bool overlapsRight = coneIsValid(rightNode.cone);
            if (overlapsRight) {
                float d2MinRight, d2MaxRight;
                overlapsRight = boxOverlapWithDistance(rightNode.box, sphere, d2MinRight, d2MaxRight);
                if (overlapsRight) {
                    float minAngleRange, maxAngleRange;
                    overlapsRight = coneOverlapsSphere(rightNode.cone, sphere.c, rightNode.box, d2MinRight, minAngleRange, maxAngleRange);
                }
            }

            // Which nodes do we overlap?
            if (overlapsLeft && overlapsRight) {
                // Assume that the left child is closer
                uint32_t closer = leftNodeIndex;
                uint32_t other = rightNodeIndex;

                // ... if the right child was actually closer, swap
                float d2MinRight;
                boxOverlapWithDistance(rightNode.box, sphere, d2MinRight, d2Max);
                if (d2MinRight < d2Min) {
                    closer = rightNodeIndex;
                    other = leftNodeIndex;
                }

                // Push the further node first, then the closer node
                stackPtr++;
                traversalStack[stackPtr] = other;

                stackPtr++;
                traversalStack[stackPtr] = closer;
            } else if (overlapsLeft) {
                stackPtr++;
                traversalStack[stackPtr] = leftNodeIndex;
            } else if (overlapsRight) {
                stackPtr++;
                traversalStack[stackPtr] = rightNodeIndex;
            }
        }
    }

    return !notFound;
}

/////////////////////////////////////////////////////////////////////////////////////////////
// BVH refit
// Source: bvh.slang lines 27-140

template<typename NodeType, typename PrimitiveType>
__device__ void refitBvhLeafNode(
    NodeType* nodes,
    const PrimitiveType* primitives,
    uint32_t nodeIndex)
{
    // Update leaf node's bounding box
    float3 pMin = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    float3 pMax = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    NodeType& node = nodes[nodeIndex];
    uint32_t nPrimitives = node.nPrimitives;

    for (uint32_t p = 0; p < nPrimitives; p++) {
        uint32_t primitiveIndex = node.offset + p;
        GPUBoundingBox primitiveBox;

        // Get bounding box based on primitive type
        if constexpr (sizeof(PrimitiveType) == sizeof(GPULineSegment)) {
            primitiveBox = lineSegmentGetBoundingBox(reinterpret_cast<const GPULineSegment&>(primitives[primitiveIndex]));
        } else {
            primitiveBox = triangleGetBoundingBox(reinterpret_cast<const GPUTriangle&>(primitives[primitiveIndex]));
        }

        pMin = minComponents(pMin, primitiveBox.pMin);
        pMax = maxComponents(pMax, primitiveBox.pMax);
    }

    node.box.pMin = pMin;
    node.box.pMax = pMax;
}

template<typename NodeType, typename PrimitiveType, typename SilhouetteType>
__device__ void refitSnchLeafNode(
    NodeType* nodes,
    const PrimitiveType* primitives,
    const SilhouetteType* silhouettes,
    uint32_t nodeIndex)
{
    // Update leaf node's bounding box
    float3 pMin = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    float3 pMax = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    NodeType& node = nodes[nodeIndex];
    uint32_t nPrimitives = node.nPrimitives;

    for (uint32_t p = 0; p < nPrimitives; p++) {
        uint32_t primitiveIndex = node.offset + p;
        GPUBoundingBox primitiveBox;

        if constexpr (sizeof(PrimitiveType) == sizeof(GPULineSegment)) {
            primitiveBox = lineSegmentGetBoundingBox(reinterpret_cast<const GPULineSegment&>(primitives[primitiveIndex]));
        } else {
            primitiveBox = triangleGetBoundingBox(reinterpret_cast<const GPUTriangle&>(primitives[primitiveIndex]));
        }

        pMin = minComponents(pMin, primitiveBox.pMin);
        pMax = maxComponents(pMax, primitiveBox.pMax);
    }

    node.box.pMin = pMin;
    node.box.pMax = pMax;

    // Update leaf node's bounding cone
    float3 axis = make_float3(0.0f, 0.0f, 0.0f);
    float3 centroid = (pMin + pMax) * 0.5f;
    float halfAngle = 0.0f;
    float radius = 0.0f;
    bool anySilhouettes = false;
    bool silhouettesHaveTwoAdjacentFaces = true;
    uint32_t nSilhouettes = node.nSilhouettes;

    for (uint32_t p = 0; p < nSilhouettes; p++) {
        uint32_t silhouetteIndex = node.silhouetteOffset + p;
        const SilhouetteType& silhouette = silhouettes[silhouetteIndex];

        float3 n0, n1, silhCentroid;
        bool hasTwoFaces;

        if constexpr (sizeof(SilhouetteType) == sizeof(GPUVertex)) {
            const GPUVertex& vertex = reinterpret_cast<const GPUVertex&>(silhouette);
            n0 = vertexGetNormal(vertex, 0);
            n1 = vertexGetNormal(vertex, 1);
            silhCentroid = vertexGetCentroid(vertex);
            hasTwoFaces = vertexHasTwoAdjacentFaces(vertex);
        } else {
            const GPUEdge& edge = reinterpret_cast<const GPUEdge&>(silhouette);
            n0 = edgeGetNormal(edge, 0);
            n1 = edgeGetNormal(edge, 1);
            silhCentroid = edgeGetCentroid(edge);
            hasTwoFaces = edgeHasTwoAdjacentFaces(edge);
        }

        axis = axis + n0;
        axis = axis + n1;
        radius = fmaxf(radius, length(silhCentroid - centroid));
        silhouettesHaveTwoAdjacentFaces = silhouettesHaveTwoAdjacentFaces && hasTwoFaces;
        anySilhouettes = true;
    }

    if (!anySilhouettes) {
        halfAngle = -M_PI;
    } else if (!silhouettesHaveTwoAdjacentFaces) {
        halfAngle = M_PI;
    } else {
        float axisNorm = length(axis);
        if (axisNorm > FLT_EPSILON) {
            axis = axis / axisNorm;

            for (uint32_t p = 0; p < nSilhouettes; p++) {
                uint32_t silhouetteIndex = node.silhouetteOffset + p;
                for (uint32_t k = 0; k < 2; k++) {
                    float3 n;
                    if constexpr (sizeof(SilhouetteType) == sizeof(GPUVertex)) {
                        n = vertexGetNormal(reinterpret_cast<const GPUVertex&>(silhouettes[silhouetteIndex]), k);
                    } else {
                        n = edgeGetNormal(reinterpret_cast<const GPUEdge&>(silhouettes[silhouetteIndex]), k);
                    }
                    float angle = acosf(fmaxf(-1.0f, fminf(1.0f, dot(axis, n))));
                    halfAngle = fmaxf(halfAngle, angle);
                }
            }
        }
    }

    node.cone.axis = axis;
    node.cone.halfAngle = halfAngle;
    node.cone.radius = radius;
}

template<typename NodeType>
__device__ void refitBvhInternalNode(
    NodeType* nodes,
    uint32_t nodeIndex)
{
    // Update internal node's bounding box
    NodeType& node = nodes[nodeIndex];
    uint32_t leftNodeIndex = nodeIndex + 1;
    uint32_t rightNodeIndex = nodeIndex + node.offset;
    const NodeType& leftNode = nodes[leftNodeIndex];
    const NodeType& rightNode = nodes[rightNodeIndex];

    GPUBoundingBox leftBox = leftNode.box;
    GPUBoundingBox rightBox = rightNode.box;
    GPUBoundingBox mergedBox = mergeBoundingBoxes(leftBox, rightBox);
    node.box = mergedBox;
}

template<typename NodeType>
__device__ void refitSnchInternalNode(
    NodeType* nodes,
    uint32_t nodeIndex)
{
    // Update internal node's bounding box
    NodeType& node = nodes[nodeIndex];
    uint32_t leftNodeIndex = nodeIndex + 1;
    uint32_t rightNodeIndex = nodeIndex + node.offset;
    const NodeType& leftNode = nodes[leftNodeIndex];
    const NodeType& rightNode = nodes[rightNodeIndex];

    GPUBoundingBox leftBox = leftNode.box;
    GPUBoundingBox rightBox = rightNode.box;
    GPUBoundingBox mergedBox = mergeBoundingBoxes(leftBox, rightBox);
    node.box = mergedBox;

    // Update internal node's bounding cone
    GPUBoundingCone leftCone = leftNode.cone;
    GPUBoundingCone rightCone = rightNode.cone;
    GPUBoundingCone mergedCone = mergeBoundingCones(leftCone, rightCone,
                                                     boxCentroid(leftBox),
                                                     boxCentroid(rightBox),
                                                     boxCentroid(mergedBox));
    node.cone = mergedCone;
}

// Refit dispatcher for BVH nodes (without silhouettes)
template<typename NodeType, typename PrimitiveType>
__device__ void refitBvhNode(
    NodeType* nodes,
    const PrimitiveType* primitives,
    uint32_t nodeIndex)
{
    if (nodes[nodeIndex].nPrimitives > 0) {
        refitBvhLeafNode(nodes, primitives, nodeIndex);
    } else {
        refitBvhInternalNode(nodes, nodeIndex);
    }
}

// Refit dispatcher for SNCH nodes (with silhouettes)
template<typename NodeType, typename PrimitiveType, typename SilhouetteType>
__device__ void refitSnchNode(
    NodeType* nodes,
    const PrimitiveType* primitives,
    const SilhouetteType* silhouettes,
    uint32_t nodeIndex)
{
    if (nodes[nodeIndex].nPrimitives > 0) {
        refitSnchLeafNode(nodes, primitives, silhouettes, nodeIndex);
    } else {
        refitSnchInternalNode(nodes, nodeIndex);
    }
}

} // namespace cuda
} // namespace fcpw
