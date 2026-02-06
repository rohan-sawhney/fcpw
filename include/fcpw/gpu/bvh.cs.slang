import fcpw;

#define FCPW_UNDEFINED_BVH 0
#define FCPW_LINE_SEGMENT_BVH 1
#define FCPW_TRIANGLE_BVH 2
#define FCPW_LINE_SEGMENT_SNCH 3
#define FCPW_TRIANGLE_SNCH 4

#ifndef FCPW_BVH_TYPE
#define FCPW_BVH_TYPE FCPW_UNDEFINED_BVH
#endif

#if FCPW_BVH_TYPE == FCPW_LINE_SEGMENT_BVH
typedef Bvh<BvhNode, LineSegment, NoSilhouette> BvhType;
#define FCPW_BVH_HAS_SILHOUETTE_DATA 0

#elif FCPW_BVH_TYPE == FCPW_TRIANGLE_BVH
typedef Bvh<BvhNode, Triangle, NoSilhouette> BvhType;
#define FCPW_BVH_HAS_SILHOUETTE_DATA 0

#elif FCPW_BVH_TYPE == FCPW_LINE_SEGMENT_SNCH
typedef Bvh<SnchNode, LineSegment, Vertex> BvhType;
#define FCPW_BVH_HAS_SILHOUETTE_DATA 1

#elif FCPW_BVH_TYPE == FCPW_TRIANGLE_SNCH
typedef Bvh<SnchNode, Triangle, Edge> BvhType;
#define FCPW_BVH_HAS_SILHOUETTE_DATA 1

#else
// Compile time error
#error FCPW_BVH_TYPE is not set to a supported type
#endif

uniform ParameterBlock<BvhType> gBvh;

[shader("compute")]
[numthreads(256, 1, 1)]
void rayIntersection(uint3 threadId: SV_DispatchThreadID,
                     uniform StructuredBuffer<Ray> rays,
                     uniform RWStructuredBuffer<Interaction> interactions,
                     uniform uint checkForOcclusion,
                     uniform uint nQueries)
{
    uint index = threadId.x;
    if (index >= nQueries)
    {
        return;
    }

    Ray r = rays[index];
    Interaction i;
    bool didIntersect = gBvh.intersect(r, checkForOcclusion != 0, i);
    if (didIntersect)
    {
        interactions[index] = i;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void robustRayIntersection(uint3 threadId: SV_DispatchThreadID,
                           uniform StructuredBuffer<Ray> rays,
                           uniform RWStructuredBuffer<Interaction> interactions,
                           uniform uint nQueries)
{
    uint index = threadId.x;
    if (index >= nQueries)
    {
        return;
    }

    Ray r = rays[index];
    Interaction i;
    bool didIntersect = gBvh.intersectRobust(r, i);
    if (didIntersect)
    {
        interactions[index] = i;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sphereIntersection(uint3 threadId: SV_DispatchThreadID,
                        uniform StructuredBuffer<BoundingSphere> boundingSpheres,
                        uniform StructuredBuffer<float3> randNums,
                        uniform RWStructuredBuffer<Interaction> interactions,
                        uniform uint nQueries)
{
    uint index = threadId.x;
    if (index >= nQueries)
    {
        return;
    }

    BoundingSphere s = boundingSpheres[index];
    float3 randNum = randNums[index];
    ConstantBranchTraversalWeight branchTraversalWeight;
    Interaction i;
    bool didIntersect = gBvh.intersect<ConstantBranchTraversalWeight>(s, randNum, branchTraversalWeight, i);
    if (didIntersect)
    {
        interactions[index] = i;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void closestPoint(uint3 threadId: SV_DispatchThreadID,
                  uniform StructuredBuffer<BoundingSphere> boundingSpheres,
                  uniform RWStructuredBuffer<Interaction> interactions,
                  uniform uint recordNormals,
                  uniform uint nQueries)
{
    uint index = threadId.x;
    if (index >= nQueries)
    {
        return;
    }

    BoundingSphere s = boundingSpheres[index];
    Interaction i;
    bool found = gBvh.findClosestPoint(s, i, recordNormals != 0);
    if (found)
    {
        interactions[index] = i;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void closestSilhouettePoint(uint3 threadId: SV_DispatchThreadID,
                            uniform StructuredBuffer<BoundingSphere> boundingSpheres,
                            uniform StructuredBuffer<uint> flipNormalOrientation,
                            uniform RWStructuredBuffer<Interaction> interactions,
                            uniform float squaredMinRadius,
                            uniform float precision,
                            uniform uint nQueries)
{
    uint index = threadId.x;
    if (index >= nQueries)
    {
        return;
    }

    Interaction i;
#if FCPW_BVH_HAS_SILHOUETTE_DATA
    BoundingSphere s = boundingSpheres[index];
    bool flipNormal = flipNormalOrientation[index] != 0;
    bool found = gBvh.findClosestSilhouettePoint(s, flipNormal, squaredMinRadius, precision, i);
    if (found)
    {
        interactions[index] = i;
    }
#endif
}

[shader("compute")]
[numthreads(256, 1, 1)]
void refit(uint3 threadId: SV_DispatchThreadID,
           uniform StructuredBuffer<uint> nodeIndices,
           uniform uint firstNodeOffset,
           uniform uint nodeCount)
{
    uint index = threadId.x;
    if (index >= nodeCount)
    {
        return;
    }

    uint nodeIndex = nodeIndices[firstNodeOffset + index];
    gBvh.refit(nodeIndex);
}