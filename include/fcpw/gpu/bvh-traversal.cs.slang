import fcpw;

#define UNDEFINED_BVH_TYPE 0
#define LINE_SEGMENT_BVH 1
#define TRIANGLE_BVH 2
#define LINE_SEGMENT_SNCH 3
#define TRIANGLE_SNCH 4

#ifndef _BVH_TYPE
#define _BVH_TYPE UNDEFINED_BVH_TYPE
#endif

#if _BVH_TYPE == LINE_SEGMENT_BVH
uniform ParameterBlock<Bvh<BvhNodes, LineSegments, NoSilhouettes>> gBvh;
#define _BVH_HAS_SILHOUETTE_DATA 0

#elif _BVH_TYPE == TRIANGLE_BVH
uniform ParameterBlock<Bvh<BvhNodes, Triangles, NoSilhouettes>> gBvh;
#define _BVH_HAS_SILHOUETTE_DATA 0

#elif _BVH_TYPE == LINE_SEGMENT_SNCH
uniform ParameterBlock<Bvh<SnchNodes, LineSegments, Vertices>> gBvh;
#define _BVH_HAS_SILHOUETTE_DATA 1

#elif _BVH_TYPE == TRIANGLE_SNCH
uniform ParameterBlock<Bvh<SnchNodes, Triangles, Edges>> gBvh;
#define _BVH_HAS_SILHOUETTE_DATA 1

#else
// Compile time error
#error _BVH_TYPE is not set to a supported type
#endif

[shader("compute")]
[numthreads(256, 1, 1)]
void rayIntersection(uint3 threadId: SV_DispatchThreadID,
                     uniform Rays rays,
                     uniform bool checkForOcclusion,
                     uniform Interactions interactions,
                     uniform uint nQueries)
{
    uint index = threadId.x;
    if (index >= nQueries)
    {
        return;
    }

    Ray r = rays[index];
    Interaction i;
    bool didIntersect = gBvh.intersect(r, checkForOcclusion, i);
    if (didIntersect)
    {
        interactions[index] = i;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sphereIntersection(uint3 threadId: SV_DispatchThreadID,
                        uniform BoundingSpheres boundingSpheres,
                        uniform Float3List randNums,
                        uniform Interactions interactions,
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
                  uniform BoundingSpheres boundingSpheres,
                  uniform Interactions interactions,
                  uniform bool recordNormals,
                  uniform uint nQueries)
{
    uint index = threadId.x;
    if (index >= nQueries)
    {
        return;
    }

    BoundingSphere s = boundingSpheres[index];
    Interaction i;
    bool found = gBvh.findClosestPoint(s, i, recordNormals);
    if (found)
    {
        interactions[index] = i;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void closestSilhouettePoint(uint3 threadId: SV_DispatchThreadID,
                            uniform BoundingSpheres boundingSpheres,
                            uniform StructuredBuffer<uint> flipNormalOrientation,
                            uniform float squaredMinRadius,
                            uniform float precision,
                            uniform Interactions interactions,
                            uniform uint nQueries)
{
    uint index = threadId.x;
    if (index >= nQueries)
    {
        return;
    }

    Interaction i;
#if _BVH_HAS_SILHOUETTE_DATA
    BoundingSphere s = boundingSpheres[index];
    bool flipNormal = flipNormalOrientation[index] == 1 ? true : false;
    bool found = gBvh.findClosestSilhouettePoint(s, flipNormal, squaredMinRadius, precision, i);
    if (found)
    {
        interactions[index] = i;
    }
#endif
}
