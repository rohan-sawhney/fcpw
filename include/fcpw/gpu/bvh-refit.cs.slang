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
typdef Bvh<BvhNode, LineSegment, NoSilhouette> BvhType;

#elif FCPW_BVH_TYPE == FCPW_TRIANGLE_BVH
typedef Bvh<BvhNode, Triangle, NoSilhouette> BvhType;

#elif FCPW_BVH_TYPE == FCPW_LINE_SEGMENT_SNCH
typedef Bvh<SnchNode, LineSegment, Vertex> BvhType;

#elif FCPW_BVH_TYPE == FCPW_TRIANGLE_SNCH
typedef Bvh<SnchNode, Triangle, Edge> BvhType;

#else
// Compile time error
#error FCPW_BVH_TYPE is not set to a supported type
#endif

uniform ParameterBlock<BvhType> gBvh;

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
