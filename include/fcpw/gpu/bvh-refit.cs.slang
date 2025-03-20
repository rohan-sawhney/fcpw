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
typdef Bvh<BvhNode, LineSegment, NoSilhouette> BvhType;

#elif _BVH_TYPE == TRIANGLE_BVH
typedef Bvh<BvhNode, Triangle, NoSilhouette> BvhType;

#elif _BVH_TYPE == LINE_SEGMENT_SNCH
typedef Bvh<SnchNode, LineSegment, Vertex> BvhType;

#elif _BVH_TYPE == TRIANGLE_SNCH
typedef Bvh<SnchNode, Triangle, Edge> BvhType;

#else
// Compile time error
#error _BVH_TYPE is not set to a supported type
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
