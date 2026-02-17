#include <fcpw/cuda/cuda_bvh_device.cuh>
#include <fcpw/cuda/cuda_kernels.h>

namespace fcpw {

///////////////////////////////////////////////////////////////////////////////
// Kernel definitions
///////////////////////////////////////////////////////////////////////////////

// each thread processes one ray intersection query
template<typename N, typename P, typename S>
__global__ void rayIntersectionKernel(const N* nodes, const P* primitives, const S* silhouettes,
                                      CUDARay* rays, CUDAInteraction* interactions,
                                      uint32_t checkForOcclusion, uint32_t nQueries)
{
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nQueries) return;

    CUDARay& cudaRay = rays[index];
    DeviceRay r;
    r.o = toFloat3(cudaRay.o);
    r.d = toFloat3(cudaRay.d);
    r.dInv = toFloat3(cudaRay.dInv);
    r.tMax = cudaRay.tMax;

    DeviceInteraction i;
    i.p = make_float3(0.0f, 0.0f, 0.0f);
    i.n = make_float3(0.0f, 0.0f, 0.0f);
    i.uv = make_float2(0.0f, 0.0f);
    i.d = FCPW_CUDA_FLT_MAX;
    i.index = FCPW_CUDA_UINT_MAX;

    bool didIntersect = bvhIntersectRay<N, P, S>(nodes, primitives, silhouettes,
                                                  r, checkForOcclusion != 0, i);
    if (didIntersect) {
        CUDAInteraction& out = interactions[index];
        out.p = fromFloat3(i.p);
        out.n = fromFloat3(i.n);
        out.uv = fromFloat2(i.uv);
        out.d = i.d;
        out.index = i.index;
    }
}

// each thread processes one sphere intersection query
template<typename N, typename P, typename S>
__global__ void sphereIntersectionKernel(const N* nodes, const P* primitives, const S* silhouettes,
                                         CUDABoundingSphere* boundingSpheres,
                                         CUDAFloat3* randNums,
                                         CUDAInteraction* interactions,
                                         uint32_t nQueries)
{
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nQueries) return;

    DeviceBoundingSphere s;
    s.c = toFloat3(boundingSpheres[index].c);
    s.r2 = boundingSpheres[index].r2;

    float3 rn = toFloat3(randNums[index]);

    DeviceInteraction i;
    i.p = make_float3(0.0f, 0.0f, 0.0f);
    i.n = make_float3(0.0f, 0.0f, 0.0f);
    i.uv = make_float2(0.0f, 0.0f);
    i.d = FCPW_CUDA_FLT_MAX;
    i.index = FCPW_CUDA_UINT_MAX;

    bool didIntersect = bvhIntersectSphere<N, P, S>(nodes, primitives, silhouettes, s, rn, i);
    if (didIntersect) {
        CUDAInteraction& out = interactions[index];
        out.p = fromFloat3(i.p);
        out.n = fromFloat3(i.n);
        out.uv = fromFloat2(i.uv);
        out.d = i.d;
        out.index = i.index;
    }
}

// each thread processes one closest point query
template<typename N, typename P, typename S>
__global__ void closestPointKernel(const N* nodes, const P* primitives, const S* silhouettes,
                                   CUDABoundingSphere* boundingSpheres,
                                   CUDAInteraction* interactions,
                                   uint32_t recordNormals, uint32_t nQueries)
{
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nQueries) return;

    DeviceBoundingSphere s;
    s.c = toFloat3(boundingSpheres[index].c);
    s.r2 = boundingSpheres[index].r2;

    DeviceInteraction i;
    i.p = make_float3(0.0f, 0.0f, 0.0f);
    i.n = make_float3(0.0f, 0.0f, 0.0f);
    i.uv = make_float2(0.0f, 0.0f);
    i.d = FCPW_CUDA_FLT_MAX;
    i.index = FCPW_CUDA_UINT_MAX;

    bool found = bvhFindClosestPoint<N, P, S>(nodes, primitives, silhouettes,
                                               s, i, recordNormals != 0);
    if (found) {
        CUDAInteraction& out = interactions[index];
        out.p = fromFloat3(i.p);
        out.n = fromFloat3(i.n);
        out.uv = fromFloat2(i.uv);
        out.d = i.d;
        out.index = i.index;
    }
}

// each thread processes one closest silhouette point query
template<typename N, typename P, typename S>
__global__ void closestSilhouettePointKernel(const N* nodes, const P* primitives,
                                             const S* silhouettes,
                                             CUDABoundingSphere* boundingSpheres,
                                             uint32_t* flipNormalOrientation,
                                             CUDAInteraction* interactions,
                                             float squaredMinRadius, float precision,
                                             uint32_t nQueries)
{
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nQueries) return;

    DeviceBoundingSphere s;
    s.c = toFloat3(boundingSpheres[index].c);
    s.r2 = boundingSpheres[index].r2;
    bool flipNormal = flipNormalOrientation[index] != 0;

    DeviceInteraction i;
    i.p = make_float3(0.0f, 0.0f, 0.0f);
    i.n = make_float3(0.0f, 0.0f, 0.0f);
    i.uv = make_float2(0.0f, 0.0f);
    i.d = FCPW_CUDA_FLT_MAX;
    i.index = FCPW_CUDA_UINT_MAX;

    bool found = bvhFindClosestSilhouettePoint<N, P, S>(nodes, primitives, silhouettes,
                                                         s, flipNormal,
                                                         squaredMinRadius, precision, i);
    if (found) {
        CUDAInteraction& out = interactions[index];
        out.p = fromFloat3(i.p);
        out.n = fromFloat3(i.n);
        out.uv = fromFloat2(i.uv);
        out.d = i.d;
        out.index = i.index;
    }
}

// each thread processes one transformed ray intersection query
template<typename N, typename P, typename S>
__global__ void transformedRayIntersectionKernel(const N* nodes, const P* primitives,
                                                 const S* silhouettes,
                                                 const CUDATransform* transform,
                                                 CUDARay* rays, CUDAInteraction* interactions,
                                                 uint32_t checkForOcclusion, uint32_t nQueries)
{
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nQueries) return;

    CUDARay& cudaRay = rays[index];
    DeviceRay r;
    r.o = toFloat3(cudaRay.o);
    r.d = toFloat3(cudaRay.d);
    r.dInv = toFloat3(cudaRay.dInv);
    r.tMax = cudaRay.tMax;

    DeviceInteraction i;
    i.p = make_float3(0.0f, 0.0f, 0.0f);
    i.n = make_float3(0.0f, 0.0f, 0.0f);
    i.uv = make_float2(0.0f, 0.0f);
    i.d = FCPW_CUDA_FLT_MAX;
    i.index = FCPW_CUDA_UINT_MAX;

    bool didIntersect = transformedBvhIntersectRay<N, P, S>(nodes, primitives, silhouettes,
                                                             transform->t, transform->tInv,
                                                             r, checkForOcclusion != 0, i);
    if (didIntersect) {
        CUDAInteraction& out = interactions[index];
        out.p = fromFloat3(i.p);
        out.n = fromFloat3(i.n);
        out.uv = fromFloat2(i.uv);
        out.d = i.d;
        out.index = i.index;
    }
}

// each thread processes one transformed sphere intersection query
template<typename N, typename P, typename S>
__global__ void transformedSphereIntersectionKernel(const N* nodes, const P* primitives,
                                                     const S* silhouettes,
                                                     const CUDATransform* transform,
                                                     CUDABoundingSphere* boundingSpheres,
                                                     CUDAFloat3* randNums,
                                                     CUDAInteraction* interactions,
                                                     uint32_t nQueries)
{
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nQueries) return;

    DeviceBoundingSphere s;
    s.c = toFloat3(boundingSpheres[index].c);
    s.r2 = boundingSpheres[index].r2;

    float3 rn = toFloat3(randNums[index]);

    DeviceInteraction i;
    i.p = make_float3(0.0f, 0.0f, 0.0f);
    i.n = make_float3(0.0f, 0.0f, 0.0f);
    i.uv = make_float2(0.0f, 0.0f);
    i.d = FCPW_CUDA_FLT_MAX;
    i.index = FCPW_CUDA_UINT_MAX;

    bool didIntersect = transformedBvhIntersectSphere<N, P, S>(nodes, primitives, silhouettes,
                                                                transform->t, transform->tInv,
                                                                s, rn, i);
    if (didIntersect) {
        CUDAInteraction& out = interactions[index];
        out.p = fromFloat3(i.p);
        out.n = fromFloat3(i.n);
        out.uv = fromFloat2(i.uv);
        out.d = i.d;
        out.index = i.index;
    }
}

// each thread processes one transformed closest point query
template<typename N, typename P, typename S>
__global__ void transformedClosestPointKernel(const N* nodes, const P* primitives,
                                              const S* silhouettes,
                                              const CUDATransform* transform,
                                              CUDABoundingSphere* boundingSpheres,
                                              CUDAInteraction* interactions,
                                              uint32_t recordNormals, uint32_t nQueries)
{
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nQueries) return;

    DeviceBoundingSphere s;
    s.c = toFloat3(boundingSpheres[index].c);
    s.r2 = boundingSpheres[index].r2;

    DeviceInteraction i;
    i.p = make_float3(0.0f, 0.0f, 0.0f);
    i.n = make_float3(0.0f, 0.0f, 0.0f);
    i.uv = make_float2(0.0f, 0.0f);
    i.d = FCPW_CUDA_FLT_MAX;
    i.index = FCPW_CUDA_UINT_MAX;

    bool found = transformedBvhFindClosestPoint<N, P, S>(nodes, primitives, silhouettes,
                                                          transform->t, transform->tInv,
                                                          s, i, recordNormals != 0);
    if (found) {
        CUDAInteraction& out = interactions[index];
        out.p = fromFloat3(i.p);
        out.n = fromFloat3(i.n);
        out.uv = fromFloat2(i.uv);
        out.d = i.d;
        out.index = i.index;
    }
}

// each thread processes one transformed closest silhouette point query
template<typename N, typename P, typename S>
__global__ void transformedClosestSilhouettePointKernel(const N* nodes, const P* primitives,
                                                         const S* silhouettes,
                                                         const CUDATransform* transform,
                                                         CUDABoundingSphere* boundingSpheres,
                                                         uint32_t* flipNormalOrientation,
                                                         CUDAInteraction* interactions,
                                                         float squaredMinRadius, float precision,
                                                         uint32_t nQueries)
{
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nQueries) return;

    DeviceBoundingSphere s;
    s.c = toFloat3(boundingSpheres[index].c);
    s.r2 = boundingSpheres[index].r2;
    bool flipNormal = flipNormalOrientation[index] != 0;

    DeviceInteraction i;
    i.p = make_float3(0.0f, 0.0f, 0.0f);
    i.n = make_float3(0.0f, 0.0f, 0.0f);
    i.uv = make_float2(0.0f, 0.0f);
    i.d = FCPW_CUDA_FLT_MAX;
    i.index = FCPW_CUDA_UINT_MAX;

    bool found = transformedBvhFindClosestSilhouettePoint<N, P, S>(nodes, primitives, silhouettes,
                                                                     transform->t, transform->tInv,
                                                                     s, flipNormal,
                                                                     squaredMinRadius, precision, i);
    if (found) {
        CUDAInteraction& out = interactions[index];
        out.p = fromFloat3(i.p);
        out.n = fromFloat3(i.n);
        out.uv = fromFloat2(i.uv);
        out.d = i.d;
        out.index = i.index;
    }
}

// each thread refits one BVH node
template<typename N, typename P, typename S>
__global__ void refitKernel(N* nodes, const P* primitives, const S* silhouettes,
                            const uint32_t* nodeIndices,
                            uint32_t firstNodeOffset, uint32_t nodeCount)
{
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nodeCount) return;

    uint32_t nodeIndex = nodeIndices[firstNodeOffset + index];
    bvhRefit<N, P, S>(nodes, primitives, silhouettes, nodeIndex);
}

///////////////////////////////////////////////////////////////////////////////
// Host launch wrappers
///////////////////////////////////////////////////////////////////////////////

static const uint32_t THREADS_PER_BLOCK = 256;

void launchRayIntersectionKernel(int bvhType,
                                 void* d_nodes, void* d_primitives, void* d_silhouettes,
                                 void* d_rays, void* d_interactions,
                                 uint32_t checkForOcclusion, uint32_t nQueries,
                                 void* stream)
{
    uint32_t numBlocks = (nQueries + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaStream_t s = static_cast<cudaStream_t>(stream);

    // dispatch to correct template based on BVH type
    switch (bvhType) {
        case FCPW_CUDA_LINE_SEGMENT_BVH:
            rayIntersectionKernel<CUDABvhNode, CUDALineSegment, CUDANoSilhouette>
                <<<numBlocks, THREADS_PER_BLOCK, 0, s>>>(
                    (CUDABvhNode*)d_nodes, (CUDALineSegment*)d_primitives,
                    (CUDANoSilhouette*)d_silhouettes,
                    (CUDARay*)d_rays, (CUDAInteraction*)d_interactions,
                    checkForOcclusion, nQueries);
            break;
        case FCPW_CUDA_TRIANGLE_BVH:
            rayIntersectionKernel<CUDABvhNode, CUDATriangle, CUDANoSilhouette>
                <<<numBlocks, THREADS_PER_BLOCK, 0, s>>>(
                    (CUDABvhNode*)d_nodes, (CUDATriangle*)d_primitives,
                    (CUDANoSilhouette*)d_silhouettes,
                    (CUDARay*)d_rays, (CUDAInteraction*)d_interactions,
                    checkForOcclusion, nQueries);
            break;
        case FCPW_CUDA_LINE_SEGMENT_SNCH:
            rayIntersectionKernel<CUDASnchNode, CUDALineSegment, CUDAVertex>
                <<<numBlocks, THREADS_PER_BLOCK, 0, s>>>(
                    (CUDASnchNode*)d_nodes, (CUDALineSegment*)d_primitives,
                    (CUDAVertex*)d_silhouettes,
                    (CUDARay*)d_rays, (CUDAInteraction*)d_interactions,
                    checkForOcclusion, nQueries);
            break;
        case FCPW_CUDA_TRIANGLE_SNCH:
            rayIntersectionKernel<CUDASnchNode, CUDATriangle, CUDAEdge>
                <<<numBlocks, THREADS_PER_BLOCK, 0, s>>>(
                    (CUDASnchNode*)d_nodes, (CUDATriangle*)d_primitives,
                    (CUDAEdge*)d_silhouettes,
                    (CUDARay*)d_rays, (CUDAInteraction*)d_interactions,
                    checkForOcclusion, nQueries);
            break;
    }
}

void launchSphereIntersectionKernel(int bvhType,
                                    void* d_nodes, void* d_primitives, void* d_silhouettes,
                                    void* d_boundingSpheres, void* d_randNums,
                                    void* d_interactions,
                                    uint32_t nQueries,
                                    void* stream)
{
    uint32_t numBlocks = (nQueries + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaStream_t s = static_cast<cudaStream_t>(stream);

    // dispatch to correct template based on BVH type
    switch (bvhType) {
        case FCPW_CUDA_LINE_SEGMENT_BVH:
            sphereIntersectionKernel<CUDABvhNode, CUDALineSegment, CUDANoSilhouette>
                <<<numBlocks, THREADS_PER_BLOCK, 0, s>>>(
                    (CUDABvhNode*)d_nodes, (CUDALineSegment*)d_primitives,
                    (CUDANoSilhouette*)d_silhouettes,
                    (CUDABoundingSphere*)d_boundingSpheres, (CUDAFloat3*)d_randNums,
                    (CUDAInteraction*)d_interactions, nQueries);
            break;
        case FCPW_CUDA_TRIANGLE_BVH:
            sphereIntersectionKernel<CUDABvhNode, CUDATriangle, CUDANoSilhouette>
                <<<numBlocks, THREADS_PER_BLOCK, 0, s>>>(
                    (CUDABvhNode*)d_nodes, (CUDATriangle*)d_primitives,
                    (CUDANoSilhouette*)d_silhouettes,
                    (CUDABoundingSphere*)d_boundingSpheres, (CUDAFloat3*)d_randNums,
                    (CUDAInteraction*)d_interactions, nQueries);
            break;
        case FCPW_CUDA_LINE_SEGMENT_SNCH:
            sphereIntersectionKernel<CUDASnchNode, CUDALineSegment, CUDAVertex>
                <<<numBlocks, THREADS_PER_BLOCK, 0, s>>>(
                    (CUDASnchNode*)d_nodes, (CUDALineSegment*)d_primitives,
                    (CUDAVertex*)d_silhouettes,
                    (CUDABoundingSphere*)d_boundingSpheres, (CUDAFloat3*)d_randNums,
                    (CUDAInteraction*)d_interactions, nQueries);
            break;
        case FCPW_CUDA_TRIANGLE_SNCH:
            sphereIntersectionKernel<CUDASnchNode, CUDATriangle, CUDAEdge>
                <<<numBlocks, THREADS_PER_BLOCK, 0, s>>>(
                    (CUDASnchNode*)d_nodes, (CUDATriangle*)d_primitives,
                    (CUDAEdge*)d_silhouettes,
                    (CUDABoundingSphere*)d_boundingSpheres, (CUDAFloat3*)d_randNums,
                    (CUDAInteraction*)d_interactions, nQueries);
            break;
    }
}

void launchClosestPointKernel(int bvhType,
                              void* d_nodes, void* d_primitives, void* d_silhouettes,
                              void* d_boundingSpheres, void* d_interactions,
                              uint32_t recordNormals, uint32_t nQueries,
                              void* stream)
{
    uint32_t numBlocks = (nQueries + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaStream_t s = static_cast<cudaStream_t>(stream);

    // dispatch to correct template based on BVH type
    switch (bvhType) {
        case FCPW_CUDA_LINE_SEGMENT_BVH:
            closestPointKernel<CUDABvhNode, CUDALineSegment, CUDANoSilhouette>
                <<<numBlocks, THREADS_PER_BLOCK, 0, s>>>(
                    (CUDABvhNode*)d_nodes, (CUDALineSegment*)d_primitives,
                    (CUDANoSilhouette*)d_silhouettes,
                    (CUDABoundingSphere*)d_boundingSpheres,
                    (CUDAInteraction*)d_interactions, recordNormals, nQueries);
            break;
        case FCPW_CUDA_TRIANGLE_BVH:
            closestPointKernel<CUDABvhNode, CUDATriangle, CUDANoSilhouette>
                <<<numBlocks, THREADS_PER_BLOCK, 0, s>>>(
                    (CUDABvhNode*)d_nodes, (CUDATriangle*)d_primitives,
                    (CUDANoSilhouette*)d_silhouettes,
                    (CUDABoundingSphere*)d_boundingSpheres,
                    (CUDAInteraction*)d_interactions, recordNormals, nQueries);
            break;
        case FCPW_CUDA_LINE_SEGMENT_SNCH:
            closestPointKernel<CUDASnchNode, CUDALineSegment, CUDAVertex>
                <<<numBlocks, THREADS_PER_BLOCK, 0, s>>>(
                    (CUDASnchNode*)d_nodes, (CUDALineSegment*)d_primitives,
                    (CUDAVertex*)d_silhouettes,
                    (CUDABoundingSphere*)d_boundingSpheres,
                    (CUDAInteraction*)d_interactions, recordNormals, nQueries);
            break;
        case FCPW_CUDA_TRIANGLE_SNCH:
            closestPointKernel<CUDASnchNode, CUDATriangle, CUDAEdge>
                <<<numBlocks, THREADS_PER_BLOCK, 0, s>>>(
                    (CUDASnchNode*)d_nodes, (CUDATriangle*)d_primitives,
                    (CUDAEdge*)d_silhouettes,
                    (CUDABoundingSphere*)d_boundingSpheres,
                    (CUDAInteraction*)d_interactions, recordNormals, nQueries);
            break;
    }
}

void launchClosestSilhouettePointKernel(int bvhType,
                                        void* d_nodes, void* d_primitives, void* d_silhouettes,
                                        void* d_boundingSpheres, void* d_flipNormalOrientation,
                                        void* d_interactions,
                                        float squaredMinRadius, float precision,
                                        uint32_t nQueries,
                                        void* stream)
{
    uint32_t numBlocks = (nQueries + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaStream_t s = static_cast<cudaStream_t>(stream);

    // dispatch to correct template based on BVH type
    switch (bvhType) {
        case FCPW_CUDA_LINE_SEGMENT_BVH:
            // No silhouettes for BVH types - no-op
            break;
        case FCPW_CUDA_TRIANGLE_BVH:
            // No silhouettes for BVH types - no-op
            break;
        case FCPW_CUDA_LINE_SEGMENT_SNCH:
            closestSilhouettePointKernel<CUDASnchNode, CUDALineSegment, CUDAVertex>
                <<<numBlocks, THREADS_PER_BLOCK, 0, s>>>(
                    (CUDASnchNode*)d_nodes, (CUDALineSegment*)d_primitives,
                    (CUDAVertex*)d_silhouettes,
                    (CUDABoundingSphere*)d_boundingSpheres,
                    (uint32_t*)d_flipNormalOrientation,
                    (CUDAInteraction*)d_interactions,
                    squaredMinRadius, precision, nQueries);
            break;
        case FCPW_CUDA_TRIANGLE_SNCH:
            closestSilhouettePointKernel<CUDASnchNode, CUDATriangle, CUDAEdge>
                <<<numBlocks, THREADS_PER_BLOCK, 0, s>>>(
                    (CUDASnchNode*)d_nodes, (CUDATriangle*)d_primitives,
                    (CUDAEdge*)d_silhouettes,
                    (CUDABoundingSphere*)d_boundingSpheres,
                    (uint32_t*)d_flipNormalOrientation,
                    (CUDAInteraction*)d_interactions,
                    squaredMinRadius, precision, nQueries);
            break;
    }
}

void launchTransformedRayIntersectionKernel(int bvhType,
                                            void* d_nodes, void* d_primitives, void* d_silhouettes,
                                            void* d_transform,
                                            void* d_rays, void* d_interactions,
                                            uint32_t checkForOcclusion, uint32_t nQueries,
                                            void* stream)
{
    uint32_t numBlocks = (nQueries + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaStream_t s = static_cast<cudaStream_t>(stream);

    switch (bvhType) {
        case FCPW_CUDA_LINE_SEGMENT_BVH:
            transformedRayIntersectionKernel<CUDABvhNode, CUDALineSegment, CUDANoSilhouette>
                <<<numBlocks, THREADS_PER_BLOCK, 0, s>>>(
                    (CUDABvhNode*)d_nodes, (CUDALineSegment*)d_primitives,
                    (CUDANoSilhouette*)d_silhouettes, (CUDATransform*)d_transform,
                    (CUDARay*)d_rays, (CUDAInteraction*)d_interactions,
                    checkForOcclusion, nQueries);
            break;
        case FCPW_CUDA_TRIANGLE_BVH:
            transformedRayIntersectionKernel<CUDABvhNode, CUDATriangle, CUDANoSilhouette>
                <<<numBlocks, THREADS_PER_BLOCK, 0, s>>>(
                    (CUDABvhNode*)d_nodes, (CUDATriangle*)d_primitives,
                    (CUDANoSilhouette*)d_silhouettes, (CUDATransform*)d_transform,
                    (CUDARay*)d_rays, (CUDAInteraction*)d_interactions,
                    checkForOcclusion, nQueries);
            break;
        case FCPW_CUDA_LINE_SEGMENT_SNCH:
            transformedRayIntersectionKernel<CUDASnchNode, CUDALineSegment, CUDAVertex>
                <<<numBlocks, THREADS_PER_BLOCK, 0, s>>>(
                    (CUDASnchNode*)d_nodes, (CUDALineSegment*)d_primitives,
                    (CUDAVertex*)d_silhouettes, (CUDATransform*)d_transform,
                    (CUDARay*)d_rays, (CUDAInteraction*)d_interactions,
                    checkForOcclusion, nQueries);
            break;
        case FCPW_CUDA_TRIANGLE_SNCH:
            transformedRayIntersectionKernel<CUDASnchNode, CUDATriangle, CUDAEdge>
                <<<numBlocks, THREADS_PER_BLOCK, 0, s>>>(
                    (CUDASnchNode*)d_nodes, (CUDATriangle*)d_primitives,
                    (CUDAEdge*)d_silhouettes, (CUDATransform*)d_transform,
                    (CUDARay*)d_rays, (CUDAInteraction*)d_interactions,
                    checkForOcclusion, nQueries);
            break;
    }
}

void launchTransformedSphereIntersectionKernel(int bvhType,
                                               void* d_nodes, void* d_primitives, void* d_silhouettes,
                                               void* d_transform,
                                               void* d_boundingSpheres, void* d_randNums,
                                               void* d_interactions,
                                               uint32_t nQueries,
                                               void* stream)
{
    uint32_t numBlocks = (nQueries + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaStream_t s = static_cast<cudaStream_t>(stream);

    switch (bvhType) {
        case FCPW_CUDA_LINE_SEGMENT_BVH:
            transformedSphereIntersectionKernel<CUDABvhNode, CUDALineSegment, CUDANoSilhouette>
                <<<numBlocks, THREADS_PER_BLOCK, 0, s>>>(
                    (CUDABvhNode*)d_nodes, (CUDALineSegment*)d_primitives,
                    (CUDANoSilhouette*)d_silhouettes, (CUDATransform*)d_transform,
                    (CUDABoundingSphere*)d_boundingSpheres, (CUDAFloat3*)d_randNums,
                    (CUDAInteraction*)d_interactions, nQueries);
            break;
        case FCPW_CUDA_TRIANGLE_BVH:
            transformedSphereIntersectionKernel<CUDABvhNode, CUDATriangle, CUDANoSilhouette>
                <<<numBlocks, THREADS_PER_BLOCK, 0, s>>>(
                    (CUDABvhNode*)d_nodes, (CUDATriangle*)d_primitives,
                    (CUDANoSilhouette*)d_silhouettes, (CUDATransform*)d_transform,
                    (CUDABoundingSphere*)d_boundingSpheres, (CUDAFloat3*)d_randNums,
                    (CUDAInteraction*)d_interactions, nQueries);
            break;
        case FCPW_CUDA_LINE_SEGMENT_SNCH:
            transformedSphereIntersectionKernel<CUDASnchNode, CUDALineSegment, CUDAVertex>
                <<<numBlocks, THREADS_PER_BLOCK, 0, s>>>(
                    (CUDASnchNode*)d_nodes, (CUDALineSegment*)d_primitives,
                    (CUDAVertex*)d_silhouettes, (CUDATransform*)d_transform,
                    (CUDABoundingSphere*)d_boundingSpheres, (CUDAFloat3*)d_randNums,
                    (CUDAInteraction*)d_interactions, nQueries);
            break;
        case FCPW_CUDA_TRIANGLE_SNCH:
            transformedSphereIntersectionKernel<CUDASnchNode, CUDATriangle, CUDAEdge>
                <<<numBlocks, THREADS_PER_BLOCK, 0, s>>>(
                    (CUDASnchNode*)d_nodes, (CUDATriangle*)d_primitives,
                    (CUDAEdge*)d_silhouettes, (CUDATransform*)d_transform,
                    (CUDABoundingSphere*)d_boundingSpheres, (CUDAFloat3*)d_randNums,
                    (CUDAInteraction*)d_interactions, nQueries);
            break;
    }
}

void launchTransformedClosestPointKernel(int bvhType,
                                         void* d_nodes, void* d_primitives, void* d_silhouettes,
                                         void* d_transform,
                                         void* d_boundingSpheres, void* d_interactions,
                                         uint32_t recordNormals, uint32_t nQueries,
                                         void* stream)
{
    uint32_t numBlocks = (nQueries + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaStream_t s = static_cast<cudaStream_t>(stream);

    switch (bvhType) {
        case FCPW_CUDA_LINE_SEGMENT_BVH:
            transformedClosestPointKernel<CUDABvhNode, CUDALineSegment, CUDANoSilhouette>
                <<<numBlocks, THREADS_PER_BLOCK, 0, s>>>(
                    (CUDABvhNode*)d_nodes, (CUDALineSegment*)d_primitives,
                    (CUDANoSilhouette*)d_silhouettes, (CUDATransform*)d_transform,
                    (CUDABoundingSphere*)d_boundingSpheres,
                    (CUDAInteraction*)d_interactions, recordNormals, nQueries);
            break;
        case FCPW_CUDA_TRIANGLE_BVH:
            transformedClosestPointKernel<CUDABvhNode, CUDATriangle, CUDANoSilhouette>
                <<<numBlocks, THREADS_PER_BLOCK, 0, s>>>(
                    (CUDABvhNode*)d_nodes, (CUDATriangle*)d_primitives,
                    (CUDANoSilhouette*)d_silhouettes, (CUDATransform*)d_transform,
                    (CUDABoundingSphere*)d_boundingSpheres,
                    (CUDAInteraction*)d_interactions, recordNormals, nQueries);
            break;
        case FCPW_CUDA_LINE_SEGMENT_SNCH:
            transformedClosestPointKernel<CUDASnchNode, CUDALineSegment, CUDAVertex>
                <<<numBlocks, THREADS_PER_BLOCK, 0, s>>>(
                    (CUDASnchNode*)d_nodes, (CUDALineSegment*)d_primitives,
                    (CUDAVertex*)d_silhouettes, (CUDATransform*)d_transform,
                    (CUDABoundingSphere*)d_boundingSpheres,
                    (CUDAInteraction*)d_interactions, recordNormals, nQueries);
            break;
        case FCPW_CUDA_TRIANGLE_SNCH:
            transformedClosestPointKernel<CUDASnchNode, CUDATriangle, CUDAEdge>
                <<<numBlocks, THREADS_PER_BLOCK, 0, s>>>(
                    (CUDASnchNode*)d_nodes, (CUDATriangle*)d_primitives,
                    (CUDAEdge*)d_silhouettes, (CUDATransform*)d_transform,
                    (CUDABoundingSphere*)d_boundingSpheres,
                    (CUDAInteraction*)d_interactions, recordNormals, nQueries);
            break;
    }
}

void launchTransformedClosestSilhouettePointKernel(int bvhType,
                                                    void* d_nodes, void* d_primitives, void* d_silhouettes,
                                                    void* d_transform,
                                                    void* d_boundingSpheres, void* d_flipNormalOrientation,
                                                    void* d_interactions,
                                                    float squaredMinRadius, float precision,
                                                    uint32_t nQueries,
                                                    void* stream)
{
    uint32_t numBlocks = (nQueries + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaStream_t s = static_cast<cudaStream_t>(stream);

    switch (bvhType) {
        case FCPW_CUDA_LINE_SEGMENT_BVH:
            // No silhouettes for BVH types - no-op
            break;
        case FCPW_CUDA_TRIANGLE_BVH:
            // No silhouettes for BVH types - no-op
            break;
        case FCPW_CUDA_LINE_SEGMENT_SNCH:
            transformedClosestSilhouettePointKernel<CUDASnchNode, CUDALineSegment, CUDAVertex>
                <<<numBlocks, THREADS_PER_BLOCK, 0, s>>>(
                    (CUDASnchNode*)d_nodes, (CUDALineSegment*)d_primitives,
                    (CUDAVertex*)d_silhouettes, (CUDATransform*)d_transform,
                    (CUDABoundingSphere*)d_boundingSpheres,
                    (uint32_t*)d_flipNormalOrientation,
                    (CUDAInteraction*)d_interactions,
                    squaredMinRadius, precision, nQueries);
            break;
        case FCPW_CUDA_TRIANGLE_SNCH:
            transformedClosestSilhouettePointKernel<CUDASnchNode, CUDATriangle, CUDAEdge>
                <<<numBlocks, THREADS_PER_BLOCK, 0, s>>>(
                    (CUDASnchNode*)d_nodes, (CUDATriangle*)d_primitives,
                    (CUDAEdge*)d_silhouettes, (CUDATransform*)d_transform,
                    (CUDABoundingSphere*)d_boundingSpheres,
                    (uint32_t*)d_flipNormalOrientation,
                    (CUDAInteraction*)d_interactions,
                    squaredMinRadius, precision, nQueries);
            break;
    }
}

void launchRefitKernel(int bvhType,
                       void* d_nodes, void* d_primitives, void* d_silhouettes,
                       uint32_t* d_nodeIndices,
                       uint32_t firstNodeOffset, uint32_t nodeCount,
                       void* stream)
{
    uint32_t numBlocks = (nodeCount + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaStream_t s = static_cast<cudaStream_t>(stream);

    // dispatch to correct template based on BVH type
    switch (bvhType) {
        case FCPW_CUDA_LINE_SEGMENT_BVH:
            refitKernel<CUDABvhNode, CUDALineSegment, CUDANoSilhouette>
                <<<numBlocks, THREADS_PER_BLOCK, 0, s>>>(
                    (CUDABvhNode*)d_nodes, (CUDALineSegment*)d_primitives,
                    (CUDANoSilhouette*)d_silhouettes,
                    d_nodeIndices, firstNodeOffset, nodeCount);
            break;
        case FCPW_CUDA_TRIANGLE_BVH:
            refitKernel<CUDABvhNode, CUDATriangle, CUDANoSilhouette>
                <<<numBlocks, THREADS_PER_BLOCK, 0, s>>>(
                    (CUDABvhNode*)d_nodes, (CUDATriangle*)d_primitives,
                    (CUDANoSilhouette*)d_silhouettes,
                    d_nodeIndices, firstNodeOffset, nodeCount);
            break;
        case FCPW_CUDA_LINE_SEGMENT_SNCH:
            refitKernel<CUDASnchNode, CUDALineSegment, CUDAVertex>
                <<<numBlocks, THREADS_PER_BLOCK, 0, s>>>(
                    (CUDASnchNode*)d_nodes, (CUDALineSegment*)d_primitives,
                    (CUDAVertex*)d_silhouettes,
                    d_nodeIndices, firstNodeOffset, nodeCount);
            break;
        case FCPW_CUDA_TRIANGLE_SNCH:
            refitKernel<CUDASnchNode, CUDATriangle, CUDAEdge>
                <<<numBlocks, THREADS_PER_BLOCK, 0, s>>>(
                    (CUDASnchNode*)d_nodes, (CUDATriangle*)d_primitives,
                    (CUDAEdge*)d_silhouettes,
                    d_nodeIndices, firstNodeOffset, nodeCount);
            break;
    }
}

} // namespace fcpw
