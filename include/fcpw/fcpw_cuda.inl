#include <fcpw/utilities/scene_data.h>
#include "oneapi/tbb/parallel_for.h"
#include "oneapi/tbb/blocked_range.h"

namespace fcpw {

// Forward declarations of kernel launchers (implementations in cuda_bvh_kernels.cu)
namespace cuda {
    // Type 1: Line Segment BVH
    void launchRayIntersectionLineSegmentBvh(const GPUBvhNode*, const GPULineSegment*,
        const GPURay*, GPUInteraction*, uint32_t, uint32_t, const KernelLaunchConfig&);
    void launchSphereIntersectionLineSegmentBvh(const GPUBvhNode*, const GPULineSegment*,
        const GPUBoundingSphere*, const float3*, GPUInteraction*, uint32_t, const KernelLaunchConfig&);
    void launchClosestPointLineSegmentBvh(const GPUBvhNode*, const GPULineSegment*,
        const GPUBoundingSphere*, GPUInteraction*, uint32_t, uint32_t, const KernelLaunchConfig&);
    void launchRefitLineSegmentBvh(GPUBvhNode*, const GPULineSegment*, const GPUNoSilhouette*,
        const uint32_t*, uint32_t, uint32_t, const KernelLaunchConfig&);

    // Type 2: Triangle BVH
    void launchRayIntersectionTriangleBvh(const GPUBvhNode*, const GPUTriangle*,
        const GPURay*, GPUInteraction*, uint32_t, uint32_t, const KernelLaunchConfig&);
    void launchSphereIntersectionTriangleBvh(const GPUBvhNode*, const GPUTriangle*,
        const GPUBoundingSphere*, const float3*, GPUInteraction*, uint32_t, const KernelLaunchConfig&);
    void launchClosestPointTriangleBvh(const GPUBvhNode*, const GPUTriangle*,
        const GPUBoundingSphere*, GPUInteraction*, uint32_t, uint32_t, const KernelLaunchConfig&);
    void launchRefitTriangleBvh(GPUBvhNode*, const GPUTriangle*, const GPUNoSilhouette*,
        const uint32_t*, uint32_t, uint32_t, const KernelLaunchConfig&);

    // Type 3: Line Segment SNCH
    void launchRayIntersectionLineSegmentSnch(const GPUSnchNode*, const GPULineSegment*,
        const GPURay*, GPUInteraction*, uint32_t, uint32_t, const KernelLaunchConfig&);
    void launchSphereIntersectionLineSegmentSnch(const GPUSnchNode*, const GPULineSegment*,
        const GPUBoundingSphere*, const float3*, GPUInteraction*, uint32_t, const KernelLaunchConfig&);
    void launchClosestPointLineSegmentSnch(const GPUSnchNode*, const GPULineSegment*,
        const GPUBoundingSphere*, GPUInteraction*, uint32_t, uint32_t, const KernelLaunchConfig&);
    void launchClosestSilhouettePointLineSegmentSnch(const GPUSnchNode*, const GPULineSegment*, const GPUVertex*,
        const GPUBoundingSphere*, const uint32_t*, GPUInteraction*, float, float, uint32_t, const KernelLaunchConfig&);
    void launchRefitLineSegmentSnch(GPUSnchNode*, const GPULineSegment*, const GPUVertex*,
        const uint32_t*, uint32_t, uint32_t, const KernelLaunchConfig&);

    // Type 4: Triangle SNCH
    void launchRayIntersectionTriangleSnch(const GPUSnchNode*, const GPUTriangle*,
        const GPURay*, GPUInteraction*, uint32_t, uint32_t, const KernelLaunchConfig&);
    void launchSphereIntersectionTriangleSnch(const GPUSnchNode*, const GPUTriangle*,
        const GPUBoundingSphere*, const float3*, GPUInteraction*, uint32_t, const KernelLaunchConfig&);
    void launchClosestPointTriangleSnch(const GPUSnchNode*, const GPUTriangle*,
        const GPUBoundingSphere*, GPUInteraction*, uint32_t, uint32_t, const KernelLaunchConfig&);
    void launchClosestSilhouettePointTriangleSnch(const GPUSnchNode*, const GPUTriangle*, const GPUEdge*,
        const GPUBoundingSphere*, const uint32_t*, GPUInteraction*, float, float, uint32_t, const KernelLaunchConfig&);
    void launchRefitTriangleSnch(GPUSnchNode*, const GPUTriangle*, const GPUEdge*,
        const uint32_t*, uint32_t, uint32_t, const KernelLaunchConfig&);
}

/////////////////////////////////////////////////////////////////////////////////////////////
// CUDAScene implementation

template<size_t DIM>
inline CUDAScene<DIM>::CUDAScene(bool printLogs_):
nThreadsPerBlock(256),
printLogs(printLogs_),
bvhType(CUDA_UNDEFINED_BVH),
context(printLogs_)
{
    // constructor
}

template<size_t DIM>
inline void CUDAScene<DIM>::transferToGPU(Scene<DIM>& scene)
{
    SceneData<DIM>* sceneData = scene.getSceneData();
    bool hasLineSegmentGeometry = sceneData->lineSegmentObjects.size() > 0;
    bool hasSilhouetteGeometry = sceneData->silhouetteVertexObjects.size() > 0 ||
                                 sceneData->silhouetteEdgeObjects.size() > 0;

    // Determine BVH type
    if (hasSilhouetteGeometry) {
        bvhType = hasLineSegmentGeometry ? CUDA_LINE_SEGMENT_SNCH : CUDA_TRIANGLE_SNCH;
    } else {
        bvhType = hasLineSegmentGeometry ? CUDA_LINE_SEGMENT_BVH : CUDA_TRIANGLE_BVH;
    }

    if (printLogs) {
        std::cout << "CUDA BVH Type: " << bvhType << std::endl;
    }

    // Initialize CUDA context
    context.initialize();

    // Allocate and upload BVH buffers
    bvhBuffers.allocate<DIM>(sceneData, true, hasSilhouetteGeometry, false);

    if (printLogs) {
        std::cout << "Transferred scene to CUDA GPU" << std::endl;
    }
}

template<size_t DIM>
inline void CUDAScene<DIM>::refit(Scene<DIM>& scene, bool updateGeometry)
{
    SceneData<DIM>* sceneData = scene.getSceneData();
    bool hasSilhouetteGeometry = sceneData->silhouetteVertexObjects.size() > 0 ||
                                 sceneData->silhouetteEdgeObjects.size() > 0;
    bool allocateSilhouetteGeometry = hasSilhouetteGeometry && updateGeometry;
    bool allocateRefitData = bvhBuffers.refitNodeIndices.isEmpty();

    // Update GPU buffers (geometry and silhouettes if updateGeometry is true)
    bvhBuffers.allocate<DIM>(sceneData, updateGeometry, allocateSilhouetteGeometry,
                             allocateRefitData);

    // Dispatch refit kernel if we have nodes to refit
    if (!bvhBuffers.refitNodeIndices.isEmpty()) {
        uint32_t nodeCount = bvhBuffers.refitNodeIndices.size();
        uint32_t firstNodeOffset = 0;

        // Compute launch configuration
        cuda::KernelLaunchConfig config = cuda::KernelLaunchConfig::compute(nodeCount, nThreadsPerBlock);

        // Launch kernel based on BVH type
        cuda::CUDATimer timer;
        if (printLogs) {
            std::cout << "Refitting " << nodeCount << " BVH nodes on GPU..." << std::endl;
            timer.start();
        }

        switch (bvhType) {
            case CUDA_LINE_SEGMENT_BVH:
                cuda::launchRefitLineSegmentBvh(
                    bvhBuffers.bvhNodes.devicePtr(), bvhBuffers.lineSegments.devicePtr(),
                    bvhBuffers.noSilhouettes.devicePtr(),
                    bvhBuffers.refitNodeIndices.devicePtr(),
                    firstNodeOffset, nodeCount, config);
                break;
            case CUDA_TRIANGLE_BVH:
                cuda::launchRefitTriangleBvh(
                    bvhBuffers.bvhNodes.devicePtr(), bvhBuffers.triangles.devicePtr(),
                    bvhBuffers.noSilhouettes.devicePtr(),
                    bvhBuffers.refitNodeIndices.devicePtr(),
                    firstNodeOffset, nodeCount, config);
                break;
            case CUDA_LINE_SEGMENT_SNCH:
                cuda::launchRefitLineSegmentSnch(
                    bvhBuffers.snchNodes.devicePtr(), bvhBuffers.lineSegments.devicePtr(),
                    bvhBuffers.vertices.devicePtr(),
                    bvhBuffers.refitNodeIndices.devicePtr(),
                    firstNodeOffset, nodeCount, config);
                break;
            case CUDA_TRIANGLE_SNCH:
                cuda::launchRefitTriangleSnch(
                    bvhBuffers.snchNodes.devicePtr(), bvhBuffers.triangles.devicePtr(),
                    bvhBuffers.edges.devicePtr(),
                    bvhBuffers.refitNodeIndices.devicePtr(),
                    firstNodeOffset, nodeCount, config);
                break;
            default:
                std::cerr << "CUDAScene::refit: Unknown BVH type" << std::endl;
                return;
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        if (printLogs) {
            timer.stop();
            std::cout << "GPU refit took " << timer.elapsedMilliseconds() << " ms" << std::endl;
        }

        // Verify refit didn't produce errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error after refit: " << cudaGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }
    }
}

template<size_t DIM>
inline void CUDAScene<DIM>::intersect(const Eigen::MatrixXf& rayOrigins,
                                      const Eigen::MatrixXf& rayDirections,
                                      const Eigen::VectorXf& rayDistanceBounds,
                                      std::vector<GPUInteraction>& interactions,
                                      bool checkForOcclusion)
{
    // Convert Eigen matrices to GPURay vector
    uint32_t nQueries = rayOrigins.cols();
    std::vector<GPURay> rays(nQueries);

    auto convertRays = [&](const tbb::blocked_range<int>& range) {
        for (int i = range.begin(); i < range.end(); ++i) {
            float3 origin = float3{rayOrigins(0, i), rayOrigins(1, i),
                                   DIM == 2 ? 0.0f : rayOrigins(2, i)};
            float3 direction = float3{rayDirections(0, i), rayDirections(1, i),
                                     DIM == 2 ? 0.0f : rayDirections(2, i)};
            float tMax = rayDistanceBounds(i);

            rays[i] = GPURay(origin, direction);
            rays[i].tMax = tMax;
        }
    };

    tbb::blocked_range<int> range(0, nQueries);
    tbb::parallel_for(range, convertRays);

    intersect(rays, interactions, checkForOcclusion);
}

template<size_t DIM>
inline void CUDAScene<DIM>::intersect(const std::vector<GPURay>& rays,
                                      std::vector<GPUInteraction>& interactions,
                                      bool checkForOcclusion)
{
    uint32_t nQueries = rays.size();
    interactions.resize(nQueries);

    // Initialize interactions
    for (uint32_t i = 0; i < nQueries; i++) {
        interactions[i].index = FCPW_GPU_UINT_MAX;
    }

    // Upload rays to GPU
    bvhBuffers.rays.allocate(nQueries);
    bvhBuffers.rays.upload(rays);

    // Allocate interaction buffer
    bvhBuffers.interactions.allocate(nQueries);
    bvhBuffers.interactions.upload(interactions);

    // Compute launch configuration
    cuda::KernelLaunchConfig config = cuda::KernelLaunchConfig::compute(nQueries, nThreadsPerBlock);

    // Launch kernel based on BVH type
    cuda::CUDATimer timer;
    if (printLogs) timer.start();

    uint32_t checkOcclusion = checkForOcclusion ? 1 : 0;

    switch (bvhType) {
        case CUDA_LINE_SEGMENT_BVH:
            cuda::launchRayIntersectionLineSegmentBvh(
                bvhBuffers.bvhNodes.devicePtr(), bvhBuffers.lineSegments.devicePtr(),
                bvhBuffers.rays.devicePtr(), bvhBuffers.interactions.devicePtr(),
                checkOcclusion, nQueries, config);
            break;
        case CUDA_TRIANGLE_BVH:
            cuda::launchRayIntersectionTriangleBvh(
                bvhBuffers.bvhNodes.devicePtr(), bvhBuffers.triangles.devicePtr(),
                bvhBuffers.rays.devicePtr(), bvhBuffers.interactions.devicePtr(),
                checkOcclusion, nQueries, config);
            break;
        case CUDA_LINE_SEGMENT_SNCH:
            cuda::launchRayIntersectionLineSegmentSnch(
                bvhBuffers.snchNodes.devicePtr(), bvhBuffers.lineSegments.devicePtr(),
                bvhBuffers.rays.devicePtr(), bvhBuffers.interactions.devicePtr(),
                checkOcclusion, nQueries, config);
            break;
        case CUDA_TRIANGLE_SNCH:
            cuda::launchRayIntersectionTriangleSnch(
                bvhBuffers.snchNodes.devicePtr(), bvhBuffers.triangles.devicePtr(),
                bvhBuffers.rays.devicePtr(), bvhBuffers.interactions.devicePtr(),
                checkOcclusion, nQueries, config);
            break;
        default:
            std::cerr << "Unknown BVH type: " << bvhType << std::endl;
            return;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    if (printLogs) {
        timer.stop();
        std::cout << nQueries << " ray intersection queries took "
                  << timer.elapsedMilliseconds() << " ms (CUDA)" << std::endl;
    }

    // Download results
    bvhBuffers.interactions.download(interactions);
}

template<size_t DIM>
inline void CUDAScene<DIM>::intersect(const Eigen::MatrixXf& sphereCenters,
                                      const Eigen::VectorXf& sphereSquaredRadii,
                                      const Eigen::MatrixXf& randNums,
                                      std::vector<GPUInteraction>& interactions)
{
    // Convert Eigen matrices to GPU structures
    uint32_t nQueries = sphereCenters.cols();
    std::vector<GPUBoundingSphere> spheres(nQueries);
    std::vector<float3> randNumsVec(nQueries);

    auto convertData = [&](const tbb::blocked_range<int>& range) {
        for (int i = range.begin(); i < range.end(); ++i) {
            float3 center = float3{sphereCenters(0, i), sphereCenters(1, i),
                                   DIM == 2 ? 0.0f : sphereCenters(2, i)};
            float r2 = sphereSquaredRadii(i);
            spheres[i] = GPUBoundingSphere(center, r2);

            randNumsVec[i] = float3{randNums(0, i), randNums(1, i),
                                    DIM == 2 ? 0.0f : randNums(2, i)};
        }
    };

    tbb::blocked_range<int> range(0, nQueries);
    tbb::parallel_for(range, convertData);

    intersect(spheres, randNumsVec, interactions);
}

template<size_t DIM>
inline void CUDAScene<DIM>::intersect(const std::vector<GPUBoundingSphere>& boundingSpheres,
                                      const std::vector<float3>& randNums,
                                      std::vector<GPUInteraction>& interactions)
{
    uint32_t nQueries = boundingSpheres.size();
    interactions.resize(nQueries);

    // Initialize interactions
    for (uint32_t i = 0; i < nQueries; i++) {
        interactions[i].index = FCPW_GPU_UINT_MAX;
    }

    // Upload data to GPU
    bvhBuffers.boundingSpheres.allocate(nQueries);
    bvhBuffers.boundingSpheres.upload(boundingSpheres);

    bvhBuffers.randNums.allocate(nQueries);
    bvhBuffers.randNums.upload(randNums);

    bvhBuffers.interactions.allocate(nQueries);
    bvhBuffers.interactions.upload(interactions);

    // Compute launch configuration
    cuda::KernelLaunchConfig config = cuda::KernelLaunchConfig::compute(nQueries, nThreadsPerBlock);

    // Launch kernel
    cuda::CUDATimer timer;
    if (printLogs) timer.start();

    switch (bvhType) {
        case CUDA_LINE_SEGMENT_BVH:
            cuda::launchSphereIntersectionLineSegmentBvh(
                bvhBuffers.bvhNodes.devicePtr(), bvhBuffers.lineSegments.devicePtr(),
                bvhBuffers.boundingSpheres.devicePtr(), bvhBuffers.randNums.devicePtr(),
                bvhBuffers.interactions.devicePtr(), nQueries, config);
            break;
        case CUDA_TRIANGLE_BVH:
            cuda::launchSphereIntersectionTriangleBvh(
                bvhBuffers.bvhNodes.devicePtr(), bvhBuffers.triangles.devicePtr(),
                bvhBuffers.boundingSpheres.devicePtr(), bvhBuffers.randNums.devicePtr(),
                bvhBuffers.interactions.devicePtr(), nQueries, config);
            break;
        case CUDA_LINE_SEGMENT_SNCH:
            cuda::launchSphereIntersectionLineSegmentSnch(
                bvhBuffers.snchNodes.devicePtr(), bvhBuffers.lineSegments.devicePtr(),
                bvhBuffers.boundingSpheres.devicePtr(), bvhBuffers.randNums.devicePtr(),
                bvhBuffers.interactions.devicePtr(), nQueries, config);
            break;
        case CUDA_TRIANGLE_SNCH:
            cuda::launchSphereIntersectionTriangleSnch(
                bvhBuffers.snchNodes.devicePtr(), bvhBuffers.triangles.devicePtr(),
                bvhBuffers.boundingSpheres.devicePtr(), bvhBuffers.randNums.devicePtr(),
                bvhBuffers.interactions.devicePtr(), nQueries, config);
            break;
        default:
            std::cerr << "Unknown BVH type: " << bvhType << std::endl;
            return;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    if (printLogs) {
        timer.stop();
        std::cout << nQueries << " sphere intersection queries took "
                  << timer.elapsedMilliseconds() << " ms (CUDA)" << std::endl;
    }

    // Download results
    bvhBuffers.interactions.download(interactions);
}

template<size_t DIM>
inline void CUDAScene<DIM>::findClosestPoints(const Eigen::MatrixXf& queryPoints,
                                               const Eigen::VectorXf& squaredMaxRadii,
                                               std::vector<GPUInteraction>& interactions,
                                               bool recordNormals)
{
    // Convert Eigen matrices to GPUBoundingSphere vector
    uint32_t nQueries = queryPoints.cols();
    std::vector<GPUBoundingSphere> spheres(nQueries);

    auto convertData = [&](const tbb::blocked_range<int>& range) {
        for (int i = range.begin(); i < range.end(); ++i) {
            float3 center = float3{queryPoints(0, i), queryPoints(1, i),
                                   DIM == 2 ? 0.0f : queryPoints(2, i)};
            float r2 = squaredMaxRadii(i);
            spheres[i] = GPUBoundingSphere(center, r2);
        }
    };

    tbb::blocked_range<int> range(0, nQueries);
    tbb::parallel_for(range, convertData);

    findClosestPoints(spheres, interactions, recordNormals);
}

template<size_t DIM>
inline void CUDAScene<DIM>::findClosestPoints(const std::vector<GPUBoundingSphere>& boundingSpheres,
                                               std::vector<GPUInteraction>& interactions,
                                               bool recordNormals)
{
    uint32_t nQueries = boundingSpheres.size();
    interactions.resize(nQueries);

    // Initialize interactions
    for (uint32_t i = 0; i < nQueries; i++) {
        interactions[i].index = FCPW_GPU_UINT_MAX;
    }

    // Upload data to GPU
    bvhBuffers.boundingSpheres.allocate(nQueries);
    bvhBuffers.boundingSpheres.upload(boundingSpheres);

    bvhBuffers.interactions.allocate(nQueries);
    bvhBuffers.interactions.upload(interactions);

    // Compute launch configuration
    cuda::KernelLaunchConfig config = cuda::KernelLaunchConfig::compute(nQueries, nThreadsPerBlock);

    // Launch kernel
    cuda::CUDATimer timer;
    if (printLogs) timer.start();

    uint32_t recordNormalsFlag = recordNormals ? 1 : 0;

    switch (bvhType) {
        case CUDA_LINE_SEGMENT_BVH:
            cuda::launchClosestPointLineSegmentBvh(
                bvhBuffers.bvhNodes.devicePtr(), bvhBuffers.lineSegments.devicePtr(),
                bvhBuffers.boundingSpheres.devicePtr(), bvhBuffers.interactions.devicePtr(),
                recordNormalsFlag, nQueries, config);
            break;
        case CUDA_TRIANGLE_BVH:
            cuda::launchClosestPointTriangleBvh(
                bvhBuffers.bvhNodes.devicePtr(), bvhBuffers.triangles.devicePtr(),
                bvhBuffers.boundingSpheres.devicePtr(), bvhBuffers.interactions.devicePtr(),
                recordNormalsFlag, nQueries, config);
            break;
        case CUDA_LINE_SEGMENT_SNCH:
            cuda::launchClosestPointLineSegmentSnch(
                bvhBuffers.snchNodes.devicePtr(), bvhBuffers.lineSegments.devicePtr(),
                bvhBuffers.boundingSpheres.devicePtr(), bvhBuffers.interactions.devicePtr(),
                recordNormalsFlag, nQueries, config);
            break;
        case CUDA_TRIANGLE_SNCH:
            cuda::launchClosestPointTriangleSnch(
                bvhBuffers.snchNodes.devicePtr(), bvhBuffers.triangles.devicePtr(),
                bvhBuffers.boundingSpheres.devicePtr(), bvhBuffers.interactions.devicePtr(),
                recordNormalsFlag, nQueries, config);
            break;
        default:
            std::cerr << "Unknown BVH type: " << bvhType << std::endl;
            return;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    if (printLogs) {
        timer.stop();
        std::cout << nQueries << " closest point queries took "
                  << timer.elapsedMilliseconds() << " ms (CUDA)" << std::endl;
    }

    // Download results
    bvhBuffers.interactions.download(interactions);
}

template<size_t DIM>
inline void CUDAScene<DIM>::findClosestSilhouettePoints(
    const Eigen::MatrixXf& queryPoints,
    const Eigen::VectorXf& squaredMaxRadii,
    const Eigen::VectorXi& flipNormalOrientation,
    std::vector<GPUInteraction>& interactions,
    float squaredMinRadius, float precision)
{
    // Convert Eigen matrices to GPU structures
    uint32_t nQueries = queryPoints.cols();
    std::vector<GPUBoundingSphere> spheres(nQueries);
    std::vector<uint32_t> flipNormals(nQueries);

    auto convertData = [&](const tbb::blocked_range<int>& range) {
        for (int i = range.begin(); i < range.end(); ++i) {
            float3 center = float3{queryPoints(0, i), queryPoints(1, i),
                                   DIM == 2 ? 0.0f : queryPoints(2, i)};
            float r2 = squaredMaxRadii(i);
            spheres[i] = GPUBoundingSphere(center, r2);
            flipNormals[i] = flipNormalOrientation(i);
        }
    };

    tbb::blocked_range<int> range(0, nQueries);
    tbb::parallel_for(range, convertData);

    findClosestSilhouettePoints(spheres, flipNormals, interactions,
                               squaredMinRadius, precision);
}

template<size_t DIM>
inline void CUDAScene<DIM>::findClosestSilhouettePoints(
    const std::vector<GPUBoundingSphere>& boundingSpheres,
    const std::vector<uint32_t>& flipNormalOrientation,
    std::vector<GPUInteraction>& interactions,
    float squaredMinRadius, float precision)
{
    uint32_t nQueries = boundingSpheres.size();
    interactions.resize(nQueries);

    // Initialize interactions
    for (uint32_t i = 0; i < nQueries; i++) {
        interactions[i].index = FCPW_GPU_UINT_MAX;
    }

    // Upload data to GPU
    bvhBuffers.boundingSpheres.allocate(nQueries);
    bvhBuffers.boundingSpheres.upload(boundingSpheres);

    bvhBuffers.flipNormalOrientation.allocate(nQueries);
    bvhBuffers.flipNormalOrientation.upload(flipNormalOrientation);

    bvhBuffers.interactions.allocate(nQueries);
    bvhBuffers.interactions.upload(interactions);

    // Compute launch configuration
    cuda::KernelLaunchConfig config = cuda::KernelLaunchConfig::compute(nQueries, nThreadsPerBlock);

    // Launch kernel
    cuda::CUDATimer timer;
    if (printLogs) timer.start();

    switch (bvhType) {
        case CUDA_LINE_SEGMENT_SNCH:
            cuda::launchClosestSilhouettePointLineSegmentSnch(
                bvhBuffers.snchNodes.devicePtr(), bvhBuffers.lineSegments.devicePtr(),
                bvhBuffers.vertices.devicePtr(), bvhBuffers.boundingSpheres.devicePtr(),
                bvhBuffers.flipNormalOrientation.devicePtr(), bvhBuffers.interactions.devicePtr(),
                squaredMinRadius, precision, nQueries, config);
            break;
        case CUDA_TRIANGLE_SNCH:
            cuda::launchClosestSilhouettePointTriangleSnch(
                bvhBuffers.snchNodes.devicePtr(), bvhBuffers.triangles.devicePtr(),
                bvhBuffers.edges.devicePtr(), bvhBuffers.boundingSpheres.devicePtr(),
                bvhBuffers.flipNormalOrientation.devicePtr(), bvhBuffers.interactions.devicePtr(),
                squaredMinRadius, precision, nQueries, config);
            break;
        default:
            std::cerr << "Silhouette queries only supported for SNCH BVH types" << std::endl;
            return;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    if (printLogs) {
        timer.stop();
        std::cout << nQueries << " closest silhouette point queries took "
                  << timer.elapsedMilliseconds() << " ms (CUDA)" << std::endl;
    }

    // Download results
    bvhBuffers.interactions.download(interactions);
}

} // namespace fcpw
