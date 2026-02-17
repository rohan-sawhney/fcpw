#include <fcpw/cuda/cuda_kernels.h>
#include <cuda_runtime.h>
#include <thread>

#define FCPW_CUDA_CHECK(call)                                                    \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "  \
                      << cudaGetErrorString(err) << std::endl;                    \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

namespace fcpw {

struct CUDAKernelTimer {
    CUDAKernelTimer(cudaStream_t stream_, bool enabled_): stream(stream_), enabled(enabled_) {
        if (enabled) {
            FCPW_CUDA_CHECK(cudaEventCreate(&start));
            FCPW_CUDA_CHECK(cudaEventCreate(&stop));
            FCPW_CUDA_CHECK(cudaEventRecord(start, stream));
        }
    }

    float elapsed() {
        if (!enabled) return 0.0f;
        FCPW_CUDA_CHECK(cudaEventRecord(stop, stream));
        FCPW_CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        FCPW_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        FCPW_CUDA_CHECK(cudaEventDestroy(start));
        FCPW_CUDA_CHECK(cudaEventDestroy(stop));
        return ms;
    }

    cudaEvent_t start, stop;
    cudaStream_t stream;
    bool enabled;
};

template<size_t DIM>
inline CUDAScene<DIM>::CUDAScene(bool printLogs_):
nThreadsPerGroup(256),
printLogs(printLogs_),
stream(nullptr)
{
    cudaStream_t cudaStream;
    FCPW_CUDA_CHECK(cudaStreamCreate(&cudaStream));
    stream = static_cast<void*>(cudaStream);
}

template<size_t DIM>
inline CUDAScene<DIM>::~CUDAScene()
{
    freeDeviceBuffers();
    if (stream) {
        cudaStreamDestroy(static_cast<cudaStream_t>(stream));
        stream = nullptr;
    }
}

template<size_t DIM>
inline void CUDAScene<DIM>::freeDeviceBuffers()
{
    if (bvhBuffers.d_nodes) { cudaFree(bvhBuffers.d_nodes); bvhBuffers.d_nodes = nullptr; }
    if (bvhBuffers.d_primitives) { cudaFree(bvhBuffers.d_primitives); bvhBuffers.d_primitives = nullptr; }
    if (bvhBuffers.d_silhouettes) { cudaFree(bvhBuffers.d_silhouettes); bvhBuffers.d_silhouettes = nullptr; }
    if (bvhBuffers.d_nodeIndices) { cudaFree(bvhBuffers.d_nodeIndices); bvhBuffers.d_nodeIndices = nullptr; }
    if (bvhBuffers.d_transform) { cudaFree(bvhBuffers.d_transform); bvhBuffers.d_transform = nullptr; }
    bvhBuffers.hasTransform = false;
}

template<size_t DIM>
template<typename NodeType, typename PrimitiveType, typename SilhouetteType,
         typename CUDANodeType, typename CUDAPrimitiveType, typename CUDASilhouetteType>
inline void CUDAScene<DIM>::allocateBuffers(const SceneData<DIM> *cpuSceneData,
                                            bool allocatePrimitiveData,
                                            bool allocateNodeData,
                                            bool allocateRefitData)
{
    // check if the aggregate is a TransformedAggregate and unwrap if so
    const Aggregate<DIM> *rawAggregate = cpuSceneData->aggregate.get();
    const TransformedAggregate<DIM> *transformedAgg =
        dynamic_cast<const TransformedAggregate<DIM> *>(rawAggregate);

    const Bvh<DIM, NodeType, PrimitiveType, SilhouetteType> *bvh = nullptr;
    if (transformedAgg != nullptr) {
        bvh = reinterpret_cast<const Bvh<DIM, NodeType, PrimitiveType, SilhouetteType> *>(
            transformedAgg->getAggregate().get());

        // extract and upload transform
        if (!bvhBuffers.hasTransform) {
            CUDATransform ct = extractCUDATransform<DIM>(transformedAgg->getTransform(),
                                                          transformedAgg->getInverseTransform());
            if (bvhBuffers.d_transform) cudaFree(bvhBuffers.d_transform);
            FCPW_CUDA_CHECK(cudaMalloc(&bvhBuffers.d_transform, sizeof(CUDATransform)));
            FCPW_CUDA_CHECK(cudaMemcpy(bvhBuffers.d_transform, &ct, sizeof(CUDATransform),
                                        cudaMemcpyHostToDevice));
            bvhBuffers.hasTransform = true;
        }
    } else {
        bvh = reinterpret_cast<const Bvh<DIM, NodeType, PrimitiveType, SilhouetteType> *>(
            rawAggregate);
    }

    CPUCUDABvhDataExtractor<DIM, NodeType, PrimitiveType, SilhouetteType,
                            CUDANodeType, CUDAPrimitiveType, CUDASilhouetteType> extractor(bvh);

    if (allocatePrimitiveData) {
        // extract and upload primitives
        std::vector<CUDAPrimitiveType> primitivesData;
        extractor.extractPrimitives(primitivesData);
        size_t primSize = primitivesData.size() * sizeof(CUDAPrimitiveType);
        if (bvhBuffers.d_primitives) cudaFree(bvhBuffers.d_primitives);
        if (primitivesData.size() > 0) {
            FCPW_CUDA_CHECK(cudaMalloc(&bvhBuffers.d_primitives, primSize));
            FCPW_CUDA_CHECK(cudaMemcpy(bvhBuffers.d_primitives, primitivesData.data(),
                                        primSize, cudaMemcpyHostToDevice));
        }
        bvhBuffers.primitivesSize = primSize;

        // extract and upload silhouettes
        std::vector<CUDASilhouetteType> silhouettesData;
        extractor.extractSilhouettes(silhouettesData);
        size_t silSize = silhouettesData.size() * sizeof(CUDASilhouetteType);
        if (bvhBuffers.d_silhouettes) cudaFree(bvhBuffers.d_silhouettes);
        if (silhouettesData.size() > 0) {
            FCPW_CUDA_CHECK(cudaMalloc(&bvhBuffers.d_silhouettes, silSize));
            FCPW_CUDA_CHECK(cudaMemcpy(bvhBuffers.d_silhouettes, silhouettesData.data(),
                                        silSize, cudaMemcpyHostToDevice));
        }
        bvhBuffers.silhouettesSize = silSize;
    }

    if (allocateNodeData) {
        // extract and upload nodes
        std::vector<CUDANodeType> nodesData;
        extractor.extractNodes(nodesData);
        size_t nodeSize = nodesData.size() * sizeof(CUDANodeType);
        if (bvhBuffers.d_nodes) cudaFree(bvhBuffers.d_nodes);
        FCPW_CUDA_CHECK(cudaMalloc(&bvhBuffers.d_nodes, nodeSize));
        FCPW_CUDA_CHECK(cudaMemcpy(bvhBuffers.d_nodes, nodesData.data(),
                                    nodeSize, cudaMemcpyHostToDevice));
        bvhBuffers.nodesSize = nodeSize;
        bvhBuffers.bvhType = extractor.getBvhType();
    }

    if (allocateRefitData) {
        CPUCUDABvhUpdateDataExtractor<DIM, NodeType, PrimitiveType, SilhouetteType>
            updateExtractor(bvh);

        bvhBuffers.updateEntryData.clear();
        std::vector<uint32_t> nodeIndicesData;
        bvhBuffers.maxUpdateDepth = updateExtractor.extract(nodeIndicesData, bvhBuffers.updateEntryData);

        size_t indexSize = nodeIndicesData.size() * sizeof(uint32_t);
        if (bvhBuffers.d_nodeIndices) cudaFree(bvhBuffers.d_nodeIndices);
        FCPW_CUDA_CHECK(cudaMalloc((void**)&bvhBuffers.d_nodeIndices, indexSize));
        FCPW_CUDA_CHECK(cudaMemcpy(bvhBuffers.d_nodeIndices, nodeIndicesData.data(),
                                    indexSize, cudaMemcpyHostToDevice));
        bvhBuffers.nodeIndicesSize = indexSize;
    }
}

template<size_t DIM>
inline void CUDAScene<DIM>::transferToGPU(Scene<DIM>& scene)
{
    SceneData<DIM> *sceneData = scene.getSceneData();
    bool hasLineSegmentGeometry = sceneData->lineSegmentObjects.size() > 0;
    bool hasSilhouetteGeometry = sceneData->silhouetteVertexObjects.size() > 0 ||
                                 sceneData->silhouetteEdgeObjects.size() > 0;

    // allocate GPU buffers
    if (hasSilhouetteGeometry) {
        if (hasLineSegmentGeometry) {
            allocateBuffers<SnchNode<2>, LineSegment, SilhouetteVertex,
                           CUDASnchNode, CUDALineSegment, CUDAVertex>(
                sceneData, true, true, false);
        } else {
            allocateBuffers<SnchNode<3>, Triangle, SilhouetteEdge,
                           CUDASnchNode, CUDATriangle, CUDAEdge>(
                sceneData, true, true, false);
        }
    } else {
        if (hasLineSegmentGeometry) {
            allocateBuffers<BvhNode<2>, LineSegment, SilhouettePrimitive<2>,
                           CUDABvhNode, CUDALineSegment, CUDANoSilhouette>(
                sceneData, true, true, false);
        } else {
            allocateBuffers<BvhNode<3>, Triangle, SilhouettePrimitive<3>,
                           CUDABvhNode, CUDATriangle, CUDANoSilhouette>(
                sceneData, true, true, false);
        }
    }

    if (printLogs) {
        std::cout << "CUDAScene::transferToGPU() bvhType: " << bvhBuffers.bvhType << std::endl;
    }
}

template<size_t DIM>
inline void CUDAScene<DIM>::refit(Scene<DIM>& scene, bool updateGeometry)
{
    SceneData<DIM> *sceneData = scene.getSceneData();
    bool hasLineSegmentGeometry = sceneData->lineSegmentObjects.size() > 0;
    bool hasSilhouetteGeometry = sceneData->silhouetteVertexObjects.size() > 0 ||
                                 sceneData->silhouetteEdgeObjects.size() > 0;
    bool allocateRefitData = bvhBuffers.updateEntryData.size() == 0;

    // allocate GPU buffers
    if (hasSilhouetteGeometry) {
        if (hasLineSegmentGeometry) {
            allocateBuffers<SnchNode<2>, LineSegment, SilhouetteVertex,
                           CUDASnchNode, CUDALineSegment, CUDAVertex>(
                sceneData, updateGeometry, false, allocateRefitData);
        } else {
            allocateBuffers<SnchNode<3>, Triangle, SilhouetteEdge,
                           CUDASnchNode, CUDATriangle, CUDAEdge>(
                sceneData, updateGeometry, false, allocateRefitData);
        }
    } else {
        if (hasLineSegmentGeometry) {
            allocateBuffers<BvhNode<2>, LineSegment, SilhouettePrimitive<2>,
                           CUDABvhNode, CUDALineSegment, CUDANoSilhouette>(
                sceneData, updateGeometry, false, allocateRefitData);
        } else {
            allocateBuffers<BvhNode<3>, Triangle, SilhouettePrimitive<3>,
                           CUDABvhNode, CUDATriangle, CUDANoSilhouette>(
                sceneData, updateGeometry, false, allocateRefitData);
        }
    }

    // run refit kernel for each depth level (bottom-up)
    cudaStream_t cudaStream = static_cast<cudaStream_t>(stream);
    CUDAKernelTimer timer(cudaStream, printLogs);
    for (int depth = bvhBuffers.maxUpdateDepth; depth >= 0; --depth) {
        uint32_t firstNodeOffset = bvhBuffers.updateEntryData[depth].first;
        uint32_t nodeCount = bvhBuffers.updateEntryData[depth].second;

        launchRefitKernel(bvhBuffers.bvhType,
                          bvhBuffers.d_nodes, bvhBuffers.d_primitives, bvhBuffers.d_silhouettes,
                          bvhBuffers.d_nodeIndices,
                          firstNodeOffset, nodeCount, stream);
        FCPW_CUDA_CHECK(cudaStreamSynchronize(cudaStream));
    }
    float ms = timer.elapsed();
    if (printLogs) {
        std::cout << "CUDA BVH refit took " << ms << " ms" << std::endl;
    }
}

template<size_t DIM>
inline void CUDAScene<DIM>::intersect(const Eigen::MatrixXf& rayOrigins,
                                       const Eigen::MatrixXf& rayDirections,
                                       const Eigen::VectorXf& rayDistanceBounds,
                                       std::vector<CUDAInteraction>& interactions,
                                       bool checkForOcclusion)
{
    int nQueries = (int)rayOrigins.rows();
    std::vector<CUDARay> rays(nQueries);

    auto callback = [&](int start, int end) {
        for (int i = start; i < end; i++) {
            CUDARay& ray = rays[i];
            ray.o = CUDAFloat3{rayOrigins(i, 0),
                               rayOrigins(i, 1),
                               DIM == 2 ? 0.0f : rayOrigins(i, 2)};
            ray.d = CUDAFloat3{rayDirections(i, 0),
                               rayDirections(i, 1),
                               DIM == 2 ? 0.0f : rayDirections(i, 2)};
            ray.dInv = CUDAFloat3{1.0f / ray.d.x,
                                  1.0f / ray.d.y,
                                  1.0f / ray.d.z};
            ray.tMax = rayDistanceBounds(i);
        }
    };

    int nThreads = std::thread::hardware_concurrency();
    int nQueriesPerThread = nQueries/nThreads;
    std::vector<std::thread> threads;

    for (int i = 0; i < nThreads; i++) {
        int start = i*nQueriesPerThread;
        int end = (i == nThreads - 1) ? nQueries : (i + 1)*nQueriesPerThread;
        threads.emplace_back(callback, start, end);
    }

    for (auto& t: threads) {
        t.join();
    }

    intersect(rays, interactions, checkForOcclusion);
}

template<size_t DIM>
inline void CUDAScene<DIM>::intersect(const std::vector<CUDARay>& rays,
                                       std::vector<CUDAInteraction>& interactions,
                                       bool checkForOcclusion)
{
    uint32_t nQueries = (uint32_t)rays.size();
    interactions.resize(nQueries);

    // allocate GPU entry point data
    CUDARay* d_rays = nullptr;
    CUDAInteraction* d_interactions = nullptr;
    size_t raysSize = nQueries * sizeof(CUDARay);
    size_t interactionsSize = nQueries * sizeof(CUDAInteraction);

    FCPW_CUDA_CHECK(cudaMalloc(&d_rays, raysSize));
    FCPW_CUDA_CHECK(cudaMalloc(&d_interactions, interactionsSize));

    FCPW_CUDA_CHECK(cudaMemcpy(d_rays, rays.data(), raysSize, cudaMemcpyHostToDevice));
    std::vector<CUDAInteraction> initInteractions(nQueries);
    FCPW_CUDA_CHECK(cudaMemcpy(d_interactions, initInteractions.data(), interactionsSize, cudaMemcpyHostToDevice));

    // launch kernel
    cudaStream_t cudaStream = static_cast<cudaStream_t>(stream);
    CUDAKernelTimer timer(cudaStream, printLogs);
    if (bvhBuffers.hasTransform) {
        launchTransformedRayIntersectionKernel(bvhBuffers.bvhType,
                                               bvhBuffers.d_nodes, bvhBuffers.d_primitives,
                                               bvhBuffers.d_silhouettes, bvhBuffers.d_transform,
                                               d_rays, d_interactions,
                                               checkForOcclusion ? 1 : 0, nQueries, stream);
    } else {
        launchRayIntersectionKernel(bvhBuffers.bvhType,
                                    bvhBuffers.d_nodes, bvhBuffers.d_primitives, bvhBuffers.d_silhouettes,
                                    d_rays, d_interactions,
                                    checkForOcclusion ? 1 : 0, nQueries, stream);
    }
    float ms = timer.elapsed();
    if (printLogs) {
        std::cout << nQueries << " CUDA ray intersection queries took "
                  << ms << " ms" << std::endl;
    }

    // read results from GPU
    FCPW_CUDA_CHECK(cudaMemcpy(interactions.data(), d_interactions, interactionsSize, cudaMemcpyDeviceToHost));

    cudaFree(d_rays);
    cudaFree(d_interactions);
}

template<size_t DIM>
inline void CUDAScene<DIM>::intersect(const Eigen::MatrixXf& sphereCenters,
                                       const Eigen::VectorXf& sphereSquaredRadii,
                                       const Eigen::MatrixXf& randNums,
                                       std::vector<CUDAInteraction>& interactions)
{
    int nQueries = (int)sphereCenters.rows();
    std::vector<CUDABoundingSphere> boundingSpheres(nQueries);
    std::vector<CUDAFloat3> randNums3(nQueries);

    auto callback = [&](int start, int end) {
        for (int i = start; i < end; i++) {
            CUDABoundingSphere& boundingSphere = boundingSpheres[i];
            boundingSphere.c = CUDAFloat3{sphereCenters(i, 0),
                                          sphereCenters(i, 1),
                                          DIM == 2 ? 0.0f : sphereCenters(i, 2)};
            boundingSphere.r2 = sphereSquaredRadii(i);

            randNums3[i] = CUDAFloat3{randNums(i, 0),
                                      randNums(i, 1),
                                      DIM == 2 ? 0.0f : randNums(i, 2)};
        }
    };

    int nThreads = std::thread::hardware_concurrency();
    int nQueriesPerThread = nQueries/nThreads;
    std::vector<std::thread> threads;

    for (int i = 0; i < nThreads; i++) {
        int start = i*nQueriesPerThread;
        int end = (i == nThreads - 1) ? nQueries : (i + 1)*nQueriesPerThread;
        threads.emplace_back(callback, start, end);
    }

    for (auto& t: threads) {
        t.join();
    }

    intersect(boundingSpheres, randNums3, interactions);
}

template<size_t DIM>
inline void CUDAScene<DIM>::intersect(const std::vector<CUDABoundingSphere>& boundingSpheres,
                                       const std::vector<CUDAFloat3>& randNums,
                                       std::vector<CUDAInteraction>& interactions)
{
    uint32_t nQueries = (uint32_t)boundingSpheres.size();
    interactions.resize(nQueries);

    // allocate GPU entry point data
    CUDABoundingSphere* d_spheres = nullptr;
    CUDAFloat3* d_randNums = nullptr;
    CUDAInteraction* d_interactions = nullptr;
    size_t spheresSize = nQueries * sizeof(CUDABoundingSphere);
    size_t randNumsSize = nQueries * sizeof(CUDAFloat3);
    size_t interactionsSize = nQueries * sizeof(CUDAInteraction);

    FCPW_CUDA_CHECK(cudaMalloc(&d_spheres, spheresSize));
    FCPW_CUDA_CHECK(cudaMalloc(&d_randNums, randNumsSize));
    FCPW_CUDA_CHECK(cudaMalloc(&d_interactions, interactionsSize));

    FCPW_CUDA_CHECK(cudaMemcpy(d_spheres, boundingSpheres.data(), spheresSize, cudaMemcpyHostToDevice));
    FCPW_CUDA_CHECK(cudaMemcpy(d_randNums, randNums.data(), randNumsSize, cudaMemcpyHostToDevice));
    std::vector<CUDAInteraction> initInteractions(nQueries);
    FCPW_CUDA_CHECK(cudaMemcpy(d_interactions, initInteractions.data(), interactionsSize, cudaMemcpyHostToDevice));

    // launch kernel
    cudaStream_t cudaStream = static_cast<cudaStream_t>(stream);
    CUDAKernelTimer timer(cudaStream, printLogs);
    if (bvhBuffers.hasTransform) {
        launchTransformedSphereIntersectionKernel(bvhBuffers.bvhType,
                                                   bvhBuffers.d_nodes, bvhBuffers.d_primitives,
                                                   bvhBuffers.d_silhouettes, bvhBuffers.d_transform,
                                                   d_spheres, d_randNums, d_interactions,
                                                   nQueries, stream);
    } else {
        launchSphereIntersectionKernel(bvhBuffers.bvhType,
                                       bvhBuffers.d_nodes, bvhBuffers.d_primitives, bvhBuffers.d_silhouettes,
                                       d_spheres, d_randNums, d_interactions,
                                       nQueries, stream);
    }
    float ms = timer.elapsed();
    if (printLogs) {
        std::cout << nQueries << " CUDA sphere intersection queries took "
                  << ms << " ms" << std::endl;
    }

    // read results from GPU
    FCPW_CUDA_CHECK(cudaMemcpy(interactions.data(), d_interactions, interactionsSize, cudaMemcpyDeviceToHost));

    cudaFree(d_spheres);
    cudaFree(d_randNums);
    cudaFree(d_interactions);
}

template<size_t DIM>
inline void CUDAScene<DIM>::findClosestPoints(const Eigen::MatrixXf& queryPoints,
                                               const Eigen::VectorXf& squaredMaxRadii,
                                               std::vector<CUDAInteraction>& interactions,
                                               bool recordNormals)
{
    int nQueries = (int)queryPoints.rows();
    std::vector<CUDABoundingSphere> boundingSpheres(nQueries);

    auto callback = [&](int start, int end) {
        for (int i = start; i < end; i++) {
            CUDABoundingSphere& boundingSphere = boundingSpheres[i];
            boundingSphere.c = CUDAFloat3{queryPoints(i, 0),
                                          queryPoints(i, 1),
                                          DIM == 2 ? 0.0f : queryPoints(i, 2)};
            boundingSphere.r2 = squaredMaxRadii(i);
        }
    };

    int nThreads = std::thread::hardware_concurrency();
    int nQueriesPerThread = nQueries/nThreads;
    std::vector<std::thread> threads;

    for (int i = 0; i < nThreads; i++) {
        int start = i*nQueriesPerThread;
        int end = (i == nThreads - 1) ? nQueries : (i + 1)*nQueriesPerThread;
        threads.emplace_back(callback, start, end);
    }

    for (auto& t: threads) {
        t.join();
    }

    findClosestPoints(boundingSpheres, interactions, recordNormals);
}

template<size_t DIM>
inline void CUDAScene<DIM>::findClosestPoints(const std::vector<CUDABoundingSphere>& boundingSpheres,
                                               std::vector<CUDAInteraction>& interactions,
                                               bool recordNormals)
{
    uint32_t nQueries = (uint32_t)boundingSpheres.size();
    interactions.resize(nQueries);

    // allocate GPU entry point data
    CUDABoundingSphere* d_spheres = nullptr;
    CUDAInteraction* d_interactions = nullptr;
    size_t spheresSize = nQueries * sizeof(CUDABoundingSphere);
    size_t interactionsSize = nQueries * sizeof(CUDAInteraction);

    FCPW_CUDA_CHECK(cudaMalloc(&d_spheres, spheresSize));
    FCPW_CUDA_CHECK(cudaMalloc(&d_interactions, interactionsSize));

    FCPW_CUDA_CHECK(cudaMemcpy(d_spheres, boundingSpheres.data(), spheresSize, cudaMemcpyHostToDevice));
    std::vector<CUDAInteraction> initInteractions(nQueries);
    FCPW_CUDA_CHECK(cudaMemcpy(d_interactions, initInteractions.data(), interactionsSize, cudaMemcpyHostToDevice));

    // launch kernel
    cudaStream_t cudaStream = static_cast<cudaStream_t>(stream);
    CUDAKernelTimer timer(cudaStream, printLogs);
    if (bvhBuffers.hasTransform) {
        launchTransformedClosestPointKernel(bvhBuffers.bvhType,
                                            bvhBuffers.d_nodes, bvhBuffers.d_primitives,
                                            bvhBuffers.d_silhouettes, bvhBuffers.d_transform,
                                            d_spheres, d_interactions,
                                            recordNormals ? 1 : 0, nQueries, stream);
    } else {
        launchClosestPointKernel(bvhBuffers.bvhType,
                                 bvhBuffers.d_nodes, bvhBuffers.d_primitives, bvhBuffers.d_silhouettes,
                                 d_spheres, d_interactions,
                                 recordNormals ? 1 : 0, nQueries, stream);
    }
    float ms = timer.elapsed();
    if (printLogs) {
        std::cout << nQueries << " CUDA closest point queries took "
                  << ms << " ms" << std::endl;
    }

    // read results from GPU
    FCPW_CUDA_CHECK(cudaMemcpy(interactions.data(), d_interactions, interactionsSize, cudaMemcpyDeviceToHost));

    cudaFree(d_spheres);
    cudaFree(d_interactions);
}

template<size_t DIM>
inline void CUDAScene<DIM>::findClosestSilhouettePoints(const Eigen::MatrixXf& queryPoints,
                                                         const Eigen::VectorXf& squaredMaxRadii,
                                                         const Eigen::VectorXi& flipNormalOrientation,
                                                         std::vector<CUDAInteraction>& interactions,
                                                         float squaredMinRadius, float precision)
{
    int nQueries = (int)queryPoints.rows();
    std::vector<CUDABoundingSphere> boundingSpheres(nQueries);
    std::vector<uint32_t> flipNormalOrientationVec(nQueries);

    auto callback = [&](int start, int end) {
        for (int i = start; i < end; i++) {
            CUDABoundingSphere& boundingSphere = boundingSpheres[i];
            boundingSphere.c = CUDAFloat3{queryPoints(i, 0),
                                          queryPoints(i, 1),
                                          DIM == 2 ? 0.0f : queryPoints(i, 2)};
            boundingSphere.r2 = squaredMaxRadii(i);

            flipNormalOrientationVec[i] = flipNormalOrientation(i);
        }
    };

    int nThreads = std::thread::hardware_concurrency();
    int nQueriesPerThread = nQueries/nThreads;
    std::vector<std::thread> threads;

    for (int i = 0; i < nThreads; i++) {
        int start = i*nQueriesPerThread;
        int end = (i == nThreads - 1) ? nQueries : (i + 1)*nQueriesPerThread;
        threads.emplace_back(callback, start, end);
    }

    for (auto& t: threads) {
        t.join();
    }

    findClosestSilhouettePoints(boundingSpheres, flipNormalOrientationVec,
                                interactions, squaredMinRadius, precision);
}

template<size_t DIM>
inline void CUDAScene<DIM>::findClosestSilhouettePoints(const std::vector<CUDABoundingSphere>& boundingSpheres,
                                                         const std::vector<uint32_t>& flipNormalOrientation,
                                                         std::vector<CUDAInteraction>& interactions,
                                                         float squaredMinRadius, float precision)
{
    uint32_t nQueries = (uint32_t)boundingSpheres.size();
    interactions.resize(nQueries);

    // allocate GPU entry point data
    CUDABoundingSphere* d_spheres = nullptr;
    uint32_t* d_flipNormal = nullptr;
    CUDAInteraction* d_interactions = nullptr;
    size_t spheresSize = nQueries * sizeof(CUDABoundingSphere);
    size_t flipSize = nQueries * sizeof(uint32_t);
    size_t interactionsSize = nQueries * sizeof(CUDAInteraction);

    FCPW_CUDA_CHECK(cudaMalloc(&d_spheres, spheresSize));
    FCPW_CUDA_CHECK(cudaMalloc(&d_flipNormal, flipSize));
    FCPW_CUDA_CHECK(cudaMalloc(&d_interactions, interactionsSize));

    FCPW_CUDA_CHECK(cudaMemcpy(d_spheres, boundingSpheres.data(), spheresSize, cudaMemcpyHostToDevice));
    FCPW_CUDA_CHECK(cudaMemcpy(d_flipNormal, flipNormalOrientation.data(), flipSize, cudaMemcpyHostToDevice));
    std::vector<CUDAInteraction> initInteractions(nQueries);
    FCPW_CUDA_CHECK(cudaMemcpy(d_interactions, initInteractions.data(), interactionsSize, cudaMemcpyHostToDevice));

    // launch kernel
    cudaStream_t cudaStream = static_cast<cudaStream_t>(stream);
    CUDAKernelTimer timer(cudaStream, printLogs);
    if (bvhBuffers.hasTransform) {
        launchTransformedClosestSilhouettePointKernel(bvhBuffers.bvhType,
                                                       bvhBuffers.d_nodes, bvhBuffers.d_primitives,
                                                       bvhBuffers.d_silhouettes, bvhBuffers.d_transform,
                                                       d_spheres, d_flipNormal, d_interactions,
                                                       squaredMinRadius, precision, nQueries, stream);
    } else {
        launchClosestSilhouettePointKernel(bvhBuffers.bvhType,
                                           bvhBuffers.d_nodes, bvhBuffers.d_primitives, bvhBuffers.d_silhouettes,
                                           d_spheres, d_flipNormal, d_interactions,
                                           squaredMinRadius, precision, nQueries, stream);
    }
    float ms = timer.elapsed();
    if (printLogs) {
        std::cout << nQueries << " CUDA closest silhouette point queries took "
                  << ms << " ms" << std::endl;
    }

    // read results from GPU
    FCPW_CUDA_CHECK(cudaMemcpy(interactions.data(), d_interactions, interactionsSize, cudaMemcpyDeviceToHost));

    cudaFree(d_spheres);
    cudaFree(d_flipNormal);
    cudaFree(d_interactions);
}

} // namespace fcpw
