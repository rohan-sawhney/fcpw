namespace fcpw {

template<size_t DIM>
inline CUDAScene<DIM>::CUDAScene(const std::string& fcpwDirectoryPath_, bool printLogs_):
stream(nullptr),
nThreadsPerBlock(256),
printLogs(printLogs_)
{
    (void)fcpwDirectoryPath_;

    // Prefer a dedicated stream, but fall back to the default stream when
    // stream creation is unsupported by the current CUDA runtime/driver setup.
    cudaError_t err = cudaStreamCreate(&stream);
    if (err == cudaSuccess) {
        return;
    }

    if (err == cudaErrorNotSupported || err == cudaErrorOperatingSystem) {
        stream = nullptr;
        cudaGetLastError(); // clear sticky error state after tolerated failure
        if (printLogs) {
            std::cerr << "CUDA: stream creation unsupported, using default stream" << std::endl;
        }
        return;
    }

    FCPW_CUDA_CHECK(err);
}

template<size_t DIM>
inline CUDAScene<DIM>::~CUDAScene()
{
    bvhBuffers.release();
    if (stream != nullptr) {
        FCPW_CUDA_CHECK(cudaStreamDestroy(stream));
        stream = nullptr;
    }
}

template<size_t DIM>
inline void CUDAScene<DIM>::transferToGPU(Scene<DIM>& scene)
{
    // Match Slang path behavior: upload node/primitive payload and refit order.
    SceneData<DIM> *sceneData = scene.getSceneData();
    bool hasSilhouetteGeometry = sceneData->silhouetteVertexObjects.size() > 0 ||
                                 sceneData->silhouetteEdgeObjects.size() > 0;

    bvhBuffers.template allocate<DIM>(sceneData,
                                      true,
                                      hasSilhouetteGeometry,
                                      true,
                                      true);
}

template<size_t DIM>
inline void CUDAScene<DIM>::refit(Scene<DIM>& scene, bool updateGeometry)
{
    // If geometry changed, refresh primitive/silhouette buffers before refit.
    SceneData<DIM> *sceneData = scene.getSceneData();
    bool hasSilhouetteGeometry = sceneData->silhouetteVertexObjects.size() > 0 ||
                                 sceneData->silhouetteEdgeObjects.size() > 0;

    bvhBuffers.template allocate<DIM>(sceneData,
                                      updateGeometry,
                                      updateGeometry && hasSilhouetteGeometry,
                                      false,
                                      bvhBuffers.updateEntryData.size() == 0);

    for (int depth = (int)bvhBuffers.maxUpdateDepth; depth >= 0; depth--) {
        // Bottom-up per-depth update, same schedule as Slang refit path.
        uint32_t firstNodeOffset = bvhBuffers.updateEntryData[depth].first;
        uint32_t nodeCount = bvhBuffers.updateEntryData[depth].second;
        launchCudaRefit(bvhBuffers.bvhType,
                        bvhBuffers.nodes,
                        bvhBuffers.primitives,
                        bvhBuffers.silhouettes,
                        bvhBuffers.nodeIndices,
                        firstNodeOffset,
                        nodeCount,
                        nThreadsPerBlock,
                        stream);
    }

    FCPW_CUDA_CHECK(cudaStreamSynchronize(stream));
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

    for (int i = 0; i < nQueries; i++) {
        rays[i] = CUDARay(float3{rayOrigins(i, 0), rayOrigins(i, 1), DIM == 2 ? 0.0f : rayOrigins(i, 2)},
                          float3{rayDirections(i, 0), rayDirections(i, 1), DIM == 2 ? 0.0f : rayDirections(i, 2)},
                          rayDistanceBounds(i));
    }

    intersect(rays, interactions, checkForOcclusion);
}

template<size_t DIM>
inline void CUDAScene<DIM>::intersect(const std::vector<CUDARay>& rays,
                                      std::vector<CUDAInteraction>& interactions,
                                      bool checkForOcclusion)
{
    // Allocate query payloads per dispatch (mirrors simple GPURun* style wrappers).
    CUDARay *d_rays = cudaAllocCopy(rays);
    CUDAInteraction *d_interactions = cudaAllocZeroed<CUDAInteraction>(rays.size());

    launchCudaRayIntersection(bvhBuffers.bvhType,
                              bvhBuffers.nodes,
                              bvhBuffers.primitives,
                              bvhBuffers.silhouettes,
                              d_rays,
                              d_interactions,
                              (uint32_t)rays.size(),
                              checkForOcclusion ? 1u : 0u,
                              nThreadsPerBlock,
                              stream);

    FCPW_CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaDownload(d_interactions, rays.size(), interactions);
    cudaFreePtr(d_rays);
    cudaFreePtr(d_interactions);
}

template<size_t DIM>
inline void CUDAScene<DIM>::intersect(const Eigen::MatrixXf& sphereCenters,
                                      const Eigen::VectorXf& sphereSquaredRadii,
                                      const Eigen::MatrixXf& randNums,
                                      std::vector<CUDAInteraction>& interactions)
{
    int nQueries = (int)sphereCenters.rows();
    std::vector<CUDABoundingSphere> boundingSpheres(nQueries);
    std::vector<float3> randNums3(nQueries);

    for (int i = 0; i < nQueries; i++) {
        boundingSpheres[i] = CUDABoundingSphere(float3{sphereCenters(i, 0),
                                                        sphereCenters(i, 1),
                                                        DIM == 2 ? 0.0f : sphereCenters(i, 2)},
                                                sphereSquaredRadii(i));
        randNums3[i] = float3{randNums(i, 0), randNums(i, 1), DIM == 2 ? 0.0f : randNums(i, 2)};
    }

    intersect(boundingSpheres, randNums3, interactions);
}

template<size_t DIM>
inline void CUDAScene<DIM>::intersect(const std::vector<CUDABoundingSphere>& boundingSpheres,
                                      const std::vector<float3>& randNums,
                                      std::vector<CUDAInteraction>& interactions)
{
    // Stochastic sphere query path.
    CUDABoundingSphere *d_spheres = cudaAllocCopy(boundingSpheres);
    float3 *d_randNums = cudaAllocCopy(randNums);
    CUDAInteraction *d_interactions = cudaAllocZeroed<CUDAInteraction>(boundingSpheres.size());

    launchCudaSphereIntersection(bvhBuffers.bvhType,
                                 bvhBuffers.nodes,
                                 bvhBuffers.primitives,
                                 bvhBuffers.silhouettes,
                                 d_spheres,
                                 d_randNums,
                                 d_interactions,
                                 (uint32_t)boundingSpheres.size(),
                                 nThreadsPerBlock,
                                 stream);

    FCPW_CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaDownload(d_interactions, boundingSpheres.size(), interactions);
    cudaFreePtr(d_spheres);
    cudaFreePtr(d_randNums);
    cudaFreePtr(d_interactions);
}

template<size_t DIM>
inline void CUDAScene<DIM>::findClosestPoints(const Eigen::MatrixXf& queryPoints,
                                              const Eigen::VectorXf& squaredMaxRadii,
                                              std::vector<CUDAInteraction>& interactions,
                                              bool recordNormals)
{
    int nQueries = (int)queryPoints.rows();
    std::vector<CUDABoundingSphere> boundingSpheres(nQueries);

    for (int i = 0; i < nQueries; i++) {
        boundingSpheres[i] = CUDABoundingSphere(float3{queryPoints(i, 0),
                                                        queryPoints(i, 1),
                                                        DIM == 2 ? 0.0f : queryPoints(i, 2)},
                                                squaredMaxRadii(i));
    }

    findClosestPoints(boundingSpheres, interactions, recordNormals);
}

template<size_t DIM>
inline void CUDAScene<DIM>::findClosestPoints(const std::vector<CUDABoundingSphere>& boundingSpheres,
                                              std::vector<CUDAInteraction>& interactions,
                                              bool recordNormals)
{
    // Closest-point query over bounding-sphere encoded query set.
    CUDABoundingSphere *d_spheres = cudaAllocCopy(boundingSpheres);
    CUDAInteraction *d_interactions = cudaAllocZeroed<CUDAInteraction>(boundingSpheres.size());

    launchCudaClosestPoint(bvhBuffers.bvhType,
                           bvhBuffers.nodes,
                           bvhBuffers.primitives,
                           bvhBuffers.silhouettes,
                           d_spheres,
                           d_interactions,
                           (uint32_t)boundingSpheres.size(),
                           recordNormals ? 1u : 0u,
                           nThreadsPerBlock,
                           stream);

    FCPW_CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaDownload(d_interactions, boundingSpheres.size(), interactions);
    cudaFreePtr(d_spheres);
    cudaFreePtr(d_interactions);
}

template<size_t DIM>
inline void CUDAScene<DIM>::findClosestSilhouettePoints(const Eigen::MatrixXf& queryPoints,
                                                        const Eigen::VectorXf& squaredMaxRadii,
                                                        const Eigen::VectorXi& flipNormalOrientation,
                                                        std::vector<CUDAInteraction>& interactions,
                                                        float squaredMinRadius,
                                                        float precision)
{
    int nQueries = (int)queryPoints.rows();
    std::vector<CUDABoundingSphere> boundingSpheres(nQueries);
    std::vector<uint32_t> flip(nQueries);

    for (int i = 0; i < nQueries; i++) {
        boundingSpheres[i] = CUDABoundingSphere(float3{queryPoints(i, 0),
                                                        queryPoints(i, 1),
                                                        DIM == 2 ? 0.0f : queryPoints(i, 2)},
                                                squaredMaxRadii(i));
        flip[i] = (uint32_t)flipNormalOrientation(i);
    }

    findClosestSilhouettePoints(boundingSpheres, flip, interactions, squaredMinRadius, precision);
}

template<size_t DIM>
inline void CUDAScene<DIM>::findClosestSilhouettePoints(const std::vector<CUDABoundingSphere>& boundingSpheres,
                                                        const std::vector<uint32_t>& flipNormalOrientation,
                                                        std::vector<CUDAInteraction>& interactions,
                                                        float squaredMinRadius,
                                                        float precision)
{
    // Silhouette-aware closest-point query path.
    CUDABoundingSphere *d_spheres = cudaAllocCopy(boundingSpheres);
    uint32_t *d_flip = cudaAllocCopy(flipNormalOrientation);
    CUDAInteraction *d_interactions = cudaAllocZeroed<CUDAInteraction>(boundingSpheres.size());

    launchCudaClosestSilhouettePoint(bvhBuffers.bvhType,
                                     bvhBuffers.nodes,
                                     bvhBuffers.primitives,
                                     bvhBuffers.silhouettes,
                                     d_spheres,
                                     d_flip,
                                     d_interactions,
                                     (uint32_t)boundingSpheres.size(),
                                     squaredMinRadius,
                                     precision,
                                     nThreadsPerBlock,
                                     stream);

    FCPW_CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaDownload(d_interactions, boundingSpheres.size(), interactions);
    cudaFreePtr(d_spheres);
    cudaFreePtr(d_flip);
    cudaFreePtr(d_interactions);
}

} // namespace fcpw
