#include <filesystem>

namespace fcpw {

template<size_t DIM>
inline GPUScene<DIM>::GPUScene(const std::string& fcpwDirectoryPath_, bool printLogs_):
nThreadsPerGroup(256),
printLogs(printLogs_)
{
    std::filesystem::path fcpwDirectoryPath(fcpwDirectoryPath_);
    std::filesystem::path shaderDirectoryPath = fcpwDirectoryPath / "include" / "fcpw" / "gpu";
    fcpwModule = (shaderDirectoryPath / "fcpw.slang").string();
    refitShaderModule = (shaderDirectoryPath / "bvh-refit.cs.slang").string();
    traversalShaderModule = (shaderDirectoryPath / "bvh-traversal.cs.slang").string();
}

template<size_t DIM>
inline void GPUScene<DIM>::transferToGPU(Scene<DIM>& scene)
{
    SceneData<DIM> *sceneData = scene.getSceneData();
    bool hasLineSegmentGeometry = sceneData->lineSegmentObjects.size() > 0;
    bool hasSilhouetteGeometry = sceneData->silhouetteVertexObjects.size() > 0 ||
                                 sceneData->silhouetteEdgeObjects.size() > 0;

    // initialize GPU context
    std::string macroValue = hasSilhouetteGeometry ? (hasLineSegmentGeometry ? "3" : "4") :
                                                     (hasLineSegmentGeometry ? "1" : "2");
    slang::PreprocessorMacroDesc macro = { "_BVH_TYPE", macroValue.c_str() };
    gpuContext.initDevice({}, 0, macro, 1);

    // create GPU buffers
    if (hasSilhouetteGeometry) {
        gpuBvhBuffers.template init<DIM, true>();
        gpuBvhBuffers.template allocateSilhouettes<DIM>(gpuContext.device, sceneData);

    } else {
        gpuBvhBuffers.template init<DIM, false>();
    }
    gpuBvhBuffers.template allocatePrimitives<DIM>(gpuContext.device, sceneData);
    gpuBvhBuffers.template allocateNodes<DIM>(gpuContext.device, sceneData);

    // initialize transient resources
    gpuContext.initTransientResources();
}

template<size_t DIM>
inline void GPUScene<DIM>::refit(Scene<DIM>& scene, bool updateGeometry)
{
    SceneData<DIM> *sceneData = scene.getSceneData();
    bool hasSilhouetteGeometry = sceneData->silhouetteVertexObjects.size() > 0 ||
                                 sceneData->silhouetteEdgeObjects.size() > 0;
    bool allocateRefitData = gpuBvhBuffers.getMaxUpdateDepth() >= 0;

    // initialize shader
    if (refitShader.reflection == nullptr) {
        loadModuleLibrary(gpuContext, fcpwModule, refitShader);
        loadShader(gpuContext, refitShaderModule, "refit", refitShader);
    }

    // update GPU buffers
    if (allocateRefitData) {
        gpuBvhBuffers.template allocateRefitData<DIM>(gpuContext.device, sceneData);
    }
    if (updateGeometry) {
        gpuBvhBuffers.template allocatePrimitives<DIM>(gpuContext.device, sceneData);
        if (hasSilhouetteGeometry) {
            gpuBvhBuffers.template allocateSilhouettes<DIM>(gpuContext.device, sceneData);
        }
    }

    // run refit shader
    runUpdate<GPUBvhBuffers>(gpuContext, refitShader, gpuBvhBuffers, printLogs);
}

inline int countThreadGroups(int nQueries, int nThreadsPerGroup, bool printLogs)
{
    int nThreadGroups = (nQueries + nThreadsPerGroup - 1)/nThreadsPerGroup;
    if (printLogs) {
        std::cout << "nQueries: " << nQueries
                  << ", nThreadGroups: " << nThreadGroups
                  << ", nThreadsPerGroup: " << nThreadsPerGroup
                  << std::endl;
    }

    return nThreadGroups;
}

template<size_t DIM>
inline void GPUScene<DIM>::intersect(const Eigen::MatrixXf& rayOrigins,
                                     const Eigen::MatrixXf& rayDirections,
                                     const Eigen::VectorXf& rayDistanceBounds,
                                     GPUInteractions& gpuInteractions,
                                     bool checkForOcclusion)
{
    /*
    int nQueries = (int)rayOrigins.rows();
    GPURay gpuRays;
    gpuRays.setSize(nQueries);

    auto callback = [&](int start, int end) {
        for (int i = start; i < end; i++) {
            gpuRays.ox[i] = rayOrigins(i, 0);
            gpuRays.oy[i] = rayOrigins(i, 1);
            gpuRays.oz[i] = DIM == 2 ? 0.0f : rayOrigins(i, 2);
            gpuRays.dx[i] = rayDirections(i, 0);
            gpuRays.dy[i] = rayDirections(i, 1);
            gpuRays.dz[i] = DIM == 2 ? 0.0f : rayDirections(i, 2);
            gpuRays.tMax[i] = rayDistanceBounds(i);
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

    intersect(gpuRays, gpuInteractions, checkForOcclusion);
    */
}

template<size_t DIM>
inline void GPUScene<DIM>::intersect(GPURays& gpuRays,
                                     GPUInteractions& gpuInteractions,
                                     bool checkForOcclusion)
{
    // initialize shader
    if (rayIntersectionShader.reflection == nullptr) {
        loadModuleLibrary(gpuContext, fcpwModule, rayIntersectionShader);
        loadShader(gpuContext, traversalShaderModule, "rayIntersection", rayIntersectionShader);
    }

    // allocate GPU buffers
    GPURayIntersectionQueryBuffers gpuQueryBuffers(gpuRays, gpuInteractions, checkForOcclusion);
    gpuQueryBuffers.allocate(gpuContext.device);

    // run ray intersection shader
    int nThreadGroups = countThreadGroups(gpuQueryBuffers.nQueries, nThreadsPerGroup, printLogs);
    runTraversal<GPUBvhBuffers, GPURayIntersectionQueryBuffers>(gpuContext, rayIntersectionShader,
                                                                gpuBvhBuffers, gpuQueryBuffers,
                                                                nThreadGroups, printLogs);
}

template<size_t DIM>
inline void GPUScene<DIM>::intersect(const Eigen::MatrixXf& sphereCenters,
                                     const Eigen::VectorXf& sphereSquaredRadii,
                                     const Eigen::MatrixXf& randNums,
                                     GPUInteractions& gpuInteractions)
{
    /*
    int nQueries = (int)sphereCenters.rows();
    GPUBoundingSpheres boundingSpheres;

    std::vector<GPUBoundingSphere> boundingSpheres(nQueries);
    std::vector<float3> randNums3(nQueries);

    auto callback = [&](int start, int end) {
        for (int i = start; i < end; i++) {
            GPUBoundingSphere& boundingSphere = boundingSpheres[i];
            boundingSphere.c = float3{sphereCenters(i, 0),
                                      sphereCenters(i, 1),
                                      DIM == 2 ? 0.0f : sphereCenters(i, 2)};
            boundingSphere.r2 = sphereSquaredRadii(i);

            randNums3[i] = float3{randNums(i, 0),
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
    */
}

template<size_t DIM>
inline void GPUScene<DIM>::intersect(GPUBoundingSpheres& gpuBoundingSpheres,
                                     GPURandNums& gpuRandNums,
                                     GPUInteractions& gpuInteractions)
{
    // initialize shader
    if (sphereIntersectionShader.reflection == nullptr) {
        loadModuleLibrary(gpuContext, fcpwModule, sphereIntersectionShader);
        loadShader(gpuContext, traversalShaderModule, "sphereIntersection", sphereIntersectionShader);
    }

    // allocate GPU buffers
    GPUSphereIntersectionQueryBuffers gpuQueryBuffers(gpuBoundingSpheres, gpuRandNums, gpuInteractions);
    gpuQueryBuffers.allocate(gpuContext.device);

    // run sphere intersection shader
    int nThreadGroups = countThreadGroups(gpuQueryBuffers.nQueries, nThreadsPerGroup, printLogs);
    runTraversal<GPUBvhBuffers, GPUSphereIntersectionQueryBuffers>(gpuContext, sphereIntersectionShader,
                                                                   gpuBvhBuffers, gpuQueryBuffers,
                                                                   nThreadGroups, printLogs);
}

template<size_t DIM>
inline void GPUScene<DIM>::findClosestPoints(const Eigen::MatrixXf& queryPoints,
                                             const Eigen::VectorXf& squaredMaxRadii,
                                             GPUInteractions& gpuInteractions,
                                             bool recordNormals)
{
    /*
    int nQueries = (int)queryPoints.rows();
    std::vector<GPUBoundingSphere> boundingSpheres(nQueries);

    auto callback = [&](int start, int end) {
        for (int i = start; i < end; i++) {
            GPUBoundingSphere& boundingSphere = boundingSpheres[i];
            boundingSphere.c = float3{queryPoints(i, 0),
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
    */
}

template<size_t DIM>
inline void GPUScene<DIM>::findClosestPoints(GPUBoundingSpheres& gpuBoundingSpheres,
                                             GPUInteractions& gpuInteractions,
                                             bool recordNormals)
{
    // initialize shader
    if (closestPointShader.reflection == nullptr) {
        loadModuleLibrary(gpuContext, fcpwModule, closestPointShader);
        loadShader(gpuContext, traversalShaderModule, "closestPoint", closestPointShader);
    }

    // allocate GPU buffers
    GPUClosestPointQueryBuffers gpuQueryBuffers(gpuBoundingSpheres, gpuInteractions, recordNormals);
    gpuQueryBuffers.allocate(gpuContext.device);

    // run closest point shader
    int nThreadGroups = countThreadGroups(gpuQueryBuffers.nQueries, nThreadsPerGroup, printLogs);
    runTraversal<GPUBvhBuffers, GPUClosestPointQueryBuffers>(gpuContext, closestPointShader,
                                                             gpuBvhBuffers, gpuQueryBuffers,
                                                             nThreadGroups, printLogs);
}

template<size_t DIM>
inline void GPUScene<DIM>::findClosestSilhouettePoints(const Eigen::MatrixXf& queryPoints,
                                                       const Eigen::VectorXf& squaredMaxRadii,
                                                       const Eigen::VectorXi& flipNormalOrientation,
                                                       GPUInteractions& gpuInteractions,
                                                       float squaredMinRadius, float precision)
{
    /*
    int nQueries = (int)queryPoints.rows();
    std::vector<GPUBoundingSphere> boundingSpheres(nQueries);
    std::vector<uint32_t> flipNormalOrientationVec(nQueries);

    auto callback = [&](int start, int end) {
        for (int i = start; i < end; i++) {
            GPUBoundingSphere& boundingSphere = boundingSpheres[i];
            boundingSphere.c = float3{queryPoints(i, 0),
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
    */
}

template<size_t DIM>
inline void GPUScene<DIM>::findClosestSilhouettePoints(GPUBoundingSpheres& gpuBoundingSpheres,
                                                       GPUFlipNormalOrientation& gpuFlipNormalOrientation,
                                                       GPUInteractions& gpuInteractions,
                                                       float squaredMinRadius, float precision)
{
    // initialize shader
    if (closestSilhouettePointShader.reflection == nullptr) {
        loadModuleLibrary(gpuContext, fcpwModule, closestSilhouettePointShader);
        loadShader(gpuContext, traversalShaderModule, "closestSilhouettePoint", closestSilhouettePointShader);
    }

    // allocate GPU buffers
    GPUClosestSilhouettePointQueryBuffers gpuQueryBuffers(
        gpuBoundingSpheres, gpuFlipNormalOrientation, gpuInteractions, squaredMinRadius, precision);
    gpuQueryBuffers.allocate(gpuContext.device);

    // run closest silhouette point shader
    int nThreadGroups = countThreadGroups(gpuQueryBuffers.nQueries, nThreadsPerGroup, printLogs);
    runTraversal<GPUBvhBuffers, GPUClosestSilhouettePointQueryBuffers>(gpuContext, closestSilhouettePointShader,
                                                                       gpuBvhBuffers, gpuQueryBuffers,
                                                                       nThreadGroups, printLogs);
}

} // namespace fcpw
