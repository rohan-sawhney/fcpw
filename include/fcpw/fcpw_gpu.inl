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
    searchPaths[0] = shaderDirectoryPath.string();
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
    const char* searchPathList[] = { searchPaths[0].c_str() };
    gpuContext.initDevice(searchPathList, 1, macro, 1);

    // create GPU buffers
    gpuBvhBuffers.template allocate<DIM>(gpuContext.device, sceneData, true,
                                         hasSilhouetteGeometry, true, false);

    // initialize transient resources
    gpuContext.initTransientResources();
}

template<size_t DIM>
inline void GPUScene<DIM>::refit(Scene<DIM>& scene, bool updateGeometry)
{
    SceneData<DIM> *sceneData = scene.getSceneData();
    bool hasSilhouetteGeometry = sceneData->silhouetteVertexObjects.size() > 0 ||
                                 sceneData->silhouetteEdgeObjects.size() > 0;
    bool allocateSilhouetteGeometry = hasSilhouetteGeometry && updateGeometry;
    bool allocateRefitData = gpuBvhBuffers.updateEntryData.size() == 0;

    // initialize shader
    if (refitShader.reflection == nullptr) {
        loadModuleLibrary(gpuContext, fcpwModule, refitShader);
        loadShader(gpuContext, refitShaderModule, "refit", refitShader);
    }

    // update GPU buffers
    gpuBvhBuffers.template allocate<DIM>(gpuContext.device, sceneData, updateGeometry,
                                         allocateSilhouetteGeometry, false,
                                         allocateRefitData);

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
inline void GPUScene<DIM>::intersect(Eigen::MatrixXf& rayOrigins,
                                     Eigen::MatrixXf& rayDirections,
                                     Eigen::VectorXf& rayDistanceBounds,
                                     std::vector<GPUInteraction>& interactions,
                                     bool checkForOcclusion)
{
    int nQueries = (int)rayOrigins.rows();
    std::vector<GPURay> rays(nQueries);

    auto callback = [&](int start, int end) {
        for (int i = start; i < end; i++) {
            GPURay& ray = rays[i];
            ray.o = float3{rayOrigins(i, 0),
                           rayOrigins(i, 1),
                           DIM == 2 ? 0.0f : rayOrigins(i, 2)};
            ray.d = float3{rayDirections(i, 0),
                           rayDirections(i, 1),
                           DIM == 2 ? 0.0f : rayDirections(i, 2)};
            ray.dInv = float3{1.0f / ray.d.x,
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
inline void GPUScene<DIM>::intersect(std::vector<GPURay>& rays,
                                     std::vector<GPUInteraction>& interactions,
                                     bool checkForOcclusion)
{
    // initialize shader
    if (rayIntersectionShader.reflection == nullptr) {
        loadModuleLibrary(gpuContext, fcpwModule, rayIntersectionShader);
        loadShader(gpuContext, traversalShaderModule, "rayIntersection", rayIntersectionShader);
    }

    // create GPU buffers
    GPUQueryRayIntersectionBuffers gpuQueryRayIntersectionBuffers;
    gpuQueryRayIntersectionBuffers.allocate(gpuContext.device, rays);
    gpuQueryRayIntersectionBuffers.checkForOcclusion = checkForOcclusion;

    // run ray intersection shader
    int nQueries = (int)rays.size();
    int nThreadGroups = countThreadGroups(nQueries, nThreadsPerGroup, printLogs);
    runTraversal<GPUBvhBuffers, GPUQueryRayIntersectionBuffers>(gpuContext, rayIntersectionShader,
                                                                gpuBvhBuffers, gpuQueryRayIntersectionBuffers,
                                                                interactions, nThreadGroups, printLogs);
}

template<size_t DIM>
inline void GPUScene<DIM>::intersect(Eigen::MatrixXf& sphereCenters,
                                     Eigen::VectorXf& sphereSquaredRadii,
                                     Eigen::MatrixXf& randNums,
                                     std::vector<GPUInteraction>& interactions)
{
    int nQueries = (int)sphereCenters.rows();
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
}

template<size_t DIM>
inline void GPUScene<DIM>::intersect(std::vector<GPUBoundingSphere>& boundingSpheres,
                                     std::vector<float3>& randNums,
                                     std::vector<GPUInteraction>& interactions)
{
    // initialize shader
    if (sphereIntersectionShader.reflection == nullptr) {
        loadModuleLibrary(gpuContext, fcpwModule, sphereIntersectionShader);
        loadShader(gpuContext, traversalShaderModule, "sphereIntersection", sphereIntersectionShader);
    }

    // create GPU buffers
    GPUQuerySphereIntersectionBuffers gpuQuerySphereIntersectionBuffers;
    gpuQuerySphereIntersectionBuffers.allocate(gpuContext.device, boundingSpheres, randNums);

    // run sphere intersection shader
    int nQueries = (int)boundingSpheres.size();
    int nThreadGroups = countThreadGroups(nQueries, nThreadsPerGroup, printLogs);
    runTraversal<GPUBvhBuffers, GPUQuerySphereIntersectionBuffers>(gpuContext, sphereIntersectionShader,
                                                                   gpuBvhBuffers, gpuQuerySphereIntersectionBuffers,
                                                                   interactions, nThreadGroups, printLogs);
}

template<size_t DIM>
inline void GPUScene<DIM>::findClosestPoints(Eigen::MatrixXf& queryPoints,
                                             Eigen::VectorXf& squaredMaxRadii,
                                             std::vector<GPUInteraction>& interactions,
                                             bool recordNormals)
{
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
}

template<size_t DIM>
inline void GPUScene<DIM>::findClosestPoints(std::vector<GPUBoundingSphere>& boundingSpheres,
                                             std::vector<GPUInteraction>& interactions,
                                             bool recordNormals)
{
    // initialize shader
    if (closestPointShader.reflection == nullptr) {
        loadModuleLibrary(gpuContext, fcpwModule, closestPointShader);
        loadShader(gpuContext, traversalShaderModule, "closestPoint", closestPointShader);
    }

    // create GPU buffers
    GPUQueryClosestPointBuffers gpuQueryClosestPointBuffers;
    gpuQueryClosestPointBuffers.allocate(gpuContext.device, boundingSpheres);
    gpuQueryClosestPointBuffers.recordNormals = recordNormals;

    // run closest point shader
    int nQueries = (int)boundingSpheres.size();
    int nThreadGroups = countThreadGroups(nQueries, nThreadsPerGroup, printLogs);
    runTraversal<GPUBvhBuffers, GPUQueryClosestPointBuffers>(gpuContext, closestPointShader,
                                                             gpuBvhBuffers, gpuQueryClosestPointBuffers,
                                                             interactions, nThreadGroups, printLogs);
}

template<size_t DIM>
inline void GPUScene<DIM>::findClosestSilhouettePoints(Eigen::MatrixXf& queryPoints,
                                                       Eigen::VectorXf& squaredMaxRadii,
                                                       Eigen::VectorXi& flipNormalOrientation,
                                                       std::vector<GPUInteraction>& interactions,
                                                       float squaredMinRadius, float precision)
{
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
}

template<size_t DIM>
inline void GPUScene<DIM>::findClosestSilhouettePoints(std::vector<GPUBoundingSphere>& boundingSpheres,
                                                       std::vector<uint32_t>& flipNormalOrientation,
                                                       std::vector<GPUInteraction>& interactions,
                                                       float squaredMinRadius, float precision)
{
    // initialize shader
    if (closestSilhouettePointShader.reflection == nullptr) {
        loadModuleLibrary(gpuContext, fcpwModule, closestSilhouettePointShader);
        loadShader(gpuContext, traversalShaderModule, "closestSilhouettePoint", closestSilhouettePointShader);
    }

    // create GPU buffers
    GPUQueryClosestSilhouettePointBuffers gpuQueryClosestSilhouettePointBuffers;
    gpuQueryClosestSilhouettePointBuffers.allocate(gpuContext.device, boundingSpheres, flipNormalOrientation);
    gpuQueryClosestSilhouettePointBuffers.squaredMinRadius = squaredMinRadius;
    gpuQueryClosestSilhouettePointBuffers.precision = precision;

    // run closest silhouette point shader
    int nQueries = (int)boundingSpheres.size();
    int nThreadGroups = countThreadGroups(nQueries, nThreadsPerGroup, printLogs);
    runTraversal<GPUBvhBuffers, GPUQueryClosestSilhouettePointBuffers>(gpuContext, closestSilhouettePointShader,
                                                                       gpuBvhBuffers, gpuQueryClosestSilhouettePointBuffers,
                                                                       interactions, nThreadGroups, printLogs);
}

} // namespace fcpw
