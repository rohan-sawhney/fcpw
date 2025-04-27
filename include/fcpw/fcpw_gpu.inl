#include <filesystem>

#define FCPW_LINE_SEGMENT_BVH 1
#define FCPW_TRIANGLE_BVH 2
#define FCPW_LINE_SEGMENT_SNCH 3
#define FCPW_TRIANGLE_SNCH 4

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
    shaderModule = (shaderDirectoryPath / "bvh.cs.slang").string();
}

template<size_t DIM>
inline void GPUScene<DIM>::transferToGPU(Scene<DIM>& scene)
{
    SceneData<DIM> *sceneData = scene.getSceneData();
    bool hasLineSegmentGeometry = sceneData->lineSegmentObjects.size() > 0;
    bool hasSilhouetteGeometry = sceneData->silhouetteVertexObjects.size() > 0 ||
                                 sceneData->silhouetteEdgeObjects.size() > 0;

    // initialize GPU context
    const char* searchPathList[] = { searchPaths[0].c_str() };
    macros[0].name = "FCPW_BVH_TYPE";
    macros[0].value = hasSilhouetteGeometry ?
                      std::to_string(hasLineSegmentGeometry ? FCPW_LINE_SEGMENT_SNCH : FCPW_TRIANGLE_SNCH).c_str() :
                      std::to_string(hasLineSegmentGeometry ? FCPW_LINE_SEGMENT_BVH : FCPW_TRIANGLE_BVH).c_str();
    gpuContext.initDevice(searchPathList, 1, macros, 1);

    // initialize transient resources
    gpuContext.initTransientResources();

    // load library module
    libraryModules.loadModule(gpuContext, fcpwModule);

    // allocate GPU buffers
    gpuBvhBuffers.template allocate<DIM>(gpuContext, sceneData, true,
                                         hasSilhouetteGeometry, true, false);
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
        std::vector<std::string> entryPointNames = { "refit" };
        refitShader.loadProgram(gpuContext, libraryModules, shaderModule, entryPointNames);
    }

    // update GPU buffers
    gpuBvhBuffers.template allocate<DIM>(gpuContext, sceneData, updateGeometry,
                                         allocateSilhouetteGeometry, false,
                                         allocateRefitData);

    // run refit shader
    runBvhUpdate<GPUBvhBuffers>(gpuContext, refitShader, gpuBvhBuffers, printLogs);
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
inline void GPUScene<DIM>::intersect(const std::vector<GPURay>& rays,
                                     std::vector<GPUInteraction>& interactions,
                                     bool checkForOcclusion)
{
    // initialize shader
    if (rayIntersectionShader.reflection == nullptr) {
        std::vector<std::string> entryPointNames = { "rayIntersection" };
        rayIntersectionShader.loadProgram(gpuContext, libraryModules, shaderModule, entryPointNames);
    }

    // allocate GPU buffers
    GPUQueryRayIntersectionBuffers gpuQueryRayIntersectionBuffers;
    gpuQueryRayIntersectionBuffers.allocate(gpuContext, rays);
    gpuQueryRayIntersectionBuffers.checkForOcclusion = checkForOcclusion ? 1 : 0;

    // run ray intersection shader
    int nQueries = (int)rays.size();
    int nThreadGroups = countThreadGroups(nQueries, nThreadsPerGroup, printLogs);
    runBvhTraversal<GPUBvhBuffers, GPUQueryRayIntersectionBuffers>(gpuContext, rayIntersectionShader,
                                                                   gpuBvhBuffers, gpuQueryRayIntersectionBuffers,
                                                                   nThreadGroups, printLogs);

    // read results from GPU
    gpuQueryRayIntersectionBuffers.read(gpuContext, interactions);
}

template<size_t DIM>
inline void GPUScene<DIM>::intersect(const Eigen::MatrixXf& sphereCenters,
                                     const Eigen::VectorXf& sphereSquaredRadii,
                                     const Eigen::MatrixXf& randNums,
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
inline void GPUScene<DIM>::intersect(const std::vector<GPUBoundingSphere>& boundingSpheres,
                                     const std::vector<float3>& randNums,
                                     std::vector<GPUInteraction>& interactions)
{
    // initialize shader
    if (sphereIntersectionShader.reflection == nullptr) {
        std::vector<std::string> entryPointNames = { "sphereIntersection" };
        sphereIntersectionShader.loadProgram(gpuContext, libraryModules, shaderModule, entryPointNames);
    }

    // allocate GPU buffers
    GPUQuerySphereIntersectionBuffers gpuQuerySphereIntersectionBuffers;
    gpuQuerySphereIntersectionBuffers.allocate(gpuContext, boundingSpheres, randNums);

    // run sphere intersection shader
    int nQueries = (int)boundingSpheres.size();
    int nThreadGroups = countThreadGroups(nQueries, nThreadsPerGroup, printLogs);
    runBvhTraversal<GPUBvhBuffers, GPUQuerySphereIntersectionBuffers>(gpuContext, sphereIntersectionShader,
                                                                      gpuBvhBuffers, gpuQuerySphereIntersectionBuffers,
                                                                      nThreadGroups, printLogs);

    // read results from GPU
    gpuQuerySphereIntersectionBuffers.read(gpuContext, interactions);
}

template<size_t DIM>
inline void GPUScene<DIM>::findClosestPoints(const Eigen::MatrixXf& queryPoints,
                                             const Eigen::VectorXf& squaredMaxRadii,
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
inline void GPUScene<DIM>::findClosestPoints(const std::vector<GPUBoundingSphere>& boundingSpheres,
                                             std::vector<GPUInteraction>& interactions,
                                             bool recordNormals)
{
    // initialize shader
    if (closestPointShader.reflection == nullptr) {
        std::vector<std::string> entryPointNames = { "closestPoint" };
        closestPointShader.loadProgram(gpuContext, libraryModules, shaderModule, entryPointNames);
    }

    // allocate GPU buffers
    GPUQueryClosestPointBuffers gpuQueryClosestPointBuffers;
    gpuQueryClosestPointBuffers.allocate(gpuContext, boundingSpheres);
    gpuQueryClosestPointBuffers.recordNormals = recordNormals ? 1 : 0;

    // run closest point shader
    int nQueries = (int)boundingSpheres.size();
    int nThreadGroups = countThreadGroups(nQueries, nThreadsPerGroup, printLogs);
    runBvhTraversal<GPUBvhBuffers, GPUQueryClosestPointBuffers>(gpuContext, closestPointShader,
                                                                gpuBvhBuffers, gpuQueryClosestPointBuffers,
                                                                nThreadGroups, printLogs);

    // read results from GPU
    gpuQueryClosestPointBuffers.read(gpuContext, interactions);
}

template<size_t DIM>
inline void GPUScene<DIM>::findClosestSilhouettePoints(const Eigen::MatrixXf& queryPoints,
                                                       const Eigen::VectorXf& squaredMaxRadii,
                                                       const Eigen::VectorXi& flipNormalOrientation,
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
inline void GPUScene<DIM>::findClosestSilhouettePoints(const std::vector<GPUBoundingSphere>& boundingSpheres,
                                                       const std::vector<uint32_t>& flipNormalOrientation,
                                                       std::vector<GPUInteraction>& interactions,
                                                       float squaredMinRadius, float precision)
{
    // initialize shader
    if (closestSilhouettePointShader.reflection == nullptr) {
        std::vector<std::string> entryPointNames = { "closestSilhouettePoint" };
        closestSilhouettePointShader.loadProgram(gpuContext, libraryModules, shaderModule, entryPointNames);
    }

    // allocate GPU buffers
    GPUQueryClosestSilhouettePointBuffers gpuQueryClosestSilhouettePointBuffers;
    gpuQueryClosestSilhouettePointBuffers.allocate(gpuContext, boundingSpheres, flipNormalOrientation);
    gpuQueryClosestSilhouettePointBuffers.squaredMinRadius = squaredMinRadius;
    gpuQueryClosestSilhouettePointBuffers.precision = precision;

    // run closest silhouette point shader
    int nQueries = (int)boundingSpheres.size();
    int nThreadGroups = countThreadGroups(nQueries, nThreadsPerGroup, printLogs);
    runBvhTraversal<GPUBvhBuffers, GPUQueryClosestSilhouettePointBuffers>(gpuContext, closestSilhouettePointShader,
                                                                          gpuBvhBuffers, gpuQueryClosestSilhouettePointBuffers,
                                                                          nThreadGroups, printLogs);

    // read results from GPU
    gpuQueryClosestSilhouettePointBuffers.read(gpuContext, interactions);
}

} // namespace fcpw
