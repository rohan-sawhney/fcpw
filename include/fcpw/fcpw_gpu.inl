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
    std::string macroValue = hasSilhouetteGeometry ?
                             std::to_string(hasLineSegmentGeometry ? FCPW_LINE_SEGMENT_SNCH : FCPW_TRIANGLE_SNCH) :
                             std::to_string(hasLineSegmentGeometry ? FCPW_LINE_SEGMENT_BVH : FCPW_TRIANGLE_BVH);
    macros[0].name = "FCPW_BVH_TYPE";
    macros[0].value = macroValue.c_str();
    context.initDevice(searchPathList, 1, macros, 1);

    // load library module
    libraryModules.loadModule(context, fcpwModule);

    // allocate GPU buffers
    bvhBuffers.template allocate<DIM>(context, sceneData, true,
                                      hasSilhouetteGeometry, true, false);

    // bind bvh resources
    bindBvhResources = [this](const ComputeShader& shader,
                              const ShaderCursor& cursor) {
        ComPtr<IShaderObject> bvhShaderObject = shader.createShaderObject(
            context, bvhBuffers.getReflectionType());
        ShaderCursor bvhCursor(bvhShaderObject);
        bvhBuffers.setResources(bvhCursor, printLogs);
        cursor.getPath("gBvh").setObject(bvhShaderObject);
    };
}

template<size_t DIM>
inline void GPUScene<DIM>::refit(Scene<DIM>& scene, bool updateGeometry)
{
    SceneData<DIM> *sceneData = scene.getSceneData();
    bool hasSilhouetteGeometry = sceneData->silhouetteVertexObjects.size() > 0 ||
                                 sceneData->silhouetteEdgeObjects.size() > 0;
    bool allocateSilhouetteGeometry = hasSilhouetteGeometry && updateGeometry;
    bool allocateRefitData = bvhBuffers.updateEntryData.size() == 0;

    // initialize shader
    if (!refitShader.isInitialized()) {
        std::vector<std::string> entryPointNames = { "refit" };
        refitShader.loadProgram(context, libraryModules, shaderModule, entryPointNames);
    }

    // update GPU buffers
    bvhBuffers.template allocate<DIM>(context, sceneData, updateGeometry,
                                      allocateSilhouetteGeometry, false,
                                      allocateRefitData);

    // run refit shader
    /*
    runBvhUpdate<GPUBvhBuffers>(context, refitShader, bvhBuffers, printLogs);
    */
}

inline uint32_t countThreadGroups(uint32_t workload, uint32_t nThreadsPerGroup, bool printLogs)
{
    uint32_t nThreadGroups = (workload + nThreadsPerGroup - 1)/nThreadsPerGroup;
    if (printLogs) {
        std::cout << "nThreadGroups: " << nThreadGroups
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
    // allocate GPU entry point data
    GPURunRayIntersectionQuery runRayIntersectionQuery;
    runRayIntersectionQuery.allocate(context, rays);
    runRayIntersectionQuery.checkForOcclusion = checkForOcclusion ? 1 : 0;

    // initialize shader
    if (!rayIntersectionShader.isInitialized()) {
        std::vector<std::string> entryPointNames = { runRayIntersectionQuery.getName() };
        rayIntersectionShader.loadProgram(context, libraryModules, shaderModule, entryPointNames);
    }

    // run ray intersection shader
    uint32_t nQueries = (uint32_t)rays.size();
    uint32_t nThreadGroups = countThreadGroups(nQueries, nThreadsPerGroup, printLogs);
    /*
    runShader<GPURunRayIntersectionQuery>(context, rayIntersectionShader,
                                          runRayIntersectionQuery, bindBvhResources,
                                          {}, nThreadGroups, 1, printLogs);
    */

    // read results from GPU
    runRayIntersectionQuery.read(context, interactions);
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
    // allocate GPU entry point data
    GPURunSphereIntersectionQuery runSphereIntersectionQuery;
    runSphereIntersectionQuery.allocate(context, boundingSpheres, randNums);

    // initialize shader
    if (!sphereIntersectionShader.isInitialized()) {
        std::vector<std::string> entryPointNames = { runSphereIntersectionQuery.getName() };
        sphereIntersectionShader.loadProgram(context, libraryModules, shaderModule, entryPointNames);
    }

    // run sphere intersection shader
    uint32_t nQueries = (uint32_t)boundingSpheres.size();
    uint32_t nThreadGroups = countThreadGroups(nQueries, nThreadsPerGroup, printLogs);
    /*
    runShader<GPURunSphereIntersectionQuery>(context, sphereIntersectionShader,
                                             runSphereIntersectionQuery, bindBvhResources,
                                             {}, nThreadGroups, 1, printLogs);
    */

    // read results from GPU
    runSphereIntersectionQuery.read(context, interactions);
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
    // allocate GPU entry point data
    GPURunClosestPointQuery runClosestPointQuery;
    runClosestPointQuery.allocate(context, boundingSpheres);
    runClosestPointQuery.recordNormals = recordNormals ? 1 : 0;

    // initialize shader
    if (!closestPointShader.isInitialized()) {
        std::vector<std::string> entryPointNames = { runClosestPointQuery.getName() };
        closestPointShader.loadProgram(context, libraryModules, shaderModule, entryPointNames);
    }

    // run closest point shader
    uint32_t nQueries = (uint32_t)boundingSpheres.size();
    uint32_t nThreadGroups = countThreadGroups(nQueries, nThreadsPerGroup, printLogs);
    /*
    runShader<GPURunClosestPointQuery>(context, closestPointShader,
                                       runClosestPointQuery, bindBvhResources,
                                       {}, nThreadGroups, 1, printLogs);
    */

    // read results from GPU
    runClosestPointQuery.read(context, interactions);
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
    // allocate GPU entry point data
    GPURunClosestSilhouettePointQuery runClosestSilhouettePointQuery;
    runClosestSilhouettePointQuery.allocate(context, boundingSpheres, flipNormalOrientation);
    runClosestSilhouettePointQuery.squaredMinRadius = squaredMinRadius;
    runClosestSilhouettePointQuery.precision = precision;

    // initialize shader
    if (!closestSilhouettePointShader.isInitialized()) {
        std::vector<std::string> entryPointNames = { runClosestSilhouettePointQuery.getName() };
        closestSilhouettePointShader.loadProgram(context, libraryModules, shaderModule, entryPointNames);
    }

    // run closest silhouette point shader
    uint32_t nQueries = (uint32_t)boundingSpheres.size();
    uint32_t nThreadGroups = countThreadGroups(nQueries, nThreadsPerGroup, printLogs);
    /*
    runShader<GPURunClosestSilhouettePointQuery>(context, closestSilhouettePointShader,
                                                 runClosestSilhouettePointQuery, bindBvhResources,
                                                 {}, nThreadGroups, 1, printLogs);
    */

    // read results from GPU
    runClosestSilhouettePointQuery.read(context, interactions);
}

} // namespace fcpw
