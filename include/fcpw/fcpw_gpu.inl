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
inline void GPUScene<DIM>::intersect(Eigen::MatrixXf& rayOrigins,
                                     Eigen::MatrixXf& rayDirections,
                                     Eigen::VectorXf& rayDistanceBounds,
                                     GPUInteractions& interactions,
                                     bool checkForOcclusion)
{
    intersect(rayOrigins.data(), rayDirections.data(), rayDistanceBounds.data(),
              interactions, checkForOcclusion);
}

template<size_t DIM>
inline void GPUScene<DIM>::intersect(std::vector<Vector<DIM>>& rayOrigins,
                                     std::vector<Vector<DIM>>& rayDirections,
                                     std::vector<float>& rayDistanceBounds,
                                     GPUInteractions& interactions,
                                     bool checkForOcclusion)
{
    /*
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
    */
}

template<size_t DIM>
inline void GPUScene<DIM>::intersect(Eigen::MatrixXf& sphereCenters,
                                     Eigen::VectorXf& sphereSquaredRadii,
                                     Eigen::MatrixXf& randNums,
                                     GPUInteractions& interactions)
{
    intersect(sphereCenters.data(), sphereSquaredRadii.data(), randNums3.data(), interactions);
}

template<size_t DIM>
inline void GPUScene<DIM>::intersect(std::vector<Vector<DIM>>& sphereCenters,
                                     std::vector<float>& sphereSquaredRadii,
                                     std::vector<Vector<DIM>>& randNums,
                                     GPUInteractions& interactions)
{
    /*
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
    */
}

template<size_t DIM>
inline void GPUScene<DIM>::findClosestPoints(Eigen::MatrixXf& queryPoints,
                                             Eigen::VectorXf& squaredMaxRadii,
                                             GPUInteractions& interactions,
                                             bool recordNormals)
{
    findClosestPoints(queryPoints.data(), squaredMaxRadii.data(), interactions, recordNormals);
}

template<size_t DIM>
inline void GPUScene<DIM>::findClosestPoints(std::vector<Vector<DIM>>& queryPoints,
                                             std::vector<float>& squaredMaxRadii,
                                             GPUInteractions& interactions,
                                             bool recordNormals)
{
    /*
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
    */
}

template<size_t DIM>
inline void GPUScene<DIM>::findClosestSilhouettePoints(Eigen::MatrixXf& queryPoints,
                                                       Eigen::VectorXf& squaredMaxRadii,
                                                       Eigen::VectorXi& flipNormalOrientation,
                                                       GPUInteractions& interactions,
                                                       float squaredMinRadius, float precision)
{
    findClosestSilhouettePoints(queryPoints.data(), squaredMaxRadii.data(),
                                flipNormalOrientation.data(), interactions,
                                squaredMinRadius, precision);
}

template<size_t DIM>
inline void GPUScene<DIM>::findClosestSilhouettePoints(std::vector<Vector<DIM>>& queryPoints,
                                                       std::vector<float>& squaredMaxRadii,
                                                       std::vector<uint32_t>& flipNormalOrientation,
                                                       GPUInteractions& interactions,
                                                       float squaredMinRadius, float precision)
{
    /*
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
    */
}

} // namespace fcpw
