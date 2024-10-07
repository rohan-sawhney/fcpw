#include <filesystem>

namespace fcpw {

template<size_t DIM>
inline GPUScene<DIM>::GPUScene(const std::string& fcpwDirectoryPath_, bool printLogs_):
nThreadsPerGroup(256),
printLogs(printLogs_)
{
    std::filesystem::path fcpwDirectoryPath(fcpwDirectoryPath_);
    std::filesystem::path shaderDirectoryPath = fcpwDirectoryPath / "include" / "fcpw" / "gpu";
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
    gpuContext.initDevice(macro, 1);

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
    size_t nRays = rayOrigins.rows();
    std::vector<GPURay> rays;
    rays.reserve(nRays);

    for (size_t i = 0; i < nRays; i++) {
        GPURay ray;
        ray.o = float3{rayOrigins(i, 0),
                       rayOrigins(i, 1),
                       DIM == 2 ? 0.0f : rayOrigins(i, 2)};
        ray.d = float3{rayDirections(i, 0),
                       rayDirections(i, 1),
                       DIM == 2 ? 0.0f : rayDirections(i, 2)};
        ray.dInv = float3{1.0f / ray.d.x,
                          1.0f / ray.d.y,
                          DIM == 2 ? 0.0f : 1.0f / ray.d.z};
        ray.tMax = rayDistanceBounds(i);
        rays.emplace_back(ray);
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
    size_t nSpheres = sphereCenters.rows();
    std::vector<GPUBoundingSphere> boundingSpheres;
    std::vector<float3> randNums3;
    boundingSpheres.reserve(nSpheres);
    randNums3.reserve(nSpheres);

    for (size_t i = 0; i < nSpheres; i++) {
        GPUBoundingSphere boundingSphere;
        boundingSphere.c = float3{sphereCenters(i, 0),
                                  sphereCenters(i, 1),
                                  DIM == 2 ? 0.0f : sphereCenters(i, 2)};
        boundingSphere.r2 = sphereSquaredRadii(i);
        boundingSpheres.emplace_back(boundingSphere);

        float3 nums{randNums(i, 0),
                    randNums(i, 1),
                    DIM == 2 ? 0.0f : randNums(i, 2)};
        randNums3.emplace_back(nums);
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
    size_t nQueryPoints = queryPoints.rows();
    std::vector<GPUBoundingSphere> boundingSpheres;
    boundingSpheres.reserve(nQueryPoints);

    for (size_t i = 0; i < nQueryPoints; i++) {
        GPUBoundingSphere boundingSphere;
        boundingSphere.c = float3{queryPoints(i, 0),
                                  queryPoints(i, 1),
                                  DIM == 2 ? 0.0f : queryPoints(i, 2)};
        boundingSphere.r2 = squaredMaxRadii(i);
        boundingSpheres.emplace_back(boundingSphere);
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
    size_t nQueryPoints = queryPoints.rows();
    std::vector<GPUBoundingSphere> boundingSpheres;
    std::vector<uint32_t> flipNormalOrientationVec;
    boundingSpheres.reserve(nQueryPoints);
    flipNormalOrientationVec.reserve(nQueryPoints);

    for (size_t i = 0; i < nQueryPoints; i++) {
        GPUBoundingSphere boundingSphere;
        boundingSphere.c = float3{queryPoints(i, 0),
                                  queryPoints(i, 1),
                                  DIM == 2 ? 0.0f : queryPoints(i, 2)};
        boundingSphere.r2 = squaredMaxRadii(i);
        boundingSpheres.emplace_back(boundingSphere);

        flipNormalOrientationVec.emplace_back(flipNormalOrientation(i));
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
