#include <fcpw/gpu/cuda/cuda_utils.h>
#include <fcpw/utilities/scene_data.h>
#include <fcpw/aggregates/bvh.h>
#include <fcpw/geometry/triangles.h>
#include <fcpw/geometry/line_segments.h>
#include <fcpw/geometry/silhouette_edges.h>
#include <fcpw/geometry/silhouette_vertices.h>

namespace fcpw {

// Extract functions - friends of BVH class
template<size_t DIM>
void extractBvhNodes(const std::vector<BvhNode<DIM>>& flatTree,
                     std::vector<GPUBvhNode>& gpuBvhNodes)
{
    int nNodes = (int)flatTree.size();
    gpuBvhNodes.resize(nNodes);

    for (int i = 0; i < nNodes; i++) {
        const BvhNode<DIM>& node = flatTree[i];
        const Vector<DIM>& pMin = node.box.pMin;
        const Vector<DIM>& pMax = node.box.pMax;
        uint32_t nPrimitives = node.nReferences;
        uint32_t offset = nPrimitives > 0 ? node.referenceOffset : node.secondChildOffset;

        float3 pMinGpu = DIM == 2 ? float3{pMin[0], pMin[1], 0.0f} : float3{pMin[0], pMin[1], pMin[2]};
        float3 pMaxGpu = DIM == 2 ? float3{pMax[0], pMax[1], 0.0f} : float3{pMax[0], pMax[1], pMax[2]};
        gpuBvhNodes[i] = GPUBvhNode(GPUBoundingBox(pMinGpu, pMaxGpu), nPrimitives, offset);
    }
}

template<size_t DIM>
void extractSnchNodes(const std::vector<SnchNode<DIM>>& flatTree,
                      std::vector<GPUSnchNode>& gpuSnchNodes)
{
    int nNodes = (int)flatTree.size();
    gpuSnchNodes.resize(nNodes);

    for (int i = 0; i < nNodes; i++) {
        const SnchNode<DIM>& node = flatTree[i];
        const Vector<DIM>& pMin = node.box.pMin;
        const Vector<DIM>& pMax = node.box.pMax;
        const Vector<DIM>& axis = node.cone.axis;
        float halfAngle = node.cone.halfAngle;
        float radius = node.cone.radius;
        uint32_t nPrimitives = node.nReferences;
        uint32_t offset = nPrimitives > 0 ? node.referenceOffset : node.secondChildOffset;
        uint32_t nSilhouettes = node.nSilhouetteReferences;
        uint32_t silhouetteOffset = node.silhouetteReferenceOffset;

        float3 pMinGpu = DIM == 2 ? float3{pMin[0], pMin[1], 0.0f} : float3{pMin[0], pMin[1], pMin[2]};
        float3 pMaxGpu = DIM == 2 ? float3{pMax[0], pMax[1], 0.0f} : float3{pMax[0], pMax[1], pMax[2]};
        float3 axisGpu = DIM == 2 ? float3{axis[0], axis[1], 0.0f} : float3{axis[0], axis[1], axis[2]};
        gpuSnchNodes[i] = GPUSnchNode(GPUBoundingBox(pMinGpu, pMaxGpu), GPUBoundingCone(axisGpu, halfAngle, radius),
                                      nPrimitives, offset, nSilhouettes, silhouetteOffset);
    }
}

void extractLineSegments(const std::vector<LineSegment *>& primitives,
                         std::vector<GPULineSegment>& gpuLineSegments)
{
    int nPrimitives = (int)primitives.size();
    gpuLineSegments.resize(nPrimitives);

    for (int i = 0; i < nPrimitives; i++) {
        const LineSegment *lineSegment = primitives[i];
        const Vector2& pa = lineSegment->soup->positions[lineSegment->indices[0]];
        const Vector2& pb = lineSegment->soup->positions[lineSegment->indices[1]];

        gpuLineSegments[i] = GPULineSegment(float3{pa[0], pa[1], 0.0f},
                                            float3{pb[0], pb[1], 0.0f},
                                            lineSegment->pIndex);
    }
}

void extractTriangles(const std::vector<Triangle *>& primitives,
                      std::vector<GPUTriangle>& gpuTriangles)
{
    int nPrimitives = (int)primitives.size();
    gpuTriangles.resize(nPrimitives);

    for (int i = 0; i < nPrimitives; i++) {
        const Triangle *triangle = primitives[i];
        const Vector3& pa = triangle->soup->positions[triangle->indices[0]];
        const Vector3& pb = triangle->soup->positions[triangle->indices[1]];
        const Vector3& pc = triangle->soup->positions[triangle->indices[2]];

        gpuTriangles[i] = GPUTriangle(float3{pa[0], pa[1], pa[2]},
                                      float3{pb[0], pb[1], pb[2]},
                                      float3{pc[0], pc[1], pc[2]},
                                      triangle->pIndex);
    }
}

void extractSilhouetteVertices(const std::vector<SilhouetteVertex *>& silhouettes,
                               std::vector<GPUVertex>& gpuVertices)
{
    int nSilhouettes = (int)silhouettes.size();
    gpuVertices.resize(nSilhouettes);

    for (int i = 0; i < nSilhouettes; i++) {
        const SilhouetteVertex *silhouetteVertex = silhouettes[i];
        const Vector2& p = silhouetteVertex->soup->positions[silhouetteVertex->indices[1]];
        Vector2 n0 = silhouetteVertex->hasFace(0) ? silhouetteVertex->normal(0) : Vector2::Zero();
        Vector2 n1 = silhouetteVertex->hasFace(1) ? silhouetteVertex->normal(1) : Vector2::Zero();
        bool hasTwoAdjacentFaces = silhouetteVertex->hasFace(0) && silhouetteVertex->hasFace(1);

        gpuVertices[i] = GPUVertex(float3{p[0], p[1], 0.0f},
                                   float3{n0[0], n0[1], 0.0f},
                                   float3{n1[0], n1[1], 0.0f},
                                   silhouetteVertex->pIndex,
                                   hasTwoAdjacentFaces == 1 ? 0 : 1);
    }
}

void extractSilhouetteEdges(const std::vector<SilhouetteEdge *>& silhouettes,
                            std::vector<GPUEdge>& gpuEdges)
{
    int nSilhouettes = (int)silhouettes.size();
    gpuEdges.resize(nSilhouettes);

    for (int i = 0; i < nSilhouettes; i++) {
        const SilhouetteEdge *silhouetteEdge = silhouettes[i];
        const Vector3& pa = silhouetteEdge->soup->positions[silhouetteEdge->indices[1]];
        const Vector3& pb = silhouetteEdge->soup->positions[silhouetteEdge->indices[2]];
        Vector3 n0 = silhouetteEdge->hasFace(0) ? silhouetteEdge->normal(0) : Vector3::Zero();
        Vector3 n1 = silhouetteEdge->hasFace(1) ? silhouetteEdge->normal(1) : Vector3::Zero();
        bool hasTwoAdjacentFaces = silhouetteEdge->hasFace(0) && silhouetteEdge->hasFace(1);

        gpuEdges[i] = GPUEdge(float3{pa[0], pa[1], pa[2]},
                              float3{pb[0], pb[1], pb[2]},
                              float3{n0[0], n0[1], n0[2]},
                              float3{n1[0], n1[1], n1[2]},
                              silhouetteEdge->pIndex,
                              hasTwoAdjacentFaces == 1 ? 0 : 1);
    }
}

namespace cuda {

/////////////////////////////////////////////////////////////////////////////////////////////
// CUDAContext implementation

CUDAContext::CUDAContext(bool printLogs_):
deviceId(-1),
initialized(false),
printLogs(printLogs_)
{
    // constructor
}

CUDAContext::~CUDAContext()
{
    // destructor - CUDA context is automatically cleaned up
}

void CUDAContext::initialize(int deviceId_)
{
    if (initialized) {
        std::cerr << "CUDAContext already initialized" << std::endl;
        return;
    }

    // check for CUDA devices
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found" << std::endl;
        exit(EXIT_FAILURE);
    }

    // validate device ID
    if (deviceId_ < 0 || deviceId_ >= deviceCount) {
        std::cerr << "Invalid CUDA device ID: " << deviceId_
                  << " (available: 0-" << deviceCount - 1 << ")" << std::endl;
        exit(EXIT_FAILURE);
    }

    deviceId = deviceId_;
    CUDA_CHECK(cudaSetDevice(deviceId));
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProps, deviceId));

    // Set stack size for kernels with deep recursion
    // BVH traversal uses a stack of 64 uint32_t (256 bytes) plus local variables
    // Set to 8 KB to handle high thread concurrency
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 8192));

    initialized = true;

    if (printLogs) {
        std::cout << "CUDA Device " << deviceId << ": " << deviceProps.name << std::endl;
        std::cout << "  Compute Capability: " << deviceProps.major << "." << deviceProps.minor << std::endl;
        std::cout << "  Total Global Memory: " << (deviceProps.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
        std::cout << "  Multiprocessors: " << deviceProps.multiProcessorCount << std::endl;
        std::cout << "  Max Threads per Block: " << deviceProps.maxThreadsPerBlock << std::endl;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////
// CUDABvhBuffers implementation

template<size_t DIM>
void CUDABvhBuffers::allocate(SceneData<DIM>* sceneData,
                               bool allocateGeometry,
                               bool allocateSilhouettes,
                               bool allocateRefitData)
{
    bool hasLineSegmentGeometry = sceneData->lineSegmentObjects.size() > 0;
    bool hasTriangleGeometry = sceneData->triangleObjects.size() > 0;
    bool hasSilhouetteGeometry = sceneData->silhouetteVertexObjects.size() > 0 ||
                                 sceneData->silhouetteEdgeObjects.size() > 0;

    if (!sceneData->aggregate) {
        std::cerr << "Scene aggregate not built" << std::endl;
        exit(EXIT_FAILURE);
    }

    // determine BVH type and extract data
    if (hasSilhouetteGeometry) {
        // SNCH type
        if (hasLineSegmentGeometry) {
            typedef Bvh<DIM, SnchNode<DIM>, LineSegment, SilhouetteVertex> SnchBvhType;
            SnchBvhType* bvh = reinterpret_cast<SnchBvhType*>(sceneData->aggregate.get());

            std::vector<GPUSnchNode> gpuSnchNodes;
            fcpw::extractSnchNodes<DIM>(bvh->flatTree, gpuSnchNodes);
            snchNodes.allocate(gpuSnchNodes.size());
            snchNodes.upload(gpuSnchNodes);

            if (allocateGeometry) {
                std::vector<GPULineSegment> gpuLineSegments;
                fcpw::extractLineSegments(bvh->primitives, gpuLineSegments);
                lineSegments.allocate(gpuLineSegments.size());
                lineSegments.upload(gpuLineSegments);
            }

            if (allocateSilhouettes) {
                std::vector<GPUVertex> gpuVertices;
                fcpw::extractSilhouetteVertices(bvh->silhouetteRefs, gpuVertices);
                vertices.allocate(gpuVertices.size());
                vertices.upload(gpuVertices);
            }
        } else if (hasTriangleGeometry) {
            typedef Bvh<DIM, SnchNode<DIM>, Triangle, SilhouetteEdge> SnchBvhType;
            SnchBvhType* bvh = reinterpret_cast<SnchBvhType*>(sceneData->aggregate.get());

            std::vector<GPUSnchNode> gpuSnchNodes;
            fcpw::extractSnchNodes<DIM>(bvh->flatTree, gpuSnchNodes);
            snchNodes.allocate(gpuSnchNodes.size());
            snchNodes.upload(gpuSnchNodes);

            if (allocateGeometry) {
                std::vector<GPUTriangle> gpuTriangles;
                fcpw::extractTriangles(bvh->primitives, gpuTriangles);
                triangles.allocate(gpuTriangles.size());
                triangles.upload(gpuTriangles);
            }

            if (allocateSilhouettes) {
                std::vector<GPUEdge> gpuEdges;
                fcpw::extractSilhouetteEdges(bvh->silhouetteRefs, gpuEdges);
                edges.allocate(gpuEdges.size());
                edges.upload(gpuEdges);
            }
        }
    } else {
        // Regular BVH type
        if (hasLineSegmentGeometry) {
            typedef Bvh<DIM, BvhNode<DIM>, LineSegment> BvhType;
            BvhType* bvh = reinterpret_cast<BvhType*>(sceneData->aggregate.get());

            std::vector<GPUBvhNode> gpuBvhNodes;
            fcpw::extractBvhNodes<DIM>(bvh->flatTree, gpuBvhNodes);
            bvhNodes.allocate(gpuBvhNodes.size());
            bvhNodes.upload(gpuBvhNodes);

            if (allocateGeometry) {
                std::vector<GPULineSegment> gpuLineSegments;
                fcpw::extractLineSegments(bvh->primitives, gpuLineSegments);
                lineSegments.allocate(gpuLineSegments.size());
                lineSegments.upload(gpuLineSegments);
            }
        } else if (hasTriangleGeometry) {
            typedef Bvh<DIM, BvhNode<DIM>, Triangle> BvhType;
            BvhType* bvh = reinterpret_cast<BvhType*>(sceneData->aggregate.get());

            std::vector<GPUBvhNode> gpuBvhNodes;
            fcpw::extractBvhNodes<DIM>(bvh->flatTree, gpuBvhNodes);
            bvhNodes.allocate(gpuBvhNodes.size());
            bvhNodes.upload(gpuBvhNodes);

            if (allocateGeometry) {
                std::vector<GPUTriangle> gpuTriangles;
                fcpw::extractTriangles(bvh->primitives, gpuTriangles);
                triangles.allocate(gpuTriangles.size());
                triangles.upload(gpuTriangles);
            }
        }
    }

    // allocate refit data if needed
    if (allocateRefitData) {
        // Extract refit node indices from BVH (bottom-up order: leaves to root)
        // The BVH is stored in pre-order in flatTree, so we reverse the order
        // to get bottom-up traversal
        std::vector<uint32_t> nodeIndices;

        // Determine BVH type and extract node indices (follow same pattern as above)
        if (hasSilhouetteGeometry) {
            // SNCH type
            if (hasLineSegmentGeometry) {
                typedef Bvh<DIM, SnchNode<DIM>, LineSegment, SilhouetteVertex> SnchBvhType;
                SnchBvhType* bvh = reinterpret_cast<SnchBvhType*>(sceneData->aggregate.get());

                const auto& flatTree = bvh->flatTree;
                nodeIndices.resize(flatTree.size());
                // Reverse order for bottom-up traversal
                for (size_t i = 0; i < flatTree.size(); i++) {
                    nodeIndices[i] = static_cast<uint32_t>(flatTree.size() - 1 - i);
                }
            } else if (hasTriangleGeometry) {
                typedef Bvh<DIM, SnchNode<DIM>, Triangle, SilhouetteEdge> SnchBvhType;
                SnchBvhType* bvh = reinterpret_cast<SnchBvhType*>(sceneData->aggregate.get());

                const auto& flatTree = bvh->flatTree;
                nodeIndices.resize(flatTree.size());
                for (size_t i = 0; i < flatTree.size(); i++) {
                    nodeIndices[i] = static_cast<uint32_t>(flatTree.size() - 1 - i);
                }
            }
        } else {
            // Regular BVH type
            if (hasLineSegmentGeometry) {
                typedef Bvh<DIM, BvhNode<DIM>, LineSegment> BvhType;
                BvhType* bvh = reinterpret_cast<BvhType*>(sceneData->aggregate.get());

                const auto& flatTree = bvh->flatTree;
                nodeIndices.resize(flatTree.size());
                for (size_t i = 0; i < flatTree.size(); i++) {
                    nodeIndices[i] = static_cast<uint32_t>(flatTree.size() - 1 - i);
                }
            } else if (hasTriangleGeometry) {
                typedef Bvh<DIM, BvhNode<DIM>, Triangle> BvhType;
                BvhType* bvh = reinterpret_cast<BvhType*>(sceneData->aggregate.get());

                const auto& flatTree = bvh->flatTree;
                nodeIndices.resize(flatTree.size());
                for (size_t i = 0; i < flatTree.size(); i++) {
                    nodeIndices[i] = static_cast<uint32_t>(flatTree.size() - 1 - i);
                }
            }
        }

        if (!nodeIndices.empty()) {
            refitNodeIndices.allocate(nodeIndices.size());
            refitNodeIndices.upload(nodeIndices);
        }
    }
}

// explicit template instantiations
template void CUDABvhBuffers::allocate<2>(SceneData<2>*, bool, bool, bool);
template void CUDABvhBuffers::allocate<3>(SceneData<3>*, bool, bool, bool);

void CUDABvhBuffers::free()
{
    bvhNodes.free();
    snchNodes.free();
    lineSegments.free();
    triangles.free();
    vertices.free();
    edges.free();
    noSilhouettes.free();
    refitNodeIndices.free();
    rays.free();
    boundingSpheres.free();
    randNums.free();
    flipNormalOrientation.free();
    interactions.free();
}

/////////////////////////////////////////////////////////////////////////////////////////////
// KernelLaunchConfig implementation

KernelLaunchConfig KernelLaunchConfig::compute(uint32_t nQueries,
                                                uint32_t threadsPerBlock)
{
    KernelLaunchConfig config;
    config.blockDim = dim3(threadsPerBlock, 1, 1);
    config.gridDim = dim3((nQueries + threadsPerBlock - 1) / threadsPerBlock, 1, 1);
    config.sharedMem = 0;
    config.stream = 0; // default stream

    return config;
}

/////////////////////////////////////////////////////////////////////////////////////////////
// CUDATimer implementation

CUDATimer::CUDATimer(): elapsed(0.0f)
{
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));
}

CUDATimer::~CUDATimer()
{
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));
}

void CUDATimer::start()
{
    CUDA_CHECK(cudaEventRecord(startEvent));
}

void CUDATimer::stop()
{
    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, startEvent, stopEvent));
}

float CUDATimer::elapsedMilliseconds() const
{
    return elapsed;
}

} // namespace cuda
} // namespace fcpw
