#pragma once

#include <fcpw/aggregates/bvh.h>
#include <fcpw/geometry/line_segments.h>
#include <fcpw/geometry/triangles.h>
#include <fcpw/geometry/silhouette_vertices.h>
#include <fcpw/geometry/silhouette_edges.h>
#include <fcpw/gpu/slang_gfx_utils.h>

#define FCPW_GPU_UINT_MAX 4294967295

namespace fcpw {

struct float2 {
    float x, y;
};

struct float3 {
    float x, y, z;
};

struct uint2 {
    uint32_t x, y;
};

struct uint3 {
    uint32_t x, y, z;
};

struct GPUBoundingBox {
    GPUBoundingBox() {
        pMin = float3{0.0f, 0.0f, 0.0f};
        pMax = float3{0.0f, 0.0f, 0.0f};
    }
    GPUBoundingBox(const float3& pMin_, const float3& pMax_): pMin(pMin_), pMax(pMax_) {}

    float3 pMin; // aabb min position
    float3 pMax; // aabb max position
};

struct GPUBoundingCone {
    GPUBoundingCone() {
        axis = float3{0.0f, 0.0f, 0.0f};
        halfAngle = M_PI;
        radius = 0.0f;
    }
    GPUBoundingCone(const float3& axis_, float halfAngle_, float radius_):
                    axis(axis_), halfAngle(halfAngle_), radius(radius_) {}

    float3 axis;     // cone axis
    float halfAngle; // cone half angle
    float radius;    // cone radius
};

struct GPUBvhNode {
    GPUBvhNode() {
        box = GPUBoundingBox();
        nPrimitives = 0;
        offset = 0;
    }
    GPUBvhNode(const GPUBoundingBox& box_, uint32_t nPrimitives_, uint32_t offset_):
               box(box_), nPrimitives(nPrimitives_), offset(offset_) {}

    GPUBoundingBox box;
    uint32_t nPrimitives;
    uint32_t offset;
};

struct GPUSnchNode {
    GPUSnchNode() {
        box = GPUBoundingBox();
        cone = GPUBoundingCone();
        nPrimitives = 0;
        offset = 0;
        nSilhouettes = 0;
        silhouetteOffset = 0;
    }
    GPUSnchNode(const GPUBoundingBox& box_, const GPUBoundingCone& cone_, uint32_t nPrimitives_,
                uint32_t offset_, uint32_t nSilhouettes_, uint32_t silhouetteOffset_):
                box(box_), cone(cone_), nPrimitives(nPrimitives_), offset(offset_),
                nSilhouettes(nSilhouettes_), silhouetteOffset(silhouetteOffset_) {}

    GPUBoundingBox box;
    GPUBoundingCone cone;
    uint32_t nPrimitives;
    uint32_t offset;
    uint32_t nSilhouettes;
    uint32_t silhouetteOffset;
};

struct GPULineSegment {
    GPULineSegment() {
        pa = float3{0.0f, 0.0f, 0.0f};
        pb = float3{0.0f, 0.0f, 0.0f};
        index = FCPW_GPU_UINT_MAX;
    }
    GPULineSegment(const float3& pa_, const float3& pb_, uint32_t index_):
                   pa(pa_), pb(pb_), index(index_) {}

    float3 pa;
    float3 pb;
    uint32_t index;
};

struct GPUTriangle {
    GPUTriangle() {
        pa = float3{0.0f, 0.0f, 0.0f};
        pb = float3{0.0f, 0.0f, 0.0f};
        pc = float3{0.0f, 0.0f, 0.0f};
        index = FCPW_GPU_UINT_MAX;
    }
    GPUTriangle(const float3& pa_, const float3& pb_, const float3& pc_, uint32_t index_):
                pa(pa_), pb(pb_), pc(pc_), index(index_) {}

    float3 pa;
    float3 pb;
    float3 pc;
    uint32_t index;
};

struct GPUVertex {
    GPUVertex() {
        p = float3{0.0f, 0.0f, 0.0f};
        n0 = float3{0.0f, 0.0f, 0.0f};
        n1 = float3{0.0f, 0.0f, 0.0f};
        index = FCPW_GPU_UINT_MAX;
        hasOneAdjacentFace = 0;
    }
    GPUVertex(const float3& p_, const float3& n0_, const float3& n1_, 
              uint32_t index_, uint32_t hasOneAdjacentFace_):
              p(p_), n0(n0_), n1(n1_), index(index_),
              hasOneAdjacentFace(hasOneAdjacentFace_) {}

    float3 p;
    float3 n0;
    float3 n1;
    uint32_t index;
    uint32_t hasOneAdjacentFace;
};

struct GPUEdge {
    GPUEdge() {
        pa = float3{0.0f, 0.0f, 0.0f};
        pb = float3{0.0f, 0.0f, 0.0f};
        n0 = float3{0.0f, 0.0f, 0.0f};
        n1 = float3{0.0f, 0.0f, 0.0f};
        index = FCPW_GPU_UINT_MAX;
        hasOneAdjacentFace = 0;
    }
    GPUEdge(const float3& pa_, const float3& pb_,
            const float3& n0_, const float3& n1_,
            uint32_t index_, uint32_t hasOneAdjacentFace_):
            pa(pa_), pb(pb_), n0(n0_), n1(n1_), index(index_),
            hasOneAdjacentFace(hasOneAdjacentFace_) {}

    float3 pa;
    float3 pb;
    float3 n0;
    float3 n1;
    uint32_t index;
    uint32_t hasOneAdjacentFace;
};

struct GPUNoSilhouette {
    GPUNoSilhouette() {
        index = FCPW_GPU_UINT_MAX;
    }

    uint32_t index;
};

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

        GPUBoundingBox gpuBoundingBox(float3{pMin[0], pMin[1], DIM == 2 ? 0.0f : pMin[2]},
                                      float3{pMax[0], pMax[1], DIM == 2 ? 0.0f : pMax[2]});
        gpuBvhNodes[i] = GPUBvhNode(gpuBoundingBox, nPrimitives, offset);
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

        GPUBoundingBox gpuBoundingBox(float3{pMin[0], pMin[1], DIM == 2 ? 0.0f : pMin[2]},
                                      float3{pMax[0], pMax[1], DIM == 2 ? 0.0f : pMax[2]});
        GPUBoundingCone gpuBoundingCone(float3{axis[0], axis[1], DIM == 2 ? 0.0f : axis[2]},
                                        halfAngle, radius);
        gpuSnchNodes[i] = GPUSnchNode(gpuBoundingBox, gpuBoundingCone, nPrimitives,
                                      offset, nSilhouettes, silhouetteOffset);
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

template<size_t DIM,
         typename NodeType,
         typename PrimitiveType,
         typename SilhouetteType,
         typename GPUNodeType,
         typename GPUPrimitiveType,
         typename GPUSilhouetteType>
class CPUBvhDataExtractor {
public:
    // constructor
    CPUBvhDataExtractor(const Bvh<DIM, NodeType, PrimitiveType, SilhouetteType> *bvh_) {
        std::cerr << "CPUBvhDataExtractor() not supported" << std::endl;
        exit(EXIT_FAILURE);
    }

    // populates GPU bvh nodes array from CPU bvh
    void extractNodes(std::vector<GPUNodeType>& gpuNodes) {
        std::cerr << "CPUBvhDataExtractor::extractNodes() not supported" << std::endl;
        exit(EXIT_FAILURE);
    }

    // populates GPU bvh primitives array from CPU bvh
    void extractPrimitives(std::vector<GPUPrimitiveType>& gpuPrimitives) {
        std::cerr << "CPUBvhDataExtractor::extractPrimitives() not supported" << std::endl;
        exit(EXIT_FAILURE);
    }

    // populates GPU bvh silhouettes array from CPU bvh
    void extractSilhouettes(std::vector<GPUSilhouetteType>& gpuSilhouettes) {
        std::cerr << "CPUBvhDataExtractor::extractSilhouettes() not supported" << std::endl;
        exit(EXIT_FAILURE);
    }

    // returns reflection type
    std::string getReflectionType() const {
        std::cerr << "CPUBvhDataExtractor::getReflectionType() not supported" << std::endl;
        exit(EXIT_FAILURE);

        return "";
    }
};

template<>
class CPUBvhDataExtractor<2, SnchNode<2>, LineSegment, SilhouetteVertex, GPUSnchNode, GPULineSegment, GPUVertex> {
public:
    // constructor
    CPUBvhDataExtractor(const Bvh<2, SnchNode<2>, LineSegment, SilhouetteVertex> *bvh_): bvh(bvh_) {}

    // populates GPU bvh nodes array from CPU bvh
    void extractNodes(std::vector<GPUSnchNode>& gpuSnchNodes) {
        extractSnchNodes<2>(bvh->flatTree, gpuSnchNodes);
    }

    // populates GPU bvh primitives array from CPU bvh
    void extractPrimitives(std::vector<GPULineSegment>& gpuLineSegments) {
        extractLineSegments(bvh->primitives, gpuLineSegments);
    }

    // populates GPU bvh silhouettes array from CPU bvh
    void extractSilhouettes(std::vector<GPUVertex>& gpuVertices) {
        extractSilhouetteVertices(bvh->silhouetteRefs, gpuVertices);
    }

    // returns reflection type
    std::string getReflectionType() const {
        return "Bvh<SnchNode, LineSegment, Vertex>";
    }

    // member
    const Bvh<2, SnchNode<2>, LineSegment, SilhouetteVertex> *bvh;
};

template<>
class CPUBvhDataExtractor<3, SnchNode<3>, Triangle, SilhouetteEdge, GPUSnchNode, GPUTriangle, GPUEdge> {
public:
    // constructor
    CPUBvhDataExtractor(const Bvh<3, SnchNode<3>, Triangle, SilhouetteEdge> *bvh_): bvh(bvh_) {}

    // populates GPU bvh nodes array from CPU bvh
    void extractNodes(std::vector<GPUSnchNode>& gpuSnchNodes) {
        extractSnchNodes<3>(bvh->flatTree, gpuSnchNodes);
    }

    // populates GPU bvh primitives array from CPU bvh
    void extractPrimitives(std::vector<GPUTriangle>& gpuTriangles) {
        extractTriangles(bvh->primitives, gpuTriangles);
    }

    // populates GPU bvh silhouettes array from CPU bvh
    void extractSilhouettes(std::vector<GPUEdge>& gpuEdges) {
        extractSilhouetteEdges(bvh->silhouetteRefs, gpuEdges);
    }

    // returns reflection type
    std::string getReflectionType() const {
        return "Bvh<SnchNode, Triangle, Edge>";
    }

    // member
    const Bvh<3, SnchNode<3>, Triangle, SilhouetteEdge> *bvh;
};

template<>
class CPUBvhDataExtractor<2, BvhNode<2>, LineSegment, SilhouettePrimitive<2>, GPUBvhNode, GPULineSegment, GPUNoSilhouette> {
public:
    // constructor
    CPUBvhDataExtractor(const Bvh<2, BvhNode<2>, LineSegment, SilhouettePrimitive<2>> *bvh_): bvh(bvh_) {}

    // populates GPU bvh nodes array from CPU bvh
    void extractNodes(std::vector<GPUBvhNode>& gpuBvhNodes) {
        extractBvhNodes<2>(bvh->flatTree, gpuBvhNodes);
    }

    // populates GPU bvh primitives array from CPU bvh
    void extractPrimitives(std::vector<GPULineSegment>& gpuLineSegments) {
        extractLineSegments(bvh->primitives, gpuLineSegments);
    }

    // populates GPU bvh silhouettes array from CPU bvh
    void extractSilhouettes(std::vector<GPUNoSilhouette>& gpuSilhouettes) {
        gpuSilhouettes.clear();
    }

    // returns reflection type
    std::string getReflectionType() const {
        return "Bvh<BvhNode, LineSegment, NoSilhouette>";
    }

    // member
    const Bvh<2, BvhNode<2>, LineSegment, SilhouettePrimitive<2>> *bvh;
};

template<>
class CPUBvhDataExtractor<3, BvhNode<3>, Triangle, SilhouettePrimitive<3>, GPUBvhNode, GPUTriangle, GPUNoSilhouette> {
public:
    // constructor
    CPUBvhDataExtractor(const Bvh<3, BvhNode<3>, Triangle, SilhouettePrimitive<3>> *bvh_): bvh(bvh_) {}

    // populates GPU bvh nodes array from CPU bvh
    void extractNodes(std::vector<GPUBvhNode>& gpuBvhNodes) {
        extractBvhNodes<3>(bvh->flatTree, gpuBvhNodes);
    }

    // populates GPU bvh primitives array from CPU bvh
    void extractPrimitives(std::vector<GPUTriangle>& gpuTriangles) {
        extractTriangles(bvh->primitives, gpuTriangles);
    }

    // populates GPU bvh silhouettes array from CPU bvh
    void extractSilhouettes(std::vector<GPUNoSilhouette>& gpuSilhouettes) {
        gpuSilhouettes.clear();
    }

    // returns reflection type
    std::string getReflectionType() const {
        return "Bvh<BvhNode, Triangle, NoSilhouette>";
    }

    // member
    const Bvh<3, BvhNode<3>, Triangle, SilhouettePrimitive<3>> *bvh;
};

template<size_t DIM,
         typename NodeType,
         typename PrimitiveType,
         typename SilhouetteType>
class CPUBvhUpdateDataExtractor {
public:
    // constructor
    CPUBvhUpdateDataExtractor(const Bvh<DIM, NodeType, PrimitiveType, SilhouetteType> *bvh_): bvh(bvh_) {}

    // populates update data from CPU bvh
    // source: https://github.com/NVIDIAGameWorks/Falcor/blob/58ce2d1eafce67b4cb9d304029068c7fb31bd831/Source/Falcor/Rendering/Lights/LightBVH.cpp#L219
    uint32_t extract(std::vector<uint32_t>& nodeIndicesData,
                     std::vector<std::pair<uint32_t, uint32_t>>& updateEntryData) {
        // count number of nodes at each level
        int maxDepth = bvh->maxDepth;
        updateEntryData.resize(maxDepth + 1, std::make_pair(0, 0));
        updateEntryData[maxDepth].second = bvh->nLeafs;
        traverseBvh(
            [&updateEntryData](int index, int depth) { ++updateEntryData[depth].second; },
            [](int index, int depth) { /* do nothing */ }
        );

        // record offsets into nodeIndicesData
        std::vector<uint32_t> offsets(maxDepth + 1, 0);
        for (uint32_t i = 1; i < maxDepth + 1; i++) {
            uint32_t currentOffset = updateEntryData[i - 1].first + updateEntryData[i - 1].second;
            offsets[i] = updateEntryData[i].first = currentOffset;
        }

        // populate nodeIndicesData such that:
        //  level 0: indices to all internal nodes at level 0
        //  ...
        //  level (maxDepth - 1): indices to all internal nodes at level (maxDepth - 1)
        //  level maxDepth: indices to all leaf nodes
        nodeIndicesData.resize(bvh->nNodes, 0);
        traverseBvh(
            [&nodeIndicesData, &offsets](int index, int depth) { nodeIndicesData[offsets[depth]++] = index; },
            [&nodeIndicesData, &offsets](int index, int depth) { nodeIndicesData[offsets.back()++] = index; }
        );

        return maxDepth;
    }

    // member
    const Bvh<DIM, NodeType, PrimitiveType, SilhouetteType> *bvh;

private:
    void traverseBvh(const std::function<void(int index, int depth)>& evalInternalNode,
                     const std::function<void(int index, int depth)>& evalLeafNode) {
        struct TraversalStack {
            int nodeIndex;
            int nodeDepth;
        };

        TraversalStack stack[FCPW_BVH_MAX_DEPTH];
        stack[0].nodeIndex = 0;
        stack[0].nodeDepth = 0;
        int stackPtr = 0;

        while (stackPtr >= 0) {
            // pop off the next node to work on
            int nodeIndex = stack[stackPtr].nodeIndex;
            int nodeDepth = stack[stackPtr].nodeDepth;
            stackPtr--;

            const NodeType& node(bvh->flatTree[nodeIndex]);
            if (node.nReferences > 0) { // leaf
                evalLeafNode(nodeIndex, nodeDepth);

            } else { // internal node
                evalInternalNode(nodeIndex, nodeDepth);

                stackPtr++;
                stack[stackPtr].nodeIndex = nodeIndex + 1;
                stack[stackPtr].nodeDepth = nodeDepth + 1;
                stackPtr++;
                stack[stackPtr].nodeIndex = nodeIndex + node.secondChildOffset;
                stack[stackPtr].nodeDepth = nodeDepth + 1;
            }
        }
    }
};

class GPUBvhBuffers : public GPUShaderObject {
public:
    GPUBuffer nodes = {};
    GPUBuffer primitives = {};
    GPUBuffer silhouettes = {};
    GPUBuffer nodeIndices = {};
    std::vector<std::pair<uint32_t, uint32_t>> updateEntryData;
    uint32_t maxUpdateDepth = 0;
    std::string reflectionType = "";

    template<size_t DIM>
    void allocate(GPUContext& gpuContext, const SceneData<DIM> *cpuSceneData,
                  bool allocatePrimitiveData, bool allocateSilhouetteData,
                  bool allocateNodeData, bool allocateRefitData) {
        std::cerr << "GPUBvhBuffers::allocate()" << std::endl;
        exit(EXIT_FAILURE);
    }

    void setResources(const ShaderCursor& cursor, bool printLogs) const {
        cursor["nodes"].setResource(nodes.view);
        cursor["primitives"].setResource(primitives.view);
        cursor["silhouettes"].setResource(silhouettes.view);
        if (printLogs) printReflectionInfo(cursor, 3, reflectionType, getName());
    }

    std::string getReflectionType() const {
        return reflectionType;
    }

    std::string getName() const {
        return "gBvh";
    }

private:
    template<size_t DIM,
             typename NodeType,
             typename PrimitiveType,
             typename SilhouetteType,
             typename GpuNodeType,
             typename GPUPrimitiveType,
             typename GPUSilhouetteType>
    void allocateGeometryBuffers(GPUContext& gpuContext, const SceneData<DIM> *cpuSceneData) {
        // extract primitives and silhouettes data from cpu bvh
        const Bvh<DIM, NodeType, PrimitiveType, SilhouetteType> *bvh =
            reinterpret_cast<const Bvh<DIM, NodeType, PrimitiveType, SilhouetteType> *>(
                cpuSceneData->aggregate.get());
        CPUBvhDataExtractor<DIM,
                            NodeType,
                            PrimitiveType,
                            SilhouetteType,
                            GpuNodeType,
                            GPUPrimitiveType,
                            GPUSilhouetteType> cpuBvhDataExtractor(bvh);

        std::vector<GPUPrimitiveType> primitivesData;
        std::vector<GPUSilhouetteType> silhouettesData;
        cpuBvhDataExtractor.extractPrimitives(primitivesData);
        cpuBvhDataExtractor.extractSilhouettes(silhouettesData);

        // allocate gpu buffers
        primitives.allocate<GPUPrimitiveType>(gpuContext, false, primitivesData);
        silhouettes.allocate<GPUSilhouetteType>(gpuContext, false, silhouettesData);
    }

    template<size_t DIM,
             typename NodeType,
             typename PrimitiveType,
             typename SilhouetteType,
             typename GpuNodeType,
             typename GPUPrimitiveType,
             typename GPUSilhouetteType>
    void allocateNodeBuffer(GPUContext& gpuContext, const SceneData<DIM> *cpuSceneData) {
        // extract nodes data from cpu bvh
        const Bvh<DIM, NodeType, PrimitiveType, SilhouetteType> *bvh =
            reinterpret_cast<const Bvh<DIM, NodeType, PrimitiveType, SilhouetteType> *>(
                cpuSceneData->aggregate.get());
        CPUBvhDataExtractor<DIM,
                            NodeType,
                            PrimitiveType,
                            SilhouetteType,
                            GpuNodeType,
                            GPUPrimitiveType,
                            GPUSilhouetteType> cpuBvhDataExtractor(bvh);

        std::vector<GpuNodeType> nodesData;
        cpuBvhDataExtractor.extractNodes(nodesData);
        reflectionType = cpuBvhDataExtractor.getReflectionType();

        // allocate gpu buffer
        nodes.allocate<GpuNodeType>(gpuContext, true, nodesData);
    }

    template<size_t DIM,
             typename NodeType,
             typename PrimitiveType,
             typename SilhouetteType>
    void allocateRefitBuffer(GPUContext& gpuContext, const SceneData<DIM> *cpuSceneData) {
        // extract update data from cpu bvh
        const Bvh<DIM, NodeType, PrimitiveType, SilhouetteType> *bvh =
            reinterpret_cast<const Bvh<DIM, NodeType, PrimitiveType, SilhouetteType> *>(
                cpuSceneData->aggregate.get());
        CPUBvhUpdateDataExtractor<DIM,
                                  NodeType,
                                  PrimitiveType,
                                  SilhouetteType> cpuBvhUpdateDataExtractor(bvh);

        updateEntryData.clear();
        std::vector<uint32_t> nodeIndicesData;
        maxUpdateDepth = cpuBvhUpdateDataExtractor.extract(nodeIndicesData, updateEntryData);

        // allocate gpu buffer
        nodeIndices.allocate<uint32_t>(gpuContext, false, nodeIndicesData);
    }
};

template<>
void GPUBvhBuffers::allocate<2>(GPUContext& gpuContext, const SceneData<2> *cpuSceneData,
                                bool allocatePrimitiveData, bool allocateSilhouetteData,
                                bool allocateNodeData, bool allocateRefitData)
{
    if (allocateSilhouetteData) {
        if (allocatePrimitiveData) {
            allocateGeometryBuffers<2, SnchNode<2>, LineSegment, SilhouetteVertex,
                                    GPUSnchNode, GPULineSegment, GPUVertex>(gpuContext, cpuSceneData);
        }

        if (allocateNodeData) {
            allocateNodeBuffer<2, SnchNode<2>, LineSegment, SilhouetteVertex,
                               GPUSnchNode, GPULineSegment, GPUVertex>(gpuContext, cpuSceneData);
        }

        if (allocateRefitData) {
            allocateRefitBuffer<2, SnchNode<2>, LineSegment, SilhouetteVertex>(gpuContext, cpuSceneData);
        }

    } else {
        if (allocatePrimitiveData) {
            allocateGeometryBuffers<2, BvhNode<2>, LineSegment, SilhouettePrimitive<2>,
                                    GPUBvhNode, GPULineSegment, GPUNoSilhouette>(gpuContext, cpuSceneData);
        }

        if (allocateNodeData) {
            allocateNodeBuffer<2, BvhNode<2>, LineSegment, SilhouettePrimitive<2>,
                               GPUBvhNode, GPULineSegment, GPUNoSilhouette>(gpuContext, cpuSceneData);
        }

        if (allocateRefitData) {
            allocateRefitBuffer<2, BvhNode<2>, LineSegment, SilhouettePrimitive<2>>(gpuContext, cpuSceneData);
        }
    }
}

template<>
void GPUBvhBuffers::allocate<3>(GPUContext& gpuContext, const SceneData<3> *cpuSceneData,
                                bool allocatePrimitiveData, bool allocateSilhouetteData,
                                bool allocateNodeData, bool allocateRefitData)
{
    if (allocateSilhouetteData) {
        if (allocatePrimitiveData) {
            allocateGeometryBuffers<3, SnchNode<3>, Triangle, SilhouetteEdge,
                                    GPUSnchNode, GPUTriangle, GPUEdge>(gpuContext, cpuSceneData);
        }

        if (allocateNodeData) {
            allocateNodeBuffer<3, SnchNode<3>, Triangle, SilhouetteEdge,
                               GPUSnchNode, GPUTriangle, GPUEdge>(gpuContext, cpuSceneData);
        }

        if (allocateRefitData) {
            allocateRefitBuffer<3, SnchNode<3>, Triangle, SilhouetteEdge>(gpuContext, cpuSceneData);
        }

    } else {
        if (allocatePrimitiveData) {
            allocateGeometryBuffers<3, BvhNode<3>, Triangle, SilhouettePrimitive<3>,
                                    GPUBvhNode, GPUTriangle, GPUNoSilhouette>(gpuContext, cpuSceneData);
        }

        if (allocateNodeData) {
            allocateNodeBuffer<3, BvhNode<3>, Triangle, SilhouettePrimitive<3>,
                               GPUBvhNode, GPUTriangle, GPUNoSilhouette>(gpuContext, cpuSceneData);
        }

        if (allocateRefitData) {
            allocateRefitBuffer<3, BvhNode<3>, Triangle, SilhouettePrimitive<3>>(gpuContext, cpuSceneData);
        }
    }
}

struct GPURay {
    GPURay() {
        o = float3{0.0f, 0.0f, 0.0f};
        d = float3{0.0f, 0.0f, 0.0f};
        dInv = float3{0.0f, 0.0f, 0.0f};
        tMax = maxFloat;
    }
    GPURay(const float3& o_, const float3& d_, float tMax_=maxFloat): o(o_), d(d_), tMax(tMax_) {
        dInv.x = 1.0f/d.x;
        dInv.y = 1.0f/d.y;
        dInv.z = 1.0f/d.z;
    }

    float3 o;    // ray origin
    float3 d;    // ray direction
    float3 dInv; // 1 over ray direction (coordinate-wise)
    float tMax;  // max ray distance
};

struct GPUBoundingSphere {
    GPUBoundingSphere() {
        c = float3{0.0f, 0.0f, 0.0f};
        r2 = 0.0f;
    }
    GPUBoundingSphere(const float3& c_, float r2_): c(c_), r2(r2_) {}

    float3 c; // sphere center
    float r2; // sphere squared radius
};

struct GPUInteraction {
    GPUInteraction() {
        p = float3{0.0f, 0.0f, 0.0f};
        n = float3{0.0f, 0.0f, 0.0f};
        uv = float2{0.0f, 0.0f};
        d = maxFloat;
        index = FCPW_GPU_UINT_MAX;
    }

    float3 p;       // interaction point associated with query
    float3 n;       // normal at interaction point
    float2 uv;      // uv coordinates of interaction point
    float d;        // distance to interaction point
    uint32_t index; // index of primitive/silhouette associated with interaction point
};

class GPURunRayIntersectionQuery : public GPUShaderEntryPoint {
public:
    GPUBuffer rays = {};
    GPUBuffer interactions = {};
    uint32_t checkForOcclusion = 0;
    uint32_t nQueries = 0;

    void allocate(GPUContext& gpuContext, const std::vector<GPURay>& raysData) {
        rays.allocate<GPURay>(gpuContext, false, raysData);
        nQueries = (uint32_t)raysData.size();
        interactions.allocate<GPUInteraction>(gpuContext, true, std::vector<GPUInteraction>(nQueries));
    }

    int setResources(const ShaderCursor& cursor) const {
        cursor.getPath("rays").setResource(rays.view);
        cursor.getPath("interactions").setResource(interactions.view);
        cursor.getPath("checkForOcclusion").setData(checkForOcclusion);
        cursor.getPath("nQueries").setData(nQueries);

        return 4;
    }

    void read(GPUContext& gpuContext, std::vector<GPUInteraction>& interactionsData) const {
        interactions.read<GPUInteraction>(gpuContext, nQueries, interactionsData);
    }

    std::string getName() const {
        return "rayIntersection";
    }
};

class GPURunSphereIntersectionQuery : public GPUShaderEntryPoint {
public:
    GPUBuffer boundingSpheres = {};
    GPUBuffer randNums = {};
    GPUBuffer interactions = {};
    uint32_t nQueries = 0;

    void allocate(GPUContext& gpuContext,
                  const std::vector<GPUBoundingSphere>& boundingSpheresData,
                  const std::vector<float3>& randNumsData) {
        boundingSpheres.allocate<GPUBoundingSphere>(gpuContext, false, boundingSpheresData);
        randNums.allocate<float3>(gpuContext, false, randNumsData);
        nQueries = (uint32_t)boundingSpheresData.size();
        interactions.allocate<GPUInteraction>(gpuContext, true, std::vector<GPUInteraction>(nQueries));
    }

    int setResources(const ShaderCursor& cursor) const {
        cursor.getPath("boundingSpheres").setResource(boundingSpheres.view);
        cursor.getPath("randNums").setResource(randNums.view);
        cursor.getPath("interactions").setResource(interactions.view);
        cursor.getPath("nQueries").setData(nQueries);

        return 4;
    }

    void read(GPUContext& gpuContext, std::vector<GPUInteraction>& interactionsData) const {
        interactions.read<GPUInteraction>(gpuContext, nQueries, interactionsData);
    }

    std::string getName() const {
        return "sphereIntersection";
    }
};

class GPURunClosestPointQuery : public GPUShaderEntryPoint {
public:
    GPUBuffer boundingSpheres = {};
    GPUBuffer interactions = {};
    uint32_t recordNormals = 0;
    uint32_t nQueries = 0;

    void allocate(GPUContext& gpuContext,
                  const std::vector<GPUBoundingSphere>& boundingSpheresData) {
        boundingSpheres.allocate<GPUBoundingSphere>(gpuContext, false, boundingSpheresData);
        nQueries = (uint32_t)boundingSpheresData.size();
        interactions.allocate<GPUInteraction>(gpuContext, true, std::vector<GPUInteraction>(nQueries));
    }

    int setResources(const ShaderCursor& cursor) const {
        cursor.getPath("boundingSpheres").setResource(boundingSpheres.view);
        cursor.getPath("interactions").setResource(interactions.view);
        cursor.getPath("recordNormals").setData(recordNormals);
        cursor.getPath("nQueries").setData(nQueries);

        return 4;
    }

    void read(GPUContext& gpuContext, std::vector<GPUInteraction>& interactionsData) const {
        interactions.read<GPUInteraction>(gpuContext, nQueries, interactionsData);
    }

    std::string getName() const {
        return "closestPoint";
    }
};

class GPURunClosestSilhouettePointQuery : public GPUShaderEntryPoint {
public:
    GPUBuffer boundingSpheres = {};
    GPUBuffer flipNormalOrientation = {};
    GPUBuffer interactions = {};
    float squaredMinRadius = 1e-6f;
    float precision = 1e-3f;
    uint32_t nQueries = 0;

    void allocate(GPUContext& gpuContext,
                  const std::vector<GPUBoundingSphere>& boundingSpheresData,
                  const std::vector<uint32_t>& flipNormalOrientationData) {
        boundingSpheres.allocate<GPUBoundingSphere>(gpuContext, false, boundingSpheresData);
        flipNormalOrientation.allocate<uint32_t>(gpuContext, false, flipNormalOrientationData);
        nQueries = (uint32_t)boundingSpheresData.size();
        interactions.allocate<GPUInteraction>(gpuContext, true, std::vector<GPUInteraction>(nQueries));
    }

    int setResources(const ShaderCursor& cursor) const {
        cursor.getPath("boundingSpheres").setResource(boundingSpheres.view);
        cursor.getPath("flipNormalOrientation").setResource(flipNormalOrientation.view);
        cursor.getPath("interactions").setResource(interactions.view);
        cursor.getPath("squaredMinRadius").setData(squaredMinRadius);
        cursor.getPath("precision").setData(precision);
        cursor.getPath("nQueries").setData(nQueries);

        return 6;
    }

    void read(GPUContext& gpuContext, std::vector<GPUInteraction>& interactionsData) const {
        interactions.read<GPUInteraction>(gpuContext, nQueries, interactionsData);
    }

    std::string getName() const {
        return "closestSilhouettePoint";
    }
};

} // namespace fcpw
