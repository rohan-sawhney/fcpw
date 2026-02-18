#pragma once

#include <cuda_runtime.h>

#include <fcpw/cuda/cuda_runtime_utils.h>
#include <fcpw/geometry/line_segments.h>
#include <fcpw/geometry/silhouette_edges.h>
#include <fcpw/geometry/silhouette_vertices.h>
#include <fcpw/geometry/triangles.h>
#include <fcpw/aggregates/bvh.h>
#include <fcpw/utilities/scene_data.h>
#include <cstdlib>
#include <iostream>
#include <functional>
#include <utility>
#include <vector>

#define FCPW_CUDA_UINT_MAX 4294967295u

namespace fcpw {

// CUDA-side mirror structs for CPU BVH/primitives/silhouettes.
// Naming intentionally parallels include/fcpw/gpu/bvh_interop_structures.h.

using float2 = ::float2;
using float3 = ::float3;

enum class CUDABvhType : uint32_t {
    LineSegmentBvh = 1,
    TriangleBvh = 2,
    LineSegmentSnch = 3,
    TriangleSnch = 4,
};

struct CUDABoundingBox {
    float3 pMin;
    float3 pMax;
};

struct CUDABoundingCone {
    float3 axis;
    float halfAngle;
    float radius;
};

struct CUDABvhNode {
    CUDABoundingBox box;
    uint32_t nPrimitives;
    uint32_t offset;
};

struct CUDASnchNode {
    CUDABoundingBox box;
    CUDABoundingCone cone;
    uint32_t nPrimitives;
    uint32_t offset;
    uint32_t nSilhouettes;
    uint32_t silhouetteOffset;
};

struct CUDALineSegment {
    float3 pa;
    float3 pb;
    uint32_t index;
};

struct CUDATriangle {
    float3 pa;
    float3 pb;
    float3 pc;
    uint32_t index;
};

struct CUDAVertex {
    float3 p;
    float3 n0;
    float3 n1;
    uint32_t index;
    uint32_t hasOneAdjacentFace;
};

struct CUDAEdge {
    float3 pa;
    float3 pb;
    float3 n0;
    float3 n1;
    uint32_t index;
    uint32_t hasOneAdjacentFace;
};

struct CUDANoSilhouette {
    uint32_t index;
};

struct CUDARay {
    float3 o;
    float3 d;
    float3 dInv;
    float tMax;

    __host__ __device__ CUDARay() {
        o = float3{0.0f, 0.0f, 0.0f};
        d = float3{0.0f, 0.0f, 0.0f};
        dInv = float3{0.0f, 0.0f, 0.0f};
        tMax = maxFloat;
    }

    __host__ __device__ CUDARay(const float3& o_, const float3& d_, float tMax_=maxFloat): o(o_), d(d_), tMax(tMax_) {
        dInv = float3{1.0f/d.x, 1.0f/d.y, 1.0f/d.z};
    }
};

struct CUDABoundingSphere {
    float3 c;
    float r2;

    __host__ __device__ CUDABoundingSphere() {
        c = float3{0.0f, 0.0f, 0.0f};
        r2 = 0.0f;
    }

    __host__ __device__ CUDABoundingSphere(const float3& c_, float r2_): c(c_), r2(r2_) {}
};

struct CUDAInteraction {
    float3 p;
    float3 n;
    float2 uv;
    float d;
    uint32_t index;

    __host__ __device__ CUDAInteraction() {
        p = float3{0.0f, 0.0f, 0.0f};
        n = float3{0.0f, 0.0f, 0.0f};
        uv = float2{0.0f, 0.0f};
        d = maxFloat;
        index = FCPW_CUDA_UINT_MAX;
    }
};

template<size_t DIM>
inline void extractBvhNodes(const std::vector<BvhNode<DIM>>& flatTree,
                            std::vector<CUDABvhNode>& cudaNodes)
{
    int nNodes = (int)flatTree.size();
    cudaNodes.resize(nNodes);

    for (int i = 0; i < nNodes; i++) {
        const BvhNode<DIM>& node = flatTree[i];
        const Vector<DIM>& pMin = node.box.pMin;
        const Vector<DIM>& pMax = node.box.pMax;
        uint32_t nPrimitives = node.nReferences;
        uint32_t offset = nPrimitives > 0 ? node.referenceOffset : node.secondChildOffset;

        cudaNodes[i] = CUDABvhNode{{{pMin[0], pMin[1], DIM == 2 ? 0.0f : pMin[2]},
                                    {pMax[0], pMax[1], DIM == 2 ? 0.0f : pMax[2]}},
                                   nPrimitives,
                                   offset};
    }
}

template<size_t DIM>
inline void extractSnchNodes(const std::vector<SnchNode<DIM>>& flatTree,
                             std::vector<CUDASnchNode>& cudaNodes)
{
    int nNodes = (int)flatTree.size();
    cudaNodes.resize(nNodes);

    for (int i = 0; i < nNodes; i++) {
        const SnchNode<DIM>& node = flatTree[i];
        const Vector<DIM>& pMin = node.box.pMin;
        const Vector<DIM>& pMax = node.box.pMax;
        const Vector<DIM>& axis = node.cone.axis;
        uint32_t nPrimitives = node.nReferences;
        uint32_t offset = nPrimitives > 0 ? node.referenceOffset : node.secondChildOffset;

        cudaNodes[i].box = CUDABoundingBox{{pMin[0], pMin[1], DIM == 2 ? 0.0f : pMin[2]},
                                           {pMax[0], pMax[1], DIM == 2 ? 0.0f : pMax[2]}};
        cudaNodes[i].cone = CUDABoundingCone{{axis[0], axis[1], DIM == 2 ? 0.0f : axis[2]},
                                             node.cone.halfAngle,
                                             node.cone.radius};
        cudaNodes[i].nPrimitives = nPrimitives;
        cudaNodes[i].offset = offset;
        cudaNodes[i].nSilhouettes = node.nSilhouetteReferences;
        cudaNodes[i].silhouetteOffset = node.silhouetteReferenceOffset;
    }
}

inline void extractLineSegments(const std::vector<LineSegment *>& primitives,
                                std::vector<CUDALineSegment>& cudaPrimitives)
{
    int nPrimitives = (int)primitives.size();
    cudaPrimitives.resize(nPrimitives);

    for (int i = 0; i < nPrimitives; i++) {
        const LineSegment *s = primitives[i];
        const Vector2& pa = s->soup->positions[s->indices[0]];
        const Vector2& pb = s->soup->positions[s->indices[1]];
        cudaPrimitives[i] = CUDALineSegment{{pa[0], pa[1], 0.0f},
                                            {pb[0], pb[1], 0.0f},
                                            (uint32_t)s->pIndex};
    }
}

inline void extractTriangles(const std::vector<Triangle *>& primitives,
                             std::vector<CUDATriangle>& cudaPrimitives)
{
    int nPrimitives = (int)primitives.size();
    cudaPrimitives.resize(nPrimitives);

    for (int i = 0; i < nPrimitives; i++) {
        const Triangle *t = primitives[i];
        const Vector3& pa = t->soup->positions[t->indices[0]];
        const Vector3& pb = t->soup->positions[t->indices[1]];
        const Vector3& pc = t->soup->positions[t->indices[2]];
        cudaPrimitives[i] = CUDATriangle{{pa[0], pa[1], pa[2]},
                                         {pb[0], pb[1], pb[2]},
                                         {pc[0], pc[1], pc[2]},
                                         (uint32_t)t->pIndex};
    }
}

inline void extractSilhouetteVertices(const std::vector<SilhouetteVertex *>& silhouettes,
                                      std::vector<CUDAVertex>& cudaSilhouettes)
{
    int nSilhouettes = (int)silhouettes.size();
    cudaSilhouettes.resize(nSilhouettes);

    for (int i = 0; i < nSilhouettes; i++) {
        const SilhouetteVertex *sv = silhouettes[i];
        const Vector2& p = sv->soup->positions[sv->indices[1]];
        Vector2 n0 = sv->hasFace(0) ? sv->normal(0) : Vector2::Zero();
        Vector2 n1 = sv->hasFace(1) ? sv->normal(1) : Vector2::Zero();
        bool hasTwoAdjacentFaces = sv->hasFace(0) && sv->hasFace(1);

        cudaSilhouettes[i] = CUDAVertex{{p[0], p[1], 0.0f},
                                        {n0[0], n0[1], 0.0f},
                                        {n1[0], n1[1], 0.0f},
                                        (uint32_t)sv->pIndex,
                                        hasTwoAdjacentFaces ? 0u : 1u};
    }
}

inline void extractSilhouetteEdges(const std::vector<SilhouetteEdge *>& silhouettes,
                                   std::vector<CUDAEdge>& cudaSilhouettes)
{
    int nSilhouettes = (int)silhouettes.size();
    cudaSilhouettes.resize(nSilhouettes);

    for (int i = 0; i < nSilhouettes; i++) {
        const SilhouetteEdge *se = silhouettes[i];
        const Vector3& pa = se->soup->positions[se->indices[1]];
        const Vector3& pb = se->soup->positions[se->indices[2]];
        Vector3 n0 = se->hasFace(0) ? se->normal(0) : Vector3::Zero();
        Vector3 n1 = se->hasFace(1) ? se->normal(1) : Vector3::Zero();
        bool hasTwoAdjacentFaces = se->hasFace(0) && se->hasFace(1);

        cudaSilhouettes[i] = CUDAEdge{{pa[0], pa[1], pa[2]},
                                      {pb[0], pb[1], pb[2]},
                                      {n0[0], n0[1], n0[2]},
                                      {n1[0], n1[1], n1[2]},
                                      (uint32_t)se->pIndex,
                                      hasTwoAdjacentFaces ? 0u : 1u};
    }
}

template<size_t DIM,
         typename NodeType,
         typename PrimitiveType,
         typename SilhouetteType,
         typename CUDA_Node,
         typename CUDA_Primitive,
         typename CUDA_Silhouette>
class CPUBvhDataExtractor {
public:
    // Generic fallback used only when a specialization is missing.
    // All supported combinations below provide concrete extractors.
    CPUBvhDataExtractor(const Bvh<DIM, NodeType, PrimitiveType, SilhouetteType> *bvh_): bvh(bvh_) {}

    void extractNodes(std::vector<CUDA_Node>& nodes) {
        std::cerr << "CPUBvhDataExtractor::extractNodes() not supported" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    void extractPrimitives(std::vector<CUDA_Primitive>& primitives) {
        std::cerr << "CPUBvhDataExtractor::extractPrimitives() not supported" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    void extractSilhouettes(std::vector<CUDA_Silhouette>& silhouettes) {
        std::cerr << "CPUBvhDataExtractor::extractSilhouettes() not supported" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    const Bvh<DIM, NodeType, PrimitiveType, SilhouetteType> *bvh;
};

template<>
class CPUBvhDataExtractor<2, SnchNode<2>, LineSegment, SilhouetteVertex, CUDASnchNode, CUDALineSegment, CUDAVertex> {
public:
    CPUBvhDataExtractor(const Bvh<2, SnchNode<2>, LineSegment, SilhouetteVertex> *bvh_): bvh(bvh_) {}
    void extractNodes(std::vector<CUDASnchNode>& nodes) { extractSnchNodes<2>(bvh->flatTree, nodes); }
    void extractPrimitives(std::vector<CUDALineSegment>& primitives) { extractLineSegments(bvh->primitives, primitives); }
    void extractSilhouettes(std::vector<CUDAVertex>& silhouettes) { extractSilhouetteVertices(bvh->silhouetteRefs, silhouettes); }
    const Bvh<2, SnchNode<2>, LineSegment, SilhouetteVertex> *bvh;
};

template<>
class CPUBvhDataExtractor<3, SnchNode<3>, Triangle, SilhouetteEdge, CUDASnchNode, CUDATriangle, CUDAEdge> {
public:
    CPUBvhDataExtractor(const Bvh<3, SnchNode<3>, Triangle, SilhouetteEdge> *bvh_): bvh(bvh_) {}
    void extractNodes(std::vector<CUDASnchNode>& nodes) { extractSnchNodes<3>(bvh->flatTree, nodes); }
    void extractPrimitives(std::vector<CUDATriangle>& primitives) { extractTriangles(bvh->primitives, primitives); }
    void extractSilhouettes(std::vector<CUDAEdge>& silhouettes) { extractSilhouetteEdges(bvh->silhouetteRefs, silhouettes); }
    const Bvh<3, SnchNode<3>, Triangle, SilhouetteEdge> *bvh;
};

template<>
class CPUBvhDataExtractor<2, BvhNode<2>, LineSegment, SilhouettePrimitive<2>, CUDABvhNode, CUDALineSegment, CUDANoSilhouette> {
public:
    CPUBvhDataExtractor(const Bvh<2, BvhNode<2>, LineSegment, SilhouettePrimitive<2>> *bvh_): bvh(bvh_) {}
    void extractNodes(std::vector<CUDABvhNode>& nodes) { extractBvhNodes<2>(bvh->flatTree, nodes); }
    void extractPrimitives(std::vector<CUDALineSegment>& primitives) { extractLineSegments(bvh->primitives, primitives); }
    void extractSilhouettes(std::vector<CUDANoSilhouette>& silhouettes) { silhouettes.clear(); }
    const Bvh<2, BvhNode<2>, LineSegment, SilhouettePrimitive<2>> *bvh;
};

template<>
class CPUBvhDataExtractor<3, BvhNode<3>, Triangle, SilhouettePrimitive<3>, CUDABvhNode, CUDATriangle, CUDANoSilhouette> {
public:
    CPUBvhDataExtractor(const Bvh<3, BvhNode<3>, Triangle, SilhouettePrimitive<3>> *bvh_): bvh(bvh_) {}
    void extractNodes(std::vector<CUDABvhNode>& nodes) { extractBvhNodes<3>(bvh->flatTree, nodes); }
    void extractPrimitives(std::vector<CUDATriangle>& primitives) { extractTriangles(bvh->primitives, primitives); }
    void extractSilhouettes(std::vector<CUDANoSilhouette>& silhouettes) { silhouettes.clear(); }
    const Bvh<3, BvhNode<3>, Triangle, SilhouettePrimitive<3>> *bvh;
};

template<size_t DIM,
         typename NodeType,
         typename PrimitiveType,
         typename SilhouetteType>
class CPUBvhUpdateDataExtractor {
public:
    // Builds level-by-level node update lists for bottom-up refit.
    CPUBvhUpdateDataExtractor(const Bvh<DIM, NodeType, PrimitiveType, SilhouetteType> *bvh_): bvh(bvh_) {}

    uint32_t extract(std::vector<uint32_t>& nodeIndicesData,
                     std::vector<std::pair<uint32_t, uint32_t>>& updateEntryData) {
        int maxDepth = bvh->maxDepth;
        updateEntryData.resize(maxDepth + 1, std::make_pair(0u, 0u));
        updateEntryData[maxDepth].second = bvh->nLeafs;

        traverseBvh([&updateEntryData](int index, int depth) {
                        (void)index;
                        ++updateEntryData[depth].second;
                    },
                    [](int index, int depth) {
                        (void)index;
                        (void)depth;
                    });

        std::vector<uint32_t> offsets(maxDepth + 1, 0);
        for (uint32_t i = 1; i < (uint32_t)maxDepth + 1u; i++) {
            uint32_t currentOffset = updateEntryData[i - 1].first + updateEntryData[i - 1].second;
            offsets[i] = updateEntryData[i].first = currentOffset;
        }

        nodeIndicesData.resize(bvh->nNodes, 0u);
        traverseBvh([&nodeIndicesData, &offsets](int index, int depth) {
                        nodeIndicesData[offsets[depth]++] = (uint32_t)index;
                    },
                    [&nodeIndicesData, &offsets](int index, int depth) {
                        (void)depth;
                        nodeIndicesData[offsets.back()++] = (uint32_t)index;
                    });

        return (uint32_t)maxDepth;
    }

private:
    void traverseBvh(const std::function<void(int, int)>& evalInternalNode,
                     const std::function<void(int, int)>& evalLeafNode) {
        // Explicit stack traversal avoids recursion and mirrors GPU traversal order.
        struct TraversalStack {
            int nodeIndex;
            int nodeDepth;
        };

        TraversalStack stack[FCPW_BVH_MAX_DEPTH];
        stack[0].nodeIndex = 0;
        stack[0].nodeDepth = 0;
        int stackPtr = 0;

        while (stackPtr >= 0) {
            int nodeIndex = stack[stackPtr].nodeIndex;
            int nodeDepth = stack[stackPtr].nodeDepth;
            stackPtr--;

            const NodeType& node(bvh->flatTree[nodeIndex]);
            if (node.nReferences > 0) {
                evalLeafNode(nodeIndex, nodeDepth);
            } else {
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

public:
    const Bvh<DIM, NodeType, PrimitiveType, SilhouetteType> *bvh;
};

class CUDABvhBuffers {
public:
    // Device buffers for one transferred BVH instance.
    // The backend currently mirrors the single-aggregate transfer constraint.
    void *nodes = nullptr;
    void *primitives = nullptr;
    void *silhouettes = nullptr;
    uint32_t *nodeIndices = nullptr;

    size_t nodeCount = 0;
    size_t primitiveCount = 0;
    size_t silhouetteCount = 0;

    std::vector<std::pair<uint32_t, uint32_t>> updateEntryData;
    uint32_t maxUpdateDepth = 0;
    CUDABvhType bvhType = CUDABvhType::TriangleBvh;

    ~CUDABvhBuffers() {
        release();
    }

    void release() {
        // Safe to call multiple times; frees all CUDA allocations owned by this object.
        if (nodes) { FCPW_CUDA_CHECK(cudaFree(nodes)); nodes = nullptr; }
        if (primitives) { FCPW_CUDA_CHECK(cudaFree(primitives)); primitives = nullptr; }
        if (silhouettes) { FCPW_CUDA_CHECK(cudaFree(silhouettes)); silhouettes = nullptr; }
        if (nodeIndices) { FCPW_CUDA_CHECK(cudaFree(nodeIndices)); nodeIndices = nullptr; }
        nodeCount = primitiveCount = silhouetteCount = 0;
        updateEntryData.clear();
        maxUpdateDepth = 0;
    }

    template<size_t DIM>
    void allocate(const SceneData<DIM> *sceneData,
                  bool allocatePrimitiveData,
                  bool allocateSilhouetteData,
                  bool allocateNodeData,
                  bool allocateRefitData) {
        // Specialized in DIM=2 and DIM=3 variants below.
        // Flags allow selective refresh (e.g. geometry-only update before refit).
        std::cerr << "CUDABvhBuffers::allocate not supported for DIM " << DIM << std::endl;
        std::exit(EXIT_FAILURE);
    }

private:
    template<size_t DIM,
             typename NodeType,
             typename PrimitiveType,
             typename SilhouetteType,
             typename CUDA_Node,
             typename CUDA_Primitive,
             typename CUDA_Silhouette>
    void allocateGeometryBuffers(const SceneData<DIM> *sceneData) {
        // Extract primitive/silhouette payloads from CPU BVH and upload compact arrays.
        const Bvh<DIM, NodeType, PrimitiveType, SilhouetteType> *bvh =
            reinterpret_cast<const Bvh<DIM, NodeType, PrimitiveType, SilhouetteType> *>(
                sceneData->aggregate.get());

        CPUBvhDataExtractor<DIM,
                            NodeType,
                            PrimitiveType,
                            SilhouetteType,
                            CUDA_Node,
                            CUDA_Primitive,
                            CUDA_Silhouette> extractor(bvh);

        std::vector<CUDA_Primitive> primitivesData;
        std::vector<CUDA_Silhouette> silhouettesData;
        extractor.extractPrimitives(primitivesData);
        extractor.extractSilhouettes(silhouettesData);

        if (primitives) {
            FCPW_CUDA_CHECK(cudaFree(primitives));
            primitives = nullptr;
        }
        if (silhouettes) {
            FCPW_CUDA_CHECK(cudaFree(silhouettes));
            silhouettes = nullptr;
        }

        primitives = cudaAllocCopy(primitivesData);
        silhouettes = cudaAllocCopy(silhouettesData);
        primitiveCount = primitivesData.size();
        silhouetteCount = silhouettesData.size();
    }

    template<size_t DIM,
             typename NodeType,
             typename PrimitiveType,
             typename SilhouetteType,
             typename CUDA_Node,
             typename CUDA_Primitive,
             typename CUDA_Silhouette>
    void allocateNodeBuffer(const SceneData<DIM> *sceneData) {
        // Extract flattened node array (BVH or SNCH variant) and upload.
        const Bvh<DIM, NodeType, PrimitiveType, SilhouetteType> *bvh =
            reinterpret_cast<const Bvh<DIM, NodeType, PrimitiveType, SilhouetteType> *>(
                sceneData->aggregate.get());

        CPUBvhDataExtractor<DIM,
                            NodeType,
                            PrimitiveType,
                            SilhouetteType,
                            CUDA_Node,
                            CUDA_Primitive,
                            CUDA_Silhouette> extractor(bvh);

        std::vector<CUDA_Node> nodesData;
        extractor.extractNodes(nodesData);

        if (nodes) {
            FCPW_CUDA_CHECK(cudaFree(nodes));
            nodes = nullptr;
        }

        nodes = cudaAllocCopy(nodesData);
        nodeCount = nodesData.size();
    }

    template<size_t DIM,
             typename NodeType,
             typename PrimitiveType,
             typename SilhouetteType>
    void allocateRefitBuffer(const SceneData<DIM> *sceneData) {
        // Build depth buckets + node index list used by refit launch loop.
        const Bvh<DIM, NodeType, PrimitiveType, SilhouetteType> *bvh =
            reinterpret_cast<const Bvh<DIM, NodeType, PrimitiveType, SilhouetteType> *>(
                sceneData->aggregate.get());

        CPUBvhUpdateDataExtractor<DIM,
                                  NodeType,
                                  PrimitiveType,
                                  SilhouetteType> extractor(bvh);

        std::vector<uint32_t> nodeIndicesData;
        updateEntryData.clear();
        maxUpdateDepth = extractor.extract(nodeIndicesData, updateEntryData);

        if (nodeIndices) {
            FCPW_CUDA_CHECK(cudaFree(nodeIndices));
            nodeIndices = nullptr;
        }

        nodeIndices = cudaAllocCopy(nodeIndicesData);
    }
};

template<>
inline void CUDABvhBuffers::allocate<2>(const SceneData<2> *sceneData,
                                        bool allocatePrimitiveData,
                                        bool allocateSilhouetteData,
                                        bool allocateNodeData,
                                        bool allocateRefitData) {
    // 2D scenes use line segments; choose SNCH when silhouette data is present.
    if (allocateSilhouetteData) {
        bvhType = CUDABvhType::LineSegmentSnch;
        if (allocatePrimitiveData) {
            allocateGeometryBuffers<2, SnchNode<2>, LineSegment, SilhouetteVertex,
                                    CUDASnchNode, CUDALineSegment, CUDAVertex>(sceneData);
        }
        if (allocateNodeData) {
            allocateNodeBuffer<2, SnchNode<2>, LineSegment, SilhouetteVertex,
                               CUDASnchNode, CUDALineSegment, CUDAVertex>(sceneData);
        }
        if (allocateRefitData) {
            allocateRefitBuffer<2, SnchNode<2>, LineSegment, SilhouetteVertex>(sceneData);
        }
    } else {
        bvhType = CUDABvhType::LineSegmentBvh;
        if (allocatePrimitiveData) {
            allocateGeometryBuffers<2, BvhNode<2>, LineSegment, SilhouettePrimitive<2>,
                                    CUDABvhNode, CUDALineSegment, CUDANoSilhouette>(sceneData);
        }
        if (allocateNodeData) {
            allocateNodeBuffer<2, BvhNode<2>, LineSegment, SilhouettePrimitive<2>,
                               CUDABvhNode, CUDALineSegment, CUDANoSilhouette>(sceneData);
        }
        if (allocateRefitData) {
            allocateRefitBuffer<2, BvhNode<2>, LineSegment, SilhouettePrimitive<2>>(sceneData);
        }
    }
}

template<>
inline void CUDABvhBuffers::allocate<3>(const SceneData<3> *sceneData,
                                        bool allocatePrimitiveData,
                                        bool allocateSilhouetteData,
                                        bool allocateNodeData,
                                        bool allocateRefitData) {
    // 3D scenes use triangles; choose SNCH when silhouette data is present.
    if (allocateSilhouetteData) {
        bvhType = CUDABvhType::TriangleSnch;
        if (allocatePrimitiveData) {
            allocateGeometryBuffers<3, SnchNode<3>, Triangle, SilhouetteEdge,
                                    CUDASnchNode, CUDATriangle, CUDAEdge>(sceneData);
        }
        if (allocateNodeData) {
            allocateNodeBuffer<3, SnchNode<3>, Triangle, SilhouetteEdge,
                               CUDASnchNode, CUDATriangle, CUDAEdge>(sceneData);
        }
        if (allocateRefitData) {
            allocateRefitBuffer<3, SnchNode<3>, Triangle, SilhouetteEdge>(sceneData);
        }
    } else {
        bvhType = CUDABvhType::TriangleBvh;
        if (allocatePrimitiveData) {
            allocateGeometryBuffers<3, BvhNode<3>, Triangle, SilhouettePrimitive<3>,
                                    CUDABvhNode, CUDATriangle, CUDANoSilhouette>(sceneData);
        }
        if (allocateNodeData) {
            allocateNodeBuffer<3, BvhNode<3>, Triangle, SilhouettePrimitive<3>,
                               CUDABvhNode, CUDATriangle, CUDANoSilhouette>(sceneData);
        }
        if (allocateRefitData) {
            allocateRefitBuffer<3, BvhNode<3>, Triangle, SilhouettePrimitive<3>>(sceneData);
        }
    }
}

} // namespace fcpw
