#pragma once

#include <fcpw/cuda/cuda_types.h>
#include <fcpw/aggregates/bvh.h>
#include <fcpw/geometry/line_segments.h>
#include <fcpw/geometry/triangles.h>
#include <fcpw/geometry/silhouette_vertices.h>
#include <fcpw/geometry/silhouette_edges.h>

namespace fcpw {

// populates CUDA bvh nodes array from CPU bvh
template<size_t DIM>
void extractCUDABvhNodes(const std::vector<BvhNode<DIM>>& flatTree,
                         std::vector<CUDABvhNode>& cudaBvhNodes)
{
    int nNodes = (int)flatTree.size();
    cudaBvhNodes.resize(nNodes);

    for (int i = 0; i < nNodes; i++) {
        const BvhNode<DIM>& node = flatTree[i];
        const Vector<DIM>& pMin = node.box.pMin;
        const Vector<DIM>& pMax = node.box.pMax;
        uint32_t nPrimitives = node.nReferences;
        uint32_t offset = nPrimitives > 0 ? node.referenceOffset : node.secondChildOffset;

        CUDABoundingBox cudaBoundingBox(CUDAFloat3{pMin[0], pMin[1], DIM == 2 ? 0.0f : pMin[2]},
                                        CUDAFloat3{pMax[0], pMax[1], DIM == 2 ? 0.0f : pMax[2]});
        cudaBvhNodes[i] = CUDABvhNode(cudaBoundingBox, nPrimitives, offset);
    }
}

// populates CUDA snch nodes array from CPU snch
template<size_t DIM>
void extractCUDASnchNodes(const std::vector<SnchNode<DIM>>& flatTree,
                          std::vector<CUDASnchNode>& cudaSnchNodes)
{
    int nNodes = (int)flatTree.size();
    cudaSnchNodes.resize(nNodes);

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

        CUDABoundingBox cudaBoundingBox(CUDAFloat3{pMin[0], pMin[1], DIM == 2 ? 0.0f : pMin[2]},
                                        CUDAFloat3{pMax[0], pMax[1], DIM == 2 ? 0.0f : pMax[2]});
        CUDABoundingCone cudaBoundingCone(CUDAFloat3{axis[0], axis[1], DIM == 2 ? 0.0f : axis[2]},
                                          halfAngle, radius);
        cudaSnchNodes[i] = CUDASnchNode(cudaBoundingBox, cudaBoundingCone, nPrimitives,
                                        offset, nSilhouettes, silhouetteOffset);
    }
}

// populates CUDA line segments array from CPU line segments
inline void extractCUDALineSegments(const std::vector<LineSegment *>& primitives,
                                    std::vector<CUDALineSegment>& cudaLineSegments)
{
    int nPrimitives = (int)primitives.size();
    cudaLineSegments.resize(nPrimitives);

    for (int i = 0; i < nPrimitives; i++) {
        const LineSegment *lineSegment = primitives[i];
        const Vector2& pa = lineSegment->soup->positions[lineSegment->indices[0]];
        const Vector2& pb = lineSegment->soup->positions[lineSegment->indices[1]];

        cudaLineSegments[i] = CUDALineSegment(CUDAFloat3{pa[0], pa[1], 0.0f},
                                              CUDAFloat3{pb[0], pb[1], 0.0f},
                                              lineSegment->pIndex);
    }
}

// populates CUDA triangles array from CPU triangles
inline void extractCUDATriangles(const std::vector<Triangle *>& primitives,
                                 std::vector<CUDATriangle>& cudaTriangles)
{
    int nPrimitives = (int)primitives.size();
    cudaTriangles.resize(nPrimitives);

    for (int i = 0; i < nPrimitives; i++) {
        const Triangle *triangle = primitives[i];
        const Vector3& pa = triangle->soup->positions[triangle->indices[0]];
        const Vector3& pb = triangle->soup->positions[triangle->indices[1]];
        const Vector3& pc = triangle->soup->positions[triangle->indices[2]];

        cudaTriangles[i] = CUDATriangle(CUDAFloat3{pa[0], pa[1], pa[2]},
                                        CUDAFloat3{pb[0], pb[1], pb[2]},
                                        CUDAFloat3{pc[0], pc[1], pc[2]},
                                        triangle->pIndex);
    }
}

// populates CUDA silhouette vertices array from CPU silhouette vertices
inline void extractCUDASilhouetteVertices(const std::vector<SilhouetteVertex *>& silhouettes,
                                          std::vector<CUDAVertex>& cudaVertices)
{
    int nSilhouettes = (int)silhouettes.size();
    cudaVertices.resize(nSilhouettes);

    for (int i = 0; i < nSilhouettes; i++) {
        const SilhouetteVertex *silhouetteVertex = silhouettes[i];
        const Vector2& p = silhouetteVertex->soup->positions[silhouetteVertex->indices[1]];
        Vector2 n0 = silhouetteVertex->hasFace(0) ? silhouetteVertex->normal(0) : Vector2::Zero();
        Vector2 n1 = silhouetteVertex->hasFace(1) ? silhouetteVertex->normal(1) : Vector2::Zero();
        bool hasTwoAdjacentFaces = silhouetteVertex->hasFace(0) && silhouetteVertex->hasFace(1);

        cudaVertices[i] = CUDAVertex(CUDAFloat3{p[0], p[1], 0.0f},
                                     CUDAFloat3{n0[0], n0[1], 0.0f},
                                     CUDAFloat3{n1[0], n1[1], 0.0f},
                                     silhouetteVertex->pIndex,
                                     hasTwoAdjacentFaces == 1 ? 0 : 1);
    }
}

// populates CUDA silhouette edges array from CPU silhouette edges
inline void extractCUDASilhouetteEdges(const std::vector<SilhouetteEdge *>& silhouettes,
                                       std::vector<CUDAEdge>& cudaEdges)
{
    int nSilhouettes = (int)silhouettes.size();
    cudaEdges.resize(nSilhouettes);

    for (int i = 0; i < nSilhouettes; i++) {
        const SilhouetteEdge *silhouetteEdge = silhouettes[i];
        const Vector3& pa = silhouetteEdge->soup->positions[silhouetteEdge->indices[1]];
        const Vector3& pb = silhouetteEdge->soup->positions[silhouetteEdge->indices[2]];
        Vector3 n0 = silhouetteEdge->hasFace(0) ? silhouetteEdge->normal(0) : Vector3::Zero();
        Vector3 n1 = silhouetteEdge->hasFace(1) ? silhouetteEdge->normal(1) : Vector3::Zero();
        bool hasTwoAdjacentFaces = silhouetteEdge->hasFace(0) && silhouetteEdge->hasFace(1);

        cudaEdges[i] = CUDAEdge(CUDAFloat3{pa[0], pa[1], pa[2]},
                                CUDAFloat3{pb[0], pb[1], pb[2]},
                                CUDAFloat3{n0[0], n0[1], n0[2]},
                                CUDAFloat3{n1[0], n1[1], n1[2]},
                                silhouetteEdge->pIndex,
                                hasTwoAdjacentFaces == 1 ? 0 : 1);
    }
}

// converts an Eigen affine transform pair to CUDATransform (3x4 row-major)
template<size_t DIM>
inline CUDATransform extractCUDATransform(const Transform<DIM>& t, const Transform<DIM>& tInv) {
    CUDATransform ct;
    auto tMat = t.matrix();       // (DIM+1) x (DIM+1)
    auto tInvMat = tInv.matrix(); // (DIM+1) x (DIM+1)

    if (DIM == 3) {
        // 4x4 -> 3x4: copy top 3 rows
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++) {
                ct.t[i][j] = tMat(i, j);
                ct.tInv[i][j] = tInvMat(i, j);
            }
        }
    } else {
        // DIM == 2: 3x3 matrix -> 3x4 with z-row/col as identity
        // 3x3 layout:  [a b tx]    3x4 layout: [a b 0 tx]
        //              [c d ty]                 [c d 0 ty]
        //              [0 0  1]                 [0 0 1  0]
        for (int i = 0; i < 2; i++) {
            ct.t[i][0] = tMat(i, 0);       ct.tInv[i][0] = tInvMat(i, 0);
            ct.t[i][1] = tMat(i, 1);       ct.tInv[i][1] = tInvMat(i, 1);
            ct.t[i][2] = 0.0f;             ct.tInv[i][2] = 0.0f;
            ct.t[i][3] = tMat(i, 2);       ct.tInv[i][3] = tInvMat(i, 2);  // translation
        }
        // z-row: identity
        ct.t[2][0] = 0.0f;   ct.tInv[2][0] = 0.0f;
        ct.t[2][1] = 0.0f;   ct.tInv[2][1] = 0.0f;
        ct.t[2][2] = 1.0f;   ct.tInv[2][2] = 1.0f;
        ct.t[2][3] = 0.0f;   ct.tInv[2][3] = 0.0f;
    }

    return ct;
}

// extracts CPU bvh data into CUDA-compatible arrays; specialized per geometry/node type
template<size_t DIM,
         typename NodeType,
         typename PrimitiveType,
         typename SilhouetteType,
         typename CUDANodeType,
         typename CUDAPrimitiveType,
         typename CUDASilhouetteType>
class CPUCUDABvhDataExtractor {
public:
    CPUCUDABvhDataExtractor(const Bvh<DIM, NodeType, PrimitiveType, SilhouetteType> *bvh_) {
        std::cerr << "CPUCUDABvhDataExtractor() not supported" << std::endl;
        exit(EXIT_FAILURE);
    }

    void extractNodes(std::vector<CUDANodeType>& cudaNodes) {
        std::cerr << "CPUCUDABvhDataExtractor::extractNodes() not supported" << std::endl;
        exit(EXIT_FAILURE);
    }

    void extractPrimitives(std::vector<CUDAPrimitiveType>& cudaPrimitives) {
        std::cerr << "CPUCUDABvhDataExtractor::extractPrimitives() not supported" << std::endl;
        exit(EXIT_FAILURE);
    }

    void extractSilhouettes(std::vector<CUDASilhouetteType>& cudaSilhouettes) {
        std::cerr << "CPUCUDABvhDataExtractor::extractSilhouettes() not supported" << std::endl;
        exit(EXIT_FAILURE);
    }

    int getBvhType() const {
        std::cerr << "CPUCUDABvhDataExtractor::getBvhType() not supported" << std::endl;
        exit(EXIT_FAILURE);
        return 0;
    }
};

template<>
class CPUCUDABvhDataExtractor<2, SnchNode<2>, LineSegment, SilhouetteVertex, CUDASnchNode, CUDALineSegment, CUDAVertex> {
public:
    CPUCUDABvhDataExtractor(const Bvh<2, SnchNode<2>, LineSegment, SilhouetteVertex> *bvh_): bvh(bvh_) {}

    void extractNodes(std::vector<CUDASnchNode>& cudaSnchNodes) {
        extractCUDASnchNodes<2>(bvh->flatTree, cudaSnchNodes);
    }

    void extractPrimitives(std::vector<CUDALineSegment>& cudaLineSegments) {
        extractCUDALineSegments(bvh->primitives, cudaLineSegments);
    }

    void extractSilhouettes(std::vector<CUDAVertex>& cudaVertices) {
        extractCUDASilhouetteVertices(bvh->silhouetteRefs, cudaVertices);
    }

    int getBvhType() const { return FCPW_CUDA_LINE_SEGMENT_SNCH; }

    const Bvh<2, SnchNode<2>, LineSegment, SilhouetteVertex> *bvh;
};

template<>
class CPUCUDABvhDataExtractor<3, SnchNode<3>, Triangle, SilhouetteEdge, CUDASnchNode, CUDATriangle, CUDAEdge> {
public:
    CPUCUDABvhDataExtractor(const Bvh<3, SnchNode<3>, Triangle, SilhouetteEdge> *bvh_): bvh(bvh_) {}

    void extractNodes(std::vector<CUDASnchNode>& cudaSnchNodes) {
        extractCUDASnchNodes<3>(bvh->flatTree, cudaSnchNodes);
    }

    void extractPrimitives(std::vector<CUDATriangle>& cudaTriangles) {
        extractCUDATriangles(bvh->primitives, cudaTriangles);
    }

    void extractSilhouettes(std::vector<CUDAEdge>& cudaEdges) {
        extractCUDASilhouetteEdges(bvh->silhouetteRefs, cudaEdges);
    }

    int getBvhType() const { return FCPW_CUDA_TRIANGLE_SNCH; }

    const Bvh<3, SnchNode<3>, Triangle, SilhouetteEdge> *bvh;
};

template<>
class CPUCUDABvhDataExtractor<2, BvhNode<2>, LineSegment, SilhouettePrimitive<2>, CUDABvhNode, CUDALineSegment, CUDANoSilhouette> {
public:
    CPUCUDABvhDataExtractor(const Bvh<2, BvhNode<2>, LineSegment, SilhouettePrimitive<2>> *bvh_): bvh(bvh_) {}

    void extractNodes(std::vector<CUDABvhNode>& cudaBvhNodes) {
        extractCUDABvhNodes<2>(bvh->flatTree, cudaBvhNodes);
    }

    void extractPrimitives(std::vector<CUDALineSegment>& cudaLineSegments) {
        extractCUDALineSegments(bvh->primitives, cudaLineSegments);
    }

    void extractSilhouettes(std::vector<CUDANoSilhouette>& cudaSilhouettes) {
        cudaSilhouettes.clear();
    }

    int getBvhType() const { return FCPW_CUDA_LINE_SEGMENT_BVH; }

    const Bvh<2, BvhNode<2>, LineSegment, SilhouettePrimitive<2>> *bvh;
};

template<>
class CPUCUDABvhDataExtractor<3, BvhNode<3>, Triangle, SilhouettePrimitive<3>, CUDABvhNode, CUDATriangle, CUDANoSilhouette> {
public:
    CPUCUDABvhDataExtractor(const Bvh<3, BvhNode<3>, Triangle, SilhouettePrimitive<3>> *bvh_): bvh(bvh_) {}

    void extractNodes(std::vector<CUDABvhNode>& cudaBvhNodes) {
        extractCUDABvhNodes<3>(bvh->flatTree, cudaBvhNodes);
    }

    void extractPrimitives(std::vector<CUDATriangle>& cudaTriangles) {
        extractCUDATriangles(bvh->primitives, cudaTriangles);
    }

    void extractSilhouettes(std::vector<CUDANoSilhouette>& cudaSilhouettes) {
        cudaSilhouettes.clear();
    }

    int getBvhType() const { return FCPW_CUDA_TRIANGLE_BVH; }

    const Bvh<3, BvhNode<3>, Triangle, SilhouettePrimitive<3>> *bvh;
};

// extracts per-depth node index arrays for bottom-up BVH refit on GPU
template<size_t DIM,
         typename NodeType,
         typename PrimitiveType,
         typename SilhouetteType>
class CPUCUDABvhUpdateDataExtractor {
public:
    CPUCUDABvhUpdateDataExtractor(const Bvh<DIM, NodeType, PrimitiveType, SilhouetteType> *bvh_): bvh(bvh_) {}

    uint32_t extract(std::vector<uint32_t>& nodeIndicesData,
                     std::vector<std::pair<uint32_t, uint32_t>>& updateEntryData) {
        int maxDepth = bvh->maxDepth;
        updateEntryData.resize(maxDepth + 1, std::make_pair(0, 0));
        updateEntryData[maxDepth].second = bvh->nLeafs;
        traverseBvh(
            [&updateEntryData](int index, int depth) { ++updateEntryData[depth].second; },
            [](int index, int depth) { /* do nothing */ }
        );

        std::vector<uint32_t> offsets(maxDepth + 1, 0);
        for (uint32_t i = 1; i < maxDepth + 1; i++) {
            uint32_t currentOffset = updateEntryData[i - 1].first + updateEntryData[i - 1].second;
            offsets[i] = updateEntryData[i].first = currentOffset;
        }

        nodeIndicesData.resize(bvh->nNodes, 0);
        traverseBvh(
            [&nodeIndicesData, &offsets](int index, int depth) { nodeIndicesData[offsets[depth]++] = index; },
            [&nodeIndicesData, &offsets](int index, int depth) { nodeIndicesData[offsets.back()++] = index; }
        );

        return maxDepth;
    }

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
};

} // namespace fcpw
