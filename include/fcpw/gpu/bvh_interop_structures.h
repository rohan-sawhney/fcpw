#pragma once

#include <fcpw/aggregates/bvh.h>
#include <fcpw/geometry/line_segments.h>
#include <fcpw/geometry/triangles.h>
#include <fcpw/geometry/silhouette_vertices.h>
#include <fcpw/geometry/silhouette_edges.h>
#include <fcpw/gpu/slang_gfx_utils.h>

#define FCPW_GPU_UINT_MAX 4294967295

namespace fcpw {

inline void exitOnError(Slang::Result result, std::string error)
{
    if (result != SLANG_OK) {
        std::cout << "failed to allocate GPUBvhNodes buffer" << std::endl;
        exit(EXIT_FAILURE);
    }
}

struct GPUBvhNodes {
    template<size_t DIM>
    void extract(const std::vector<BvhNode<DIM>>& flatTree) {
        int nNodes = (int)flatTree.size();
        pxMin.resize(nNodes, 0.0f); pyMin.resize(nNodes, 0.0f); pzMin.resize(nNodes, 0.0f);
        pxMax.resize(nNodes, 0.0f); pyMax.resize(nNodes, 0.0f); pzMax.resize(nNodes, 0.0f);
        nPrimitives.resize(nNodes, 0);
        offsets.resize(nNodes, 0);

        for (int i = 0; i < nNodes; i++) {
            const BvhNode<DIM>& node = flatTree[i];
            const Vector<DIM>& pMin = node.box.pMin;
            const Vector<DIM>& pMax = node.box.pMax;

            pxMin[i] = pMin[0]; pyMin[i] = pMin[1]; pzMin[i] = DIM == 2 ? 0.0f : pMin[2];
            pxMax[i] = pMax[0]; pyMax[i] = pMax[1]; pzMax[i] = DIM == 2 ? 0.0f : pMax[2];
            nPrimitives[i] = node.nReferences;
            offsets[i] = node.nReferences > 0 ? node.referenceOffset : node.secondChildOffset;
        }
    }

    void allocate(ComPtr<IDevice>& device) {
        Slang::Result result = pxMinBuffer.create<float>(device, false, pxMin.data(), pxMin.size());
        exitOnError(result, "failed to allocate GPUBvhNodes buffer");
        result = pyMinBuffer.create<float>(device, false, pyMin.data(), pyMin.size());
        exitOnError(result, "failed to allocate GPUBvhNodes buffer");
        result = pzMinBuffer.create<float>(device, false, pzMin.data(), pzMin.size());
        exitOnError(result, "failed to allocate GPUBvhNodes buffer");
        result = pxMaxBuffer.create<float>(device, false, pxMax.data(), pxMax.size());
        exitOnError(result, "failed to allocate GPUBvhNodes buffer");
        result = pyMaxBuffer.create<float>(device, false, pyMax.data(), pyMax.size());
        exitOnError(result, "failed to allocate GPUBvhNodes buffer");
        result = pzMaxBuffer.create<float>(device, false, pzMax.data(), pzMax.size());
        exitOnError(result, "failed to allocate GPUBvhNodes buffer");
        result = nPrimitivesBuffer.create<uint32_t>(device, false, nPrimitives.data(), nPrimitives.size());
        exitOnError(result, "failed to allocate GPUBvhNodes buffer");
        result = offsetsBuffer.create<uint32_t>(device, false, offsets.data(), offsets.size());
        exitOnError(result, "failed to allocate GPUBvhNodes buffer");
    }

    void setResources(ShaderCursor& nodesCursor) const {
        nodesCursor["pxMin"].setResource(pxMinBuffer.view);
        nodesCursor["pyMin"].setResource(pyMinBuffer.view);
        nodesCursor["pzMin"].setResource(pzMinBuffer.view);
        nodesCursor["pxMax"].setResource(pxMaxBuffer.view);
        nodesCursor["pyMax"].setResource(pyMaxBuffer.view);
        nodesCursor["pzMax"].setResource(pzMaxBuffer.view);
        nodesCursor["nPrimitives"].setResource(nPrimitivesBuffer.view);
        nodesCursor["offsets"].setResource(offsetsBuffer.view);
    }

private:
    std::vector<float> pxMin, pyMin, pzMin;
    std::vector<float> pxMax, pyMax, pzMax;
    std::vector<uint32_t> nPrimitives;
    std::vector<uint32_t> offsets;
    GPUBuffer pxMinBuffer = {}, pyMinBuffer = {}, pzMinBuffer = {};
    GPUBuffer pxMaxBuffer = {}, pyMaxBuffer = {}, pzMaxBuffer = {};
    GPUBuffer nPrimitivesBuffer = {};
    GPUBuffer offsetsBuffer = {};
};

struct GPUSnchNodes {
    template<size_t DIM>
    void extract(const std::vector<SnchNode<DIM>>& flatTree) {
        int nNodes = (int)flatTree.size();
        pxMin.resize(nNodes, 0.0f); pyMin.resize(nNodes, 0.0f); pzMin.resize(nNodes, 0.0f);
        pxMax.resize(nNodes, 0.0f); pyMax.resize(nNodes, 0.0f); pzMax.resize(nNodes, 0.0f);
        axesx.resize(nNodes, 0.0f); axesy.resize(nNodes, 0.0f); axesz.resize(nNodes, 0.0f);
        halfAngles.resize(nNodes, 0.0f);
        radii.resize(nNodes, 0.0f);
        nPrimitives.resize(nNodes, 0);
        offsets.resize(nNodes, 0);
        nSilhouettes.resize(nNodes, 0);
        silhouetteOffsets.resize(nNodes, 0);

        for (int i = 0; i < nNodes; i++) {
            const SnchNode<DIM>& node = flatTree[i];
            const Vector<DIM>& pMin = node.box.pMin;
            const Vector<DIM>& pMax = node.box.pMax;
            const Vector<DIM>& axis = node.cone.axis;

            pxMin[i] = pMin[0]; pyMin[i] = pMin[1]; pzMin[i] = DIM == 2 ? 0.0f : pMin[2];
            pxMax[i] = pMax[0]; pyMax[i] = pMax[1]; pzMax[i] = DIM == 2 ? 0.0f : pMax[2];
            axesx[i] = axis[0]; axesy[i] = axis[1]; axesz[i] = DIM == 2 ? 0.0f : axis[2];
            halfAngles[i] = node.cone.halfAngle;
            radii[i] = node.cone.radius;
            nPrimitives[i] = node.nReferences;
            offsets[i] = node.nReferences > 0 ? node.referenceOffset : node.secondChildOffset;
            nSilhouettes[i] = node.nSilhouetteReferences;
            silhouetteOffsets[i] = node.silhouetteReferenceOffset;
        }
    }

    void allocate(ComPtr<IDevice>& device) {
        Slang::Result result = pxMinBuffer.create<float>(device, false, pxMin.data(), pxMin.size());
        exitOnError(result, "failed to allocate GPUSnchNodes buffer");
        result = pyMinBuffer.create<float>(device, false, pyMin.data(), pyMin.size());
        exitOnError(result, "failed to allocate GPUSnchNodes buffer");
        result = pzMinBuffer.create<float>(device, false, pzMin.data(), pzMin.size());
        exitOnError(result, "failed to allocate GPUSnchNodes buffer");
        result = pxMaxBuffer.create<float>(device, false, pxMax.data(), pxMax.size());
        exitOnError(result, "failed to allocate GPUSnchNodes buffer");
        result = pyMaxBuffer.create<float>(device, false, pyMax.data(), pyMax.size());
        exitOnError(result, "failed to allocate GPUSnchNodes buffer");
        result = pzMaxBuffer.create<float>(device, false, pzMax.data(), pzMax.size());
        exitOnError(result, "failed to allocate GPUSnchNodes buffer");
        result = axesxBuffer.create<float>(device, false, axesx.data(), axesx.size());
        exitOnError(result, "failed to allocate GPUSnchNodes buffer");
        result = axesyBuffer.create<float>(device, false, axesy.data(), axesy.size());
        exitOnError(result, "failed to allocate GPUSnchNodes buffer");
        result = axeszBuffer.create<float>(device, false, axesz.data(), axesz.size());
        exitOnError(result, "failed to allocate GPUSnchNodes buffer");
        result = halfAnglesBuffer.create<float>(device, false, halfAngles.data(), halfAngles.size());
        exitOnError(result, "failed to allocate GPUSnchNodes buffer");
        result = radiiBuffer.create<float>(device, false, radii.data(), radii.size());
        exitOnError(result, "failed to allocate GPUSnchNodes buffer");
        result = nPrimitivesBuffer.create<uint32_t>(device, false, nPrimitives.data(), nPrimitives.size());
        exitOnError(result, "failed to allocate GPUSnchNodes buffer");
        result = offsetsBuffer.create<uint32_t>(device, false, offsets.data(), offsets.size());
        exitOnError(result, "failed to allocate GPUSnchNodes buffer");
        result = nSilhouettesBuffer.create<uint32_t>(device, false, nSilhouettes.data(), nSilhouettes.size());
        exitOnError(result, "failed to allocate GPUSnchNodes buffer");
        result = silhouetteOffsetsBuffer.create<uint32_t>(device, false, silhouetteOffsets.data(), silhouetteOffsets.size());
        exitOnError(result, "failed to allocate GPUSnchNodes buffer");
    }

    void setResources(ShaderCursor& nodesCursor) const {
        nodesCursor["pxMin"].setResource(pxMinBuffer.view);
        nodesCursor["pyMin"].setResource(pyMinBuffer.view);
        nodesCursor["pzMin"].setResource(pzMinBuffer.view);
        nodesCursor["pxMax"].setResource(pxMaxBuffer.view);
        nodesCursor["pyMax"].setResource(pyMaxBuffer.view);
        nodesCursor["pzMax"].setResource(pzMaxBuffer.view);
        nodesCursor["axesx"].setResource(axesxBuffer.view);
        nodesCursor["axesy"].setResource(axesyBuffer.view);
        nodesCursor["axesz"].setResource(axeszBuffer.view);
        nodesCursor["halfAngles"].setResource(halfAnglesBuffer.view);
        nodesCursor["radii"].setResource(radiiBuffer.view);
        nodesCursor["nPrimitives"].setResource(nPrimitivesBuffer.view);
        nodesCursor["offsets"].setResource(offsetsBuffer.view);
        nodesCursor["nSilhouettes"].setResource(nSilhouettesBuffer.view);
        nodesCursor["silhouetteOffsets"].setResource(silhouetteOffsetsBuffer.view);
    }

private:
    std::vector<float> pxMin, pyMin, pzMin;
    std::vector<float> pxMax, pyMax, pzMax;
    std::vector<float> axesx, axesy, axesz;
    std::vector<float> halfAngles;
    std::vector<float> radii;
    std::vector<uint32_t> nPrimitives;
    std::vector<uint32_t> offsets;
    std::vector<uint32_t> nSilhouettes;
    std::vector<uint32_t> silhouetteOffsets;
    GPUBuffer pxMinBuffer = {}, pyMinBuffer = {}, pzMinBuffer = {};
    GPUBuffer pxMaxBuffer = {}, pyMaxBuffer = {}, pzMaxBuffer = {};
    GPUBuffer axesxBuffer = {}, axesyBuffer = {}, axeszBuffer = {};
    GPUBuffer halfAnglesBuffer = {};
    GPUBuffer radiiBuffer = {};
    GPUBuffer nPrimitivesBuffer = {};
    GPUBuffer offsetsBuffer = {};
    GPUBuffer nSilhouettesBuffer = {};
    GPUBuffer silhouetteOffsetsBuffer = {};
};

struct GPULineSegments {
    void extract(const std::vector<LineSegment *>& primitives) {
        int nPrimitives = (int)primitives.size();
        pax.resize(nPrimitives, 0.0f); pay.resize(nPrimitives, 0.0f);
        pbx.resize(nPrimitives, 0.0f); pby.resize(nPrimitives, 0.0f);
        indices.resize(nPrimitives, 0);

        for (int i = 0; i < nPrimitives; i++) {
            const LineSegment *lineSegment = primitives[i];
            const Vector2& pa = lineSegment->soup->positions[lineSegment->indices[0]];
            const Vector2& pb = lineSegment->soup->positions[lineSegment->indices[1]];

            pax[i] = pa[0]; pay[i] = pa[1];
            pbx[i] = pb[0]; pby[i] = pb[1];
            indices[i] = lineSegment->pIndex;
        }
    }

    void allocate(ComPtr<IDevice>& device) {
        Slang::Result result = paxBuffer.create<float>(device, false, pax.data(), pax.size());
        exitOnError(result, "failed to allocate GPULineSegments buffer");
        result = payBuffer.create<float>(device, false, pay.data(), pay.size());
        exitOnError(result, "failed to allocate GPULineSegments buffer");
        result = pbxBuffer.create<float>(device, false, pbx.data(), pbx.size());
        exitOnError(result, "failed to allocate GPULineSegments buffer");
        result = pbyBuffer.create<float>(device, false, pby.data(), pby.size());
        exitOnError(result, "failed to allocate GPULineSegments buffer");
        result = indicesBuffer.create<uint32_t>(device, false, indices.data(), indices.size());
        exitOnError(result, "failed to allocate GPULineSegments buffer");
    }

    void setResources(ShaderCursor& primitivesCursor) const {
        primitivesCursor["pax"].setResource(paxBuffer.view);
        primitivesCursor["pay"].setResource(payBuffer.view);
        primitivesCursor["pbx"].setResource(pbxBuffer.view);
        primitivesCursor["pby"].setResource(pbyBuffer.view);
        primitivesCursor["indices"].setResource(indicesBuffer.view);
    }

private:
    std::vector<float> pax, pay;
    std::vector<float> pbx, pby;
    std::vector<uint32_t> indices;
    GPUBuffer paxBuffer = {}, payBuffer = {};
    GPUBuffer pbxBuffer = {}, pbyBuffer = {};
    GPUBuffer indicesBuffer = {};
};

struct GPUTriangles {
    void extract(const std::vector<Triangle *>& primitives) {
        int nPrimitives = (int)primitives.size();
        pax.resize(nPrimitives, 0.0f); pay.resize(nPrimitives, 0.0f); paz.resize(nPrimitives, 0.0f);
        pbx.resize(nPrimitives, 0.0f); pby.resize(nPrimitives, 0.0f); pbz.resize(nPrimitives, 0.0f);
        pcx.resize(nPrimitives, 0.0f); pcy.resize(nPrimitives, 0.0f); pcz.resize(nPrimitives, 0.0f);
        indices.resize(nPrimitives, 0);

        for (int i = 0; i < nPrimitives; i++) {
            const Triangle *triangle = primitives[i];
            const Vector3& pa = triangle->soup->positions[triangle->indices[0]];
            const Vector3& pb = triangle->soup->positions[triangle->indices[1]];
            const Vector3& pc = triangle->soup->positions[triangle->indices[2]];

            pax[i] = pa[0]; pay[i] = pa[1]; paz[i] = pa[2];
            pbx[i] = pb[0]; pby[i] = pb[1]; pbz[i] = pb[2];
            pcx[i] = pc[0]; pcy[i] = pc[1]; pcz[i] = pc[2];
            indices[i] = triangle->pIndex;
        }
    }

    void allocate(ComPtr<IDevice>& device) {
        Slang::Result result = paxBuffer.create<float>(device, false, pax.data(), pax.size());
        exitOnError(result, "failed to allocate GPUTriangles buffer");
        result = payBuffer.create<float>(device, false, pay.data(), pay.size());
        exitOnError(result, "failed to allocate GPUTriangles buffer");
        result = pazBuffer.create<float>(device, false, paz.data(), paz.size());
        exitOnError(result, "failed to allocate GPUTriangles buffer");
        result = pbxBuffer.create<float>(device, false, pbx.data(), pbx.size());
        exitOnError(result, "failed to allocate GPUTriangles buffer");
        result = pbyBuffer.create<float>(device, false, pby.data(), pby.size());
        exitOnError(result, "failed to allocate GPUTriangles buffer");
        result = pbzBuffer.create<float>(device, false, pbz.data(), pbz.size());
        exitOnError(result, "failed to allocate GPUTriangles buffer");
        result = pcxBuffer.create<float>(device, false, pcx.data(), pcx.size());
        exitOnError(result, "failed to allocate GPUTriangles buffer");
        result = pcyBuffer.create<float>(device, false, pcy.data(), pcy.size());
        exitOnError(result, "failed to allocate GPUTriangles buffer");
        result = pczBuffer.create<float>(device, false, pcz.data(), pcz.size());
        exitOnError(result, "failed to allocate GPUTriangles buffer");
        result = indicesBuffer.create<uint32_t>(device, false, indices.data(), indices.size());
        exitOnError(result, "failed to allocate GPUTriangles buffer");
    }

    void setResources(ShaderCursor& primitivesCursor) const {
        primitivesCursor["pax"].setResource(paxBuffer.view);
        primitivesCursor["pay"].setResource(payBuffer.view);
        primitivesCursor["paz"].setResource(pazBuffer.view);
        primitivesCursor["pbx"].setResource(pbxBuffer.view);
        primitivesCursor["pby"].setResource(pbyBuffer.view);
        primitivesCursor["pbz"].setResource(pbzBuffer.view);
        primitivesCursor["pcx"].setResource(pcxBuffer.view);
        primitivesCursor["pcy"].setResource(pcyBuffer.view);
        primitivesCursor["pcz"].setResource(pczBuffer.view);
        primitivesCursor["indices"].setResource(indicesBuffer.view);
    }

private:
    std::vector<float> pax, pay, paz;
    std::vector<float> pbx, pby, pbz;
    std::vector<float> pcx, pcy, pcz;
    std::vector<uint32_t> indices;
    GPUBuffer paxBuffer = {}, payBuffer = {}, pazBuffer = {};
    GPUBuffer pbxBuffer = {}, pbyBuffer = {}, pbzBuffer = {};
    GPUBuffer pcxBuffer = {}, pcyBuffer = {}, pczBuffer = {};
    GPUBuffer indicesBuffer = {};
};

struct GPUNoSilhouettes {
    void extract(const std::vector<SilhouetteVertex *>& silhouettes) {
        // do nothing
    }

    void allocate(ComPtr<IDevice>& device) {
        // do nothing
    }

    void setResources(ShaderCursor& silhouettesCursor) const {
        // do nothing
    }
};

struct GPUSilhouetteVertices {
    void extract(const std::vector<SilhouetteVertex *>& silhouettes) {
        int nSilhouettes = (int)silhouettes.size();
        px.resize(nSilhouettes, 0.0f); py.resize(nSilhouettes, 0.0f);
        n0x.resize(nSilhouettes, 0.0f); n0y.resize(nSilhouettes, 0.0f);
        n1x.resize(nSilhouettes, 0.0f); n1y.resize(nSilhouettes, 0.0f);
        indices.resize(nSilhouettes, 0);
        hasOneAdjacentFace.resize(nSilhouettes, 0);

        for (int i = 0; i < nSilhouettes; i++) {
            const SilhouetteVertex *silhouetteVertex = silhouettes[i];
            const Vector2& p = silhouetteVertex->soup->positions[silhouetteVertex->indices[1]];
            Vector2 n0 = silhouetteVertex->hasFace(0) ? silhouetteVertex->normal(0) : Vector2::Zero();
            Vector2 n1 = silhouetteVertex->hasFace(1) ? silhouetteVertex->normal(1) : Vector2::Zero();

            px[i] = p[0]; py[i] = p[1];
            n0x[i] = n0[0]; n0y[i] = n0[1];
            n1x[i] = n1[0]; n1y[i] = n1[1];
            indices[i] = silhouetteVertex->pIndex;
            hasOneAdjacentFace[i] = silhouetteVertex->hasFace(0) && silhouetteVertex->hasFace(1) ? 0 : 1;
        }
    }

    void allocate(ComPtr<IDevice>& device) {
        Slang::Result result = pxBuffer.create<float>(device, false, px.data(), px.size());
        exitOnError(result, "failed to allocate GPUSilhouetteVertices buffer");
        result = pyBuffer.create<float>(device, false, py.data(), py.size());
        exitOnError(result, "failed to allocate GPUSilhouetteVertices buffer");
        result = n0xBuffer.create<float>(device, false, n0x.data(), n0x.size());
        exitOnError(result, "failed to allocate GPUSilhouetteVertices buffer");
        result = n0yBuffer.create<float>(device, false, n0y.data(), n0y.size());
        exitOnError(result, "failed to allocate GPUSilhouetteVertices buffer");
        result = n1xBuffer.create<float>(device, false, n1x.data(), n1x.size());
        exitOnError(result, "failed to allocate GPUSilhouetteVertices buffer");
        result = n1yBuffer.create<float>(device, false, n1y.data(), n1y.size());
        exitOnError(result, "failed to allocate GPUSilhouetteVertices buffer");
        result = indicesBuffer.create<uint32_t>(device, false, indices.data(), indices.size());
        exitOnError(result, "failed to allocate GPUSilhouetteVertices buffer");
        result = hasOneAdjacentFaceBuffer.create<uint32_t>(device, false, hasOneAdjacentFace.data(), hasOneAdjacentFace.size());
        exitOnError(result, "failed to allocate GPUSilhouetteVertices buffer");
    }

    void setResources(ShaderCursor& silhouettesCursor) const {
        silhouettesCursor["px"].setResource(pxBuffer.view);
        silhouettesCursor["py"].setResource(pyBuffer.view);
        silhouettesCursor["n0x"].setResource(n0xBuffer.view);
        silhouettesCursor["n0y"].setResource(n0yBuffer.view);
        silhouettesCursor["n1x"].setResource(n1xBuffer.view);
        silhouettesCursor["n1y"].setResource(n1yBuffer.view);
        silhouettesCursor["indices"].setResource(indicesBuffer.view);
        silhouettesCursor["hasOneAdjacentFace"].setResource(hasOneAdjacentFaceBuffer.view);
    }

private:
    std::vector<float> px, py;
    std::vector<float> n0x, n0y;
    std::vector<float> n1x, n1y;
    std::vector<uint32_t> indices;
    std::vector<uint32_t> hasOneAdjacentFace;
    GPUBuffer pxBuffer = {}, pyBuffer = {};
    GPUBuffer n0xBuffer = {}, n0yBuffer = {};
    GPUBuffer n1xBuffer = {}, n1yBuffer = {};
    GPUBuffer indicesBuffer = {};
    GPUBuffer hasOneAdjacentFaceBuffer = {};
};

struct GPUSilhouetteEdges {
    void extract(const std::vector<SilhouetteEdge *>& silhouettes) {
        int nSilhouettes = (int)silhouettes.size();
        pax.resize(nSilhouettes, 0.0f); pay.resize(nSilhouettes, 0.0f); paz.resize(nSilhouettes, 0.0f);
        pbx.resize(nSilhouettes, 0.0f); pby.resize(nSilhouettes, 0.0f); pbz.resize(nSilhouettes, 0.0f);
        n0x.resize(nSilhouettes, 0.0f); n0y.resize(nSilhouettes, 0.0f); n0z.resize(nSilhouettes, 0.0f);
        n1x.resize(nSilhouettes, 0.0f); n1y.resize(nSilhouettes, 0.0f); n1z.resize(nSilhouettes, 0.0f);
        indices.resize(nSilhouettes, 0);
        hasOneAdjacentFace.resize(nSilhouettes, 0);

        for (int i = 0; i < nSilhouettes; i++) {
            const SilhouetteEdge *silhouetteEdge = silhouettes[i];
            const Vector3& pa = silhouetteEdge->soup->positions[silhouetteEdge->indices[1]];
            const Vector3& pb = silhouetteEdge->soup->positions[silhouetteEdge->indices[2]];
            Vector3 n0 = silhouetteEdge->hasFace(0) ? silhouetteEdge->normal(0) : Vector3::Zero();
            Vector3 n1 = silhouetteEdge->hasFace(1) ? silhouetteEdge->normal(1) : Vector3::Zero();

            pax[i] = pa[0]; pay[i] = pa[1]; paz[i] = pa[2];
            pbx[i] = pb[0]; pby[i] = pb[1]; pbz[i] = pb[2];
            n0x[i] = n0[0]; n0y[i] = n0[1]; n0z[i] = n0[2];
            n1x[i] = n1[0]; n1y[i] = n1[1]; n1z[i] = n1[2];
            indices[i] = silhouetteEdge->pIndex;
            hasOneAdjacentFace[i] = silhouetteEdge->hasFace(0) && silhouetteEdge->hasFace(1) ? 0 : 1;
        }
    }

    void allocate(ComPtr<IDevice>& device) {
        Slang::Result result = paxBuffer.create<float>(device, false, pax.data(), pax.size());
        exitOnError(result, "failed to allocate GPUSilhouetteEdges buffer");
        result = payBuffer.create<float>(device, false, pay.data(), pay.size());
        exitOnError(result, "failed to allocate GPUSilhouetteEdges buffer");
        result = pazBuffer.create<float>(device, false, paz.data(), paz.size());
        exitOnError(result, "failed to allocate GPUSilhouetteEdges buffer");
        result = pbxBuffer.create<float>(device, false, pbx.data(), pbx.size());
        exitOnError(result, "failed to allocate GPUSilhouetteEdges buffer");
        result = pbyBuffer.create<float>(device, false, pby.data(), pby.size());
        exitOnError(result, "failed to allocate GPUSilhouetteEdges buffer");
        result = pbzBuffer.create<float>(device, false, pbz.data(), pbz.size());
        exitOnError(result, "failed to allocate GPUSilhouetteEdges buffer");
        result = n0xBuffer.create<float>(device, false, n0x.data(), n0x.size());
        exitOnError(result, "failed to allocate GPUSilhouetteEdges buffer");
        result = n0yBuffer.create<float>(device, false, n0y.data(), n0y.size());
        exitOnError(result, "failed to allocate GPUSilhouetteEdges buffer");
        result = n0zBuffer.create<float>(device, false, n0z.data(), n0z.size());
        exitOnError(result, "failed to allocate GPUSilhouetteEdges buffer");
        result = n1xBuffer.create<float>(device, false, n1x.data(), n1x.size());
        exitOnError(result, "failed to allocate GPUSilhouetteEdges buffer");
        result = n1yBuffer.create<float>(device, false, n1y.data(), n1y.size());
        exitOnError(result, "failed to allocate GPUSilhouetteEdges buffer");
        result = n1zBuffer.create<float>(device, false, n1z.data(), n1z.size());
        exitOnError(result, "failed to allocate GPUSilhouetteEdges buffer");
        result = indicesBuffer.create<uint32_t>(device, false, indices.data(), indices.size());
        exitOnError(result, "failed to allocate GPUSilhouetteEdges buffer");
        result = hasOneAdjacentFaceBuffer.create<uint32_t>(device, false, hasOneAdjacentFace.data(), hasOneAdjacentFace.size());
        exitOnError(result, "failed to allocate GPUSilhouetteEdges buffer");
    }

    void setResources(ShaderCursor& silhouettesCursor) const {
        silhouettesCursor["pax"].setResource(paxBuffer.view);
        silhouettesCursor["pay"].setResource(payBuffer.view);
        silhouettesCursor["paz"].setResource(pazBuffer.view);
        silhouettesCursor["pbx"].setResource(pbxBuffer.view);
        silhouettesCursor["pby"].setResource(pbyBuffer.view);
        silhouettesCursor["pbz"].setResource(pbzBuffer.view);
        silhouettesCursor["n0x"].setResource(n0xBuffer.view);
        silhouettesCursor["n0y"].setResource(n0yBuffer.view);
        silhouettesCursor["n0z"].setResource(n0zBuffer.view);
        silhouettesCursor["n1x"].setResource(n1xBuffer.view);
        silhouettesCursor["n1y"].setResource(n1yBuffer.view);
        silhouettesCursor["n1z"].setResource(n1zBuffer.view);
        silhouettesCursor["indices"].setResource(indicesBuffer.view);
        silhouettesCursor["hasOneAdjacentFace"].setResource(hasOneAdjacentFaceBuffer.view);
    }

private:
    std::vector<float> pax, pay, paz;
    std::vector<float> pbx, pby, pbz;
    std::vector<float> n0x, n0y, n0z;
    std::vector<float> n1x, n1y, n1z;
    std::vector<uint32_t> indices;
    std::vector<uint32_t> hasOneAdjacentFace;
    GPUBuffer paxBuffer = {}, payBuffer = {}, pazBuffer = {};
    GPUBuffer pbxBuffer = {}, pbyBuffer = {}, pbzBuffer = {};
    GPUBuffer n0xBuffer = {}, n0yBuffer = {}, n0zBuffer = {};
    GPUBuffer n1xBuffer = {}, n1yBuffer = {}, n1zBuffer = {};
    GPUBuffer indicesBuffer = {};
    GPUBuffer hasOneAdjacentFaceBuffer = {};
};

template<size_t DIM,
         typename NodeType,
         typename PrimitiveType,
         typename SilhouetteType,
         typename GPUNodesType,
         typename GPUPrimitivesType,
         typename GPUSilhouettesType>
class CPUBvhDataExtractor {
public:
    CPUBvhDataExtractor(const Bvh<DIM, NodeType, PrimitiveType, SilhouetteType> *bvh_) {
        std::cerr << "CPUBvhDataExtractor() not supported" << std::endl;
        exit(EXIT_FAILURE);
    }

    void extractNodes(GPUNodesType& gpuNodes) {
        std::cerr << "CPUBvhDataExtractor::extractNodes() not supported" << std::endl;
        exit(EXIT_FAILURE);
    }

    void extractPrimitives(GPUPrimitivesType& gpuPrimitives) {
        std::cerr << "CPUBvhDataExtractor::extractPrimitives() not supported" << std::endl;
        exit(EXIT_FAILURE);
    }

    void extractSilhouettes(GPUSilhouettesType& gpuSilhouettes) {
        std::cerr << "CPUBvhDataExtractor::extractSilhouettes() not supported" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string getReflectionType() const {
        std::cerr << "CPUBvhDataExtractor::getReflectionType() not supported" << std::endl;
        exit(EXIT_FAILURE);

        return "";
    }
};

template<>
class CPUBvhDataExtractor<2, BvhNode<2>, LineSegment, SilhouettePrimitive<2>, GPUBvhNodes, GPULineSegments, GPUNoSilhouettes> {
public:
    CPUBvhDataExtractor(const Bvh<2, BvhNode<2>, LineSegment, SilhouettePrimitive<2>> *bvh_): bvh(bvh_) {}

    void extractNodes(GPUBvhNodes& gpuBvhNodes) {
        gpuBvhNodes.extract<2>(bvh->flatTree);
    }

    void extractPrimitives(GPULineSegments& gpuLineSegments) {
        gpuLineSegments.extract(bvh->primitives);
    }

    void extractSilhouettes(GPUNoSilhouettes& gpuSilhouettes) {
        // do nothing
    }

    std::string getReflectionType() const {
        return "Bvh<BvhNodes, LineSegments, NoSilhouettes>";
    }

    const Bvh<2, BvhNode<2>, LineSegment, SilhouettePrimitive<2>> *bvh;
};

template<>
class CPUBvhDataExtractor<2, SnchNode<2>, LineSegment, SilhouetteVertex, GPUSnchNodes, GPULineSegments, GPUSilhouetteVertices> {
public:
    CPUBvhDataExtractor(const Bvh<2, SnchNode<2>, LineSegment, SilhouetteVertex> *bvh_): bvh(bvh_) {}

    void extractNodes(GPUSnchNodes& gpuSnchNodes) {
        gpuSnchNodes.extract<2>(bvh->flatTree);
    }

    void extractPrimitives(GPULineSegments& gpuLineSegments) {
        gpuLineSegments.extract(bvh->primitives);
    }

    void extractSilhouettes(GPUSilhouetteVertices& gpuSilhouetteVertices) {
        gpuSilhouetteVertices.extract(bvh->silhouetteRefs);
    }

    std::string getReflectionType() const {
        return "Bvh<SnchNodes, LineSegments, Vertices>";
    }

    const Bvh<2, SnchNode<2>, LineSegment, SilhouetteVertex> *bvh;
};

template<>
class CPUBvhDataExtractor<3, BvhNode<3>, Triangle, SilhouettePrimitive<3>, GPUBvhNodes, GPUTriangles, GPUNoSilhouettes> {
public:
    CPUBvhDataExtractor(const Bvh<3, BvhNode<3>, Triangle, SilhouettePrimitive<3>> *bvh_): bvh(bvh_) {}

    void extractNodes(GPUBvhNodes& gpuBvhNodes) {
        gpuBvhNodes.extract<3>(bvh->flatTree);
    }

    void extractPrimitives(GPUTriangles& gpuTriangles) {
        gpuTriangles.extract(bvh->primitives);
    }

    void extractSilhouettes(GPUNoSilhouettes& gpuSilhouettes) {
        // do nothing
    }

    std::string getReflectionType() const {
        return "Bvh<BvhNodes, Triangles, NoSilhouettes>";
    }

    const Bvh<3, BvhNode<3>, Triangle, SilhouettePrimitive<3>> *bvh;
};

template<>
class CPUBvhDataExtractor<3, SnchNode<3>, Triangle, SilhouetteEdge, GPUSnchNodes, GPUTriangles, GPUSilhouetteEdges> {
public:
    CPUBvhDataExtractor(const Bvh<3, SnchNode<3>, Triangle, SilhouetteEdge> *bvh_): bvh(bvh_) {}

    void extractNodes(GPUSnchNodes& gpuSnchNodes) {
        gpuSnchNodes.extract<3>(bvh->flatTree);
    }

    void extractPrimitives(GPUTriangles& gpuTriangles) {
        gpuTriangles.extract(bvh->primitives);
    }

    void extractSilhouettes(GPUSilhouetteEdges& gpuSilhouetteEdges) {
        gpuSilhouetteEdges.extract(bvh->silhouetteRefs);
    }

    std::string getReflectionType() const {
        return "Bvh<SnchNodes, Triangles, Edges>";
    }

    const Bvh<3, SnchNode<3>, Triangle, SilhouetteEdge> *bvh;
};

template<size_t DIM,
         typename NodeType,
         typename PrimitiveType,
         typename SilhouetteType>
class CPUBvhUpdateDataExtractor {
public:
    CPUBvhUpdateDataExtractor(const Bvh<DIM, NodeType, PrimitiveType, SilhouetteType> *bvh_): bvh(bvh_) {}

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

template<size_t DIM,
         typename NodeType,
         typename PrimitiveType,
         typename SilhouetteType,
         typename GPUNodesType,
         typename GPUPrimitivesType,
         typename GPUSilhouettesType>
class GPUBvhBuffersImpl {
public:
    void allocateNodes(ComPtr<IDevice>& device, const SceneData<DIM> *cpuSceneData) {
        // extract nodes data from cpu bvh
        const Bvh<DIM, NodeType, PrimitiveType, SilhouetteType> *bvh =
            reinterpret_cast<const Bvh<DIM, NodeType, PrimitiveType, SilhouetteType> *>(
                cpuSceneData->aggregate.get());
        CPUBvhDataExtractor<DIM,
                            NodeType,
                            PrimitiveType,
                            SilhouetteType,
                            GPUNodesType,
                            GPUPrimitivesType,
                            GPUSilhouettesType> cpuBvhDataExtractor(bvh);
        cpuBvhDataExtractor.extractNodes(nodes);
        reflectionType = cpuBvhDataExtractor.getReflectionType();

        // allocate gpu buffers
        nodes.allocate(device);
    }

    void allocatePrimitives(ComPtr<IDevice>& device, const SceneData<DIM> *cpuSceneData) {
        // extract primitives data from cpu bvh
        const Bvh<DIM, NodeType, PrimitiveType, SilhouetteType> *bvh =
            reinterpret_cast<const Bvh<DIM, NodeType, PrimitiveType, SilhouetteType> *>(
                cpuSceneData->aggregate.get());
        CPUBvhDataExtractor<DIM,
                            NodeType,
                            PrimitiveType,
                            SilhouetteType,
                            GPUNodesType,
                            GPUPrimitivesType,
                            GPUSilhouettesType> cpuBvhDataExtractor(bvh);
        cpuBvhDataExtractor.extractPrimitives(primitives);

        // allocate gpu buffers
        primitives.allocate(device);
    }

    void allocateSilhouettes(ComPtr<IDevice>& device, const SceneData<DIM> *cpuSceneData) {
        // extract silhouettes data from cpu bvh
        const Bvh<DIM, NodeType, PrimitiveType, SilhouetteType> *bvh =
            reinterpret_cast<const Bvh<DIM, NodeType, PrimitiveType, SilhouetteType> *>(
                cpuSceneData->aggregate.get());
        CPUBvhDataExtractor<DIM,
                            NodeType,
                            PrimitiveType,
                            SilhouetteType,
                            GPUNodesType,
                            GPUPrimitivesType,
                            GPUSilhouettesType> cpuBvhDataExtractor(bvh);
        cpuBvhDataExtractor.extractSilhouettes(silhouettes);

        // allocate gpu buffers
        silhouettes.allocate(device);
    }

    void allocateRefitData(ComPtr<IDevice>& device, const SceneData<DIM> *cpuSceneData) {
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
        Slang::Result result = nodeIndices.create<uint32_t>(
            device, false, nodeIndicesData.data(), nodeIndicesData.size());
        exitOnError(result, "failed to create nodeIndices object");
    }

    ComPtr<IShaderObject> createShaderObject(ComPtr<IDevice>& device, const Shader& shader) const {
        // create shader object
        ComPtr<IShaderObject> shaderObject;
        Slang::Result result = device->createShaderObject(
            shader.reflection->findTypeByName(reflectionType.c_str()),
            ShaderObjectContainerType::None, shaderObject.writeRef());
        exitOnError(result, "failed to create bvh shader object");

        return shaderObject;
    }

    int setResources(ShaderCursor& shaderCursor) const {
        ShaderCursor nodesCursor = shaderCursor["nodes"];
        nodes.setResources(nodesCursor);
        ShaderCursor primitivesCursor = shaderCursor["primitives"];
        primitives.setResources(primitivesCursor);
        ShaderCursor silhouettesCursor = shaderCursor["silhouettes"];
        silhouettes.setResources(silhouettesCursor);

        return 3;
    }

    int setRefitResources(ShaderCursor& entryPointCursor) const {
        entryPointCursor.getPath("nodeIndices").setResource(nodeIndices.view);

        return 1;
    }

    uint32_t getMaxUpdateDepth() const {
        return maxUpdateDepth;
    }

    std::pair<uint32_t, uint32_t> getUpdateEntryData(uint32_t depth) {
        return updateEntryData[depth];
    }

private:
    GPUNodesType nodes;
    GPUPrimitivesType primitives;
    GPUSilhouettesType silhouettes;
    GPUBuffer nodeIndices = {};
    std::vector<std::pair<uint32_t, uint32_t>> updateEntryData;
    uint32_t maxUpdateDepth = 0;
    std::string reflectionType = "";
};

class GPUBvhBuffers {
public:
    template<size_t DIM, bool hasSilhouettes>
    void init() {
        std::cerr << "GPUBvhBuffers::init()" << std::endl;
        exit(EXIT_FAILURE);
    }

    void allocateNodes(ComPtr<IDevice>& device, const SceneData<DIM> *cpuSceneData) {
        if (bvh2DBuffers != nullptr) bvh2DBuffers->allocateNodes(device, cpuSceneData);
        else if (snch2DBuffers != nullptr) snch2DBuffers->allocateNodes(device, cpuSceneData);
        else if (bvh3DBuffers != nullptr) bvh3DBuffers->allocateNodes(device, cpuSceneData);
        else if (snch3DBuffers != nullptr) snch3DBuffers->allocateNodes(device, cpuSceneData);
    }

    void allocatePrimitives(ComPtr<IDevice>& device, const SceneData<DIM> *cpuSceneData) {
        if (bvh2DBuffers != nullptr) bvh2DBuffers->allocatePrimitives(device, cpuSceneData);
        else if (snch2DBuffers != nullptr) snch2DBuffers->allocatePrimitives(device, cpuSceneData);
        else if (bvh3DBuffers != nullptr) bvh3DBuffers->allocatePrimitives(device, cpuSceneData);
        else if (snch3DBuffers != nullptr) snch3DBuffers->allocatePrimitives(device, cpuSceneData);
    }

    void allocateSilhouettes(ComPtr<IDevice>& device, const SceneData<DIM> *cpuSceneData) {
        if (bvh2DBuffers != nullptr) bvh2DBuffers->allocateSilhouettes(device, cpuSceneData);
        else if (snch2DBuffers != nullptr) snch2DBuffers->allocateSilhouettes(device, cpuSceneData);
        else if (bvh3DBuffers != nullptr) bvh3DBuffers->allocateSilhouettes(device, cpuSceneData);
        else if (snch3DBuffers != nullptr) snch3DBuffers->allocateSilhouettes(device, cpuSceneData);
    }

    void allocateRefitData(ComPtr<IDevice>& device, const SceneData<DIM> *cpuSceneData) {
        if (bvh2DBuffers != nullptr) bvh2DBuffers->allocateRefitData(device, cpuSceneData);
        else if (snch2DBuffers != nullptr) snch2DBuffers->allocateRefitData(device, cpuSceneData);
        else if (bvh3DBuffers != nullptr) bvh3DBuffers->allocateRefitData(device, cpuSceneData);
        else if (snch3DBuffers != nullptr) snch3DBuffers->allocateRefitData(device, cpuSceneData);
    }

    ComPtr<IShaderObject> createShaderObject(ComPtr<IDevice>& device, const Shader& shader) const {
        if (bvh2DBuffers != nullptr) return bvh2DBuffers->createShaderObject(device, shader);
        else if (snch2DBuffers != nullptr) return snch2DBuffers->createShaderObject(device, shader);
        else if (bvh3DBuffers != nullptr) return bvh3DBuffers->createShaderObject(device, shader);
        else if (snch3DBuffers != nullptr) return snch3DBuffers->createShaderObject(device, shader);
    }

    int setResources(ShaderCursor& shaderCursor) const {
        if (bvh2DBuffers != nullptr) return bvh2DBuffers->setResources(shaderCursor);
        else if (snch2DBuffers != nullptr) return snch2DBuffers->setResources(shaderCursor);
        else if (bvh3DBuffers != nullptr) return bvh3DBuffers->setResources(shaderCursor);
        else if (snch3DBuffers != nullptr) return snch3DBuffers->setResources(shaderCursor);
    }

    int setRefitResources(ShaderCursor& entryPointCursor) const {
        if (bvh2DBuffers != nullptr) return bvh2DBuffers->setRefitResources(entryPointCursor);
        else if (snch2DBuffers != nullptr) return snch2DBuffers->setRefitResources(entryPointCursor);
        else if (bvh3DBuffers != nullptr) return bvh3DBuffers->setRefitResources(entryPointCursor);
        else if (snch3DBuffers != nullptr) return snch3DBuffers->setRefitResources(entryPointCursor);
    }

    uint32_t getMaxUpdateDepth() const {
        if (bvh2DBuffers != nullptr) return bvh2DBuffers->getMaxUpdateDepth();
        else if (snch2DBuffers != nullptr) return snch2DBuffers->getMaxUpdateDepth();
        else if (bvh3DBuffers != nullptr) return bvh3DBuffers->getMaxUpdateDepth();
        else if (snch3DBuffers != nullptr) return snch3DBuffers->getMaxUpdateDepth();
    }

    std::pair<uint32_t, uint32_t> getUpdateEntryData(uint32_t depth) {
        if (bvh2DBuffers != nullptr) return bvh2DBuffers->getUpdateEntryData(depth);
        else if (snch2DBuffers != nullptr) return snch2DBuffers->getUpdateEntryData(depth);
        else if (bvh3DBuffers != nullptr) return bvh3DBuffers->getUpdateEntryData(depth);
        else if (snch3DBuffers != nullptr) return snch3DBuffers->getUpdateEntryData(depth);
    }

private:
    void reset() {
        bvh2DBuffers = nullptr;
        snch2DBuffers = nullptr;
        bvh3DBuffers = nullptr;
        snch3DBuffers = nullptr;
    }

    typedef GPUBvhBuffersImpl<2, BvhNode<2>, LineSegment, SilhouettePrimitive<2>, GPUBvhNodes, GPULineSegments, GPUNoSilhouettes> Bvh2DBuffers;
    std::unique_ptr<Bvh2DBuffers> bvh2DBuffers = nullptr;
    typedef GPUBvhBuffersImpl<2, SnchNode<2>, LineSegment, SilhouetteVertex, GPUSnchNodes, GPULineSegments, GPUSilhouetteVertices> Snch2DBuffers;
    std::unique_ptr<Snch2DBuffers> snch2DBuffers = nullptr;
    typedef GPUBvhBuffersImpl<3, BvhNode<3>, Triangle, SilhouettePrimitive<3>, GPUBvhNodes, GPUTriangles, GPUNoSilhouettes> Bvh3DBuffers;
    std::unique_ptr<Bvh3DBuffers> bvh3DBuffers = nullptr;
    typedef GPUBvhBuffersImpl<3, SnchNode<3>, Triangle, SilhouetteEdge, GPUSnchNodes, GPUTriangles, GPUSilhouetteEdges> Snch3DBuffers;
    std::unique_ptr<Snch3DBuffers> snch3DBuffers = nullptr;
};

template<>
void GPUBvhBuffers::init<2, false>()
{
    reset();
    bvh2DBuffers = std::unique_ptr<Bvh2DBuffers>(new Bvh2DBuffers());
}

template<>
void GPUBvhBuffers::init<2, true>()
{
    reset();
    snch2DBuffers = std::unique_ptr<Snch2DBuffers>(new Snch2DBuffers());
}

template<>
void GPUBvhBuffers::init<3, false>()
{
    reset();
    bvh3DBuffers = std::unique_ptr<Bvh3DBuffers>(new Bvh3DBuffers());
}

template<>
void GPUBvhBuffers::init<3, true>()
{
    reset();
    snch3DBuffers = std::unique_ptr<Snch3DBuffers>(new Snch3DBuffers());
}

struct GPURays {
    void setSize(int size) {
        ox.resize(size, 0.0f);
        oy.resize(size, 0.0f);
        oz.resize(size, 0.0f);
        dx.resize(size, 0.0f);
        dy.resize(size, 0.0f);
        dz.resize(size, 0.0f);
        tMax.resize(size, 0.0f);
    }

    void allocate(ComPtr<IDevice>& device) {
        Slang::Result result = oxBuffer.create<float>(device, false, ox.data(), ox.size());
        exitOnError(result, "failed to allocate GPURays buffer");
        result = oyBuffer.create<float>(device, false, oy.data(), oy.size());
        exitOnError(result, "failed to allocate GPURays buffer");
        result = ozBuffer.create<float>(device, false, oz.data(), oz.size());
        exitOnError(result, "failed to allocate GPURays buffer");
        result = dxBuffer.create<float>(device, false, dx.data(), dx.size());
        exitOnError(result, "failed to allocate GPURays buffer");
        result = dyBuffer.create<float>(device, false, dy.data(), dy.size());
        exitOnError(result, "failed to allocate GPURays buffer");
        result = dzBuffer.create<float>(device, false, dz.data(), dz.size());
        exitOnError(result, "failed to allocate GPURays buffer");
        result = tMaxBuffer.create<float>(device, false, tMax.data(), tMax.size());
        exitOnError(result, "failed to allocate GPURays buffer");
    }

    void setResources(ShaderCursor& raysCursor) const {
        raysCursor["ox"].setResource(oxBuffer.view);
        raysCursor["oy"].setResource(oyBuffer.view);
        raysCursor["oz"].setResource(ozBuffer.view);
        raysCursor["dx"].setResource(dxBuffer.view);
        raysCursor["dy"].setResource(dyBuffer.view);
        raysCursor["dz"].setResource(dzBuffer.view);
        raysCursor["tMax"].setResource(tMaxBuffer.view);
    }

    std::vector<float> ox, oy, oz;
    std::vector<float> dx, dy, dz;
    std::vector<float> tMax;

private:
    GPUBuffer oxBuffer = {}, oyBuffer = {}, ozBuffer = {};
    GPUBuffer dxBuffer = {}, dyBuffer = {}, dzBuffer = {};
    GPUBuffer tMaxBuffer = {};
};

struct GPUBoundingSpheres {
    void setSize(int size) {
        cx.resize(size, 0.0f);
        cy.resize(size, 0.0f);
        cz.resize(size, 0.0f);
        r2.resize(size, 0.0f);
    }

    void allocate(ComPtr<IDevice>& device) {
        Slang::Result result = cxBuffer.create<float>(device, false, cx.data(), cx.size());
        exitOnError(result, "failed to allocate GPUBoundingSpheres buffer");
        result = cyBuffer.create<float>(device, false, cy.data(), cy.size());
        exitOnError(result, "failed to allocate GPUBoundingSpheres buffer");
        result = czBuffer.create<float>(device, false, cz.data(), cz.size());
        exitOnError(result, "failed to allocate GPUBoundingSpheres buffer");
        result = r2Buffer.create<float>(device, false, r2.data(), r2.size());
        exitOnError(result, "failed to allocate GPUBoundingSpheres buffer");
    }

    void setResources(ShaderCursor& boundingSpheresCursor) const {
        boundingSpheresCursor["cx"].setResource(cxBuffer.view);
        boundingSpheresCursor["cy"].setResource(cyBuffer.view);
        boundingSpheresCursor["cz"].setResource(czBuffer.view);
        boundingSpheresCursor["r2"].setResource(r2Buffer.view);
    }

    std::vector<float> cx, cy, cz;
    std::vector<float> r2;

private:
    GPUBuffer cxBuffer = {}, cyBuffer = {}, czBuffer = {};
    GPUBuffer r2Buffer = {};
};

struct GPUFloat3List {
    void setSize(int size) {
        x.resize(size, 0.0f);
        y.resize(size, 0.0f);
        z.resize(size, 0.0f);
    }

    void allocate(ComPtr<IDevice>& device) {
        Slang::Result result = xBuffer.create<float>(device, false, x.data(), x.size());
        exitOnError(result, "failed to allocate GPUFloat3List buffer");
        result = yBuffer.create<float>(device, false, y.data(), y.size());
        exitOnError(result, "failed to allocate GPUFloat3List buffer");
        result = zBuffer.create<float>(device, false, z.data(), z.size());
        exitOnError(result, "failed to allocate GPUFloat3List buffer");
    }

    void setResources(ShaderCursor& randNumsCursor) const {
        randNumsCursor["x"].setResource(xBuffer.view);
        randNumsCursor["y"].setResource(yBuffer.view);
        randNumsCursor["z"].setResource(zBuffer.view);
    }

    std::vector<float> x, y, z;

private:
    GPUBuffer xBuffer = {}, yBuffer = {}, zBuffer = {};
};

struct GPUInteractions {
    void setSize(int size) {
        px.resize(size, 0.0f);
        py.resize(size, 0.0f);
        pz.resize(size, 0.0f);
        nx.resize(size, 0.0f);
        ny.resize(size, 0.0f);
        nz.resize(size, 0.0f);
        uvx.resize(size, 0.0f);
        uvy.resize(size, 0.0f);
        d.resize(size, 0.0f);
        indices.resize(size, 0);
    }

    void allocate(ComPtr<IDevice>& device) {
        Slang::Result result = pxBuffer.create<float>(device, true, px.data(), px.size());
        exitOnError(result, "failed to allocate GPUInteractions buffer");
        result = pyBuffer.create<float>(device, true, py.data(), py.size());
        exitOnError(result, "failed to allocate GPUInteractions buffer");
        result = pzBuffer.create<float>(device, true, pz.data(), pz.size());
        exitOnError(result, "failed to allocate GPUInteractions buffer");
        result = nxBuffer.create<float>(device, true, nx.data(), nx.size());
        exitOnError(result, "failed to allocate GPUInteractions buffer");
        result = nyBuffer.create<float>(device, true, ny.data(), ny.size());
        exitOnError(result, "failed to allocate GPUInteractions buffer");
        result = nzBuffer.create<float>(device, true, nz.data(), nz.size());
        exitOnError(result, "failed to allocate GPUInteractions buffer");
        result = uvxBuffer.create<float>(device, true, uvx.data(), uvx.size());
        exitOnError(result, "failed to allocate GPUInteractions buffer");
        result = uvyBuffer.create<float>(device, true, uvy.data(), uvy.size());
        exitOnError(result, "failed to allocate GPUInteractions buffer");
        result = dBuffer.create<float>(device, true, d.data(), d.size());
        exitOnError(result, "failed to allocate GPUInteractions buffer");
        result = indicesBuffer.create<uint32_t>(device, true, indices.data(), indices.size());
        exitOnError(result, "failed to allocate GPUInteractions buffer");
    }

    void setResources(ShaderCursor& interactionsCursor) const {
        interactionsCursor["px"].setResource(pxBuffer.view);
        interactionsCursor["py"].setResource(pyBuffer.view);
        interactionsCursor["pz"].setResource(pzBuffer.view);
        interactionsCursor["nx"].setResource(nxBuffer.view);
        interactionsCursor["ny"].setResource(nyBuffer.view);
        interactionsCursor["nz"].setResource(nzBuffer.view);
        interactionsCursor["uvx"].setResource(uvxBuffer.view);
        interactionsCursor["uvy"].setResource(uvyBuffer.view);
        interactionsCursor["d"].setResource(dBuffer.view);
        interactionsCursor["indices"].setResource(indicesBuffer.view);
    }

    void read(ComPtr<IDevice>& device) {
        Slang::Result result = pxBuffer.read<float>(device, px.size(), px);
        exitOnError(result, "failed to read GPUInteractions buffer");
        result = pyBuffer.read<float>(device, py.size(), py);
        exitOnError(result, "failed to read GPUInteractions buffer");
        result = pzBuffer.read<float>(device, pz.size(), pz);
        exitOnError(result, "failed to read GPUInteractions buffer");
        result = nxBuffer.read<float>(device, nx.size(), nx);
        exitOnError(result, "failed to read GPUInteractions buffer");
        result = nyBuffer.read<float>(device, ny.size(), ny);
        exitOnError(result, "failed to read GPUInteractions buffer");
        result = nzBuffer.read<float>(device, nz.size(), nz);
        exitOnError(result, "failed to read GPUInteractions buffer");
        result = uvxBuffer.read<float>(device, uvx.size(), uvx);
        exitOnError(result, "failed to read GPUInteractions buffer");
        result = uvyBuffer.read<float>(device, uvy.size(), uvy);
        exitOnError(result, "failed to read GPUInteractions buffer");
        result = dBuffer.read<float>(device, d.size(), d);
        exitOnError(result, "failed to read GPUInteractions buffer");
        result = indicesBuffer.read<uint32_t>(device, indices.size(), indices);
        exitOnError(result, "failed to read GPUInteractions buffer");
    }

    std::vector<float> px, py, pz;
    std::vector<float> nx, ny, nz;
    std::vector<float> uvx, uvy;
    std::vector<float> d;
    std::vector<uint32_t> indices;

private:
    GPUBuffer pxBuffer = {}, pyBuffer = {}, pzBuffer = {};
    GPUBuffer nxBuffer = {}, nyBuffer = {}, nzBuffer = {};
    GPUBuffer uvxBuffer = {}, uvyBuffer = {};
    GPUBuffer dBuffer = {};
    GPUBuffer indicesBuffer = {};
};

struct GPURayQueryBuffers {
    GPURays rays;
    bool checkForOcclusion = false;
    GPUInteractions interactions;
    int nQueries;

    void setQueryCount(int nQueries_) {
        nQueries = nQueries_;
        rays.setSize(nQueries);
        interactions.setSize(nQueries);
    }

    void allocate(ComPtr<IDevice>& device) {
        rays.allocate(device);
        interactions.allocate(device);
    }

    int setResources(ShaderCursor& entryPointCursor) const {
        ShaderCursor raysCursor = entryPointCursor.getPath("rays");
        rays.setResources(raysCursor);
        entryPointCursor.getPath("checkForOcclusion").setData(checkForOcclusion);
        ShaderCursor interactionsCursor = entryPointCursor.getPath("interactions");
        interactions.setResources(interactionsCursor);
        entryPointCursor.getPath("nQueries").setData(nQueries);

        return 4;
    }
};

struct GPUSphereIntersectionQueryBuffers {
    GPUBoundingSpheres boundingSpheres;
    GPUFloat3List randNums;
    GPUInteractions interactions;
    int nQueries;

    void setQueryCount(int nQueries_) {
        nQueries = nQueries_;
        boundingSpheres.setSize(nQueries);
        randNums.setSize(nQueries);
        interactions.setSize(nQueries);
    }

    void allocate(ComPtr<IDevice>& device) {
        boundingSpheres.allocate(device);
        randNums.allocate(device);
        interactions.allocate(device);
    }

    int setResources(ShaderCursor& entryPointCursor) const {
        ShaderCursor boundingSpheresCursor = entryPointCursor.getPath("boundingSpheres");
        boundingSpheres.setResources(boundingSpheresCursor);
        ShaderCursor randNumsCursor = entryPointCursor.getPath("randNums");
        randNums.setResources(randNumsCursor);
        ShaderCursor interactionsCursor = entryPointCursor.getPath("interactions");
        interactions.setResources(interactionsCursor);
        entryPointCursor.getPath("nQueries").setData(nQueries);

        return 4;
    }
};

struct GPUClosestPointQueryBuffers {
    GPUBoundingSpheres boundingSpheres;
    GPUInteractions interactions;
    float recordNormals = false;
    int nQueries;

    void setQueryCount(int nQueries_) {
        nQueries = nQueries_;
        boundingSpheres.setSize(nQueries);
        interactions.setSize(nQueries);
    }

    void allocate(ComPtr<IDevice>& device) {
        boundingSpheres.allocate(device);
        interactions.allocate(device);
    }

    int setResources(ShaderCursor& entryPointCursor) const {
        ShaderCursor boundingSpheresCursor = entryPointCursor.getPath("boundingSpheres");
        boundingSpheres.setResources(boundingSpheresCursor);
        ShaderCursor interactionsCursor = entryPointCursor.getPath("interactions");
        interactions.setResources(interactionsCursor);
        entryPointCursor.getPath("recordNormals").setData(recordNormals);
        entryPointCursor.getPath("nQueries").setData(nQueries);

        return 4;
    }
};

struct GPUClosestSilhouettePointQueryBuffers {
    GPUBoundingSpheres boundingSpheres;
    std::vector<uint32_t> flipNormalOrientationData;
    float squaredMinRadius = 1e-6f;
    float precision = 1e-3f;
    GPUInteractions interactions;
    int nQueries;

    void setQueryCount(int nQueries_) {
        nQueries = nQueries_;
        boundingSpheres.setSize(nQueries);
        flipNormalOrientationData.resize(nQueries, 0);
        interactions.setSize(nQueries);
    }

    void allocate(ComPtr<IDevice>& device) {
        boundingSpheres.allocate(device);
        Slang::Result result = flipNormalOrientation.create<uint32_t>(
            device, false, flipNormalOrientationData.data(), flipNormalOrientationData.size());
        exitOnError(result, "failed to allocate GPUClosestSilhouettePointQueryBuffers buffer");
        interactions.allocate(device);
    }

    int setResources(ShaderCursor& entryPointCursor) const {
        ShaderCursor boundingSpheresCursor = entryPointCursor.getPath("boundingSpheres");
        boundingSpheres.setResources(boundingSpheresCursor);
        entryPointCursor.getPath("flipNormalOrientation").setResource(flipNormalOrientation.view);
        entryPointCursor.getPath("squaredMinRadius").setData(squaredMinRadius);
        entryPointCursor.getPath("precision").setData(precision);
        ShaderCursor interactionsCursor = entryPointCursor.getPath("interactions");
        interactions.setResources(interactionsCursor);
        entryPointCursor.getPath("nQueries").setData(nQueries);

        return 6;
    }

private:
    GPUBuffer flipNormalOrientation = {};
};

} // namespace fcpw
