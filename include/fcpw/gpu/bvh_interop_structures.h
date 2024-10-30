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

    void setResources(ShaderCursor& shaderCursor) const {
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
    // constructor
    CPUBvhDataExtractor(const Bvh<DIM, NodeType, PrimitiveType, SilhouetteType> *bvh_) {
        std::cerr << "CPUBvhDataExtractor() not supported" << std::endl;
        exit(EXIT_FAILURE);
    }

    // extracts CPU bvh nodes data
    void extractNodes(GPUNodesType& gpuNodes) {
        std::cerr << "CPUBvhDataExtractor::extractNodes() not supported" << std::endl;
        exit(EXIT_FAILURE);
    }

    // extracts CPU bvh primitives data
    void extractPrimitives(GPUPrimitivesType& gpuPrimitives) {
        std::cerr << "CPUBvhDataExtractor::extractPrimitives() not supported" << std::endl;
        exit(EXIT_FAILURE);
    }

    // extracts CPU bvh silhouettes data
    void extractSilhouettes(GPUSilhouettesType& gpuSilhouettes) {
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
class CPUBvhDataExtractor<2, BvhNode<2>, LineSegment, SilhouettePrimitive<2>, GPUBvhNodes, GPULineSegments, GPUNoSilhouettes> {
public:
    // constructor
    CPUBvhDataExtractor(const Bvh<2, BvhNode<2>, LineSegment, SilhouettePrimitive<2>> *bvh_): bvh(bvh_) {}

    // extracts CPU bvh nodes data
    void extractNodes(GPUBvhNodes& gpuBvhNodes) {
        gpuBvhNodes.extract<2>(bvh->flatTree);
    }

    // extracts CPU bvh primitives data
    void extractPrimitives(GPULineSegments& gpuLineSegments) {
        gpuLineSegments.extract(bvh->primitives);
    }

    // extracts CPU bvh silhouettes data
    void extractSilhouettes(GPUNoSilhouettes& gpuSilhouettes) {
        // do nothing
    }

    // returns reflection type
    std::string getReflectionType() const {
        return "Bvh<BvhNodes, LineSegments, NoSilhouettes>";
    }

    // member
    const Bvh<2, BvhNode<2>, LineSegment, SilhouettePrimitive<2>> *bvh;
};

template<>
class CPUBvhDataExtractor<2, SnchNode<2>, LineSegment, SilhouetteVertex, GPUSnchNodes, GPULineSegments, GPUSilhouetteVertices> {
public:
    // constructor
    CPUBvhDataExtractor(const Bvh<2, SnchNode<2>, LineSegment, SilhouetteVertex> *bvh_): bvh(bvh_) {}

    // extracts CPU bvh nodes data
    void extractNodes(GPUSnchNodes& gpuSnchNodes) {
        gpuSnchNodes.extract<2>(bvh->flatTree);
    }

    // extracts CPU bvh primitives data
    void extractPrimitives(GPULineSegments& gpuLineSegments) {
        gpuLineSegments.extract(bvh->primitives);
    }

    // extracts CPU bvh silhouettes data
    void extractSilhouettes(GPUSilhouetteVertices& gpuSilhouetteVertices) {
        gpuSilhouetteVertices.extract(bvh->silhouetteRefs);
    }

    // returns reflection type
    std::string getReflectionType() const {
        return "Bvh<SnchNodes, LineSegments, Vertices>";
    }

    // member
    const Bvh<2, SnchNode<2>, LineSegment, SilhouetteVertex> *bvh;
};

template<>
class CPUBvhDataExtractor<3, BvhNode<3>, Triangle, SilhouettePrimitive<3>, GPUBvhNodes, GPUTriangles, GPUNoSilhouettes> {
public:
    // constructor
    CPUBvhDataExtractor(const Bvh<3, BvhNode<3>, Triangle, SilhouettePrimitive<3>> *bvh_): bvh(bvh_) {}

    // extracts CPU bvh nodes data
    void extractNodes(GPUBvhNodes& gpuBvhNodes) {
        gpuBvhNodes.extract<3>(bvh->flatTree);
    }

    // extracts CPU bvh primitives data
    void extractPrimitives(GPUTriangles& gpuTriangles) {
        gpuTriangles.extract(bvh->primitives);
    }

    // extracts CPU bvh silhouettes data
    void extractSilhouettes(GPUNoSilhouettes& gpuSilhouettes) {
        // do nothing
    }

    // returns reflection type
    std::string getReflectionType() const {
        return "Bvh<BvhNodes, Triangles, NoSilhouettes>";
    }

    // member
    const Bvh<3, BvhNode<3>, Triangle, SilhouettePrimitive<3>> *bvh;
};

template<>
class CPUBvhDataExtractor<3, SnchNode<3>, Triangle, SilhouetteEdge, GPUSnchNodes, GPUTriangles, GPUSilhouetteEdges> {
public:
    // constructor
    CPUBvhDataExtractor(const Bvh<3, SnchNode<3>, Triangle, SilhouetteEdge> *bvh_): bvh(bvh_) {}

    // extracts CPU bvh nodes data
    void extractNodes(GPUSnchNodes& gpuSnchNodes) {
        gpuSnchNodes.extract<3>(bvh->flatTree);
    }

    // extracts CPU bvh primitives data
    void extractPrimitives(GPUTriangles& gpuTriangles) {
        gpuTriangles.extract(bvh->primitives);
    }

    // extracts CPU bvh silhouettes data
    void extractSilhouettes(GPUSilhouetteEdges& gpuSilhouetteEdges) {
        gpuSilhouetteEdges.extract(bvh->silhouetteRefs);
    }

    // returns reflection type
    std::string getReflectionType() const {
        return "Bvh<SnchNodes, Triangles, Edges>";
    }

    // member
    const Bvh<3, SnchNode<3>, Triangle, SilhouetteEdge> *bvh;
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

    void allocateGeometry(ComPtr<IDevice>& device, const SceneData<DIM> *cpuSceneData) {
        // extract primitives and silhouettes data from cpu bvh
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
        cpuBvhDataExtractor.extractSilhouettes(silhouettes);

        // allocate gpu buffers
        primitives.allocate(device);
        silhouettes.allocate(device);
    }

    void allocateRefitData(ComPtr<IDevice>& device, const SceneData<DIM> *cpuSceneData) {
        // extract update data from cpu bvh
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
        updateEntryData.clear();
        std::vector<uint32_t> nodeIndicesData;
        maxUpdateDepth = cpuBvhUpdateDataExtractor.extract(nodeIndicesData, updateEntryData);

        // allocate gpu buffer
        Slang::Result result = nodeIndices.create<uint32_t>(
            device, false, nodeIndicesData.data(), nodeIndicesData.size());
        exitOnError(result, "failed to create nodeIndices object");
    }

    ComPtr<IShaderObject> createShaderObject(ComPtr<IDevice>& device,
                                             const Shader& shader,
                                             bool printLogs) const {
        // create shader object
        ComPtr<IShaderObject> shaderObject;
        Slang::Result result = device->createShaderObject(
            shader.reflection->findTypeByName(reflectionType.c_str()),
            ShaderObjectContainerType::None, shaderObject.writeRef());
        exitOnError(result, "failed to create bvh shader object");

        // set shader object resources
        ShaderCursor shaderCursor(shaderObject);
        ShaderCursor nodesCursor = shaderCursor["nodes"];
        nodes.setResources(nodesCursor);
        ShaderCursor primitivesCursor = shaderCursor["primitives"];
        primitives.setResources(primitivesCursor);
        ShaderCursor silhouettesCursor = shaderCursor["silhouettes"];
        silhouettes.setResources(silhouettesCursor);

        if (printLogs) {
            std::cout << "BvhReflectionType: " << shaderObject->getElementTypeLayout()->getName() << std::endl;
            std::cout << "\tcursor[0]: " << shaderCursor.getTypeLayout()->getFieldByIndex(0)->getName() << std::endl;
            std::cout << "\tcursor[1]: " << shaderCursor.getTypeLayout()->getFieldByIndex(1)->getName() << std::endl;
            std::cout << "\tcursor[2]: " << shaderCursor.getTypeLayout()->getFieldByIndex(2)->getName() << std::endl;   
        }

        return shaderObject;
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

    void setQueryCount(int nQueries_) {
        nQueries = nQueries_;
        rays.setSize(nQueries);
        interactions.setSize(nQueries);
    }

    void allocate(ComPtr<IDevice>& device) {
        rays.allocate(device);
        interactions.allocate(device);
    }

    int setResources(ShaderCursor& shaderCursor) const {
        ShaderCursor raysCursor = shaderCursor.getPath("rays");
        rays.setResources(raysCursor);
        shaderCursor.getPath("checkForOcclusion").setData(checkForOcclusion);
        ShaderCursor interactionsCursor = shaderCursor.getPath("interactions");
        interactions.setResources(interactionsCursor);
        shaderCursor.getPath("nQueries").setData(nQueries);

        return 5;
    }

private:
    int nQueries;
};

struct GPUSphereIntersectionQueryBuffers {
    GPUBoundingSpheres boundingSpheres;
    GPUFloat3List randNums;
    GPUInteractions interactions;

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

    int setResources(ShaderCursor& shaderCursor) const {
        ShaderCursor boundingSpheresCursor = shaderCursor.getPath("boundingSpheres");
        boundingSpheres.setResources(boundingSpheresCursor);
        ShaderCursor randNumsCursor = shaderCursor.getPath("randNums");
        randNums.setResources(randNumsCursor);
        ShaderCursor interactionsCursor = shaderCursor.getPath("interactions");
        interactions.setResources(interactionsCursor);
        shaderCursor.getPath("nQueries").setData(nQueries);

        return 5;
    }

private:
    int nQueries;
};

struct GPUClosestPointQueryBuffers {
    GPUBoundingSpheres boundingSpheres;
    GPUInteractions interactions;
    float recordNormals = false;

    void setQueryCount(int nQueries_) {
        nQueries = nQueries_;
        boundingSpheres.setSize(nQueries);
        interactions.setSize(nQueries);
    }

    void allocate(ComPtr<IDevice>& device) {
        boundingSpheres.allocate(device);
        interactions.allocate(device);
    }

    int setResources(ShaderCursor& shaderCursor) const {
        ShaderCursor boundingSpheresCursor = shaderCursor.getPath("boundingSpheres");
        boundingSpheres.setResources(boundingSpheresCursor);
        ShaderCursor interactionsCursor = shaderCursor.getPath("interactions");
        interactions.setResources(interactionsCursor);
        shaderCursor.getPath("recordNormals").setData(recordNormals);
        shaderCursor.getPath("nQueries").setData(nQueries);

        return 5;
    }

private:
    int nQueries;
};

struct GPUClosestSilhouettePointQueryBuffers {
    GPUBoundingSpheres boundingSpheres;
    std::vector<uint32_t> flipNormalOrientationData;
    float squaredMinRadius = 1e-6f;
    float precision = 1e-3f;
    GPUInteractions interactions;

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

    int setResources(ShaderCursor& shaderCursor) const {
        ShaderCursor boundingSpheresCursor = shaderCursor.getPath("boundingSpheres");
        boundingSpheres.setResources(boundingSpheresCursor);
        shaderCursor.getPath("flipNormalOrientation").setResource(flipNormalOrientation.view);
        shaderCursor.getPath("squaredMinRadius").setData(squaredMinRadius);
        shaderCursor.getPath("precision").setData(precision);
        ShaderCursor interactionsCursor = shaderCursor.getPath("interactions");
        interactions.setResources(interactionsCursor);
        shaderCursor.getPath("nQueries").setData(nQueries);

        return 7;
    }

private:
    GPUBuffer flipNormalOrientation = {};
    int nQueries;
};

// ==========================================================================================

struct float3 {
    float x, y, z;
};

struct float2 {
    float x, y;
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

class GPUBvhBuffers {
public:
    GPUBuffer nodes = {};
    GPUBuffer primitives = {};
    GPUBuffer silhouettes = {};
    GPUBuffer nodeIndices = {};
    std::vector<std::pair<uint32_t, uint32_t>> updateEntryData;
    uint32_t maxUpdateDepth = 0;
    std::string reflectionType = "";

    template<size_t DIM>
    void allocate(ComPtr<IDevice>& device, const SceneData<DIM> *cpuSceneData,
                  bool allocatePrimitiveData, bool allocateSilhouetteData,
                  bool allocateNodeData, bool allocateRefitData) {
        std::cerr << "GPUBvhBuffers::allocate()" << std::endl;
        exit(EXIT_FAILURE);
    }

    ComPtr<IShaderObject> createShaderObject(ComPtr<IDevice>& device, const Shader& shader,
                                             bool printLogs) const {
        // create shader object
        ComPtr<IShaderObject> shaderObject;
        Slang::Result createShaderObjectResult = device->createShaderObject(
            shader.reflection->findTypeByName(reflectionType.c_str()),
            ShaderObjectContainerType::None, shaderObject.writeRef());
        if (createShaderObjectResult != SLANG_OK) {
            std::cout << "failed to create bvh shader object" << std::endl;
            exit(EXIT_FAILURE);
        }

        // set shader object resources
        ShaderCursor cursor(shaderObject);
        cursor["nodes"].setResource(nodes.view);
        cursor["primitives"].setResource(primitives.view);
        cursor["silhouettes"].setResource(silhouettes.view);
        if (printLogs) {
            std::cout << "BvhReflectionType: " << shaderObject->getElementTypeLayout()->getName() << std::endl;
            std::cout << "\tcursor[0]: " << cursor.getTypeLayout()->getFieldByIndex(0)->getName() << std::endl;
            std::cout << "\tcursor[1]: " << cursor.getTypeLayout()->getFieldByIndex(1)->getName() << std::endl;
            std::cout << "\tcursor[2]: " << cursor.getTypeLayout()->getFieldByIndex(2)->getName() << std::endl;   
        }

        return shaderObject;
    }

private:
    template<size_t DIM,
             typename NodeType,
             typename PrimitiveType,
             typename SilhouetteType,
             typename GpuNodeType,
             typename GPUPrimitiveType,
             typename GPUSilhouetteType>
    void allocateGeometryBuffers(ComPtr<IDevice>& device, const SceneData<DIM> *cpuSceneData) {
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
        Slang::Result createBufferResult = primitives.create<GPUPrimitiveType>(
            device, false, primitivesData.data(), primitivesData.size());
        if (createBufferResult != SLANG_OK) {
            std::cout << "failed to create primitives buffer" << std::endl;
            exit(EXIT_FAILURE);
        }

        createBufferResult = silhouettes.create<GPUSilhouetteType>(
            device, false, silhouettesData.data(), silhouettesData.size());
        if (createBufferResult != SLANG_OK) {
            std::cout << "failed to create silhouettes buffer" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    template<size_t DIM,
             typename NodeType,
             typename PrimitiveType,
             typename SilhouetteType,
             typename GpuNodeType,
             typename GPUPrimitiveType,
             typename GPUSilhouetteType>
    void allocateNodeBuffer(ComPtr<IDevice>& device, const SceneData<DIM> *cpuSceneData) {
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
        Slang::Result createBufferResult = nodes.create<GpuNodeType>(
            device, true, nodesData.data(), nodesData.size());
        if (createBufferResult != SLANG_OK) {
            std::cout << "failed to create nodes buffer" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    template<size_t DIM,
             typename NodeType,
             typename PrimitiveType,
             typename SilhouetteType>
    void allocateRefitBuffer(ComPtr<IDevice>& device, const SceneData<DIM> *cpuSceneData) {
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
        Slang::Result createBufferResult = nodeIndices.create<uint32_t>(
            device, false, nodeIndicesData.data(), nodeIndicesData.size());
        if (createBufferResult != SLANG_OK) {
            std::cout << "failed to create nodeIndices buffer" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
};

template<>
void GPUBvhBuffers::allocate<2>(ComPtr<IDevice>& device, const SceneData<2> *cpuSceneData,
                                bool allocatePrimitiveData, bool allocateSilhouetteData,
                                bool allocateNodeData, bool allocateRefitData)
{
    if (allocateSilhouetteData) {
        if (allocatePrimitiveData) {
            allocateGeometryBuffers<2, SnchNode<2>, LineSegment, SilhouetteVertex,
                                    GPUSnchNode, GPULineSegment, GPUVertex>(device, cpuSceneData);
        }

        if (allocateNodeData) {
            allocateNodeBuffer<2, SnchNode<2>, LineSegment, SilhouetteVertex,
                               GPUSnchNode, GPULineSegment, GPUVertex>(device, cpuSceneData);
        }

        if (allocateRefitData) {
            allocateRefitBuffer<2, SnchNode<2>, LineSegment, SilhouetteVertex>(device, cpuSceneData);
        }

    } else {
        if (allocatePrimitiveData) {
            allocateGeometryBuffers<2, BvhNode<2>, LineSegment, SilhouettePrimitive<2>,
                                    GPUBvhNode, GPULineSegment, GPUNoSilhouette>(device, cpuSceneData);
        }

        if (allocateNodeData) {
            allocateNodeBuffer<2, BvhNode<2>, LineSegment, SilhouettePrimitive<2>,
                               GPUBvhNode, GPULineSegment, GPUNoSilhouette>(device, cpuSceneData);
        }

        if (allocateRefitData) {
            allocateRefitBuffer<2, BvhNode<2>, LineSegment, SilhouettePrimitive<2>>(device, cpuSceneData);
        }
    }
}

template<>
void GPUBvhBuffers::allocate<3>(ComPtr<IDevice>& device, const SceneData<3> *cpuSceneData,
                                bool allocatePrimitiveData, bool allocateSilhouetteData,
                                bool allocateNodeData, bool allocateRefitData)
{
    if (allocateSilhouetteData) {
        if (allocatePrimitiveData) {
            allocateGeometryBuffers<3, SnchNode<3>, Triangle, SilhouetteEdge,
                                    GPUSnchNode, GPUTriangle, GPUEdge>(device, cpuSceneData);
        }

        if (allocateNodeData) {
            allocateNodeBuffer<3, SnchNode<3>, Triangle, SilhouetteEdge,
                               GPUSnchNode, GPUTriangle, GPUEdge>(device, cpuSceneData);
        }

        if (allocateRefitData) {
            allocateRefitBuffer<3, SnchNode<3>, Triangle, SilhouetteEdge>(device, cpuSceneData);
        }

    } else {
        if (allocatePrimitiveData) {
            allocateGeometryBuffers<3, BvhNode<3>, Triangle, SilhouettePrimitive<3>,
                                    GPUBvhNode, GPUTriangle, GPUNoSilhouette>(device, cpuSceneData);
        }

        if (allocateNodeData) {
            allocateNodeBuffer<3, BvhNode<3>, Triangle, SilhouettePrimitive<3>,
                               GPUBvhNode, GPUTriangle, GPUNoSilhouette>(device, cpuSceneData);
        }

        if (allocateRefitData) {
            allocateRefitBuffer<3, BvhNode<3>, Triangle, SilhouettePrimitive<3>>(device, cpuSceneData);
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

class GPUInteractionsBuffer {
public:
    GPUBuffer interactions = {};
    uint32_t nInteractions = 0;

    void allocate(ComPtr<IDevice>& device) {
        std::vector<GPUInteraction> interactionsData(nInteractions);
        Slang::Result createBufferResult = interactions.create<GPUInteraction>(
            device, true, interactionsData.data(), interactionsData.size());
        if (createBufferResult != SLANG_OK) {
            std::cout << "failed to create interactions buffer" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    void read(ComPtr<IDevice>& device, std::vector<GPUInteraction>& interactionsData) const {
        interactionsData.resize(nInteractions);
        Slang::Result readBufferResult = interactions.read<GPUInteraction>(
            device, nInteractions, interactionsData);
        if (readBufferResult != SLANG_OK) {
            std::cout << "failed to read interactions buffer from GPU" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
};

class GPUQueryRayIntersectionBuffers {
public:
    GPUBuffer rays = {};
    bool checkForOcclusion = false;
    GPUInteractionsBuffer interactionsBuffer;

    void allocate(ComPtr<IDevice>& device, std::vector<GPURay>& raysData) {
        Slang::Result createBufferResult = rays.create<GPURay>(
            device, false, raysData.data(), raysData.size());
        if (createBufferResult != SLANG_OK) {
            std::cout << "failed to create rays buffer" << std::endl;
            exit(EXIT_FAILURE);
        }

        interactionsBuffer.nInteractions = (uint32_t)raysData.size();
        interactionsBuffer.allocate(device);
    }

    int setResources(ShaderCursor& cursor) const {
        cursor.getPath("rays").setResource(rays.view);
        cursor.getPath("checkForOcclusion").setData(checkForOcclusion);
        cursor.getPath("interactions").setResource(interactionsBuffer.interactions.view);
        cursor.getPath("nQueries").setData(interactionsBuffer.nInteractions);

        return 5;
    }

    void read(ComPtr<IDevice>& device, std::vector<GPUInteraction>& interactionsData) const {
        interactionsBuffer.read(device, interactionsData);
    }
};

class GPUQuerySphereIntersectionBuffers {
public:
    GPUBuffer boundingSpheres = {};
    GPUBuffer randNums = {};
    GPUInteractionsBuffer interactionsBuffer;

    void allocate(ComPtr<IDevice>& device,
                  std::vector<GPUBoundingSphere>& boundingSpheresData,
                  std::vector<float3>& randNumsData) {
        Slang::Result createBufferResult = boundingSpheres.create<GPUBoundingSphere>(
            device, false, boundingSpheresData.data(), boundingSpheresData.size());
        if (createBufferResult != SLANG_OK) {
            std::cout << "failed to create boundingSpheres buffer" << std::endl;
            exit(EXIT_FAILURE);
        }

        createBufferResult = randNums.create<float3>(
            device, false, randNumsData.data(), randNumsData.size());
        if (createBufferResult != SLANG_OK) {
            std::cout << "failed to create randNums buffer" << std::endl;
            exit(EXIT_FAILURE);
        }

        interactionsBuffer.nInteractions = (uint32_t)boundingSpheresData.size();
        interactionsBuffer.allocate(device);
    }

    int setResources(ShaderCursor& cursor) const {
        cursor.getPath("boundingSpheres").setResource(boundingSpheres.view);
        cursor.getPath("randNums").setResource(randNums.view);
        cursor.getPath("interactions").setResource(interactionsBuffer.interactions.view);
        cursor.getPath("nQueries").setData(interactionsBuffer.nInteractions);

        return 5;
    }

    void read(ComPtr<IDevice>& device, std::vector<GPUInteraction>& interactionsData) const {
        interactionsBuffer.read(device, interactionsData);
    }
};

class GPUQueryClosestPointBuffers {
public:
    GPUBuffer boundingSpheres = {};
    GPUInteractionsBuffer interactionsBuffer;
    float recordNormals = false;

    void allocate(ComPtr<IDevice>& device,
                  std::vector<GPUBoundingSphere>& boundingSpheresData) {
        Slang::Result createBufferResult = boundingSpheres.create<GPUBoundingSphere>(
            device, false, boundingSpheresData.data(), boundingSpheresData.size());
        if (createBufferResult != SLANG_OK) {
            std::cout << "failed to create boundingSpheres buffer" << std::endl;
            exit(EXIT_FAILURE);
        }

        interactionsBuffer.nInteractions = (uint32_t)boundingSpheresData.size();
        interactionsBuffer.allocate(device);
    }

    int setResources(ShaderCursor& cursor) const {
        cursor.getPath("boundingSpheres").setResource(boundingSpheres.view);
        cursor.getPath("interactions").setResource(interactionsBuffer.interactions.view);
        cursor.getPath("recordNormals").setData(recordNormals);
        cursor.getPath("nQueries").setData(interactionsBuffer.nInteractions);

        return 5;
    }

    void read(ComPtr<IDevice>& device, std::vector<GPUInteraction>& interactionsData) const {
        interactionsBuffer.read(device, interactionsData);
    }
};

class GPUQueryClosestSilhouettePointBuffers {
public:
    GPUBuffer boundingSpheres = {};
    GPUBuffer flipNormalOrientation = {};
    float squaredMinRadius = 1e-6f;
    float precision = 1e-3f;
    GPUInteractionsBuffer interactionsBuffer;

    void allocate(ComPtr<IDevice>& device,
                  std::vector<GPUBoundingSphere>& boundingSpheresData,
                  std::vector<uint32_t>& flipNormalOrientationData) {
        Slang::Result createBufferResult = boundingSpheres.create<GPUBoundingSphere>(
            device, false, boundingSpheresData.data(), boundingSpheresData.size());
        if (createBufferResult != SLANG_OK) {
            std::cout << "failed to create boundingSpheres buffer" << std::endl;
            exit(EXIT_FAILURE);
        }

        createBufferResult = flipNormalOrientation.create<uint32_t>(
            device, false, flipNormalOrientationData.data(), flipNormalOrientationData.size());
        if (createBufferResult != SLANG_OK) {
            std::cout << "failed to create flipNormalOrientation buffer" << std::endl;
            exit(EXIT_FAILURE);
        }

        interactionsBuffer.nInteractions = (uint32_t)boundingSpheresData.size();
        interactionsBuffer.allocate(device);
    }

    int setResources(ShaderCursor& cursor) const {
        cursor.getPath("boundingSpheres").setResource(boundingSpheres.view);
        cursor.getPath("flipNormalOrientation").setResource(flipNormalOrientation.view);
        cursor.getPath("squaredMinRadius").setData(squaredMinRadius);
        cursor.getPath("precision").setData(precision);
        cursor.getPath("interactions").setResource(interactionsBuffer.interactions.view);
        cursor.getPath("nQueries").setData(interactionsBuffer.nInteractions);

        return 7;
    }

    void read(ComPtr<IDevice>& device, std::vector<GPUInteraction>& interactionsData) const {
        interactionsBuffer.read(device, interactionsData);
    }
};

} // namespace fcpw