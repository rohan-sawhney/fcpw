#pragma once

// Host-side POD types for the CUDA backend. These types mirror the GPU backend's
// fcpw::float2/float3 (in bvh_interop_structures.h) and avoid a dependency on
// cuda_runtime.h in host code. The "CUDA" prefix prevents name collisions with
// CUDA's built-in float2/float3 types; device code in cuda_bvh_device.cuh uses
// the CUDA built-ins directly, with toFloat3()/fromFloat3() conversion helpers.

#include <cstdint>
#include <cmath>
#include <vector>
#include <utility>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define FCPW_CUDA_UINT_MAX 4294967295
#define FCPW_CUDA_LINE_SEGMENT_BVH 1
#define FCPW_CUDA_TRIANGLE_BVH 2
#define FCPW_CUDA_LINE_SEGMENT_SNCH 3
#define FCPW_CUDA_TRIANGLE_SNCH 4

namespace fcpw {

struct CUDAFloat2 {
    float x, y;
};

struct CUDAFloat3 {
    float x, y, z;
};

struct CUDABoundingBox {
    CUDABoundingBox() {
        pMin = CUDAFloat3{0.0f, 0.0f, 0.0f};
        pMax = CUDAFloat3{0.0f, 0.0f, 0.0f};
    }
    CUDABoundingBox(const CUDAFloat3& pMin_, const CUDAFloat3& pMax_): pMin(pMin_), pMax(pMax_) {}

    CUDAFloat3 pMin; // aabb min position
    CUDAFloat3 pMax; // aabb max position
};

struct CUDABoundingCone {
    CUDABoundingCone() {
        axis = CUDAFloat3{0.0f, 0.0f, 0.0f};
        halfAngle = (float)M_PI;
        radius = 0.0f;
    }
    CUDABoundingCone(const CUDAFloat3& axis_, float halfAngle_, float radius_):
                     axis(axis_), halfAngle(halfAngle_), radius(radius_) {}

    CUDAFloat3 axis;     // cone axis
    float halfAngle;     // cone half angle
    float radius;        // cone radius
};

struct CUDABvhNode {
    CUDABvhNode() {
        box = CUDABoundingBox();
        nPrimitives = 0;
        offset = 0;
    }
    CUDABvhNode(const CUDABoundingBox& box_, uint32_t nPrimitives_, uint32_t offset_):
                box(box_), nPrimitives(nPrimitives_), offset(offset_) {}

    CUDABoundingBox box;
    uint32_t nPrimitives;
    uint32_t offset;
};

struct CUDASnchNode {
    CUDASnchNode() {
        box = CUDABoundingBox();
        cone = CUDABoundingCone();
        nPrimitives = 0;
        offset = 0;
        nSilhouettes = 0;
        silhouetteOffset = 0;
    }
    CUDASnchNode(const CUDABoundingBox& box_, const CUDABoundingCone& cone_, uint32_t nPrimitives_,
                 uint32_t offset_, uint32_t nSilhouettes_, uint32_t silhouetteOffset_):
                 box(box_), cone(cone_), nPrimitives(nPrimitives_), offset(offset_),
                 nSilhouettes(nSilhouettes_), silhouetteOffset(silhouetteOffset_) {}

    CUDABoundingBox box;
    CUDABoundingCone cone;
    uint32_t nPrimitives;
    uint32_t offset;
    uint32_t nSilhouettes;
    uint32_t silhouetteOffset;
};

struct CUDALineSegment {
    CUDALineSegment() {
        pa = CUDAFloat3{0.0f, 0.0f, 0.0f};
        pb = CUDAFloat3{0.0f, 0.0f, 0.0f};
        index = FCPW_CUDA_UINT_MAX;
    }
    CUDALineSegment(const CUDAFloat3& pa_, const CUDAFloat3& pb_, uint32_t index_):
                    pa(pa_), pb(pb_), index(index_) {}

    CUDAFloat3 pa;
    CUDAFloat3 pb;
    uint32_t index;
};

struct CUDATriangle {
    CUDATriangle() {
        pa = CUDAFloat3{0.0f, 0.0f, 0.0f};
        pb = CUDAFloat3{0.0f, 0.0f, 0.0f};
        pc = CUDAFloat3{0.0f, 0.0f, 0.0f};
        index = FCPW_CUDA_UINT_MAX;
    }
    CUDATriangle(const CUDAFloat3& pa_, const CUDAFloat3& pb_, const CUDAFloat3& pc_, uint32_t index_):
                 pa(pa_), pb(pb_), pc(pc_), index(index_) {}

    CUDAFloat3 pa;
    CUDAFloat3 pb;
    CUDAFloat3 pc;
    uint32_t index;
};

struct CUDAVertex {
    CUDAVertex() {
        p = CUDAFloat3{0.0f, 0.0f, 0.0f};
        n0 = CUDAFloat3{0.0f, 0.0f, 0.0f};
        n1 = CUDAFloat3{0.0f, 0.0f, 0.0f};
        index = FCPW_CUDA_UINT_MAX;
        hasOneAdjacentFace = 0;
    }
    CUDAVertex(const CUDAFloat3& p_, const CUDAFloat3& n0_, const CUDAFloat3& n1_,
               uint32_t index_, uint32_t hasOneAdjacentFace_):
               p(p_), n0(n0_), n1(n1_), index(index_),
               hasOneAdjacentFace(hasOneAdjacentFace_) {}

    CUDAFloat3 p;
    CUDAFloat3 n0;
    CUDAFloat3 n1;
    uint32_t index;
    uint32_t hasOneAdjacentFace;
};

struct CUDAEdge {
    CUDAEdge() {
        pa = CUDAFloat3{0.0f, 0.0f, 0.0f};
        pb = CUDAFloat3{0.0f, 0.0f, 0.0f};
        n0 = CUDAFloat3{0.0f, 0.0f, 0.0f};
        n1 = CUDAFloat3{0.0f, 0.0f, 0.0f};
        index = FCPW_CUDA_UINT_MAX;
        hasOneAdjacentFace = 0;
    }
    CUDAEdge(const CUDAFloat3& pa_, const CUDAFloat3& pb_,
             const CUDAFloat3& n0_, const CUDAFloat3& n1_,
             uint32_t index_, uint32_t hasOneAdjacentFace_):
             pa(pa_), pb(pb_), n0(n0_), n1(n1_), index(index_),
             hasOneAdjacentFace(hasOneAdjacentFace_) {}

    CUDAFloat3 pa;
    CUDAFloat3 pb;
    CUDAFloat3 n0;
    CUDAFloat3 n1;
    uint32_t index;
    uint32_t hasOneAdjacentFace;
};

struct CUDANoSilhouette {
    CUDANoSilhouette() {
        index = FCPW_CUDA_UINT_MAX;
    }

    uint32_t index;
};

struct CUDATransform {
    float t[3][4];    // forward transform (row-major 3x4)
    float tInv[3][4]; // inverse transform (row-major 3x4)
};

struct CUDARay {
    CUDARay() {
        o = CUDAFloat3{0.0f, 0.0f, 0.0f};
        d = CUDAFloat3{0.0f, 0.0f, 0.0f};
        dInv = CUDAFloat3{0.0f, 0.0f, 0.0f};
        tMax = 1e30f;
    }
    CUDARay(const CUDAFloat3& o_, const CUDAFloat3& d_, float tMax_=1e30f): o(o_), d(d_), tMax(tMax_) {
        dInv.x = 1.0f/d.x;
        dInv.y = 1.0f/d.y;
        dInv.z = 1.0f/d.z;
    }

    CUDAFloat3 o;    // ray origin
    CUDAFloat3 d;    // ray direction
    CUDAFloat3 dInv; // 1 over ray direction
    float tMax;      // max ray distance
};

struct CUDABoundingSphere {
    CUDABoundingSphere() {
        c = CUDAFloat3{0.0f, 0.0f, 0.0f};
        r2 = 0.0f;
    }
    CUDABoundingSphere(const CUDAFloat3& c_, float r2_): c(c_), r2(r2_) {}

    CUDAFloat3 c; // sphere center
    float r2;     // sphere squared radius
};

struct CUDAInteraction {
    CUDAInteraction() {
        p = CUDAFloat3{0.0f, 0.0f, 0.0f};
        n = CUDAFloat3{0.0f, 0.0f, 0.0f};
        uv = CUDAFloat2{0.0f, 0.0f};
        d = 1e30f;
        index = FCPW_CUDA_UINT_MAX;
    }

    CUDAFloat3 p;    // interaction point
    CUDAFloat3 n;    // normal
    CUDAFloat2 uv;   // uv coordinates
    float d;         // distance
    uint32_t index;  // primitive index
};

struct CUDABvhBuffers {
    void *d_nodes = nullptr;          // device bvh nodes
    void *d_primitives = nullptr;     // device primitives (line segments or triangles)
    void *d_silhouettes = nullptr;    // device silhouettes (vertices or edges)
    uint32_t *d_nodeIndices = nullptr; // device node indices for refit
    int bvhType = 0;                  // FCPW_CUDA_*_BVH or FCPW_CUDA_*_SNCH
    std::vector<std::pair<uint32_t, uint32_t>> updateEntryData; // per-depth (offset, count) for refit
    uint32_t maxUpdateDepth = 0;      // max BVH depth for refit traversal
    size_t nodesSize = 0;
    size_t primitivesSize = 0;
    size_t silhouettesSize = 0;
    size_t nodeIndicesSize = 0;
    void *d_transform = nullptr;  // device CUDATransform (null if no transform)
    bool hasTransform = false;    // whether a transform is applied
};

} // namespace fcpw
