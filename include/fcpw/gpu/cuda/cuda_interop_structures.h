#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// This file contains GPU data structures for CUDA backend
// These mirror the structures in bvh_interop_structures.h but use CUDA's built-in types
// CUDA provides float2, float3, uint2, uint3, etc. natively, so we don't redefine them

namespace fcpw {

#define FCPW_GPU_UINT_MAX 4294967295

// Use CUDA's built-in vector types (float2, float3, uint2, uint3)
// These are defined in cuda_runtime.h via vector_types.h

#ifndef FCPW_GPU_STRUCTURES_DEFINED
#define FCPW_GPU_STRUCTURES_DEFINED

struct GPUBoundingBox {
    __device__ __host__ GPUBoundingBox() {
        pMin = float3{0.0f, 0.0f, 0.0f};
        pMax = float3{0.0f, 0.0f, 0.0f};
    }
    __device__ __host__ GPUBoundingBox(const float3& pMin_, const float3& pMax_): pMin(pMin_), pMax(pMax_) {}

    float3 pMin;
    float3 pMax;
};

struct GPUBoundingSphere {
    __device__ __host__ GPUBoundingSphere() {
        c = float3{0.0f, 0.0f, 0.0f};
        r2 = 0.0f;
    }
    __device__ __host__ GPUBoundingSphere(const float3& c_, float r2_): c(c_), r2(r2_) {}

    float3 c;
    float r2;
};

struct GPUBoundingCone {
    __device__ __host__ GPUBoundingCone() {
        axis = float3{0.0f, 0.0f, 0.0f};
        halfAngle = 3.14159265358979323846f;
        radius = 0.0f;
    }
    __device__ __host__ GPUBoundingCone(const float3& axis_, float halfAngle_, float radius_):
                    axis(axis_), halfAngle(halfAngle_), radius(radius_) {}

    float3 axis;
    float halfAngle;
    float radius;
};

struct GPUBvhNode {
    __device__ __host__ GPUBvhNode() {
        box = GPUBoundingBox();
        nPrimitives = 0;
        offset = 0;
    }
    __device__ __host__ GPUBvhNode(const GPUBoundingBox& box_, uint32_t nPrimitives_, uint32_t offset_):
               box(box_), nPrimitives(nPrimitives_), offset(offset_) {}

    GPUBoundingBox box;
    uint32_t nPrimitives;
    uint32_t offset;
};

struct GPUSnchNode {
    __device__ __host__ GPUSnchNode() {
        box = GPUBoundingBox();
        cone = GPUBoundingCone();
        nPrimitives = 0;
        offset = 0;
        nSilhouettes = 0;
        silhouetteOffset = 0;
    }
    __device__ __host__ GPUSnchNode(const GPUBoundingBox& box_, const GPUBoundingCone& cone_, uint32_t nPrimitives_,
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
    __device__ __host__ GPULineSegment() {
        pa = float3{0.0f, 0.0f, 0.0f};
        pb = float3{0.0f, 0.0f, 0.0f};
        index = FCPW_GPU_UINT_MAX;
    }
    __device__ __host__ GPULineSegment(const float3& pa_, const float3& pb_, uint32_t index_):
                   pa(pa_), pb(pb_), index(index_) {}

    float3 pa;
    float3 pb;
    uint32_t index;
};

struct GPUTriangle {
    __device__ __host__ GPUTriangle() {
        pa = float3{0.0f, 0.0f, 0.0f};
        pb = float3{0.0f, 0.0f, 0.0f};
        pc = float3{0.0f, 0.0f, 0.0f};
        index = FCPW_GPU_UINT_MAX;
    }
    __device__ __host__ GPUTriangle(const float3& pa_, const float3& pb_, const float3& pc_, uint32_t index_):
                pa(pa_), pb(pb_), pc(pc_), index(index_) {}

    float3 pa;
    float3 pb;
    float3 pc;
    uint32_t index;
};

struct GPUVertex {
    __device__ __host__ GPUVertex() {
        p = float3{0.0f, 0.0f, 0.0f};
        n0 = float3{0.0f, 0.0f, 0.0f};
        n1 = float3{0.0f, 0.0f, 0.0f};
        index = FCPW_GPU_UINT_MAX;
        hasOneAdjacentFace = 0;
    }
    __device__ __host__ GPUVertex(const float3& p_, const float3& n0_, const float3& n1_,
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
    __device__ __host__ GPUEdge() {
        pa = float3{0.0f, 0.0f, 0.0f};
        pb = float3{0.0f, 0.0f, 0.0f};
        n0 = float3{0.0f, 0.0f, 0.0f};
        n1 = float3{0.0f, 0.0f, 0.0f};
        index = FCPW_GPU_UINT_MAX;
        hasOneAdjacentFace = 0;
    }
    __device__ __host__ GPUEdge(const float3& pa_, const float3& pb_,
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
    __device__ __host__ GPUNoSilhouette() {
        index = FCPW_GPU_UINT_MAX;
    }

    uint32_t index;
};

struct GPURay {
    __device__ __host__ GPURay() {
        o = float3{0.0f, 0.0f, 0.0f};
        d = float3{0.0f, 0.0f, 1.0f};
        invD = float3{0.0f, 0.0f, 1.0f};
        tMax = 1e30f;
    }
    __device__ __host__ GPURay(const float3& o_, const float3& d_): o(o_), d(d_) {
        invD = float3{1.0f / d.x, 1.0f / d.y, 1.0f / d.z};
        tMax = 1e30f;
    }

    float3 o;
    float3 d;
    float3 invD;
    float tMax;
};

struct GPUInteraction {
    __device__ __host__ GPUInteraction() {
        p = float3{0.0f, 0.0f, 0.0f};
        n = float3{0.0f, 0.0f, 0.0f};
        uv = float2{0.0f, 0.0f};
        d = 0.0f;
        index = FCPW_GPU_UINT_MAX;
    }

    float3 p;
    float3 n;
    float2 uv;
    float d;
    uint32_t index;
};

#endif // FCPW_GPU_STRUCTURES_DEFINED

} // namespace fcpw
