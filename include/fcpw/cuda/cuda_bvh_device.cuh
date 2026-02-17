#pragma once

#include <cuda_runtime.h>
#include <math_constants.h>
#include <fcpw/cuda/cuda_types.h>

namespace fcpw {

// constants
#define FCPW_CUDA_BVH_MAX_DEPTH 64
#define FCPW_CUDA_FLT_MAX 3.402823466e+38f
#define FCPW_CUDA_FLT_EPSILON 1.192092896e-07f
#define FCPW_CUDA_M_PI 3.14159265358979323846f
#define FCPW_CUDA_M_PI_2 1.57079632679489661923f

///////////////////////////////////////////////////////////////////////////////
// Device types and conversion helpers
///////////////////////////////////////////////////////////////////////////////

struct DeviceRay {
    float3 o;
    float3 d;
    float3 dInv;
    float tMax;
};

struct DeviceBoundingSphere {
    float3 c;
    float r2;
};

struct DeviceBoundingBox {
    float3 pMin;
    float3 pMax;
};

struct DeviceBoundingCone {
    float3 axis;
    float halfAngle;
    float radius;
};

struct DeviceInteraction {
    float3 p;
    float3 n;
    float2 uv;
    float d;
    unsigned int index;
};

__device__ __forceinline__ float3 toFloat3(const CUDAFloat3& v) {
    return make_float3(v.x, v.y, v.z);
}

__device__ __forceinline__ CUDAFloat3 fromFloat3(const float3& v) {
    CUDAFloat3 r; r.x = v.x; r.y = v.y; r.z = v.z; return r;
}

__device__ __forceinline__ CUDAFloat2 fromFloat2(const float2& v) {
    CUDAFloat2 r; r.x = v.x; r.y = v.y; return r;
}

///////////////////////////////////////////////////////////////////////////////
// Float3 math helpers
///////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 operator*(float s, const float3& a) {
    return make_float3(s * a.x, s * a.y, s * a.z);
}

__device__ __forceinline__ float3 operator*(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __forceinline__ float3 operator/(const float3& a, float s) {
    float inv = 1.0f / s;
    return make_float3(a.x * inv, a.y * inv, a.z * inv);
}

__device__ __forceinline__ float3 operator-(const float3& a) {
    return make_float3(-a.x, -a.y, -a.z);
}

__device__ __forceinline__ float dot3(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float3 cross3(const float3& a, const float3& b) {
    return make_float3(a.y * b.z - a.z * b.y,
                       a.z * b.x - a.x * b.z,
                       a.x * b.y - a.y * b.x);
}

__device__ __forceinline__ float length3(const float3& a) {
    return sqrtf(dot3(a, a));
}

__device__ __forceinline__ float3 normalize3(const float3& a) {
    float len = length3(a);
    return (len > 0.0f) ? a / len : make_float3(0.0f, 0.0f, 0.0f);
}

__device__ __forceinline__ float3 fmin3(const float3& a, const float3& b) {
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

__device__ __forceinline__ float3 fmax3(const float3& a, const float3& b) {
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

__device__ __forceinline__ float3 fabs3(const float3& a) {
    return make_float3(fabsf(a.x), fabsf(a.y), fabsf(a.z));
}

///////////////////////////////////////////////////////////////////////////////
// Bounding volume operations
///////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ bool bbox_overlap_sphere(const DeviceBoundingBox& box,
                                                    const DeviceBoundingSphere& s,
                                                    float& d2Min, float& d2Max) {
    float3 u = box.pMin - s.c;
    float3 v = s.c - box.pMax;
    float3 a = fmax3(fmax3(u, v), make_float3(0.0f, 0.0f, 0.0f));
    float3 b = fmin3(u, v);
    d2Min = dot3(a, a);
    d2Max = dot3(b, b);
    return d2Min <= s.r2;
}

__device__ __forceinline__ bool bbox_overlap_sphere(const DeviceBoundingBox& box,
                                                    const DeviceBoundingSphere& s,
                                                    float& d2Min) {
    float3 u = box.pMin - s.c;
    float3 v = s.c - box.pMax;
    float3 a = fmax3(fmax3(u, v), make_float3(0.0f, 0.0f, 0.0f));
    d2Min = dot3(a, a);
    return d2Min <= s.r2;
}

__device__ __forceinline__ bool bbox_intersect_ray(const DeviceBoundingBox& box,
                                                   const DeviceRay& r,
                                                   float& tMin, float& tMax) {
    float3 t0 = make_float3((box.pMin.x - r.o.x) * r.dInv.x,
                             (box.pMin.y - r.o.y) * r.dInv.y,
                             (box.pMin.z - r.o.z) * r.dInv.z);
    float3 t1 = make_float3((box.pMax.x - r.o.x) * r.dInv.x,
                             (box.pMax.y - r.o.y) * r.dInv.y,
                             (box.pMax.z - r.o.z) * r.dInv.z);
    float3 tNear = fmin3(t0, t1);
    float3 tFar = fmax3(t0, t1);

    float tNearMax = fmaxf(0.0f, fmaxf(tNear.x, fmaxf(tNear.y, tNear.z)));
    float tFarMin = fminf(r.tMax, fminf(tFar.x, fminf(tFar.y, tFar.z)));
    if (tNearMax > tFarMin) {
        tMin = FCPW_CUDA_FLT_MAX;
        tMax = FCPW_CUDA_FLT_MAX;
        return false;
    }

    tMin = tNearMax;
    tMax = tFarMin;
    return true;
}

__device__ __forceinline__ float3 bbox_centroid(const DeviceBoundingBox& box) {
    return 0.5f * (box.pMin + box.pMax);
}

__device__ __forceinline__ bool cone_is_valid(const DeviceBoundingCone& cone) {
    return cone.halfAngle >= 0.0f;
}

__device__ __forceinline__ bool cudaInRange(float val, float low, float high) {
    return val >= low && val <= high;
}

__device__ __forceinline__ void computeOrthonormalBasis(const float3& n, float3& b1, float3& b2) {
    float sign = n.z >= 0.0f ? 1.0f : -1.0f;
    float a = -1.0f / (sign + n.z);
    float b = n.x * n.y * a;
    b1 = make_float3(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
    b2 = make_float3(b, sign + n.y * n.y * a, -n.y);
}

__device__ __forceinline__ float projectToPlane(const float3& n, const float3& e) {
    float3 b1, b2;
    computeOrthonormalBasis(n, b1, b2);
    float r1 = dot3(e, fabs3(b1));
    float r2 = dot3(e, fabs3(b2));
    return sqrtf(r1 * r1 + r2 * r2);
}

__device__ __forceinline__ bool cone_overlap(const DeviceBoundingCone& cone,
                                             const float3& o,
                                             const DeviceBoundingBox& b,
                                             float distToBox,
                                             float& minAngleRange,
                                             float& maxAngleRange) {
    minAngleRange = 0.0f;
    maxAngleRange = FCPW_CUDA_M_PI_2;

    if (cone.halfAngle >= FCPW_CUDA_M_PI_2 || distToBox < FCPW_CUDA_FLT_EPSILON) {
        return true;
    }

    float3 c = bbox_centroid(b);
    float3 viewConeAxis = c - o;
    float l = length3(viewConeAxis);
    viewConeAxis = viewConeAxis / l;

    float dAxisAngle = acosf(fmaxf(-1.0f, fminf(1.0f, dot3(cone.axis, viewConeAxis))));
    if (cudaInRange(FCPW_CUDA_M_PI_2, dAxisAngle - cone.halfAngle, dAxisAngle + cone.halfAngle)) {
        return true;
    }

    if (l > cone.radius) {
        float viewConeHalfAngle = asinf(cone.radius / l);
        float halfAngleSum = cone.halfAngle + viewConeHalfAngle;
        minAngleRange = dAxisAngle - halfAngleSum;
        maxAngleRange = dAxisAngle + halfAngleSum;
        return halfAngleSum >= FCPW_CUDA_M_PI_2 ? true : cudaInRange(FCPW_CUDA_M_PI_2, minAngleRange, maxAngleRange);
    }

    float3 e = b.pMax - c;
    float d = dot3(e, fabs3(viewConeAxis));
    float s = l - d;
    if (s <= 0.0f) {
        return true;
    }

    d = projectToPlane(viewConeAxis, e);
    float viewConeHalfAngle = atan2f(d, s);
    float halfAngleSum = cone.halfAngle + viewConeHalfAngle;
    minAngleRange = dAxisAngle - halfAngleSum;
    maxAngleRange = dAxisAngle + halfAngleSum;
    return halfAngleSum >= FCPW_CUDA_M_PI_2 ? true : cudaInRange(FCPW_CUDA_M_PI_2, minAngleRange, maxAngleRange);
}

__device__ __forceinline__ DeviceBoundingBox mergeBoundingBoxes(const DeviceBoundingBox& a,
                                                                const DeviceBoundingBox& b) {
    DeviceBoundingBox result;
    result.pMin = fmin3(a.pMin, b.pMin);
    result.pMax = fmax3(a.pMax, b.pMax);
    return result;
}

__device__ __forceinline__ float3 rotateAxis(const float3& u, const float3& v, float theta) {
    float cosTheta = cosf(theta);
    float sinTheta = sinf(theta);
    float3 w = make_float3(length3(cross3(u, v)), length3(cross3(u, v)), length3(cross3(u, v)));
    // Simplified: use Rodrigues' rotation formula
    // rotate u by theta around the axis perpendicular to u and v
    float3 k = normalize3(cross3(u, v));
    float kLen = length3(cross3(u, v));
    if (kLen < FCPW_CUDA_FLT_EPSILON) {
        return u;
    }
    // Rodrigues: v_rot = v*cos(theta) + (k x v)*sin(theta) + k*(k.v)*(1-cos(theta))
    float3 result = u * cosTheta + cross3(k, u) * sinTheta + k * (dot3(k, u) * (1.0f - cosTheta));
    return result;
}

__device__ __forceinline__ DeviceBoundingCone mergeBoundingCones(const DeviceBoundingCone& coneA,
                                                                 const DeviceBoundingCone& coneB,
                                                                 const float3& originA,
                                                                 const float3& originB,
                                                                 const float3& newOrigin) {
    DeviceBoundingCone cone;
    cone.axis = make_float3(0.0f, 0.0f, 0.0f);
    cone.halfAngle = FCPW_CUDA_M_PI;
    cone.radius = 0.0f;

    if (cone_is_valid(coneA) && cone_is_valid(coneB)) {
        float3 axisA = coneA.axis;
        float3 axisB = coneB.axis;
        float halfAngleA = coneA.halfAngle;
        float halfAngleB = coneB.halfAngle;
        float3 dOriginA = newOrigin - originA;
        float3 dOriginB = newOrigin - originB;
        cone.radius = sqrtf(fmaxf(coneA.radius * coneA.radius + dot3(dOriginA, dOriginA),
                                   coneB.radius * coneB.radius + dot3(dOriginB, dOriginB)));

        if (halfAngleB > halfAngleA) {
            float3 tmpAxis = axisA; axisA = axisB; axisB = tmpAxis;
            float tmpHalfAngle = halfAngleA; halfAngleA = halfAngleB; halfAngleB = tmpHalfAngle;
        }

        float theta = acosf(fmaxf(-1.0f, fminf(1.0f, dot3(axisA, axisB))));
        if (fminf(theta + halfAngleB, FCPW_CUDA_M_PI) <= halfAngleA) {
            cone.axis = axisA;
            cone.halfAngle = halfAngleA;
            return cone;
        }

        float oTheta = (halfAngleA + theta + halfAngleB) / 2.0f;
        if (oTheta >= FCPW_CUDA_M_PI) {
            cone.axis = axisA;
            return cone;
        }

        float rTheta = oTheta - halfAngleA;
        cone.axis = rotateAxis(axisA, axisB, rTheta);
        cone.halfAngle = oTheta;
    } else if (cone_is_valid(coneA)) {
        cone = coneA;
    } else if (cone_is_valid(coneB)) {
        cone = coneB;
    } else {
        cone.halfAngle = -FCPW_CUDA_M_PI;
    }

    return cone;
}

///////////////////////////////////////////////////////////////////////////////
// Node traits
///////////////////////////////////////////////////////////////////////////////

template<typename N>
struct NodeTraits {};

template<>
struct NodeTraits<CUDABvhNode> {
    __device__ static DeviceBoundingBox getBoundingBox(const CUDABvhNode& node) {
        DeviceBoundingBox box;
        box.pMin = toFloat3(node.box.pMin);
        box.pMax = toFloat3(node.box.pMax);
        return box;
    }
    __device__ static bool isLeaf(const CUDABvhNode& node) {
        return node.nPrimitives > 0;
    }
    __device__ static unsigned int getRightChildOffset(const CUDABvhNode& node) {
        return node.offset;
    }
    __device__ static unsigned int getNumPrimitives(const CUDABvhNode& node) {
        return node.nPrimitives;
    }
    __device__ static unsigned int getPrimitiveOffset(const CUDABvhNode& node) {
        return node.offset;
    }
    __device__ static bool hasBoundingCone(const CUDABvhNode&) { return false; }
    __device__ static DeviceBoundingCone getBoundingCone(const CUDABvhNode&) {
        DeviceBoundingCone cone;
        cone.axis = make_float3(0.0f, 0.0f, 0.0f);
        cone.halfAngle = FCPW_CUDA_M_PI;
        cone.radius = 0.0f;
        return cone;
    }
    __device__ static unsigned int getNumSilhouettes(const CUDABvhNode&) { return 0; }
    __device__ static unsigned int getSilhouetteOffset(const CUDABvhNode&) { return 0; }
    __device__ static void setBoundingBox(CUDABvhNode& node, const DeviceBoundingBox& box) {
        node.box.pMin = fromFloat3(box.pMin);
        node.box.pMax = fromFloat3(box.pMax);
    }
    __device__ static void setBoundingCone(CUDABvhNode&, const DeviceBoundingCone&) {}
};

template<>
struct NodeTraits<CUDASnchNode> {
    __device__ static DeviceBoundingBox getBoundingBox(const CUDASnchNode& node) {
        DeviceBoundingBox box;
        box.pMin = toFloat3(node.box.pMin);
        box.pMax = toFloat3(node.box.pMax);
        return box;
    }
    __device__ static bool isLeaf(const CUDASnchNode& node) {
        return node.nPrimitives > 0;
    }
    __device__ static unsigned int getRightChildOffset(const CUDASnchNode& node) {
        return node.offset;
    }
    __device__ static unsigned int getNumPrimitives(const CUDASnchNode& node) {
        return node.nPrimitives;
    }
    __device__ static unsigned int getPrimitiveOffset(const CUDASnchNode& node) {
        return node.offset;
    }
    __device__ static bool hasBoundingCone(const CUDASnchNode&) { return true; }
    __device__ static DeviceBoundingCone getBoundingCone(const CUDASnchNode& node) {
        DeviceBoundingCone cone;
        cone.axis = toFloat3(node.cone.axis);
        cone.halfAngle = node.cone.halfAngle;
        cone.radius = node.cone.radius;
        return cone;
    }
    __device__ static unsigned int getNumSilhouettes(const CUDASnchNode& node) {
        return node.nSilhouettes;
    }
    __device__ static unsigned int getSilhouetteOffset(const CUDASnchNode& node) {
        return node.silhouetteOffset;
    }
    __device__ static void setBoundingBox(CUDASnchNode& node, const DeviceBoundingBox& box) {
        node.box.pMin = fromFloat3(box.pMin);
        node.box.pMax = fromFloat3(box.pMax);
    }
    __device__ static void setBoundingCone(CUDASnchNode& node, const DeviceBoundingCone& cone) {
        node.cone.axis = fromFloat3(cone.axis);
        node.cone.halfAngle = cone.halfAngle;
        node.cone.radius = cone.radius;
    }
};

///////////////////////////////////////////////////////////////////////////////
// Primitive operations
///////////////////////////////////////////////////////////////////////////////

// Free geometry functions
__device__ __forceinline__ bool intersectLineSegmentFunc(
    const float3& pa, const float3& pb,
    const float3& ro, const float3& rd, float rtMax, bool checkForOcclusion,
    float3& p, float3& n, float2& uv, float& d)
{
    float3 u = pa - ro;
    float3 v = pb - pa;

    float dv = cross3(rd, v).z;
    if (fabsf(dv) <= FCPW_CUDA_FLT_EPSILON) return false;

    float ud = cross3(u, rd).z;
    float s = ud / dv;

    if (s >= 0.0f && s <= 1.0f) {
        float t = cross3(u, v).z / dv;
        if (t >= 0.0f && t <= rtMax) {
            if (checkForOcclusion) return true;
            p = pa + s * v;
            float vlen = length3(v);
            if (vlen > 0.0f) {
                n = normalize3(make_float3(v.y, -v.x, 0.0f));
            }
            uv = make_float2(s, 0.0f);
            d = t;
            return true;
        }
    }
    return false;
}

__device__ __forceinline__ float findClosestPointLineSegmentFunc(
    const float3& pa, const float3& pb, const float3& x,
    float3& p, float& t)
{
    float3 u = pb - pa;
    float3 v = x - pa;

    float c1 = dot3(u, v);
    if (c1 <= 0.0f) {
        t = 0.0f;
        p = pa;
        return length3(x - p);
    }

    float c2 = dot3(u, u);
    if (c2 <= c1) {
        t = 1.0f;
        p = pb;
        return length3(x - p);
    }

    t = c1 / c2;
    p = pa + u * t;
    return length3(x - p);
}

__device__ __forceinline__ bool intersectTriangleFunc(
    const float3& pa, const float3& pb, const float3& pc,
    const float3& ro, const float3& rd, float rtMax, bool checkForOcclusion,
    float3& p, float3& n, float2& uv, float& d)
{
    float3 v1 = pb - pa;
    float3 v2 = pc - pa;
    float3 q = cross3(rd, v2);
    float det = dot3(v1, q);

    if (fabsf(det) <= FCPW_CUDA_FLT_EPSILON) return false;
    float invDet = 1.0f / det;

    float3 r = ro - pa;
    float v = dot3(r, q) * invDet;
    if (v < 0.0f || v > 1.0f) return false;

    float3 s = cross3(r, v1);
    float w = dot3(rd, s) * invDet;
    if (w < 0.0f || v + w > 1.0f) return false;

    float t = dot3(v2, s) * invDet;
    if (t >= 0.0f && t <= rtMax) {
        if (checkForOcclusion) return true;
        p = pa + v1 * v + v2 * w;
        n = normalize3(cross3(v1, v2));
        uv = make_float2(1.0f - v - w, v);
        d = t;
        return true;
    }
    return false;
}

__device__ __forceinline__ float findClosestPointTriangleFunc(
    const float3& pa, const float3& pb, const float3& pc,
    const float3& x, float3& p, float2& t)
{
    float3 ab = pb - pa;
    float3 ac = pc - pa;
    float3 ax = x - pa;
    float d1 = dot3(ab, ax);
    float d2 = dot3(ac, ax);
    if (d1 <= 0.0f && d2 <= 0.0f) {
        t = make_float2(1.0f, 0.0f);
        p = pa;
        return length3(x - p);
    }

    float3 bx = x - pb;
    float d3 = dot3(ab, bx);
    float d4 = dot3(ac, bx);
    if (d3 >= 0.0f && d4 <= d3) {
        t = make_float2(0.0f, 1.0f);
        p = pb;
        return length3(x - p);
    }

    float3 cx = x - pc;
    float d5 = dot3(ab, cx);
    float d6 = dot3(ac, cx);
    if (d6 >= 0.0f && d5 <= d6) {
        t = make_float2(0.0f, 0.0f);
        p = pc;
        return length3(x - p);
    }

    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        float v = d1 / (d1 - d3);
        t = make_float2(1.0f - v, v);
        p = pa + ab * v;
        return length3(x - p);
    }

    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        float w = d2 / (d2 - d6);
        t = make_float2(1.0f - w, 0.0f);
        p = pa + ac * w;
        return length3(x - p);
    }

    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        t = make_float2(0.0f, 1.0f - w);
        p = pb + (pc - pb) * w;
        return length3(x - p);
    }

    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom;
    float w = vc * denom;
    t = make_float2(1.0f - v - w, v);
    p = pa + ab * v + ac * w;
    return length3(x - p);
}

// PrimitiveOps traits
template<typename P>
struct PrimitiveOps {};

template<>
struct PrimitiveOps<CUDALineSegment> {
    __device__ static DeviceBoundingBox getBoundingBox(const CUDALineSegment& ls) {
        float3 pa = toFloat3(ls.pa);
        float3 pb = toFloat3(ls.pb);
        float3 eps = make_float3(FCPW_CUDA_FLT_EPSILON, FCPW_CUDA_FLT_EPSILON, 0.0f);
        DeviceBoundingBox box;
        box.pMin = fmin3(pa, pb) - eps;
        box.pMax = fmax3(pa, pb) + eps;
        return box;
    }

    __device__ static float getSurfaceArea(const CUDALineSegment& ls) {
        return length3(toFloat3(ls.pb) - toFloat3(ls.pa));
    }

    __device__ static float3 getNormal(const CUDALineSegment& ls) {
        float3 s = toFloat3(ls.pb) - toFloat3(ls.pa);
        return normalize3(make_float3(s.y, -s.x, 0.0f));
    }

    __device__ static unsigned int getIndex(const CUDALineSegment& ls) {
        return ls.index;
    }

    __device__ static bool intersectRay(const CUDALineSegment& ls, DeviceRay& r,
                                        bool checkForOcclusion, DeviceInteraction& i) {
        float3 pa = toFloat3(ls.pa);
        float3 pb = toFloat3(ls.pb);
        bool hit = intersectLineSegmentFunc(pa, pb, r.o, r.d, r.tMax, checkForOcclusion,
                                            i.p, i.n, i.uv, i.d);
        if (hit) {
            i.index = ls.index;
            return true;
        }
        return false;
    }

    __device__ static bool intersectSphere(const CUDALineSegment& ls, const DeviceBoundingSphere& s,
                                           DeviceInteraction& i) {
        float3 pa = toFloat3(ls.pa);
        float3 pb = toFloat3(ls.pb);
        float t;
        float d = findClosestPointLineSegmentFunc(pa, pb, s.c, i.p, t);
        if (d * d <= s.r2) {
            i.uv.x = t;
            i.d = getSurfaceArea(ls);
            i.index = ls.index;
            return true;
        }
        return false;
    }

    __device__ static bool findClosestPoint(const CUDALineSegment& ls, const DeviceBoundingSphere& s,
                                            DeviceInteraction& i) {
        float3 pa = toFloat3(ls.pa);
        float3 pb = toFloat3(ls.pb);
        float t;
        float d = findClosestPointLineSegmentFunc(pa, pb, s.c, i.p, t);
        if (d * d <= s.r2) {
            i.uv = make_float2(t, 0.0f);
            i.d = d;
            i.index = ls.index;
            return true;
        }
        return false;
    }

    __device__ static float samplePoint(const CUDALineSegment& ls, const float2& randNums,
                                        float2& uv, float3& p, float3& n) {
        float3 pa = toFloat3(ls.pa);
        float3 pb = toFloat3(ls.pb);
        float3 s = pb - pa;
        float area = length3(s);
        float u = randNums.x;
        uv = make_float2(u, 0.0f);
        p = pa + u * s;
        n = make_float3(s.y, -s.x, 0.0f) / area;
        return 1.0f / area;
    }
};

template<>
struct PrimitiveOps<CUDATriangle> {
    __device__ static DeviceBoundingBox getBoundingBox(const CUDATriangle& tri) {
        float3 pa = toFloat3(tri.pa);
        float3 pb = toFloat3(tri.pb);
        float3 pc = toFloat3(tri.pc);
        float3 eps = make_float3(FCPW_CUDA_FLT_EPSILON, FCPW_CUDA_FLT_EPSILON, FCPW_CUDA_FLT_EPSILON);
        DeviceBoundingBox box;
        box.pMin = fmin3(fmin3(pa, pb), pc) - eps;
        box.pMax = fmax3(fmax3(pa, pb), pc) + eps;
        return box;
    }

    __device__ static float getSurfaceArea(const CUDATriangle& tri) {
        float3 pa = toFloat3(tri.pa);
        float3 pb = toFloat3(tri.pb);
        float3 pc = toFloat3(tri.pc);
        return 0.5f * length3(cross3(pb - pa, pc - pa));
    }

    __device__ static float3 getNormal(const CUDATriangle& tri) {
        float3 pa = toFloat3(tri.pa);
        float3 pb = toFloat3(tri.pb);
        float3 pc = toFloat3(tri.pc);
        return normalize3(cross3(pb - pa, pc - pa));
    }

    __device__ static unsigned int getIndex(const CUDATriangle& tri) {
        return tri.index;
    }

    __device__ static bool intersectRay(const CUDATriangle& tri, DeviceRay& r,
                                        bool checkForOcclusion, DeviceInteraction& i) {
        float3 pa = toFloat3(tri.pa);
        float3 pb = toFloat3(tri.pb);
        float3 pc = toFloat3(tri.pc);
        bool hit = intersectTriangleFunc(pa, pb, pc, r.o, r.d, r.tMax, checkForOcclusion,
                                         i.p, i.n, i.uv, i.d);
        if (hit) {
            i.index = tri.index;
            return true;
        }
        return false;
    }

    __device__ static bool intersectSphere(const CUDATriangle& tri, const DeviceBoundingSphere& s,
                                           DeviceInteraction& i) {
        float3 pa = toFloat3(tri.pa);
        float3 pb = toFloat3(tri.pb);
        float3 pc = toFloat3(tri.pc);
        float d = findClosestPointTriangleFunc(pa, pb, pc, s.c, i.p, i.uv);
        if (d * d <= s.r2) {
            i.d = getSurfaceArea(tri);
            i.index = tri.index;
            return true;
        }
        return false;
    }

    __device__ static bool findClosestPoint(const CUDATriangle& tri, const DeviceBoundingSphere& s,
                                            DeviceInteraction& i) {
        float3 pa = toFloat3(tri.pa);
        float3 pb = toFloat3(tri.pb);
        float3 pc = toFloat3(tri.pc);
        float d = findClosestPointTriangleFunc(pa, pb, pc, s.c, i.p, i.uv);
        if (d * d <= s.r2) {
            i.d = d;
            i.index = tri.index;
            return true;
        }
        return false;
    }

    __device__ static float samplePoint(const CUDATriangle& tri, const float2& randNums,
                                        float2& uv, float3& p, float3& n) {
        float3 pa = toFloat3(tri.pa);
        float3 pb = toFloat3(tri.pb);
        float3 pc = toFloat3(tri.pc);
        n = cross3(pb - pa, pc - pa);
        float area = length3(n);
        float u1 = sqrtf(randNums.x);
        float u2 = randNums.y;
        float u = 1.0f - u1;
        float v = u2 * u1;
        float w = 1.0f - u - v;
        uv = make_float2(u, v);
        p = pa * u + pb * v + pc * w;
        n = n / area;
        return 2.0f / area;
    }
};

///////////////////////////////////////////////////////////////////////////////
// Silhouette operations
///////////////////////////////////////////////////////////////////////////////

template<typename S>
struct SilhouetteOps {};

template<>
struct SilhouetteOps<CUDANoSilhouette> {
    __device__ static float3 getCentroid(const CUDANoSilhouette&) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    __device__ static bool hasTwoAdjacentFaces(const CUDANoSilhouette&) { return false; }
    __device__ static float3 getNormal(const CUDANoSilhouette&, unsigned int) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    __device__ static unsigned int getIndex(const CUDANoSilhouette&) { return FCPW_CUDA_UINT_MAX; }
    __device__ static bool findClosestSilhouettePoint(const CUDANoSilhouette&,
                                                      const DeviceBoundingSphere&,
                                                      bool, float, float,
                                                      DeviceInteraction&) {
        return false;
    }
};

__device__ __forceinline__ bool isSilhouetteVertexFunc(
    const float3& n0, const float3& n1, const float3& viewDir,
    float d, bool flipNormalOrientation, float precision)
{
    float sign = flipNormalOrientation ? 1.0f : -1.0f;

    if (d <= precision) {
        float det = n0.x * n1.y - n1.x * n0.y;
        return sign * det > precision;
    }

    float3 viewDirUnit = viewDir / d;
    float dot0 = dot3(viewDirUnit, n0);
    float dot1 = dot3(viewDirUnit, n1);

    bool isZeroDot0 = fabsf(dot0) <= precision;
    if (isZeroDot0) return sign * dot1 > precision;

    bool isZeroDot1 = fabsf(dot1) <= precision;
    if (isZeroDot1) return sign * dot0 > precision;

    return dot0 * dot1 < 0.0f;
}

template<>
struct SilhouetteOps<CUDAVertex> {
    __device__ static float3 getCentroid(const CUDAVertex& v) {
        return toFloat3(v.p);
    }
    __device__ static bool hasTwoAdjacentFaces(const CUDAVertex& v) {
        return v.hasOneAdjacentFace == 0;
    }
    __device__ static float3 getNormal(const CUDAVertex& v, unsigned int fIndex) {
        return fIndex == 0 ? toFloat3(v.n0) : toFloat3(v.n1);
    }
    __device__ static unsigned int getIndex(const CUDAVertex& v) { return v.index; }

    __device__ static bool findClosestSilhouettePoint(const CUDAVertex& vert,
                                                      const DeviceBoundingSphere& s,
                                                      bool flipNormalOrientation,
                                                      float squaredMinRadius, float precision,
                                                      DeviceInteraction& i) {
        if (squaredMinRadius >= s.r2) return false;

        float3 p = toFloat3(vert.p);
        float3 viewDir = s.c - p;
        float d = length3(viewDir);
        if (d * d > s.r2) return false;

        bool process = vert.hasOneAdjacentFace == 1;
        if (!process) {
            process = isSilhouetteVertexFunc(toFloat3(vert.n0), toFloat3(vert.n1),
                                             viewDir, d, flipNormalOrientation, precision);
        }

        if (process && d * d <= s.r2) {
            i.p = p;
            i.uv = make_float2(0.0f, 0.0f);
            i.d = d;
            i.index = vert.index;
            return true;
        }
        return false;
    }
};

__device__ __forceinline__ bool isSilhouetteEdgeFunc(
    const float3& pa, const float3& pb, const float3& n0, const float3& n1,
    const float3& viewDir, float d, bool flipNormalOrientation, float precision)
{
    float sign = flipNormalOrientation ? 1.0f : -1.0f;

    if (d <= precision) {
        float3 edgeDir = normalize3(pb - pa);
        float signedDihedralAngle = atan2f(dot3(edgeDir, cross3(n0, n1)), dot3(n0, n1));
        return sign * signedDihedralAngle > precision;
    }

    float3 viewDirUnit = viewDir / d;
    float dot0 = dot3(viewDirUnit, n0);
    float dot1 = dot3(viewDirUnit, n1);

    bool isZeroDot0 = fabsf(dot0) <= precision;
    if (isZeroDot0) return sign * dot1 > precision;

    bool isZeroDot1 = fabsf(dot1) <= precision;
    if (isZeroDot1) return sign * dot0 > precision;

    return dot0 * dot1 < 0.0f;
}

template<>
struct SilhouetteOps<CUDAEdge> {
    __device__ static float3 getCentroid(const CUDAEdge& e) {
        return 0.5f * (toFloat3(e.pa) + toFloat3(e.pb));
    }
    __device__ static bool hasTwoAdjacentFaces(const CUDAEdge& e) {
        return e.hasOneAdjacentFace == 0;
    }
    __device__ static float3 getNormal(const CUDAEdge& e, unsigned int fIndex) {
        return fIndex == 0 ? toFloat3(e.n0) : toFloat3(e.n1);
    }
    __device__ static unsigned int getIndex(const CUDAEdge& e) { return e.index; }

    __device__ static bool findClosestSilhouettePoint(const CUDAEdge& edge,
                                                      const DeviceBoundingSphere& s,
                                                      bool flipNormalOrientation,
                                                      float squaredMinRadius, float precision,
                                                      DeviceInteraction& i) {
        if (squaredMinRadius >= s.r2) return false;

        float3 pa = toFloat3(edge.pa);
        float3 pb = toFloat3(edge.pb);
        float t;
        float d = findClosestPointLineSegmentFunc(pa, pb, s.c, i.p, t);
        if (d * d > s.r2) return false;

        bool process = edge.hasOneAdjacentFace == 1;
        if (!process) {
            float3 viewDir = s.c - i.p;
            process = isSilhouetteEdgeFunc(pa, pb, toFloat3(edge.n0), toFloat3(edge.n1),
                                           viewDir, d, flipNormalOrientation, precision);
        }

        if (process && d * d <= s.r2) {
            i.uv = make_float2(t, 0.0f);
            i.d = d;
            i.index = edge.index;
            return true;
        }
        return false;
    }
};

///////////////////////////////////////////////////////////////////////////////
// BVH traversal functions
///////////////////////////////////////////////////////////////////////////////

template<typename N, typename P, typename S>
__device__ bool bvhIntersectRay(const N* nodes, const P* primitives, const S* silhouettes,
                                DeviceRay& r, bool checkForOcclusion, DeviceInteraction& i)
{
    struct TraversalEntry { unsigned int node; float distance; };
    TraversalEntry traversalStack[FCPW_CUDA_BVH_MAX_DEPTH];
    float4 distToChildNodes = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    DeviceBoundingBox rootBox = NodeTraits<N>::getBoundingBox(nodes[0]);
    bool didIntersect = false;

    float tMin, tMax;
    if (bbox_intersect_ray(rootBox, r, tMin, tMax)) {
        traversalStack[0].node = 0;
        traversalStack[0].distance = tMin;
        int stackPtr = 0;

        while (stackPtr >= 0) {
            // pop off the next node to work on
            unsigned int currentNodeIndex = traversalStack[stackPtr].node;
            float currentDist = traversalStack[stackPtr].distance;
            stackPtr--;

            // if this node is further than the closest found intersection, continue
            if (currentDist > r.tMax) continue;

            N node = nodes[currentNodeIndex];
            if (NodeTraits<N>::isLeaf(node)) {
                // intersect primitives in leaf node
                unsigned int nPrimitives = NodeTraits<N>::getNumPrimitives(node);
                for (unsigned int p = 0; p < nPrimitives; p++) {
                    DeviceInteraction c;
                    c.p = make_float3(0.0f, 0.0f, 0.0f);
                    c.n = make_float3(0.0f, 0.0f, 0.0f);
                    c.uv = make_float2(0.0f, 0.0f);
                    c.d = FCPW_CUDA_FLT_MAX;
                    c.index = FCPW_CUDA_UINT_MAX;
                    unsigned int primitiveIndex = NodeTraits<N>::getPrimitiveOffset(node) + p;
                    bool didIntersectPrimitive = PrimitiveOps<P>::intersectRay(primitives[primitiveIndex], r, checkForOcclusion, c);

                    if (didIntersectPrimitive) {
                        if (checkForOcclusion) {
                            i.index = c.index;
                            return true;
                        }
                        didIntersect = true;
                        r.tMax = fminf(r.tMax, c.d);
                        i = c;
                    }
                }
            } else {
                // intersect child nodes
                unsigned int leftNodeIndex = currentNodeIndex + 1;
                DeviceBoundingBox leftBox = NodeTraits<N>::getBoundingBox(nodes[leftNodeIndex]);
                float leftTMin, leftTMax;
                bool didIntersectLeft = bbox_intersect_ray(leftBox, r, leftTMin, leftTMax);

                unsigned int rightNodeIndex = currentNodeIndex + NodeTraits<N>::getRightChildOffset(node);
                DeviceBoundingBox rightBox = NodeTraits<N>::getBoundingBox(nodes[rightNodeIndex]);
                float rightTMin, rightTMax;
                bool didIntersectRight = bbox_intersect_ray(rightBox, r, rightTMin, rightTMax);

                if (didIntersectLeft && didIntersectRight) {
                    // assume left child is closer; swap if right is actually closer
                    unsigned int closer = leftNodeIndex;
                    unsigned int other = rightNodeIndex;
                    float closerDist = leftTMin;
                    float otherDist = rightTMin;

                    if (rightTMin < leftTMin) {
                        closer = rightNodeIndex;
                        other = leftNodeIndex;
                        closerDist = rightTMin;
                        otherDist = leftTMin;
                    }

                    stackPtr++;
                    traversalStack[stackPtr].node = other;
                    traversalStack[stackPtr].distance = otherDist;
                    stackPtr++;
                    traversalStack[stackPtr].node = closer;
                    traversalStack[stackPtr].distance = closerDist;
                } else if (didIntersectLeft) {
                    stackPtr++;
                    traversalStack[stackPtr].node = leftNodeIndex;
                    traversalStack[stackPtr].distance = leftTMin;
                } else if (didIntersectRight) {
                    stackPtr++;
                    traversalStack[stackPtr].node = rightNodeIndex;
                    traversalStack[stackPtr].distance = rightTMin;
                }
            }
        }
    }

    return didIntersect;
}

template<typename N, typename P, typename S>
__device__ bool bvhIntersectSphere(const N* nodes, const P* primitives, const S* silhouettes,
                                   DeviceBoundingSphere s, float3 randNums, DeviceInteraction& i)
{
    float4 distToChildNodes = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    DeviceBoundingBox rootBox = NodeTraits<N>::getBoundingBox(nodes[0]);
    unsigned int currentNodeIndex = 0;
    unsigned int selectedPrimitiveIndex = FCPW_CUDA_UINT_MAX;
    bool didIntersect = false;

    float d2Min, d2Max;
    if (bbox_overlap_sphere(rootBox, s, d2Min, d2Max)) {
        float maxDistToChildNode = d2Max;
        float traversalPdf = 1.0f;
        float u = randNums.x;
        int stackPtr = 0;

        while (stackPtr >= 0) {
            stackPtr--;

            N node = nodes[currentNodeIndex];
            if (NodeTraits<N>::isLeaf(node)) {
                float totalPrimitiveWeight = 0.0f;
                unsigned int nPrimitives = NodeTraits<N>::getNumPrimitives(node);
                for (unsigned int p = 0; p < nPrimitives; p++) {
                    DeviceInteraction c;
                    c.p = make_float3(0.0f, 0.0f, 0.0f);
                    c.n = make_float3(0.0f, 0.0f, 0.0f);
                    c.uv = make_float2(0.0f, 0.0f);
                    c.d = FCPW_CUDA_FLT_MAX;
                    c.index = FCPW_CUDA_UINT_MAX;
                    bool didIntersectPrimitive = false;
                    unsigned int primitiveIndex = NodeTraits<N>::getPrimitiveOffset(node) + p;
                    P primitive = primitives[primitiveIndex];

                    if (maxDistToChildNode <= s.r2) {
                        didIntersectPrimitive = true;
                        c.d = PrimitiveOps<P>::getSurfaceArea(primitive);
                        c.index = PrimitiveOps<P>::getIndex(primitive);
                    } else {
                        didIntersectPrimitive = PrimitiveOps<P>::intersectSphere(primitive, s, c);
                    }

                    if (didIntersectPrimitive) {
                        didIntersect = true;
                        totalPrimitiveWeight += c.d;
                        float selectionProb = c.d / totalPrimitiveWeight;

                        if (u < selectionProb) {
                            u = u / selectionProb;
                            i = c;
                            i.d *= traversalPdf;
                            selectedPrimitiveIndex = primitiveIndex;
                        } else {
                            u = (u - selectionProb) / (1.0f - selectionProb);
                        }
                    }
                }

                if (totalPrimitiveWeight > 0.0f) {
                    i.d /= totalPrimitiveWeight;
                }
            } else {
                unsigned int leftNodeIndex = currentNodeIndex + 1;
                DeviceBoundingBox leftBox = NodeTraits<N>::getBoundingBox(nodes[leftNodeIndex]);
                float leftD2Min, leftD2Max;
                bool overlapsLeft = bbox_overlap_sphere(leftBox, s, leftD2Min, leftD2Max);
                float weightLeft = overlapsLeft ? 1.0f : 0.0f;
                if (weightLeft > 0.0f) {
                    float3 uv = s.c - bbox_centroid(leftBox);
                    weightLeft *= 1.0f; // constant branch traversal weight
                }

                unsigned int rightNodeIndex = currentNodeIndex + NodeTraits<N>::getRightChildOffset(node);
                DeviceBoundingBox rightBox = NodeTraits<N>::getBoundingBox(nodes[rightNodeIndex]);
                float rightD2Min, rightD2Max;
                bool overlapsRight = bbox_overlap_sphere(rightBox, s, rightD2Min, rightD2Max);
                float weightRight = overlapsRight ? 1.0f : 0.0f;
                if (weightRight > 0.0f) {
                    float3 uv = s.c - bbox_centroid(rightBox);
                    weightRight *= 1.0f; // constant branch traversal weight
                }

                float totalTraversalWeight = weightLeft + weightRight;
                if (totalTraversalWeight > 0.0f) {
                    stackPtr++;
                    float traversalProbLeft = weightLeft / totalTraversalWeight;
                    float traversalProbRight = 1.0f - traversalProbLeft;

                    if (u < traversalProbLeft) {
                        u = u / traversalProbLeft;
                        currentNodeIndex = leftNodeIndex;
                        traversalPdf *= traversalProbLeft;
                        maxDistToChildNode = leftD2Max;
                    } else {
                        u = (u - traversalProbLeft) / traversalProbRight;
                        currentNodeIndex = rightNodeIndex;
                        traversalPdf *= traversalProbRight;
                        maxDistToChildNode = rightD2Max;
                    }
                }
            }
        }
    }

    if (didIntersect) {
        if (i.index == FCPW_CUDA_UINT_MAX || selectedPrimitiveIndex == FCPW_CUDA_UINT_MAX) {
            didIntersect = false;
        } else {
            float2 uv;
            float3 p, n;
            float2 sampleRandNums = make_float2(randNums.y, randNums.z);
            float samplingPdf = PrimitiveOps<P>::samplePoint(primitives[selectedPrimitiveIndex],
                                                              sampleRandNums, uv, p, n);
            i.uv = uv;
            i.p = p;
            i.n = n;
            i.d *= samplingPdf;
        }
    }

    return didIntersect;
}

template<typename N, typename P, typename S>
__device__ bool bvhFindClosestPoint(const N* nodes, const P* primitives, const S* silhouettes,
                                    DeviceBoundingSphere& s, DeviceInteraction& i,
                                    bool recordNormal)
{
    unsigned int traversalStack[FCPW_CUDA_BVH_MAX_DEPTH];
    DeviceBoundingBox rootBox = NodeTraits<N>::getBoundingBox(nodes[0]);
    unsigned int selectedPrimitiveIndex = FCPW_CUDA_UINT_MAX;
    bool notFound = true;

    float d2Min, d2Max;
    if (bbox_overlap_sphere(rootBox, s, d2Min, d2Max)) {
        s.r2 = fminf(s.r2, d2Max);
        traversalStack[0] = 0;
        int stackPtr = 0;

        while (stackPtr >= 0) {
            // pop off the next node to work on
            unsigned int currentNodeIndex = traversalStack[stackPtr];
            N node = nodes[currentNodeIndex];
            stackPtr--;

            // if this node is further than the closest found point, continue
            DeviceBoundingBox currentBox = NodeTraits<N>::getBoundingBox(node);
            float cd2Min;
            if (!bbox_overlap_sphere(currentBox, s, cd2Min)) continue;

            if (NodeTraits<N>::isLeaf(node)) {
                // find closest point in leaf node
                unsigned int nPrimitives = NodeTraits<N>::getNumPrimitives(node);
                for (unsigned int p = 0; p < nPrimitives; p++) {
                    DeviceInteraction c;
                    c.p = make_float3(0.0f, 0.0f, 0.0f);
                    c.n = make_float3(0.0f, 0.0f, 0.0f);
                    c.uv = make_float2(0.0f, 0.0f);
                    c.d = FCPW_CUDA_FLT_MAX;
                    c.index = FCPW_CUDA_UINT_MAX;
                    unsigned int primitiveIndex = NodeTraits<N>::getPrimitiveOffset(node) + p;
                    bool found = PrimitiveOps<P>::findClosestPoint(primitives[primitiveIndex], s, c);

                    if (found) {
                        notFound = false;
                        s.r2 = fminf(s.r2, c.d * c.d);
                        i = c;
                        selectedPrimitiveIndex = primitiveIndex;
                    }
                }
            } else {
                // intersect child nodes
                unsigned int leftNodeIndex = currentNodeIndex + 1;
                DeviceBoundingBox leftBox = NodeTraits<N>::getBoundingBox(nodes[leftNodeIndex]);
                float leftD2Min, leftD2Max;
                bool overlapsLeft = bbox_overlap_sphere(leftBox, s, leftD2Min, leftD2Max);
                s.r2 = fminf(s.r2, leftD2Max);

                unsigned int rightNodeIndex = currentNodeIndex + NodeTraits<N>::getRightChildOffset(node);
                DeviceBoundingBox rightBox = NodeTraits<N>::getBoundingBox(nodes[rightNodeIndex]);
                float rightD2Min, rightD2Max;
                bool overlapsRight = bbox_overlap_sphere(rightBox, s, rightD2Min, rightD2Max);
                s.r2 = fminf(s.r2, rightD2Max);

                if (overlapsLeft && overlapsRight) {
                    // assume left child is closer; swap if right is actually closer
                    unsigned int closer = leftNodeIndex;
                    unsigned int other = rightNodeIndex;

                    if (rightD2Min < leftD2Min) {
                        closer = rightNodeIndex;
                        other = leftNodeIndex;
                    }

                    stackPtr++;
                    traversalStack[stackPtr] = other;
                    stackPtr++;
                    traversalStack[stackPtr] = closer;
                } else if (overlapsLeft) {
                    stackPtr++;
                    traversalStack[stackPtr] = leftNodeIndex;
                } else if (overlapsRight) {
                    stackPtr++;
                    traversalStack[stackPtr] = rightNodeIndex;
                }
            }
        }
    }

    if (!notFound && recordNormal) {
        i.n = PrimitiveOps<P>::getNormal(primitives[selectedPrimitiveIndex]);
    }

    return !notFound;
}

template<typename N, typename P, typename S>
__device__ bool bvhFindClosestSilhouettePoint(const N* nodes, const P* primitives,
                                               const S* silhouettes,
                                               DeviceBoundingSphere& s,
                                               bool flipNormalOrientation,
                                               float squaredMinRadius, float precision,
                                               DeviceInteraction& i)
{
    if (squaredMinRadius >= s.r2) return false;

    unsigned int traversalStack[FCPW_CUDA_BVH_MAX_DEPTH];
    float2 distToChildNodes = make_float2(0.0f, 0.0f);
    DeviceBoundingBox rootBox = NodeTraits<N>::getBoundingBox(nodes[0]);
    bool notFound = true;

    float d2Min;
    if (bbox_overlap_sphere(rootBox, s, d2Min)) {
        traversalStack[0] = 0;
        int stackPtr = 0;

        while (stackPtr >= 0) {
            // pop off the next node to work on
            unsigned int currentNodeIndex = traversalStack[stackPtr];
            N node = nodes[currentNodeIndex];
            stackPtr--;

            // if this node is further than the closest found silhouette point, continue
            DeviceBoundingBox currentBox = NodeTraits<N>::getBoundingBox(node);
            float cd2Min;
            if (!bbox_overlap_sphere(currentBox, s, cd2Min)) continue;

            if (NodeTraits<N>::isLeaf(node)) {
                // find closest silhouette point in leaf node
                unsigned int nSilhouettes = NodeTraits<N>::getNumSilhouettes(node);
                for (unsigned int p = 0; p < nSilhouettes; p++) {
                    unsigned int silhouetteIndex = NodeTraits<N>::getSilhouetteOffset(node) + p;
                    S silhouette = silhouettes[silhouetteIndex];
                    if (SilhouetteOps<S>::getIndex(silhouette) == i.index) continue;

                    DeviceInteraction c;
                    c.p = make_float3(0.0f, 0.0f, 0.0f);
                    c.n = make_float3(0.0f, 0.0f, 0.0f);
                    c.uv = make_float2(0.0f, 0.0f);
                    c.d = FCPW_CUDA_FLT_MAX;
                    c.index = FCPW_CUDA_UINT_MAX;
                    bool found = SilhouetteOps<S>::findClosestSilhouettePoint(
                        silhouette, s, flipNormalOrientation, squaredMinRadius, precision, c);

                    if (found) {
                        notFound = false;
                        s.r2 = fminf(s.r2, c.d * c.d);
                        i = c;

                        if (squaredMinRadius >= s.r2) break;
                    }
                }
            } else {
                // intersect child nodes with cone pruning
                unsigned int leftNodeIndex = currentNodeIndex + 1;
                N leftNode = nodes[leftNodeIndex];
                DeviceBoundingCone leftCone = NodeTraits<N>::getBoundingCone(leftNode);
                bool overlapsLeft = cone_is_valid(leftCone);
                if (overlapsLeft) {
                    DeviceBoundingBox leftBox = NodeTraits<N>::getBoundingBox(leftNode);
                    float leftD2Min;
                    overlapsLeft = bbox_overlap_sphere(leftBox, s, leftD2Min);
                    if (overlapsLeft) {
                        float minAngleRange, maxAngleRange;
                        overlapsLeft = cone_overlap(leftCone, s.c, leftBox, leftD2Min,
                                                    minAngleRange, maxAngleRange);
                    }
                    distToChildNodes.x = leftD2Min;
                }

                unsigned int rightNodeIndex = currentNodeIndex + NodeTraits<N>::getRightChildOffset(node);
                N rightNode = nodes[rightNodeIndex];
                DeviceBoundingCone rightCone = NodeTraits<N>::getBoundingCone(rightNode);
                bool overlapsRight = cone_is_valid(rightCone);
                if (overlapsRight) {
                    DeviceBoundingBox rightBox = NodeTraits<N>::getBoundingBox(rightNode);
                    float rightD2Min;
                    overlapsRight = bbox_overlap_sphere(rightBox, s, rightD2Min);
                    if (overlapsRight) {
                        float minAngleRange, maxAngleRange;
                        overlapsRight = cone_overlap(rightCone, s.c, rightBox, rightD2Min,
                                                     minAngleRange, maxAngleRange);
                    }
                    distToChildNodes.y = rightD2Min;
                }

                if (overlapsLeft && overlapsRight) {
                    // assume left child is closer; swap if right is actually closer
                    unsigned int closer = leftNodeIndex;
                    unsigned int other = rightNodeIndex;
                    if (distToChildNodes.y < distToChildNodes.x) {
                        closer = rightNodeIndex;
                        other = leftNodeIndex;
                    }
                    stackPtr++;
                    traversalStack[stackPtr] = other;
                    stackPtr++;
                    traversalStack[stackPtr] = closer;
                } else if (overlapsLeft) {
                    stackPtr++;
                    traversalStack[stackPtr] = leftNodeIndex;
                } else if (overlapsRight) {
                    stackPtr++;
                    traversalStack[stackPtr] = rightNodeIndex;
                }
            }
        }
    }

    return !notFound;
}

///////////////////////////////////////////////////////////////////////////////
// Transform utilities
///////////////////////////////////////////////////////////////////////////////

// threshold for treating ray tMax / sphere r2 as "infinite";
// must be above CUDARay's default tMax (1e30) and CUDABoundingSphere's maxFloat
#define FCPW_CUDA_INF_THRESHOLD 1e29f

__device__ __forceinline__ float3 transformPoint(const float m[3][4], const float3& p) {
    return make_float3(m[0][0] * p.x + m[0][1] * p.y + m[0][2] * p.z + m[0][3],
                       m[1][0] * p.x + m[1][1] * p.y + m[1][2] * p.z + m[1][3],
                       m[2][0] * p.x + m[2][1] * p.y + m[2][2] * p.z + m[2][3]);
}

__device__ __forceinline__ DeviceRay transformRay(const float m[3][4], const DeviceRay& r) {
    float3 o = transformPoint(m, r.o);
    bool isFinite = r.tMax < FCPW_CUDA_INF_THRESHOLD;
    float3 endPt = r.o + r.d * (isFinite ? r.tMax : 1.0f);
    float3 d = transformPoint(m, endPt) - o;
    float dNorm = length3(d);

    DeviceRay result;
    result.o = o;
    result.d = d / dNorm;
    result.tMax = isFinite ? dNorm : r.tMax;
    result.dInv = make_float3(1.0f / result.d.x, 1.0f / result.d.y, 1.0f / result.d.z);
    return result;
}

__device__ __forceinline__ DeviceBoundingSphere transformSphere(const float m[3][4],
                                                                const DeviceBoundingSphere& s) {
    float3 c = transformPoint(m, s.c);
    float r2 = s.r2;
    if (s.r2 < FCPW_CUDA_INF_THRESHOLD) {
        float3 offset = make_float3(s.c.x + sqrtf(s.r2), s.c.y, s.c.z);
        float3 d = transformPoint(m, offset) - c;
        r2 = dot3(d, d);
    }

    DeviceBoundingSphere result;
    result.c = c;
    result.r2 = r2;
    return result;
}

__device__ __forceinline__ void transformInteraction(const float t[3][4], const float tInv[3][4],
                                                     const float3& x, bool overwriteDistance,
                                                     DeviceInteraction& i) {
    float3 p = transformPoint(t, i.p);
    // n = normalize(transpose(linear(tInv)) * n)
    float3 n = make_float3(tInv[0][0] * i.n.x + tInv[1][0] * i.n.y + tInv[2][0] * i.n.z,
                           tInv[0][1] * i.n.x + tInv[1][1] * i.n.y + tInv[2][1] * i.n.z,
                           tInv[0][2] * i.n.x + tInv[1][2] * i.n.y + tInv[2][2] * i.n.z);
    n = normalize3(n);

    i.p = p;
    i.n = n;
    if (overwriteDistance) {
        i.d = length3(p - x);
    }
}

///////////////////////////////////////////////////////////////////////////////
// TransformedAggregate wrapper functions
///////////////////////////////////////////////////////////////////////////////

template<typename N, typename P, typename S>
__device__ bool transformedBvhIntersectRay(const N* nodes, const P* primitives,
                                           const S* silhouettes,
                                           const float t[3][4], const float tInv[3][4],
                                           DeviceRay& r, bool checkForOcclusion,
                                           DeviceInteraction& i) {
    // apply inverse transform to ray
    DeviceRay rInv = transformRay(tInv, r);

    // intersect
    bool didIntersect = bvhIntersectRay<N, P, S>(nodes, primitives, silhouettes,
                                                  rInv, checkForOcclusion, i);

    // apply transform to ray and interaction
    r.tMax = transformRay(t, rInv).tMax;
    if (didIntersect) {
        transformInteraction(t, tInv, r.o, true, i);
    }

    return didIntersect;
}

template<typename N, typename P, typename S>
__device__ bool transformedBvhIntersectSphere(const N* nodes, const P* primitives,
                                              const S* silhouettes,
                                              const float t[3][4], const float tInv[3][4],
                                              DeviceBoundingSphere s, const float3& randNums,
                                              DeviceInteraction& i) {
    // apply inverse transform to sphere
    DeviceBoundingSphere sInv = transformSphere(tInv, s);

    // intersect
    bool didIntersect = bvhIntersectSphere<N, P, S>(nodes, primitives, silhouettes,
                                                     sInv, randNums, i);

    // apply transform to interaction
    if (didIntersect) {
        transformInteraction(t, tInv, s.c, false, i);
    }

    return didIntersect;
}

template<typename N, typename P, typename S>
__device__ bool transformedBvhFindClosestPoint(const N* nodes, const P* primitives,
                                               const S* silhouettes,
                                               const float t[3][4], const float tInv[3][4],
                                               DeviceBoundingSphere& s, DeviceInteraction& i,
                                               bool recordNormal) {
    // apply inverse transform to sphere
    DeviceBoundingSphere sInv = transformSphere(tInv, s);

    // find closest point
    bool found = bvhFindClosestPoint<N, P, S>(nodes, primitives, silhouettes,
                                               sInv, i, recordNormal);

    // apply transform to sphere and interaction
    s.r2 = transformSphere(t, sInv).r2;
    if (found) {
        transformInteraction(t, tInv, s.c, true, i);
    }

    return found;
}

template<typename N, typename P, typename S>
__device__ bool transformedBvhFindClosestSilhouettePoint(const N* nodes, const P* primitives,
                                                          const S* silhouettes,
                                                          const float t[3][4], const float tInv[3][4],
                                                          DeviceBoundingSphere& s,
                                                          bool flipNormalOrientation,
                                                          float squaredMinRadius, float precision,
                                                          DeviceInteraction& i) {
    // apply inverse transform to sphere
    DeviceBoundingSphere sInv = transformSphere(tInv, s);
    DeviceBoundingSphere sMin;
    sMin.c = s.c;
    sMin.r2 = squaredMinRadius;
    DeviceBoundingSphere sMinInv = transformSphere(tInv, sMin);

    // find closest silhouette point
    bool found = bvhFindClosestSilhouettePoint<N, P, S>(nodes, primitives, silhouettes,
                                                         sInv, flipNormalOrientation,
                                                         sMinInv.r2, precision, i);

    // apply transform to sphere and interaction
    s.r2 = transformSphere(t, sInv).r2;
    if (found) {
        transformInteraction(t, tInv, s.c, true, i);
    }

    return found;
}

///////////////////////////////////////////////////////////////////////////////
// BVH refit
///////////////////////////////////////////////////////////////////////////////

template<typename N, typename P, typename S>
__device__ void bvhRefitLeafNode(N* nodes, const P* primitives, const S* silhouettes,
                                  unsigned int nodeIndex)
{
    float3 pMin = make_float3(FCPW_CUDA_FLT_MAX, FCPW_CUDA_FLT_MAX, FCPW_CUDA_FLT_MAX);
    float3 pMax = make_float3(-FCPW_CUDA_FLT_MAX, -FCPW_CUDA_FLT_MAX, -FCPW_CUDA_FLT_MAX);
    N node = nodes[nodeIndex];
    unsigned int nPrimitives = NodeTraits<N>::getNumPrimitives(node);

    for (unsigned int p = 0; p < nPrimitives; p++) {
        unsigned int primitiveIndex = NodeTraits<N>::getPrimitiveOffset(node) + p;
        DeviceBoundingBox primitiveBox = PrimitiveOps<P>::getBoundingBox(primitives[primitiveIndex]);
        pMin = fmin3(pMin, primitiveBox.pMin);
        pMax = fmax3(pMax, primitiveBox.pMax);
    }

    DeviceBoundingBox newBox;
    newBox.pMin = pMin;
    newBox.pMax = pMax;
    NodeTraits<N>::setBoundingBox(nodes[nodeIndex], newBox);

    if (NodeTraits<N>::hasBoundingCone(node)) {
        float3 axis = make_float3(0.0f, 0.0f, 0.0f);
        float3 centroid = 0.5f * (pMin + pMax);
        float halfAngle = 0.0f;
        float radius = 0.0f;
        bool anySilhouettes = false;
        bool silhouettesHaveTwoAdjacentFaces = true;
        unsigned int nSilhouettes = NodeTraits<N>::getNumSilhouettes(node);

        for (unsigned int p = 0; p < nSilhouettes; p++) {
            unsigned int silhouetteIndex = NodeTraits<N>::getSilhouetteOffset(node) + p;
            S silhouette = silhouettes[silhouetteIndex];
            axis = axis + SilhouetteOps<S>::getNormal(silhouette, 0);
            axis = axis + SilhouetteOps<S>::getNormal(silhouette, 1);
            radius = fmaxf(radius, length3(SilhouetteOps<S>::getCentroid(silhouette) - centroid));
            silhouettesHaveTwoAdjacentFaces = silhouettesHaveTwoAdjacentFaces &&
                                              SilhouetteOps<S>::hasTwoAdjacentFaces(silhouette);
            anySilhouettes = true;
        }

        if (!anySilhouettes) {
            halfAngle = -FCPW_CUDA_M_PI;
        } else if (!silhouettesHaveTwoAdjacentFaces) {
            halfAngle = FCPW_CUDA_M_PI;
        } else {
            float axisNorm = length3(axis);
            if (axisNorm > FCPW_CUDA_FLT_EPSILON) {
                axis = axis / axisNorm;

                for (unsigned int p = 0; p < nSilhouettes; p++) {
                    unsigned int silhouetteIndex = NodeTraits<N>::getSilhouetteOffset(node) + p;
                    for (unsigned int k = 0; k < 2; k++) {
                        float3 n = SilhouetteOps<S>::getNormal(silhouettes[silhouetteIndex], k);
                        float angle = acosf(fmaxf(-1.0f, fminf(1.0f, dot3(axis, n))));
                        halfAngle = fmaxf(halfAngle, angle);
                    }
                }
            }
        }

        DeviceBoundingCone newCone;
        newCone.axis = axis;
        newCone.halfAngle = halfAngle;
        newCone.radius = radius;
        NodeTraits<N>::setBoundingCone(nodes[nodeIndex], newCone);
    }
}

template<typename N, typename P, typename S>
__device__ void bvhRefitInternalNode(N* nodes, const P* primitives, const S* silhouettes,
                                      unsigned int nodeIndex)
{
    N node = nodes[nodeIndex];
    unsigned int leftNodeIndex = nodeIndex + 1;
    unsigned int rightNodeIndex = nodeIndex + NodeTraits<N>::getRightChildOffset(node);
    N leftNode = nodes[leftNodeIndex];
    N rightNode = nodes[rightNodeIndex];

    DeviceBoundingBox leftBox = NodeTraits<N>::getBoundingBox(leftNode);
    DeviceBoundingBox rightBox = NodeTraits<N>::getBoundingBox(rightNode);
    DeviceBoundingBox mergedBox = mergeBoundingBoxes(leftBox, rightBox);
    NodeTraits<N>::setBoundingBox(nodes[nodeIndex], mergedBox);

    if (NodeTraits<N>::hasBoundingCone(node)) {
        DeviceBoundingCone leftCone = NodeTraits<N>::getBoundingCone(leftNode);
        DeviceBoundingCone rightCone = NodeTraits<N>::getBoundingCone(rightNode);
        DeviceBoundingCone mergedCone = mergeBoundingCones(leftCone, rightCone,
                                                           bbox_centroid(leftBox),
                                                           bbox_centroid(rightBox),
                                                           bbox_centroid(mergedBox));
        NodeTraits<N>::setBoundingCone(nodes[nodeIndex], mergedCone);
    }
}

template<typename N, typename P, typename S>
__device__ void bvhRefit(N* nodes, const P* primitives, const S* silhouettes,
                         unsigned int nodeIndex)
{
    if (NodeTraits<N>::isLeaf(nodes[nodeIndex])) {
        bvhRefitLeafNode<N, P, S>(nodes, primitives, silhouettes, nodeIndex);
    } else {
        bvhRefitInternalNode<N, P, S>(nodes, primitives, silhouettes, nodeIndex);
    }
}

} // namespace fcpw
