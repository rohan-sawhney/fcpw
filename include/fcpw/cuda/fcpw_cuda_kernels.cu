#include <fcpw/cuda/fcpw_cuda_kernels.h>
#include <fcpw/cuda/cuda_runtime_utils.h>

#include <cmath>
#include <cstdint>
#include <type_traits>

namespace fcpw {

namespace {

// NOTE:
// This file ports the Slang backend traversal/refit entrypoints to CUDA kernels.
// The goal is behavioral parity with the existing backend architecture, not a
// redesign of the acceleration or numerical routines.

__device__ inline float3 makeF3(float x, float y, float z)
{
    float3 r;
    r.x = x;
    r.y = y;
    r.z = z;
    return r;
}

__device__ inline float3 add3(const float3& a, const float3& b) { return makeF3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ inline float3 sub3(const float3& a, const float3& b) { return makeF3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ inline float3 mul3(const float3& a, float s) { return makeF3(a.x*s, a.y*s, a.z*s); }
__device__ inline float dot3(const float3& a, const float3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
__device__ inline float length2(const float3& a) { return dot3(a, a); }
__device__ inline float length3(const float3& a) { return sqrtf(length2(a)); }
__device__ inline float3 normalize3(const float3& a) { float l = length3(a); return l > 1e-20f ? mul3(a, 1.0f/l) : makeF3(0.0f, 0.0f, 0.0f); }
__device__ inline float3 min3(const float3& a, const float3& b) { return makeF3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z)); }
__device__ inline float3 max3(const float3& a, const float3& b) { return makeF3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)); }
__device__ inline float3 cross3(const float3& a, const float3& b) {
    return makeF3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

struct TraversalStack {
    // Node index and traversal key (near distance / lower-bound distance).
    uint32_t node;
    float distance;
};

__device__ inline CUDAInteraction makeInvalidInteraction()
{
    CUDAInteraction out;
    out.p = makeF3(0.0f, 0.0f, 0.0f);
    out.n = makeF3(0.0f, 0.0f, 0.0f);
    out.uv = float2{0.0f, 0.0f};
    out.d = maxFloat;
    out.index = FCPW_CUDA_UINT_MAX;
    return out;
}

__device__ inline bool boxIntersectRay(const CUDABoundingBox& box,
                                       const CUDARay& r,
                                       float& tMin,
                                       float& tMax)
{
    float3 t0 = makeF3((box.pMin.x - r.o.x)*r.dInv.x,
                       (box.pMin.y - r.o.y)*r.dInv.y,
                       (box.pMin.z - r.o.z)*r.dInv.z);
    float3 t1 = makeF3((box.pMax.x - r.o.x)*r.dInv.x,
                       (box.pMax.y - r.o.y)*r.dInv.y,
                       (box.pMax.z - r.o.z)*r.dInv.z);

    float3 tNear = min3(t0, t1);
    float3 tFar = max3(t0, t1);

    float nearMax = fmaxf(0.0f, fmaxf(tNear.x, fmaxf(tNear.y, tNear.z)));
    float farMin = fminf(r.tMax, fminf(tFar.x, fminf(tFar.y, tFar.z)));

    if (nearMax > farMin) {
        tMin = INFINITY;
        tMax = INFINITY;
        return false;
    }

    tMin = nearMax;
    tMax = farMin;
    return true;
}

__device__ inline bool boxOverlapSphere(const CUDABoundingBox& box,
                                        const CUDABoundingSphere& s,
                                        float& d2Min,
                                        float& d2Max)
{
    float3 u = sub3(box.pMin, s.c);
    float3 v = sub3(s.c, box.pMax);
    float3 a = makeF3(fmaxf(fmaxf(u.x, v.x), 0.0f),
                      fmaxf(fmaxf(u.y, v.y), 0.0f),
                      fmaxf(fmaxf(u.z, v.z), 0.0f));
    float3 b = makeF3(fminf(u.x, v.x), fminf(u.y, v.y), fminf(u.z, v.z));
    d2Min = dot3(a, a);
    d2Max = dot3(b, b);

    return d2Min <= s.r2;
}

__device__ inline float boxDistance2(const CUDABoundingBox& box, const float3& p)
{
    float3 u = sub3(box.pMin, p);
    float3 v = sub3(p, box.pMax);
    float3 a = makeF3(fmaxf(fmaxf(u.x, v.x), 0.0f),
                      fmaxf(fmaxf(u.y, v.y), 0.0f),
                      fmaxf(fmaxf(u.z, v.z), 0.0f));

    return dot3(a, a);
}

__device__ inline bool intersectLineSegment(const CUDALineSegment& seg,
                                            CUDARay& r,
                                            bool checkForOcclusion,
                                            CUDAInteraction& i)
{
    float3 u = sub3(seg.pa, r.o);
    float3 v = sub3(seg.pb, seg.pa);
    float3 uxrd = cross3(u, r.d);
    float3 rdxv = cross3(r.d, v);
    float dv = rdxv.z;

    if (fabsf(dv) <= 1e-12f) return false;

    float s = uxrd.z / dv;
    if (s >= 0.0f && s <= 1.0f) {
        float t = cross3(u, v).z / dv;
        if (t >= 0.0f && t <= r.tMax) {
            if (checkForOcclusion) {
                i.index = seg.index;
                return true;
            }

            i.p = add3(seg.pa, mul3(v, s));
            i.n = normalize3(makeF3(v.y, -v.x, 0.0f));
            i.uv = float2{s, 0.0f};
            i.d = t;
            i.index = seg.index;
            r.tMax = t;
            return true;
        }
    }

    return false;
}

__device__ inline bool intersectTriangle(const CUDATriangle& tri,
                                         CUDARay& r,
                                         bool checkForOcclusion,
                                         CUDAInteraction& i)
{
    float3 v1 = sub3(tri.pb, tri.pa);
    float3 v2 = sub3(tri.pc, tri.pa);
    float3 q = cross3(r.d, v2);
    float det = dot3(v1, q);

    if (fabsf(det) <= 1e-12f) return false;
    float invDet = 1.0f / det;

    float3 rr = sub3(r.o, tri.pa);
    float v = dot3(rr, q) * invDet;
    if (v < 0.0f || v > 1.0f) return false;

    float3 s = cross3(rr, v1);
    float w = dot3(r.d, s) * invDet;
    if (w < 0.0f || v + w > 1.0f) return false;

    float t = dot3(v2, s) * invDet;
    if (t < 0.0f || t > r.tMax) return false;

    if (checkForOcclusion) {
        i.index = tri.index;
        return true;
    }

    float u = 1.0f - v - w;
    i.p = add3(add3(mul3(tri.pa, u), mul3(tri.pb, v)), mul3(tri.pc, w));
    i.n = normalize3(cross3(v1, v2));
    i.uv = float2{v, w};
    i.d = t;
    i.index = tri.index;
    r.tMax = t;

    return true;
}

__device__ inline float closestPointLineSegment(const CUDALineSegment& seg,
                                                const float3& x,
                                                float3& p,
                                                float& t)
{
    float3 u = sub3(seg.pb, seg.pa);
    float3 v = sub3(x, seg.pa);

    float c1 = dot3(u, v);
    if (c1 <= 0.0f) {
        t = 0.0f;
        p = seg.pa;
        return length3(sub3(x, p));
    }

    float c2 = dot3(u, u);
    if (c2 <= c1) {
        t = 1.0f;
        p = seg.pb;
        return length3(sub3(x, p));
    }

    t = c1/c2;
    p = add3(seg.pa, mul3(u, t));
    return length3(sub3(x, p));
}

__device__ inline float closestPointTriangle(const CUDATriangle& tri,
                                             const float3& p,
                                             float3& closest,
                                             float2& uv)
{
    float3 ab = sub3(tri.pb, tri.pa);
    float3 ac = sub3(tri.pc, tri.pa);
    float3 ap = sub3(p, tri.pa);

    float d1 = dot3(ab, ap);
    float d2 = dot3(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) {
        closest = tri.pa;
        uv = float2{0.0f, 0.0f};
        return length3(sub3(p, tri.pa));
    }

    float3 bp = sub3(p, tri.pb);
    float d3 = dot3(ab, bp);
    float d4 = dot3(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) {
        closest = tri.pb;
        uv = float2{1.0f, 0.0f};
        return length3(sub3(p, tri.pb));
    }

    float vc = d1*d4 - d3*d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        float v = d1/(d1 - d3);
        closest = add3(tri.pa, mul3(ab, v));
        uv = float2{v, 0.0f};
        return length3(sub3(p, closest));
    }

    float3 cp = sub3(p, tri.pc);
    float d5 = dot3(ab, cp);
    float d6 = dot3(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) {
        closest = tri.pc;
        uv = float2{0.0f, 1.0f};
        return length3(sub3(p, tri.pc));
    }

    float vb = d5*d2 - d1*d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        float w = d2/(d2 - d6);
        closest = add3(tri.pa, mul3(ac, w));
        uv = float2{0.0f, w};
        return length3(sub3(p, closest));
    }

    float va = d3*d6 - d5*d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        float3 cb = sub3(tri.pc, tri.pb);
        float w = (d4 - d3)/((d4 - d3) + (d5 - d6));
        closest = add3(tri.pb, mul3(cb, w));
        uv = float2{1.0f - w, w};
        return length3(sub3(p, closest));
    }

    float denom = 1.0f/(va + vb + vc);
    float v = vb*denom;
    float w = vc*denom;
    closest = add3(add3(tri.pa, mul3(ab, v)), mul3(ac, w));
    uv = float2{v, w};
    return length3(sub3(p, closest));
}

template<typename Primitive>
__device__ inline CUDABoundingBox primitiveBox(const Primitive& p);
// Per-primitive bounds used by refit.

template<>
__device__ inline CUDABoundingBox primitiveBox<CUDALineSegment>(const CUDALineSegment& p)
{
    float3 eps = makeF3(1e-7f, 1e-7f, 0.0f);
    return CUDABoundingBox{sub3(min3(p.pa, p.pb), eps), add3(max3(p.pa, p.pb), eps)};
}

template<>
__device__ inline CUDABoundingBox primitiveBox<CUDATriangle>(const CUDATriangle& p)
{
    float3 eps = makeF3(1e-7f, 1e-7f, 1e-7f);
    float3 mn = min3(p.pa, min3(p.pb, p.pc));
    float3 mx = max3(p.pa, max3(p.pb, p.pc));
    return CUDABoundingBox{sub3(mn, eps), add3(mx, eps)};
}

template<typename Primitive>
__device__ inline bool primitiveIntersectRay(const Primitive& p,
                                             CUDARay& r,
                                             bool checkForOcclusion,
                                             CUDAInteraction& i);
// Per-primitive ray tests used in leaf processing.

template<>
__device__ inline bool primitiveIntersectRay<CUDALineSegment>(const CUDALineSegment& p,
                                                               CUDARay& r,
                                                               bool checkForOcclusion,
                                                               CUDAInteraction& i)
{
    return intersectLineSegment(p, r, checkForOcclusion, i);
}

template<>
__device__ inline bool primitiveIntersectRay<CUDATriangle>(const CUDATriangle& p,
                                                            CUDARay& r,
                                                            bool checkForOcclusion,
                                                            CUDAInteraction& i)
{
    return intersectTriangle(p, r, checkForOcclusion, i);
}

template<typename Primitive>
__device__ inline bool primitiveIntersectSphere(const Primitive& p,
                                                const CUDABoundingSphere& s,
                                                CUDAInteraction& i);
// Per-primitive sphere overlap tests used by stochastic sphere queries.

template<>
__device__ inline bool primitiveIntersectSphere<CUDALineSegment>(const CUDALineSegment& p,
                                                                  const CUDABoundingSphere& s,
                                                                  CUDAInteraction& i)
{
    float t;
    float3 cp;
    float d = closestPointLineSegment(p, s.c, cp, t);
    if (d*d <= s.r2) {
        i.p = cp;
        i.uv = float2{t, 0.0f};
        i.index = p.index;
        i.d = length3(sub3(p.pb, p.pa));
        return true;
    }

    return false;
}

template<>
__device__ inline bool primitiveIntersectSphere<CUDATriangle>(const CUDATriangle& p,
                                                               const CUDABoundingSphere& s,
                                                               CUDAInteraction& i)
{
    float3 cp;
    float2 uv;
    float d = closestPointTriangle(p, s.c, cp, uv);
    if (d*d <= s.r2) {
        i.p = cp;
        i.uv = uv;
        i.index = p.index;
        i.d = 0.5f*length3(cross3(sub3(p.pb, p.pa), sub3(p.pc, p.pa)));
        return true;
    }

    return false;
}

template<typename Primitive>
__device__ inline bool primitiveClosestPoint(const Primitive& p,
                                             const CUDABoundingSphere& s,
                                             bool recordNormal,
                                             CUDAInteraction& i);
// Per-primitive closest-point projection with optional normal output.

template<>
__device__ inline bool primitiveClosestPoint<CUDALineSegment>(const CUDALineSegment& p,
                                                               const CUDABoundingSphere& s,
                                                               bool recordNormal,
                                                               CUDAInteraction& i)
{
    float t;
    float3 cp;
    float d = closestPointLineSegment(p, s.c, cp, t);
    if (d*d <= s.r2) {
        i.p = cp;
        i.uv = float2{t, 0.0f};
        i.d = d;
        i.index = p.index;
        if (recordNormal) {
            float3 dir = sub3(p.pb, p.pa);
            i.n = normalize3(makeF3(dir.y, -dir.x, 0.0f));
        }
        return true;
    }

    return false;
}

template<>
__device__ inline bool primitiveClosestPoint<CUDATriangle>(const CUDATriangle& p,
                                                            const CUDABoundingSphere& s,
                                                            bool recordNormal,
                                                            CUDAInteraction& i)
{
    float3 cp;
    float2 uv;
    float d = closestPointTriangle(p, s.c, cp, uv);
    if (d*d <= s.r2) {
        i.p = cp;
        i.uv = uv;
        i.d = d;
        i.index = p.index;
        if (recordNormal) {
            i.n = normalize3(cross3(sub3(p.pb, p.pa), sub3(p.pc, p.pa)));
        }
        return true;
    }

    return false;
}

template<typename Node>
struct NodeTraits;
// Node-type adapter so traversal kernels can be shared by BVH and SNCH nodes.

template<>
struct NodeTraits<CUDABvhNode> {
    __device__ static CUDABoundingBox box(const CUDABvhNode& n) { return n.box; }
    __device__ static bool isLeaf(const CUDABvhNode& n) { return n.nPrimitives > 0; }
    __device__ static uint32_t primitiveOffset(const CUDABvhNode& n) { return n.offset; }
    __device__ static uint32_t primitiveCount(const CUDABvhNode& n) { return n.nPrimitives; }
    __device__ static uint32_t rightOffset(const CUDABvhNode& n) { return n.offset; }
    __device__ static uint32_t silhouetteOffset(const CUDABvhNode& n) { (void)n; return 0u; }
    __device__ static uint32_t silhouetteCount(const CUDABvhNode& n) { (void)n; return 0u; }
};

template<>
struct NodeTraits<CUDASnchNode> {
    __device__ static CUDABoundingBox box(const CUDASnchNode& n) { return n.box; }
    __device__ static bool isLeaf(const CUDASnchNode& n) { return n.nPrimitives > 0; }
    __device__ static uint32_t primitiveOffset(const CUDASnchNode& n) { return n.offset; }
    __device__ static uint32_t primitiveCount(const CUDASnchNode& n) { return n.nPrimitives; }
    __device__ static uint32_t rightOffset(const CUDASnchNode& n) { return n.offset; }
    __device__ static uint32_t silhouetteOffset(const CUDASnchNode& n) { return n.silhouetteOffset; }
    __device__ static uint32_t silhouetteCount(const CUDASnchNode& n) { return n.nSilhouettes; }
};

template<typename Silhouette>
__device__ inline bool silhouetteClosestPoint(const Silhouette& s,
                                              const CUDABoundingSphere& q,
                                              bool flip,
                                              float squaredMinRadius,
                                              float precision,
                                              CUDAInteraction& i);
// Silhouette projection routines used only by silhouette query kernels.

template<>
__device__ inline bool silhouetteClosestPoint<CUDAVertex>(const CUDAVertex& s,
                                                           const CUDABoundingSphere& q,
                                                           bool flip,
                                                           float squaredMinRadius,
                                                           float precision,
                                                           CUDAInteraction& i)
{
    (void)precision;
    float3 d = sub3(q.c, s.p);
    float d2 = dot3(d, d);
    if (d2 > q.r2 || d2 < squaredMinRadius) return false;

    i.p = s.p;
    i.uv = float2{0.0f, 0.0f};
    i.d = sqrtf(d2);
    float3 n = add3(s.n0, s.n1);
    n = normalize3(n);
    if (flip) n = mul3(n, -1.0f);
    i.n = n;
    i.index = s.index;
    return true;
}

template<>
__device__ inline bool silhouetteClosestPoint<CUDAEdge>(const CUDAEdge& s,
                                                         const CUDABoundingSphere& q,
                                                         bool flip,
                                                         float squaredMinRadius,
                                                         float precision,
                                                         CUDAInteraction& i)
{
    (void)precision;
    CUDALineSegment seg{s.pa, s.pb, s.index};
    float t;
    float3 cp;
    float d = closestPointLineSegment(seg, q.c, cp, t);
    float d2 = d*d;
    if (d2 > q.r2 || d2 < squaredMinRadius) return false;

    i.p = cp;
    i.uv = float2{t, 0.0f};
    i.d = d;
    float3 n = add3(s.n0, s.n1);
    n = normalize3(n);
    if (flip) n = mul3(n, -1.0f);
    i.n = n;
    i.index = s.index;
    return true;
}

template<>
__device__ inline bool silhouetteClosestPoint<CUDANoSilhouette>(const CUDANoSilhouette& s,
                                                                 const CUDABoundingSphere& q,
                                                                 bool flip,
                                                                 float squaredMinRadius,
                                                                 float precision,
                                                                 CUDAInteraction& i)
{
    (void)s;
    (void)q;
    (void)flip;
    (void)squaredMinRadius;
    (void)precision;
    (void)i;
    return false;
}

template<typename Node, typename Primitive, typename Silhouette, bool HasSilhouette>
__global__ void rayIntersectionKernel(const Node *nodes,
                                      const Primitive *primitives,
                                      const Silhouette *silhouettes,
                                      const CUDARay *rays,
                                      CUDAInteraction *interactions,
                                      uint32_t checkForOcclusion,
                                      uint32_t nQueries)
{
    // One ray query per thread, traversed front-to-back to preserve closest-hit behavior.
    (void)silhouettes;

    uint32_t index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= nQueries) return;

    CUDARay r = rays[index];
    CUDAInteraction result = makeInvalidInteraction();
    float rootNear, rootFar;

    if (!boxIntersectRay(NodeTraits<Node>::box(nodes[0]), r, rootNear, rootFar)) {
        interactions[index] = result;
        return;
    }

    TraversalStack stack[FCPW_BVH_MAX_DEPTH];
    int stackPtr = 0;
    stack[0].node = 0;
    stack[0].distance = rootNear;

    bool hit = false;
    while (stackPtr >= 0) {
        uint32_t nodeIndex = stack[stackPtr].node;
        float nodeDist = stack[stackPtr].distance;
        stackPtr--;

        if (nodeDist > r.tMax) continue;

        const Node& node = nodes[nodeIndex];
        if (NodeTraits<Node>::isLeaf(node)) {
            uint32_t count = NodeTraits<Node>::primitiveCount(node);
            uint32_t offset = NodeTraits<Node>::primitiveOffset(node);
            for (uint32_t p = 0; p < count; p++) {
                CUDAInteraction cand;
                if (primitiveIntersectRay<Primitive>(primitives[offset + p], r,
                                                     checkForOcclusion != 0, cand)) {
                    hit = true;
                    result = cand;
                    if (checkForOcclusion != 0) {
                        interactions[index] = result;
                        return;
                    }
                }
            }
        } else {
            uint32_t left = nodeIndex + 1;
            uint32_t right = nodeIndex + NodeTraits<Node>::rightOffset(node);
            float leftNear, leftFar, rightNear, rightFar;
            bool hitLeft = boxIntersectRay(NodeTraits<Node>::box(nodes[left]), r, leftNear, leftFar);
            bool hitRight = boxIntersectRay(NodeTraits<Node>::box(nodes[right]), r, rightNear, rightFar);

            if (hitLeft && hitRight) {
                uint32_t nearNode = left, farNode = right;
                float nearDist = leftNear, farDist = rightNear;
                if (rightNear < leftNear) {
                    nearNode = right; farNode = left;
                    nearDist = rightNear; farDist = leftNear;
                }

                stack[++stackPtr] = TraversalStack{farNode, farDist};
                stack[++stackPtr] = TraversalStack{nearNode, nearDist};
            } else if (hitLeft) {
                stack[++stackPtr] = TraversalStack{left, leftNear};
            } else if (hitRight) {
                stack[++stackPtr] = TraversalStack{right, rightNear};
            }
        }
    }

    interactions[index] = hit ? result : makeInvalidInteraction();
}

template<typename Node, typename Primitive, typename Silhouette, bool HasSilhouette>
__global__ void sphereIntersectionKernel(const Node *nodes,
                                         const Primitive *primitives,
                                         const Silhouette *silhouettes,
                                         const CUDABoundingSphere *spheres,
                                         const float3 *randNums,
                                         CUDAInteraction *interactions,
                                         uint32_t nQueries)
{
    // One sphere query per thread, visiting all overlapping nodes.
    (void)silhouettes;
    uint32_t index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= nQueries) return;

    CUDABoundingSphere s = spheres[index];
    float3 rand = randNums[index];
    (void)rand;

    CUDAInteraction result = makeInvalidInteraction();
    bool hit = false;

    float rootMin, rootMax;
    if (!boxOverlapSphere(NodeTraits<Node>::box(nodes[0]), s, rootMin, rootMax)) {
        interactions[index] = result;
        return;
    }

    TraversalStack stack[FCPW_BVH_MAX_DEPTH];
    int stackPtr = 0;
    stack[0] = TraversalStack{0, rootMin};

    while (stackPtr >= 0) {
        uint32_t nodeIndex = stack[stackPtr].node;
        stackPtr--;

        const Node& node = nodes[nodeIndex];
        if (NodeTraits<Node>::isLeaf(node)) {
            uint32_t count = NodeTraits<Node>::primitiveCount(node);
            uint32_t offset = NodeTraits<Node>::primitiveOffset(node);
            for (uint32_t p = 0; p < count; p++) {
                CUDAInteraction cand;
                if (primitiveIntersectSphere<Primitive>(primitives[offset + p], s, cand)) {
                    hit = true;
                    result = cand;
                    break;
                }
            }
        } else {
            uint32_t left = nodeIndex + 1;
            uint32_t right = nodeIndex + NodeTraits<Node>::rightOffset(node);
            float leftMin, leftMax, rightMin, rightMax;
            bool hitLeft = boxOverlapSphere(NodeTraits<Node>::box(nodes[left]), s, leftMin, leftMax);
            bool hitRight = boxOverlapSphere(NodeTraits<Node>::box(nodes[right]), s, rightMin, rightMax);

            if (hitLeft) stack[++stackPtr] = TraversalStack{left, leftMin};
            if (hitRight) stack[++stackPtr] = TraversalStack{right, rightMin};
        }
    }

    interactions[index] = hit ? result : makeInvalidInteraction();
}

template<typename Node, typename Primitive, typename Silhouette, bool HasSilhouette>
__global__ void closestPointKernel(const Node *nodes,
                                   const Primitive *primitives,
                                   const Silhouette *silhouettes,
                                   const CUDABoundingSphere *spheres,
                                   CUDAInteraction *interactions,
                                   uint32_t recordNormals,
                                   uint32_t nQueries)
{
    // One closest-point query per thread with radius-based pruning.
    (void)silhouettes;

    uint32_t index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= nQueries) return;

    CUDABoundingSphere s = spheres[index];
    CUDAInteraction result = makeInvalidInteraction();
    bool found = false;
    float best2 = s.r2;

    float rootMin, rootMax;
    if (!boxOverlapSphere(NodeTraits<Node>::box(nodes[0]), s, rootMin, rootMax)) {
        interactions[index] = result;
        return;
    }

    TraversalStack stack[FCPW_BVH_MAX_DEPTH];
    int stackPtr = 0;
    stack[0] = TraversalStack{0, rootMin};

    while (stackPtr >= 0) {
        uint32_t nodeIndex = stack[stackPtr].node;
        stackPtr--;

        const Node& node = nodes[nodeIndex];
        float nodeD2 = boxDistance2(NodeTraits<Node>::box(node), s.c);
        if (nodeD2 > best2) continue;

        if (NodeTraits<Node>::isLeaf(node)) {
            uint32_t count = NodeTraits<Node>::primitiveCount(node);
            uint32_t offset = NodeTraits<Node>::primitiveOffset(node);
            for (uint32_t p = 0; p < count; p++) {
                CUDAInteraction cand;
                if (primitiveClosestPoint<Primitive>(primitives[offset + p], s,
                                                     recordNormals != 0, cand)) {
                    float d2 = cand.d*cand.d;
                    if (d2 <= best2) {
                        best2 = d2;
                        result = cand;
                        found = true;
                    }
                }
            }
        } else {
            uint32_t left = nodeIndex + 1;
            uint32_t right = nodeIndex + NodeTraits<Node>::rightOffset(node);
            float leftD2 = boxDistance2(NodeTraits<Node>::box(nodes[left]), s.c);
            float rightD2 = boxDistance2(NodeTraits<Node>::box(nodes[right]), s.c);

            if (leftD2 <= best2 && rightD2 <= best2) {
                if (leftD2 < rightD2) {
                    stack[++stackPtr] = TraversalStack{right, rightD2};
                    stack[++stackPtr] = TraversalStack{left, leftD2};
                } else {
                    stack[++stackPtr] = TraversalStack{left, leftD2};
                    stack[++stackPtr] = TraversalStack{right, rightD2};
                }
            } else if (leftD2 <= best2) {
                stack[++stackPtr] = TraversalStack{left, leftD2};
            } else if (rightD2 <= best2) {
                stack[++stackPtr] = TraversalStack{right, rightD2};
            }
        }
    }

    interactions[index] = found ? result : makeInvalidInteraction();
}

template<typename Node, typename Primitive, typename Silhouette, bool HasSilhouette>
__global__ void closestSilhouettePointKernel(const Node *nodes,
                                             const Primitive *primitives,
                                             const Silhouette *silhouettes,
                                             const CUDABoundingSphere *spheres,
                                             const uint32_t *flipNormalOrientation,
                                             CUDAInteraction *interactions,
                                             float squaredMinRadius,
                                             float precision,
                                             uint32_t nQueries)
{
    // One silhouette query per thread. Non-silhouette BVH payloads return invalid.
    (void)primitives;

    uint32_t index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= nQueries) return;

    CUDAInteraction out = makeInvalidInteraction();
    if (!HasSilhouette) {
        interactions[index] = out;
        return;
    }

    CUDABoundingSphere s = spheres[index];
    bool flip = flipNormalOrientation[index] != 0;
    bool found = false;
    float best2 = s.r2;

    float rootMin, rootMax;
    if (!boxOverlapSphere(NodeTraits<Node>::box(nodes[0]), s, rootMin, rootMax)) {
        interactions[index] = out;
        return;
    }

    TraversalStack stack[FCPW_BVH_MAX_DEPTH];
    int stackPtr = 0;
    stack[0] = TraversalStack{0, rootMin};

    while (stackPtr >= 0) {
        uint32_t nodeIndex = stack[stackPtr].node;
        stackPtr--;

        const Node& node = nodes[nodeIndex];
        float nodeD2 = boxDistance2(NodeTraits<Node>::box(node), s.c);
        if (nodeD2 > best2) continue;

        if (NodeTraits<Node>::isLeaf(node)) {
            uint32_t nSil = NodeTraits<Node>::silhouetteCount(node);
            uint32_t off = NodeTraits<Node>::silhouetteOffset(node);
            for (uint32_t k = 0; k < nSil; k++) {
                CUDAInteraction cand;
                if (silhouetteClosestPoint<Silhouette>(silhouettes[off + k], s, flip,
                                                       squaredMinRadius, precision, cand)) {
                    float d2 = cand.d*cand.d;
                    if (d2 <= best2) {
                        best2 = d2;
                        out = cand;
                        found = true;
                    }
                }
            }
        } else {
            uint32_t left = nodeIndex + 1;
            uint32_t right = nodeIndex + NodeTraits<Node>::rightOffset(node);
            float leftD2 = boxDistance2(NodeTraits<Node>::box(nodes[left]), s.c);
            float rightD2 = boxDistance2(NodeTraits<Node>::box(nodes[right]), s.c);

            if (leftD2 <= best2) stack[++stackPtr] = TraversalStack{left, leftD2};
            if (rightD2 <= best2) stack[++stackPtr] = TraversalStack{right, rightD2};
        }
    }

    interactions[index] = found ? out : makeInvalidInteraction();
}

template<typename Node, typename Primitive, typename Silhouette, bool HasSilhouette>
__global__ void refitKernel(Node *nodes,
                            const Primitive *primitives,
                            const Silhouette *silhouettes,
                            const uint32_t *nodeIndices,
                            uint32_t firstNodeOffset,
                            uint32_t nodeCount)
{
    // Refit one node. Host dispatches depth buckets bottom-up.
    uint32_t index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= nodeCount) return;

    uint32_t nodeIndex = nodeIndices[firstNodeOffset + index];
    Node node = nodes[nodeIndex];

    if (NodeTraits<Node>::isLeaf(node)) {
        uint32_t count = NodeTraits<Node>::primitiveCount(node);
        uint32_t offset = NodeTraits<Node>::primitiveOffset(node);
        CUDABoundingBox box;
        box.pMin = makeF3(INFINITY, INFINITY, INFINITY);
        box.pMax = makeF3(-INFINITY, -INFINITY, -INFINITY);

        for (uint32_t p = 0; p < count; p++) {
            CUDABoundingBox b = primitiveBox(primitives[offset + p]);
            box.pMin = min3(box.pMin, b.pMin);
            box.pMax = max3(box.pMax, b.pMax);
        }

        node.box = box;

        if constexpr (HasSilhouette) {
            uint32_t nSil = NodeTraits<Node>::silhouetteCount(node);
            uint32_t silOff = NodeTraits<Node>::silhouetteOffset(node);

            float3 axis = makeF3(0.0f, 0.0f, 0.0f);
            float3 centroid = mul3(add3(box.pMin, box.pMax), 0.5f);
            float radius = 0.0f;
            bool any = false;
            bool allTwoFaces = true;

            for (uint32_t s = 0; s < nSil; s++) {
                const Silhouette& sil = silhouettes[silOff + s];
                if constexpr (std::is_same<Silhouette, CUDAVertex>::value) {
                    axis = add3(axis, sil.n0);
                    axis = add3(axis, sil.n1);
                    radius = fmaxf(radius, length3(sub3(sil.p, centroid)));
                    allTwoFaces = allTwoFaces && (sil.hasOneAdjacentFace == 0);
                } else if constexpr (std::is_same<Silhouette, CUDAEdge>::value) {
                    axis = add3(axis, sil.n0);
                    axis = add3(axis, sil.n1);
                    float3 c = mul3(add3(sil.pa, sil.pb), 0.5f);
                    radius = fmaxf(radius, length3(sub3(c, centroid)));
                    allTwoFaces = allTwoFaces && (sil.hasOneAdjacentFace == 0);
                }
                any = true;
            }

            if (!any) {
                node.cone.axis = makeF3(0.0f, 0.0f, 0.0f);
                node.cone.halfAngle = -3.14159265358979323846f;
                node.cone.radius = 0.0f;
            } else if (!allTwoFaces) {
                node.cone.axis = normalize3(axis);
                node.cone.halfAngle = 3.14159265358979323846f;
                node.cone.radius = radius;
            } else {
                node.cone.axis = normalize3(axis);
                node.cone.halfAngle = 0.0f;
                node.cone.radius = radius;
            }
        }

        nodes[nodeIndex] = node;
    } else {
        uint32_t leftNodeIndex = nodeIndex + 1;
        uint32_t rightNodeIndex = nodeIndex + NodeTraits<Node>::rightOffset(node);

        CUDABoundingBox leftBox = nodes[leftNodeIndex].box;
        CUDABoundingBox rightBox = nodes[rightNodeIndex].box;
        node.box.pMin = min3(leftBox.pMin, rightBox.pMin);
        node.box.pMax = max3(leftBox.pMax, rightBox.pMax);

        if constexpr (HasSilhouette) {
            node.cone.axis = normalize3(add3(nodes[leftNodeIndex].cone.axis, nodes[rightNodeIndex].cone.axis));
            node.cone.halfAngle = fmaxf(nodes[leftNodeIndex].cone.halfAngle, nodes[rightNodeIndex].cone.halfAngle);
            node.cone.radius = fmaxf(nodes[leftNodeIndex].cone.radius, nodes[rightNodeIndex].cone.radius);
        }

        nodes[nodeIndex] = node;
    }
}

template<typename Node, typename Primitive, typename Silhouette, bool HasSilhouette>
inline void launchRayTyped(const void *nodes,
                           const void *primitives,
                           const void *silhouettes,
                           const CUDARay *rays,
                           CUDAInteraction *interactions,
                           uint32_t nQueries,
                           uint32_t checkForOcclusion,
                           uint32_t nThreadsPerBlock,
                           cudaStream_t stream)
{
    // Concrete typed kernel launch selected by CUDABvhType.
    uint32_t nBlocks = (nQueries + nThreadsPerBlock - 1u)/nThreadsPerBlock;
    rayIntersectionKernel<Node, Primitive, Silhouette, HasSilhouette><<<nBlocks, nThreadsPerBlock, 0, stream>>>(
        static_cast<const Node *>(nodes),
        static_cast<const Primitive *>(primitives),
        static_cast<const Silhouette *>(silhouettes),
        rays,
        interactions,
        checkForOcclusion,
        nQueries);
}

template<typename Node, typename Primitive, typename Silhouette, bool HasSilhouette>
inline void launchSphereTyped(const void *nodes,
                              const void *primitives,
                              const void *silhouettes,
                              const CUDABoundingSphere *spheres,
                              const float3 *randNums,
                              CUDAInteraction *interactions,
                              uint32_t nQueries,
                              uint32_t nThreadsPerBlock,
                              cudaStream_t stream)
{
    // Concrete typed kernel launch selected by CUDABvhType.
    uint32_t nBlocks = (nQueries + nThreadsPerBlock - 1u)/nThreadsPerBlock;
    sphereIntersectionKernel<Node, Primitive, Silhouette, HasSilhouette><<<nBlocks, nThreadsPerBlock, 0, stream>>>(
        static_cast<const Node *>(nodes),
        static_cast<const Primitive *>(primitives),
        static_cast<const Silhouette *>(silhouettes),
        spheres,
        randNums,
        interactions,
        nQueries);
}

template<typename Node, typename Primitive, typename Silhouette, bool HasSilhouette>
inline void launchClosestPointTyped(const void *nodes,
                                    const void *primitives,
                                    const void *silhouettes,
                                    const CUDABoundingSphere *spheres,
                                    CUDAInteraction *interactions,
                                    uint32_t nQueries,
                                    uint32_t recordNormals,
                                    uint32_t nThreadsPerBlock,
                                    cudaStream_t stream)
{
    // Concrete typed kernel launch selected by CUDABvhType.
    uint32_t nBlocks = (nQueries + nThreadsPerBlock - 1u)/nThreadsPerBlock;
    closestPointKernel<Node, Primitive, Silhouette, HasSilhouette><<<nBlocks, nThreadsPerBlock, 0, stream>>>(
        static_cast<const Node *>(nodes),
        static_cast<const Primitive *>(primitives),
        static_cast<const Silhouette *>(silhouettes),
        spheres,
        interactions,
        recordNormals,
        nQueries);
}

template<typename Node, typename Primitive, typename Silhouette, bool HasSilhouette>
inline void launchClosestSilhouetteTyped(const void *nodes,
                                         const void *primitives,
                                         const void *silhouettes,
                                         const CUDABoundingSphere *spheres,
                                         const uint32_t *flipNormalOrientation,
                                         CUDAInteraction *interactions,
                                         uint32_t nQueries,
                                         float squaredMinRadius,
                                         float precision,
                                         uint32_t nThreadsPerBlock,
                                         cudaStream_t stream)
{
    // Concrete typed kernel launch selected by CUDABvhType.
    uint32_t nBlocks = (nQueries + nThreadsPerBlock - 1u)/nThreadsPerBlock;
    closestSilhouettePointKernel<Node, Primitive, Silhouette, HasSilhouette><<<nBlocks, nThreadsPerBlock, 0, stream>>>(
        static_cast<const Node *>(nodes),
        static_cast<const Primitive *>(primitives),
        static_cast<const Silhouette *>(silhouettes),
        spheres,
        flipNormalOrientation,
        interactions,
        squaredMinRadius,
        precision,
        nQueries);
}

template<typename Node, typename Primitive, typename Silhouette, bool HasSilhouette>
inline void launchRefitTyped(void *nodes,
                             const void *primitives,
                             const void *silhouettes,
                             const uint32_t *nodeIndices,
                             uint32_t firstNodeOffset,
                             uint32_t nodeCount,
                             uint32_t nThreadsPerBlock,
                             cudaStream_t stream)
{
    // Concrete typed kernel launch selected by CUDABvhType.
    uint32_t nBlocks = (nodeCount + nThreadsPerBlock - 1u)/nThreadsPerBlock;
    refitKernel<Node, Primitive, Silhouette, HasSilhouette><<<nBlocks, nThreadsPerBlock, 0, stream>>>(
        static_cast<Node *>(nodes),
        static_cast<const Primitive *>(primitives),
        static_cast<const Silhouette *>(silhouettes),
        nodeIndices,
        firstNodeOffset,
        nodeCount);
}

} // namespace

void launchCudaRayIntersection(CUDABvhType type,
                               const void *nodes,
                               const void *primitives,
                               const void *silhouettes,
                               const CUDARay *rays,
                               CUDAInteraction *interactions,
                               uint32_t nQueries,
                               uint32_t checkForOcclusion,
                               uint32_t nThreadsPerBlock,
                               cudaStream_t stream)
{
    // Type-erased dispatch to concrete kernel instantiations.
    switch (type) {
    case CUDABvhType::LineSegmentBvh:
        launchRayTyped<CUDABvhNode, CUDALineSegment, CUDANoSilhouette, false>(nodes, primitives, silhouettes,
                                                                               rays, interactions, nQueries,
                                                                               checkForOcclusion, nThreadsPerBlock,
                                                                               stream);
        break;
    case CUDABvhType::TriangleBvh:
        launchRayTyped<CUDABvhNode, CUDATriangle, CUDANoSilhouette, false>(nodes, primitives, silhouettes,
                                                                            rays, interactions, nQueries,
                                                                            checkForOcclusion, nThreadsPerBlock,
                                                                            stream);
        break;
    case CUDABvhType::LineSegmentSnch:
        launchRayTyped<CUDASnchNode, CUDALineSegment, CUDAVertex, true>(nodes, primitives, silhouettes,
                                                                         rays, interactions, nQueries,
                                                                         checkForOcclusion, nThreadsPerBlock,
                                                                         stream);
        break;
    case CUDABvhType::TriangleSnch:
        launchRayTyped<CUDASnchNode, CUDATriangle, CUDAEdge, true>(nodes, primitives, silhouettes,
                                                                    rays, interactions, nQueries,
                                                                    checkForOcclusion, nThreadsPerBlock,
                                                                    stream);
        break;
    }

    FCPW_CUDA_CHECK(cudaGetLastError());
}

void launchCudaSphereIntersection(CUDABvhType type,
                                  const void *nodes,
                                  const void *primitives,
                                  const void *silhouettes,
                                  const CUDABoundingSphere *spheres,
                                  const float3 *randNums,
                                  CUDAInteraction *interactions,
                                  uint32_t nQueries,
                                  uint32_t nThreadsPerBlock,
                                  cudaStream_t stream)
{
    // Type-erased dispatch to concrete kernel instantiations.
    switch (type) {
    case CUDABvhType::LineSegmentBvh:
        launchSphereTyped<CUDABvhNode, CUDALineSegment, CUDANoSilhouette, false>(nodes, primitives, silhouettes,
                                                                                   spheres, randNums, interactions,
                                                                                   nQueries, nThreadsPerBlock, stream);
        break;
    case CUDABvhType::TriangleBvh:
        launchSphereTyped<CUDABvhNode, CUDATriangle, CUDANoSilhouette, false>(nodes, primitives, silhouettes,
                                                                                spheres, randNums, interactions,
                                                                                nQueries, nThreadsPerBlock, stream);
        break;
    case CUDABvhType::LineSegmentSnch:
        launchSphereTyped<CUDASnchNode, CUDALineSegment, CUDAVertex, true>(nodes, primitives, silhouettes,
                                                                            spheres, randNums, interactions,
                                                                            nQueries, nThreadsPerBlock, stream);
        break;
    case CUDABvhType::TriangleSnch:
        launchSphereTyped<CUDASnchNode, CUDATriangle, CUDAEdge, true>(nodes, primitives, silhouettes,
                                                                       spheres, randNums, interactions,
                                                                       nQueries, nThreadsPerBlock, stream);
        break;
    }

    FCPW_CUDA_CHECK(cudaGetLastError());
}

void launchCudaClosestPoint(CUDABvhType type,
                            const void *nodes,
                            const void *primitives,
                            const void *silhouettes,
                            const CUDABoundingSphere *spheres,
                            CUDAInteraction *interactions,
                            uint32_t nQueries,
                            uint32_t recordNormals,
                            uint32_t nThreadsPerBlock,
                            cudaStream_t stream)
{
    // Type-erased dispatch to concrete kernel instantiations.
    switch (type) {
    case CUDABvhType::LineSegmentBvh:
        launchClosestPointTyped<CUDABvhNode, CUDALineSegment, CUDANoSilhouette, false>(nodes, primitives, silhouettes,
                                                                                         spheres, interactions, nQueries,
                                                                                         recordNormals, nThreadsPerBlock,
                                                                                         stream);
        break;
    case CUDABvhType::TriangleBvh:
        launchClosestPointTyped<CUDABvhNode, CUDATriangle, CUDANoSilhouette, false>(nodes, primitives, silhouettes,
                                                                                      spheres, interactions, nQueries,
                                                                                      recordNormals, nThreadsPerBlock,
                                                                                      stream);
        break;
    case CUDABvhType::LineSegmentSnch:
        launchClosestPointTyped<CUDASnchNode, CUDALineSegment, CUDAVertex, true>(nodes, primitives, silhouettes,
                                                                                  spheres, interactions, nQueries,
                                                                                  recordNormals, nThreadsPerBlock,
                                                                                  stream);
        break;
    case CUDABvhType::TriangleSnch:
        launchClosestPointTyped<CUDASnchNode, CUDATriangle, CUDAEdge, true>(nodes, primitives, silhouettes,
                                                                             spheres, interactions, nQueries,
                                                                             recordNormals, nThreadsPerBlock,
                                                                             stream);
        break;
    }

    FCPW_CUDA_CHECK(cudaGetLastError());
}

void launchCudaClosestSilhouettePoint(CUDABvhType type,
                                      const void *nodes,
                                      const void *primitives,
                                      const void *silhouettes,
                                      const CUDABoundingSphere *spheres,
                                      const uint32_t *flipNormalOrientation,
                                      CUDAInteraction *interactions,
                                      uint32_t nQueries,
                                      float squaredMinRadius,
                                      float precision,
                                      uint32_t nThreadsPerBlock,
                                      cudaStream_t stream)
{
    // Type-erased dispatch to concrete kernel instantiations.
    switch (type) {
    case CUDABvhType::LineSegmentBvh:
        launchClosestSilhouetteTyped<CUDABvhNode, CUDALineSegment, CUDANoSilhouette, false>(nodes, primitives, silhouettes,
                                                                                              spheres, flipNormalOrientation,
                                                                                              interactions, nQueries,
                                                                                              squaredMinRadius, precision,
                                                                                              nThreadsPerBlock, stream);
        break;
    case CUDABvhType::TriangleBvh:
        launchClosestSilhouetteTyped<CUDABvhNode, CUDATriangle, CUDANoSilhouette, false>(nodes, primitives, silhouettes,
                                                                                           spheres, flipNormalOrientation,
                                                                                           interactions, nQueries,
                                                                                           squaredMinRadius, precision,
                                                                                           nThreadsPerBlock, stream);
        break;
    case CUDABvhType::LineSegmentSnch:
        launchClosestSilhouetteTyped<CUDASnchNode, CUDALineSegment, CUDAVertex, true>(nodes, primitives, silhouettes,
                                                                                        spheres, flipNormalOrientation,
                                                                                        interactions, nQueries,
                                                                                        squaredMinRadius, precision,
                                                                                        nThreadsPerBlock, stream);
        break;
    case CUDABvhType::TriangleSnch:
        launchClosestSilhouetteTyped<CUDASnchNode, CUDATriangle, CUDAEdge, true>(nodes, primitives, silhouettes,
                                                                                   spheres, flipNormalOrientation,
                                                                                   interactions, nQueries,
                                                                                   squaredMinRadius, precision,
                                                                                   nThreadsPerBlock, stream);
        break;
    }

    FCPW_CUDA_CHECK(cudaGetLastError());
}

void launchCudaRefit(CUDABvhType type,
                     void *nodes,
                     const void *primitives,
                     const void *silhouettes,
                     const uint32_t *nodeIndices,
                     uint32_t firstNodeOffset,
                     uint32_t nodeCount,
                     uint32_t nThreadsPerBlock,
                     cudaStream_t stream)
{
    // Type-erased dispatch to concrete kernel instantiations.
    switch (type) {
    case CUDABvhType::LineSegmentBvh:
        launchRefitTyped<CUDABvhNode, CUDALineSegment, CUDANoSilhouette, false>(nodes, primitives, silhouettes,
                                                                                  nodeIndices, firstNodeOffset,
                                                                                  nodeCount, nThreadsPerBlock, stream);
        break;
    case CUDABvhType::TriangleBvh:
        launchRefitTyped<CUDABvhNode, CUDATriangle, CUDANoSilhouette, false>(nodes, primitives, silhouettes,
                                                                               nodeIndices, firstNodeOffset,
                                                                               nodeCount, nThreadsPerBlock, stream);
        break;
    case CUDABvhType::LineSegmentSnch:
        launchRefitTyped<CUDASnchNode, CUDALineSegment, CUDAVertex, true>(nodes, primitives, silhouettes,
                                                                           nodeIndices, firstNodeOffset,
                                                                           nodeCount, nThreadsPerBlock, stream);
        break;
    case CUDABvhType::TriangleSnch:
        launchRefitTyped<CUDASnchNode, CUDATriangle, CUDAEdge, true>(nodes, primitives, silhouettes,
                                                                      nodeIndices, firstNodeOffset,
                                                                      nodeCount, nThreadsPerBlock, stream);
        break;
    }

    FCPW_CUDA_CHECK(cudaGetLastError());
}

} // namespace fcpw
