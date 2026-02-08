#pragma once

#include <cuda_runtime.h>
#include <cfloat>
#include <fcpw/gpu/cuda/cuda_interop_structures.h>

namespace fcpw {

/////////////////////////////////////////////////////////////////////////////////////////////
// Helper functions (from bounding-volumes.slang and math functions)

__device__ __forceinline__ float3 cross(const float3& a, const float3& b)
{
    return make_float3(a.y * b.z - a.z * b.y,
                      a.z * b.x - a.x * b.z,
                      a.x * b.y - a.y * b.x);
}

__device__ __forceinline__ float dot(const float3& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float lengthSquared(const float3& v)
{
    return dot(v, v);
}

__device__ __forceinline__ float length(const float3& v)
{
    return sqrtf(lengthSquared(v));
}

__device__ __forceinline__ float3 normalize(const float3& v)
{
    float len = length(v);
    return make_float3(v.x / len, v.y / len, v.z / len);
}

__device__ __forceinline__ float3 operator+(const float3& a, const float3& b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 operator-(const float3& a, const float3& b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 operator*(const float3& v, float s)
{
    return make_float3(v.x * s, v.y * s, v.z * s);
}

__device__ __forceinline__ float3 operator*(float s, const float3& v)
{
    return v * s;
}

__device__ __forceinline__ float3 operator*(const float3& a, const float3& b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ __forceinline__ float3 operator/(const float3& v, float s)
{
    return make_float3(v.x / s, v.y / s, v.z / s);
}

__device__ __forceinline__ float2 operator+(const float2& a, const float2& b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}

/////////////////////////////////////////////////////////////////////////////////////////////
// Line Segment operations (from geometry.slang)

// Source: geometry.slang lines 37-77
__device__ __forceinline__ bool lineSegmentIntersectsRay(
    const float3& pa, const float3& pb,
    const float3& ro, const float3& rd, float rtMax, bool checkForOcclusion,
    float3& p, float3& n, float2& uv, float& d)
{
    float3 u = pa - ro;
    float3 v = pb - pa;

    // return if line segment and ray are parallel
    float dv = cross(rd, v).z;
    if (fabsf(dv) <= FLT_EPSILON) {
        return false;
    }

    // solve ro + t*rd = pa + s*(pb - pa) for t >= 0 && 0 <= s <= 1
    // s = (u x rd)/(rd x v)
    float ud = cross(u, rd).z;
    float s = ud / dv;

    if (s >= 0.0f && s <= 1.0f) {
        // t = (u x v)/(rd x v)
        float t = cross(u, v).z / dv;

        if (t >= 0.0f && t <= rtMax) {
            if (checkForOcclusion) {
                return true;
            }

            p = pa + s * v;
            n = normalize(make_float3(v.y, -v.x, 0.0f));
            uv = make_float2(s, 0.0f);
            d = t;
            return true;
        }
    }

    return false;
}

// Source: geometry.slang lines 79-103
__device__ __forceinline__ float lineSegmentFindClosestPoint(
    const float3& pa, const float3& pb, const float3& x,
    float3& p, float& t)
{
    float3 u = pb - pa;
    float3 v = x - pa;

    float c1 = dot(u, v);
    if (c1 <= 0.0f) {
        t = 0.0f;
        p = pa;
        return length(x - p);
    }

    float c2 = dot(u, u);
    if (c2 <= c1) {
        t = 1.0f;
        p = pb;
        return length(x - p);
    }

    t = c1 / c2;
    p = pa + u * t;
    return length(x - p);
}

__device__ __forceinline__ bool lineSegmentIntersectsSphere(const GPULineSegment& seg,
                                                            const GPUBoundingSphere& sphere)
{
    float3 p;
    float t;
    float d = lineSegmentFindClosestPoint(seg.pa, seg.pb, sphere.c, p, t);
    return (d * d <= sphere.r2);
}

__device__ __forceinline__ float lineSegmentFindClosestPoint(const GPULineSegment& seg,
                                                             const float3& queryPoint,
                                                             GPUInteraction& interaction)
{
    float t;
    float d = lineSegmentFindClosestPoint(seg.pa, seg.pb, queryPoint, interaction.p, t);

    interaction.uv = make_float2(t, 0.0f);
    interaction.d = d;
    interaction.index = seg.index;

    // Compute normal (perpendicular to segment)
    float3 s = seg.pb - seg.pa;
    float len = length(s);
    interaction.n = make_float3(s.y / len, -s.x / len, 0.0f);

    return d * d;
}

__device__ __forceinline__ float3 lineSegmentSample(const GPULineSegment& seg,
                                                    const float3& randNum)
{
    // Source: geometry.slang lines 183-193
    float3 s = seg.pb - seg.pa;
    float u = randNum.x;
    return seg.pa + u * s;
}

__device__ __forceinline__ float lineSegmentSurfaceArea(const GPULineSegment& seg)
{
    // Source: geometry.slang line 136
    return length(seg.pb - seg.pa);
}

/////////////////////////////////////////////////////////////////////////////////////////////
// Triangle operations (from geometry.slang)

// Source: geometry.slang lines 202-249 (Möller–Trumbore intersection algorithm)
__device__ __forceinline__ bool triangleIntersectsRay(
    const float3& pa, const float3& pb, const float3& pc,
    GPURay& ray, bool checkForOcclusion,
    float3& p, float3& n, float2& uv, float& d)
{
    // Möller–Trumbore intersection algorithm
    float3 v1 = pb - pa;
    float3 v2 = pc - pa;
    float3 q = cross(ray.d, v2);
    float det = dot(v1, q);

    // ray and triangle are parallel if det is close to 0
    if (fabsf(det) <= FLT_EPSILON) {
        return false;
    }
    float invDet = 1.0f / det;

    float3 r = ray.o - pa;
    float v = dot(r, q) * invDet;
    if (v < 0.0f || v > 1.0f) {
        return false;
    }

    float3 s = cross(r, v1);
    float w = dot(ray.d, s) * invDet;
    if (w < 0.0f || v + w > 1.0f) {
        return false;
    }

    float t = dot(v2, s) * invDet;
    if (t >= 0.0f && t <= ray.tMax) {
        if (checkForOcclusion) {
            return true;
        }

        p = pa + v1 * v + v2 * w;
        n = normalize(cross(v1, v2));
        uv = make_float2(1.0f - v - w, v);
        d = t;
        return true;
    }

    return false;
}

// Source: geometry.slang lines 251-332 (Real-Time Collision Detection algorithm)
__device__ __forceinline__ float triangleFindClosestPoint(
    const float3& pa, const float3& pb, const float3& pc, const float3& x,
    float3& p, float2& t)
{
    // source: real time collision detection
    // check if x in vertex region outside pa
    float3 ab = pb - pa;
    float3 ac = pc - pa;
    float3 ax = x - pa;
    float d1 = dot(ab, ax);
    float d2 = dot(ac, ax);
    if (d1 <= 0.0f && d2 <= 0.0f) {
        // barycentric coordinates (1, 0, 0)
        t = make_float2(1.0f, 0.0f);
        p = pa;
        return length(x - p);
    }

    // check if x in vertex region outside pb
    float3 bx = x - pb;
    float d3 = dot(ab, bx);
    float d4 = dot(ac, bx);
    if (d3 >= 0.0f && d4 <= d3) {
        // barycentric coordinates (0, 1, 0)
        t = make_float2(0.0f, 1.0f);
        p = pb;
        return length(x - p);
    }

    // check if x in vertex region outside pc
    float3 cx = x - pc;
    float d5 = dot(ab, cx);
    float d6 = dot(ac, cx);
    if (d6 >= 0.0f && d5 <= d6) {
        // barycentric coordinates (0, 0, 1)
        t = make_float2(0.0f, 0.0f);
        p = pc;
        return length(x - p);
    }

    // check if x in edge region of ab, if so return projection of x onto ab
    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        // barycentric coordinates (1 - v, v, 0)
        float v = d1 / (d1 - d3);
        t = make_float2(1.0f - v, v);
        p = pa + ab * v;
        return length(x - p);
    }

    // check if x in edge region of ac, if so return projection of x onto ac
    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        // barycentric coordinates (1 - w, 0, w)
        float w = d2 / (d2 - d6);
        t = make_float2(1.0f - w, 0.0f);
        p = pa + ac * w;
        return length(x - p);
    }

    // check if x in edge region of bc, if so return projection of x onto bc
    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        // barycentric coordinates (0, 1 - w, w)
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        t = make_float2(0.0f, 1.0f - w);
        p = pb + (pc - pb) * w;
        return length(x - p);
    }

    // x inside face region. Compute p through its barycentric coordinates (u, v, w)
    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom;
    float w = vc * denom;
    t = make_float2(1.0f - v - w, v);
    p = pa + ab * v + ac * w; //= u*a + v*b + w*c, u = va*denom = 1.0f - v - w
    return length(x - p);
}

__device__ __forceinline__ bool triangleIntersectsRay(const GPUTriangle& tri, GPURay& ray,
                                                      GPUInteraction& interaction)
{
    float3 p, n;
    float2 uv;
    float d;
    bool hit = triangleIntersectsRay(tri.pa, tri.pb, tri.pc, ray, false, p, n, uv, d);

    if (hit) {
        interaction.p = p;
        interaction.n = n;
        interaction.uv = uv;
        interaction.d = d;
        interaction.index = tri.index;
        ray.tMax = d;
    }

    return hit;
}

__device__ __forceinline__ bool triangleIntersectsSphere(const GPUTriangle& tri,
                                                         const GPUBoundingSphere& sphere)
{
    float3 p;
    float2 t;
    float d = triangleFindClosestPoint(tri.pa, tri.pb, tri.pc, sphere.c, p, t);
    return (d * d <= sphere.r2);
}

__device__ __forceinline__ float triangleFindClosestPoint(const GPUTriangle& tri,
                                                          const float3& queryPoint,
                                                          GPUInteraction& interaction)
{
    float d = triangleFindClosestPoint(tri.pa, tri.pb, tri.pc, queryPoint,
                                       interaction.p, interaction.uv);

    interaction.d = d;
    interaction.index = tri.index;

    // Compute normal
    float3 v1 = tri.pb - tri.pa;
    float3 v2 = tri.pc - tri.pa;
    interaction.n = normalize(cross(v1, v2));

    return d * d;
}

__device__ __forceinline__ float3 triangleSample(const GPUTriangle& tri,
                                                 const float3& randNum)
{
    // Source: geometry.slang lines 410-424 (proper uniform sampling)
    float u1 = sqrtf(randNum.x);
    float u2 = randNum.y;
    float u = 1.0f - u1;
    float v = u2 * u1;
    float w = 1.0f - u - v;
    return tri.pa * u + tri.pb * v + tri.pc * w;
}

__device__ __forceinline__ float triangleSurfaceArea(const GPUTriangle& tri)
{
    // Source: geometry.slang line 357
    return 0.5f * length(cross(tri.pb - tri.pa, tri.pc - tri.pa));
}

/////////////////////////////////////////////////////////////////////////////////////////////
// Silhouette operations (from geometry.slang)

// Source: geometry.slang lines 490-522
__device__ __forceinline__ bool isSilhouetteVertex(
    const float3& n0, const float3& n1, const float3& viewDir, float d,
    bool flipNormalOrientation, float precision)
{
    float sign = flipNormalOrientation ? 1.0f : -1.0f;

    // Vertex is a silhouette point if it is concave and the query point lies on the vertex
    if (d <= precision) {
        float det = n0.x * n1.y - n1.x * n0.y;
        return sign * det > precision;
    }

    // Vertex is a silhouette point if the query point lies on the halfplane
    // defined by an adjacent line segment and the other segment is backfacing
    float3 viewDirUnit = viewDir / d;
    float dot0 = dot(viewDirUnit, n0);
    float dot1 = dot(viewDirUnit, n1);

    bool isZeroDot0 = fabsf(dot0) <= precision;
    if (isZeroDot0) {
        return sign * dot1 > precision;
    }

    bool isZeroDot1 = fabsf(dot1) <= precision;
    if (isZeroDot1) {
        return sign * dot0 > precision;
    }

    // Vertex is a silhouette point if an adjacent line segment is frontfacing
    // w.r.t. the query point and the other segment is backfacing
    return dot0 * dot1 < 0.0f;
}

// Source: geometry.slang lines 599-633
__device__ __forceinline__ bool isSilhouetteEdge(
    const float3& pa, const float3& pb, const float3& n0, const float3& n1,
    const float3& viewDir, float d, bool flipNormalOrientation, float precision)
{
    float sign = flipNormalOrientation ? 1.0f : -1.0f;

    // Edge is a silhouette if it is concave and the query point lies on the edge
    if (d <= precision) {
        float3 edgeDir = normalize(pb - pa);
        float signedDihedralAngle = atan2f(dot(edgeDir, cross(n0, n1)), dot(n0, n1));
        return sign * signedDihedralAngle > precision;
    }

    // Edge is a silhouette if the query point lies on the halfplane defined
    // by an adjacent triangle and the other triangle is backfacing
    float3 viewDirUnit = viewDir / d;
    float dot0 = dot(viewDirUnit, n0);
    float dot1 = dot(viewDirUnit, n1);

    bool isZeroDot0 = fabsf(dot0) <= precision;
    if (isZeroDot0) {
        return sign * dot1 > precision;
    }

    bool isZeroDot1 = fabsf(dot1) <= precision;
    if (isZeroDot1) {
        return sign * dot0 > precision;
    }

    // Edge is a silhouette if an adjacent triangle is frontfacing w.r.t. the
    // query point and the other triangle is backfacing
    return dot0 * dot1 < 0.0f;
}

// Source: geometry.slang lines 555-590 (Vertex::findClosestSilhouettePoint)
__device__ __forceinline__ bool vertexFindClosestSilhouettePoint(
    const GPUVertex& vertex, const GPUBoundingSphere& sphere,
    bool flipNormalOrientation, float squaredMinRadius, float precision,
    GPUInteraction& interaction)
{
    if (squaredMinRadius >= sphere.r2) {
        return false;
    }

    // Compute view direction
    float3 viewDir = sphere.c - vertex.p;
    float d = length(viewDir);
    if (d * d > sphere.r2) {
        return false;
    }

    // Check if vertex is a silhouette point from view direction
    bool process = (vertex.hasOneAdjacentFace == 1) ? true : false;
    if (!process) {
        process = isSilhouetteVertex(vertex.n0, vertex.n1, viewDir, d, flipNormalOrientation, precision);
    }

    if (process && d * d <= sphere.r2) {
        interaction.p = vertex.p;
        interaction.uv = make_float2(0.0f, 0.0f);
        interaction.d = d;
        interaction.index = vertex.index;
        return true;
    }

    return false;
}

// Source: geometry.slang lines 667-701 (Edge::findClosestSilhouettePoint)
__device__ __forceinline__ bool edgeFindClosestSilhouettePoint(
    const GPUEdge& edge, const GPUBoundingSphere& sphere,
    bool flipNormalOrientation, float squaredMinRadius, float precision,
    GPUInteraction& interaction)
{
    if (squaredMinRadius >= sphere.r2) {
        return false;
    }

    // Compute view direction
    float t;
    float d = lineSegmentFindClosestPoint(edge.pa, edge.pb, sphere.c, interaction.p, t);
    if (d * d > sphere.r2) {
        return false;
    }

    // Check if edge is a silhouette from view direction
    bool process = (edge.hasOneAdjacentFace == 1) ? true : false;
    if (!process) {
        float3 viewDir = sphere.c - interaction.p;
        process = isSilhouetteEdge(edge.pa, edge.pb, edge.n0, edge.n1, viewDir, d, flipNormalOrientation, precision);
    }

    if (process && d * d <= sphere.r2) {
        interaction.uv = make_float2(t, 0.0f);
        interaction.d = d;
        interaction.index = edge.index;
        return true;
    }

    return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////
// Bounding box computation for primitives

__device__ __forceinline__ GPUBoundingBox lineSegmentGetBoundingBox(const GPULineSegment& seg)
{
    GPUBoundingBox box;
    box.pMin = make_float3(fminf(seg.pa.x, seg.pb.x),
                           fminf(seg.pa.y, seg.pb.y),
                           fminf(seg.pa.z, seg.pb.z));
    box.pMax = make_float3(fmaxf(seg.pa.x, seg.pb.x),
                           fmaxf(seg.pa.y, seg.pb.y),
                           fmaxf(seg.pa.z, seg.pb.z));
    return box;
}

__device__ __forceinline__ GPUBoundingBox triangleGetBoundingBox(const GPUTriangle& tri)
{
    GPUBoundingBox box;
    box.pMin = make_float3(fminf(fminf(tri.pa.x, tri.pb.x), tri.pc.x),
                           fminf(fminf(tri.pa.y, tri.pb.y), tri.pc.y),
                           fminf(fminf(tri.pa.z, tri.pb.z), tri.pc.z));
    box.pMax = make_float3(fmaxf(fmaxf(tri.pa.x, tri.pb.x), tri.pc.x),
                           fmaxf(fmaxf(tri.pa.y, tri.pb.y), tri.pc.y),
                           fmaxf(fmaxf(tri.pa.z, tri.pb.z), tri.pc.z));
    return box;
}

__device__ __forceinline__ GPUBoundingBox vertexGetBoundingBox(const GPUVertex& vertex)
{
    GPUBoundingBox box;
    box.pMin = vertex.p;
    box.pMax = vertex.p;
    return box;
}

__device__ __forceinline__ GPUBoundingBox edgeGetBoundingBox(const GPUEdge& edge)
{
    GPUBoundingBox box;
    box.pMin = make_float3(fminf(edge.pa.x, edge.pb.x),
                           fminf(edge.pa.y, edge.pb.y),
                           fminf(edge.pa.z, edge.pb.z));
    box.pMax = make_float3(fmaxf(edge.pa.x, edge.pb.x),
                           fmaxf(edge.pa.y, edge.pb.y),
                           fmaxf(edge.pa.z, edge.pb.z));
    return box;
}

/////////////////////////////////////////////////////////////////////////////////////////////
// Centroid and normal accessors for silhouettes

__device__ __forceinline__ float3 vertexGetCentroid(const GPUVertex& vertex)
{
    return vertex.p;
}

__device__ __forceinline__ float3 edgeGetCentroid(const GPUEdge& edge)
{
    return (edge.pa + edge.pb) * 0.5f;
}

__device__ __forceinline__ float3 vertexGetNormal(const GPUVertex& vertex, uint32_t fIndex)
{
    return (fIndex == 0) ? vertex.n0 : vertex.n1;
}

__device__ __forceinline__ float3 edgeGetNormal(const GPUEdge& edge, uint32_t fIndex)
{
    return (fIndex == 0) ? edge.n0 : edge.n1;
}

__device__ __forceinline__ bool vertexHasTwoAdjacentFaces(const GPUVertex& vertex)
{
    return vertex.hasOneAdjacentFace == 0;
}

__device__ __forceinline__ bool edgeHasTwoAdjacentFaces(const GPUEdge& edge)
{
    return edge.hasOneAdjacentFace == 0;
}

} // namespace fcpw
