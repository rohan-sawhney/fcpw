#pragma once

#include <fcpw/gpu/cuda/cuda_interop_structures.h>
#include <cuda_runtime.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38F
#endif

#ifndef FLT_EPSILON
#define FLT_EPSILON 1.192092896e-07F
#endif

namespace fcpw {
namespace cuda {

/////////////////////////////////////////////////////////////////////////////////////////////
// Basic math operations

__device__ __forceinline__ float3 make_float3(float x, float y, float z)
{
    float3 v;
    v.x = x;
    v.y = y;
    v.z = z;
    return v;
}

__device__ __forceinline__ float3 operator+(const float3& a, const float3& b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 operator-(const float3& a, const float3& b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 operator*(const float3& a, float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __forceinline__ float3 operator*(float s, const float3& a)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __forceinline__ float3 operator*(const float3& a, const float3& b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ __forceinline__ float3 operator/(const float3& a, float s)
{
    float invS = 1.0f / s;
    return make_float3(a.x * invS, a.y * invS, a.z * invS);
}

__device__ __forceinline__ float dot(const float3& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float3 cross(const float3& a, const float3& b)
{
    return make_float3(a.y * b.z - a.z * b.y,
                       a.z * b.x - a.x * b.z,
                       a.x * b.y - a.y * b.x);
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
    if (len > FLT_EPSILON) {
        return v / len;
    }
    return make_float3(0.0f, 0.0f, 0.0f);
}

__device__ __forceinline__ float minComponent(const float3& v)
{
    return fminf(fminf(v.x, v.y), v.z);
}

__device__ __forceinline__ float maxComponent(const float3& v)
{
    return fmaxf(fmaxf(v.x, v.y), v.z);
}

__device__ __forceinline__ float3 minComponents(const float3& a, const float3& b)
{
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

__device__ __forceinline__ float3 maxComponents(const float3& a, const float3& b)
{
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

/////////////////////////////////////////////////////////////////////////////////////////////
// BoundingBox operations

__device__ __forceinline__ float3 boxCentroid(const GPUBoundingBox& box)
{
    return (box.pMin + box.pMax) * 0.5f;
}

__device__ __forceinline__ float3 boxExtent(const GPUBoundingBox& box)
{
    return box.pMax - box.pMin;
}

__device__ __forceinline__ float boxSurfaceArea(const GPUBoundingBox& box)
{
    float3 e = boxExtent(box);
    return 2.0f * (e.x * e.y + e.y * e.z + e.z * e.x);
}

__device__ __forceinline__ bool boxIntersectsRay(const GPUBoundingBox& box,
                                                  const GPURay& ray,
                                                  float& tMin, float& tMax)
{
    // slab method
    float3 invDir = make_float3(1.0f / ray.d.x, 1.0f / ray.d.y, 1.0f / ray.d.z);
    float3 t0 = (box.pMin - ray.o) * invDir;
    float3 t1 = (box.pMax - ray.o) * invDir;

    float3 tNear = minComponents(t0, t1);
    float3 tFar = maxComponents(t0, t1);

    tMin = fmaxf(fmaxf(tNear.x, tNear.y), tNear.z);
    tMax = fminf(fminf(tFar.x, tFar.y), tFar.z);

    return tMin <= tMax && tMax >= 0.0f;
}

__device__ __forceinline__ float boxDistanceSquared(const GPUBoundingBox& box,
                                                     const float3& point)
{
    // distance squared from point to box
    float3 closest = maxComponents(box.pMin, minComponents(point, box.pMax));
    float3 diff = closest - point;
    return lengthSquared(diff);
}

__device__ __forceinline__ GPUBoundingBox mergeBoundingBoxes(const GPUBoundingBox& box1,
                                                              const GPUBoundingBox& box2)
{
    GPUBoundingBox merged;
    merged.pMin = minComponents(box1.pMin, box2.pMin);
    merged.pMax = maxComponents(box1.pMax, box2.pMax);
    return merged;
}

__device__ __forceinline__ bool boxContainsPoint(const GPUBoundingBox& box,
                                                 const float3& point)
{
    return (point.x >= box.pMin.x && point.x <= box.pMax.x &&
            point.y >= box.pMin.y && point.y <= box.pMax.y &&
            point.z >= box.pMin.z && point.z <= box.pMax.z);
}

/////////////////////////////////////////////////////////////////////////////////////////////
// BoundingSphere operations

__device__ __forceinline__ bool sphereContainsPoint(const GPUBoundingSphere& sphere,
                                                    const float3& point)
{
    float3 diff = point - sphere.c;
    return lengthSquared(diff) <= sphere.r2;
}

__device__ __forceinline__ bool sphereOverlapsBox(const GPUBoundingSphere& sphere,
                                                  const GPUBoundingBox& box)
{
    float distSq = boxDistanceSquared(box, sphere.c);
    return distSq <= sphere.r2;
}

// Source: bounding-volumes.slang lines 43-53
__device__ __forceinline__ bool boxOverlapWithDistance(const GPUBoundingBox& box,
                                                       const GPUBoundingSphere& sphere,
                                                       float& d2Min, float& d2Max)
{
    float3 u = box.pMin - sphere.c;
    float3 v = sphere.c - box.pMax;
    float3 a = maxComponents(maxComponents(u, v), make_float3(0.0f, 0.0f, 0.0f));
    float3 b = minComponents(u, v);
    d2Min = dot(a, a);  // Minimum squared distance from sphere center to box
    d2Max = dot(b, b);  // Maximum squared distance from sphere center to box
    return d2Min <= sphere.r2;
}

/////////////////////////////////////////////////////////////////////////////////////////////
// BoundingCone operations (for SNCH)

__device__ __forceinline__ bool coneIsValid(const GPUBoundingCone& cone)
{
    return cone.halfAngle >= 0.0f && cone.halfAngle <= M_PI;
}

__device__ __forceinline__ bool coneHasNegativeHalfAngle(const GPUBoundingCone& cone)
{
    return cone.halfAngle < 0.0f;
}

__device__ __forceinline__ bool coneOverlapsCone(const GPUBoundingCone& cone1,
                                                 const GPUBoundingCone& cone2)
{
    // check if either cone is invalid
    if (!coneIsValid(cone1) || !coneIsValid(cone2)) {
        return false;
    }

    // check for negative half angles (no silhouettes)
    if (coneHasNegativeHalfAngle(cone1) || coneHasNegativeHalfAngle(cone2)) {
        return false;
    }

    // check if either cone has half angle >= PI (contains all directions)
    if (cone1.halfAngle >= M_PI || cone2.halfAngle >= M_PI) {
        return true;
    }

    // compute angle between cone axes
    float axisAlignment = dot(cone1.axis, cone2.axis);
    float angleBetweenAxes = acosf(fmaxf(-1.0f, fminf(1.0f, axisAlignment)));

    // cones overlap if angle between axes is less than sum of half angles
    return angleBetweenAxes <= (cone1.halfAngle + cone2.halfAngle);
}

__device__ __forceinline__ bool inRange(float val, float low, float high)
{
    return val >= low && val <= high;
}

__device__ __forceinline__ void computeOrthonormalBasis(const float3& n, float3& b1, float3& b2)
{
    // Source: https://graphics.pixar.com/library/OrthonormalB/paper.pdf
    float sign = n.z >= 0.0f ? 1.0f : -1.0f;
    float a = -1.0f / (sign + n.z);
    float b = n.x * n.y * a;

    b1 = make_float3(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
    b2 = make_float3(b, sign + n.y * n.y * a, -n.y);
}

__device__ __forceinline__ float projectToPlane(const float3& n, const float3& e)
{
    // Compute orthonormal basis
    float3 b1, b2;
    computeOrthonormalBasis(n, b1, b2);

    // Compute maximal projection radius
    float r1 = dot(e, make_float3(fabsf(b1.x), fabsf(b1.y), fabsf(b1.z)));
    float r2 = dot(e, make_float3(fabsf(b2.x), fabsf(b2.y), fabsf(b2.z)));
    return sqrtf(r1 * r1 + r2 * r2);
}

// Source: bounding-volumes.slang lines 157-214
__device__ __forceinline__ bool coneOverlapsSphere(
    const GPUBoundingCone& cone, const float3& o, const GPUBoundingBox& b,
    float distToBox, float& minAngleRange, float& maxAngleRange)
{
    // Initialize angle bounds
    minAngleRange = 0.0f;
    maxAngleRange = M_PI / 2.0f;

    // There's overlap if this cone's halfAngle is greater than 90 degrees, or
    // if the box contains the view cone origin (since the view cone is invalid)
    if (cone.halfAngle >= M_PI / 2.0f || distToBox < FLT_EPSILON) {
        return true;
    }

    // Compute the view cone axis
    float3 c = boxCentroid(b);
    float3 viewConeAxis = c - o;
    float l = length(viewConeAxis);
    viewConeAxis = viewConeAxis / l;

    // Check for overlap between the view cone axis and this cone
    float dAxisAngle = acosf(fmaxf(-1.0f, fminf(1.0f, dot(cone.axis, viewConeAxis)))); // [0, 180]
    if (inRange(M_PI / 2.0f, dAxisAngle - cone.halfAngle, dAxisAngle + cone.halfAngle)) {
        return true;
    }

    // Check if the view cone origin lies outside this cone's bounding sphere;
    // if it does, compute the view cone halfAngle and check for overlap
    if (l > cone.radius) {
        float viewConeHalfAngle = asinf(cone.radius / l);
        float halfAngleSum = cone.halfAngle + viewConeHalfAngle;
        minAngleRange = dAxisAngle - halfAngleSum;
        maxAngleRange = dAxisAngle + halfAngleSum;
        return halfAngleSum >= M_PI / 2.0f ? true : inRange(M_PI / 2.0f, minAngleRange, maxAngleRange);
    }

    // The view cone origin lies inside the box's bounding sphere, so check if
    // the plane defined by the view cone axis intersects the box; if it does, then
    // there's overlap since the view cone has a halfAngle greater than 90 degrees
    float3 e = b.pMax - c;
    float d = dot(e, make_float3(fabsf(viewConeAxis.x), fabsf(viewConeAxis.y), fabsf(viewConeAxis.z))); // max projection length onto axis
    float s = l - d;
    if (s <= 0.0f) {
        return true;
    }

    // Compute the view cone halfAngle by projecting the max extents of the box
    // onto the plane, and check for overlap
    d = projectToPlane(viewConeAxis, e);
    float viewConeHalfAngle = atan2f(d, s);
    float halfAngleSum = cone.halfAngle + viewConeHalfAngle;
    minAngleRange = dAxisAngle - halfAngleSum;
    maxAngleRange = dAxisAngle + halfAngleSum;
    return halfAngleSum >= M_PI / 2.0f ? true : inRange(M_PI / 2.0f, minAngleRange, maxAngleRange);
}

__device__ __forceinline__ GPUBoundingCone mergeBoundingCones(const GPUBoundingCone& cone1,
                                                              const GPUBoundingCone& cone2,
                                                              const float3& centroid1,
                                                              const float3& centroid2,
                                                              const float3& mergedCentroid)
{
    GPUBoundingCone merged;

    // check for invalid cones
    if (!coneIsValid(cone1) && !coneIsValid(cone2)) {
        merged.axis = make_float3(0.0f, 0.0f, 0.0f);
        merged.halfAngle = M_PI;
        merged.radius = 0.0f;
        return merged;
    }

    if (!coneIsValid(cone1)) return cone2;
    if (!coneIsValid(cone2)) return cone1;

    // check for negative half angles
    if (coneHasNegativeHalfAngle(cone1) && coneHasNegativeHalfAngle(cone2)) {
        merged.axis = make_float3(0.0f, 0.0f, 0.0f);
        merged.halfAngle = -M_PI;
        merged.radius = 0.0f;
        return merged;
    }

    if (coneHasNegativeHalfAngle(cone1)) return cone2;
    if (coneHasNegativeHalfAngle(cone2)) return cone1;

    // check if either cone has half angle >= PI
    if (cone1.halfAngle >= M_PI || cone2.halfAngle >= M_PI) {
        merged.axis = make_float3(0.0f, 0.0f, 0.0f);
        merged.halfAngle = M_PI;
        merged.radius = fmaxf(cone1.radius, cone2.radius) +
                       length(mergedCentroid - (cone1.radius > cone2.radius ? centroid1 : centroid2));
        return merged;
    }

    // compute merged axis as average of cone axes
    float3 axis = normalize(cone1.axis + cone2.axis);
    float axisLen = length(cone1.axis + cone2.axis);

    if (axisLen < FLT_EPSILON) {
        // axes are opposite, cone covers all directions
        merged.axis = make_float3(0.0f, 0.0f, 0.0f);
        merged.halfAngle = M_PI;
        merged.radius = fmaxf(cone1.radius, cone2.radius) +
                       length(mergedCentroid - (cone1.radius > cone2.radius ? centroid1 : centroid2));
        return merged;
    }

    // compute half angle as max angle to any of the original cone directions
    float halfAngle = 0.0f;
    float angle1 = acosf(fmaxf(-1.0f, fminf(1.0f, dot(axis, cone1.axis))));
    float angle2 = acosf(fmaxf(-1.0f, fminf(1.0f, dot(axis, cone2.axis))));
    halfAngle = fmaxf(angle1 + cone1.halfAngle, angle2 + cone2.halfAngle);

    // compute merged radius
    float r1 = cone1.radius + length(centroid1 - mergedCentroid);
    float r2 = cone2.radius + length(centroid2 - mergedCentroid);
    float radius = fmaxf(r1, r2);

    merged.axis = axis;
    merged.halfAngle = fminf(halfAngle, M_PI);
    merged.radius = radius;

    return merged;
}

} // namespace cuda
} // namespace fcpw
