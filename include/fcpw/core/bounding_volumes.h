#pragma once

#include <fcpw/core/ray.h>

namespace fcpw {

template<size_t DIM>
struct BoundingSphere {
    // constructor
    BoundingSphere(const Vector<DIM>& c_, float r2_): c(c_), r2(r2_) {}

    // computes transformed sphere
    BoundingSphere<DIM> transform(const Transform<DIM>& t) const {
        Vector<DIM> tc = t*c;
        float tr2 = maxFloat;

        if (r2 < maxFloat) {
            Vector<DIM> direction = Vector<DIM>::Zero();
            direction[0] = 1;
            tr2 = (t*(c + direction*std::sqrt(r2)) - tc).squaredNorm();
        }

        return BoundingSphere<DIM>(tc, tr2);
    }

    // members
    Vector<DIM> c;
    float r2;
};

template<size_t DIM>
struct BoundingBox {
    // constructor
    BoundingBox(): pMin(Vector<DIM>::Constant(maxFloat)),
                   pMax(Vector<DIM>::Constant(minFloat)) {}

    // constructor
    BoundingBox(const Vector<DIM>& p) {
        Vector<DIM> epsilonVector = Vector<DIM>::Constant(epsilon);
        pMin = p - epsilonVector;
        pMax = p + epsilonVector;
    }

    // expands volume to include point
    void expandToInclude(const Vector<DIM>& p) {
        Vector<DIM> epsilonVector = Vector<DIM>::Constant(epsilon);
        pMin = pMin.cwiseMin(p - epsilonVector);
        pMax = pMax.cwiseMax(p + epsilonVector);
    }

    // expands volume to include box
    void expandToInclude(const BoundingBox<DIM>& b) {
        pMin = pMin.cwiseMin(b.pMin);
        pMax = pMax.cwiseMax(b.pMax);
    }

    // returns box extent
    Vector<DIM> extent() const {
        return pMax - pMin;
    }

    // computes min and max squared distance to point;
    // min squared distance is 0 if point is inside box
    void computeSquaredDistance(const Vector<DIM>& p, float& d2Min, float& d2Max) const {
        Vector<DIM> u = pMin - p;
        Vector<DIM> v = p - pMax;
        d2Min = u.cwiseMax(v).cwiseMax(0.0f).squaredNorm();
        d2Max = u.cwiseMin(v).squaredNorm();
    }

    // checks whether box contains point
    bool contains(const Vector<DIM>& p) const {
        return (p.array() >= pMin.array()).all() &&
               (p.array() <= pMax.array()).all();
    }

    // checks for overlap with sphere
    bool overlap(const BoundingSphere<DIM>& s, float& d2Min, float& d2Max) const {
        computeSquaredDistance(s.c, d2Min, d2Max);
        return d2Min <= s.r2;
    }

    // checks for overlap with sphere
    bool overlap(const BoundingSphere<DIM>& s, float& d2Min) const {
        Vector<DIM> u = pMin - s.c;
        Vector<DIM> v = s.c - pMax;
        d2Min = u.cwiseMax(v).cwiseMax(0.0f).squaredNorm();

        return d2Min <= s.r2;
    }

    // checks for overlap with bounding box
    bool overlap(const BoundingBox<DIM>& b) const {
        return (b.pMax.array() >= pMin.array()).all() &&
               (b.pMin.array() <= pMax.array()).all();
    }

    // checks for ray intersection
    bool intersect(const Ray<DIM>& r, float& tMin, float& tMax) const {
        // slab test for ray box intersection
        // source: http://www.jcgt.org/published/0007/03/04/paper-lowres.pdf
        Vector<DIM> t0 = (pMin - r.o).cwiseProduct(r.invD);
        Vector<DIM> t1 = (pMax - r.o).cwiseProduct(r.invD);
        Vector<DIM> tNear = t0.cwiseMin(t1);
        Vector<DIM> tFar = t0.cwiseMax(t1);

        float tNearMax = std::max(0.0f, tNear.maxCoeff());
        float tFarMin = std::min(r.tMax, tFar.minCoeff());
        if (tNearMax > tFarMin) return false;

        tMin = tNearMax;
        tMax = tFarMin;
        return true;
    }

    // checks for ray intersection (generic fallback for non-3D)
    bool intersectConservative(const Ray<DIM>& r, float& tMin, float& tMax) const {
        // Fall back to regular intersection for non-3D
        return intersect(r, tMin, tMax);
    }

    // checks whether bounding box is valid
    bool isValid() const {
        return (pMax.array() >= pMin.array()).all();
    }

    // returns max dimension
    int maxDimension() const {
        int index;
        float maxLength = (pMax - pMin).maxCoeff(&index);

        return index;
    }

    // returns centroid
    Vector<DIM> centroid() const {
        return (pMin + pMax)*0.5f;
    }

    // returns surface area
    float surfaceArea() const {
        Vector<DIM> e = extent().cwiseMax(1e-5f); // the 1e-5f is to prevent division by zero
        return 2.0f*Vector<DIM>::Constant(e.prod()).cwiseQuotient(e).sum();
    }

    // returns volume
    float volume() const {
        return extent().prod();
    }

    // computes bounding sphere
    BoundingSphere<DIM> boundingSphere() const {
        Vector<DIM> c = centroid();
        Vector<DIM> e = pMax - c;

        return BoundingSphere<DIM>(c, e.squaredNorm());
    }

    // computes transformed box
    BoundingBox<DIM> transform(const Transform<DIM>& t) const {
        BoundingBox<DIM> b;
        int nCorners = 1 << DIM;

        for (int i = 0; i < nCorners; i++) {
            Vector<DIM> p = Vector<DIM>::Zero();
            int temp = i;

            for (size_t j = 0; j < DIM; j++) {
                int idx = temp%2;
                p[j] = idx == 0 ? pMin[j] : pMax[j];
                temp /= 2;
            }

            b.expandToInclude(t*p);
        }

        return b;
    }

    // returns the intersection of two bounding boxes
    BoundingBox<DIM> intersect(const BoundingBox<DIM>& b) const {
        BoundingBox<DIM> bIntersect;
        bIntersect.pMin = pMin.cwiseMax(b.pMin);
        bIntersect.pMax = pMax.cwiseMin(b.pMax);

        return bIntersect;
    }

    // members
    Vector<DIM> pMin, pMax;
};

// Conservative ray-box intersection for watertight ray tracing
// Based on: Woop, Benthin, Wald. "Watertight Ray/Triangle Intersection" JCGT 2013
// This ensures no triangles are missed during BVH traversal when using watertight triangle intersection
inline bool intersectBoxConservative(const BoundingBox<3>& box, const Ray<3>& r, float& tMin, float& tMax)
{
    const WatertightRayData& wd = r.getWatertightData();
    const int kx = wd.kx;
    const int ky = wd.ky;
    const int kz = wd.kz;

    // Conservative rounding constants
    constexpr float p = 1.0f + 1.1920929e-7f;  // 1 + 2^-23
    constexpr float m = 1.0f - 1.1920929e-7f;  // 1 - 2^-23

    // Helper functions for conservative rounding
    auto up = [](float a) { return a > 0.0f ? a * p : a * m; };
    auto dn = [](float a) { return a > 0.0f ? a * m : a * p; };
    auto Up = [](float a) { return a * p; };
    auto Dn = [](float a) { return a * m; };

    // Get box bounds as array for indexed access
    // box stored as [pMin.x, pMin.y, pMin.z, pMax.x, pMax.y, pMax.z]
    float boxBounds[6] = {
        box.pMin[0], box.pMin[1], box.pMin[2],
        box.pMax[0], box.pMax[1], box.pMax[2]
    };

    // Calculate corrected origin for near- and far-plane distance calculations
    constexpr float eps = 5.0f * 5.9604645e-8f;  // 5 * 2^-24

    float lower_kx = Dn(std::fabs(r.o[kx] - box.pMin[kx]));
    float lower_ky = Dn(std::fabs(r.o[ky] - box.pMin[ky]));
    float lower_kz = Dn(std::fabs(r.o[kz] - box.pMin[kz]));
    float upper_kx = Up(std::fabs(r.o[kx] - box.pMax[kx]));
    float upper_ky = Up(std::fabs(r.o[ky] - box.pMax[ky]));
    float upper_kz = Up(std::fabs(r.o[kz] - box.pMax[kz]));

    float max_z = std::max(lower_kz, upper_kz);

    float err_near_x = Up(lower_kx + max_z);
    float err_near_y = Up(lower_ky + max_z);
    float org_near_x = up(r.o[kx] + Up(eps * err_near_x));
    float org_near_y = up(r.o[ky] + Up(eps * err_near_y));
    float org_near_z = r.o[kz];

    float err_far_x = Up(upper_kx + max_z);
    float err_far_y = Up(upper_ky + max_z);
    float org_far_x = dn(r.o[kx] - Up(eps * err_far_x));
    float org_far_y = dn(r.o[ky] - Up(eps * err_far_y));
    float org_far_z = r.o[kz];

    // Swap near/far origins based on direction signs
    if (r.d[kx] < 0.0f) std::swap(org_near_x, org_far_x);
    if (r.d[ky] < 0.0f) std::swap(org_near_y, org_far_y);

    // Get conservative reciprocal directions
    float rdir_near_x = wd.rdirNear[0];
    float rdir_near_y = wd.rdirNear[1];
    float rdir_near_z = wd.rdirNear[2];
    float rdir_far_x = wd.rdirFar[0];
    float rdir_far_y = wd.rdirFar[1];
    float rdir_far_z = wd.rdirFar[2];

    // Calculate near/far t values
    float tNearX = (boxBounds[wd.nearX] - org_near_x) * rdir_near_x;
    float tNearY = (boxBounds[wd.nearY] - org_near_y) * rdir_near_y;
    float tNearZ = (boxBounds[wd.nearZ] - org_near_z) * rdir_near_z;
    float tFarX = (boxBounds[wd.farX] - org_far_x) * rdir_far_x;
    float tFarY = (boxBounds[wd.farY] - org_far_y) * rdir_far_y;
    float tFarZ = (boxBounds[wd.farZ] - org_far_z) * rdir_far_z;

    float tNearMax = std::max(std::max(tNearX, tNearY), std::max(tNearZ, 0.0f));
    float tFarMin = std::min(std::min(tFarX, tFarY), std::min(tFarZ, r.tMax));

    if (tNearMax > tFarMin) return false;

    tMin = tNearMax;
    tMax = tFarMin;
    return true;
}

inline void computeOrthonormalBasis(const Vector3& n, Vector3& b1, Vector3& b2)
{
    // source: https://graphics.pixar.com/library/OrthonormalB/paper.pdf
    float sign = std::copysignf(1.0f, n[2]);
    const float a = -1.0f/(sign + n[2]);
    const float b = n[0]*n[1]*a;

    b1 = Vector3(1.0f + sign*n[0]*n[0]*a, sign*b, -sign*n[0]);
    b2 = Vector3(b, sign + n[1]*n[1]*a, -n[1]);
}

template<size_t DIM>
inline float projectToPlane(const Vector<DIM>& n, const Vector<DIM>& e)
{
    std::cerr << "projectToPlane(): DIM: " << DIM << " not supported" << std::endl;
    exit(EXIT_FAILURE);

    return 0.0f;
}

template<>
inline float projectToPlane<2>(const Vector2& n, const Vector2& e)
{
    // compute orthonormal basis
    Vector2 b(-n[1], n[0]);

    // compute maximal projection radius
    float r = e.dot(b.cwiseAbs());
    return std::abs(r);
}

template<>
inline float projectToPlane<3>(const Vector3& n, const Vector3& e)
{
    // compute orthonormal basis
    Vector3 b1, b2;
    computeOrthonormalBasis(n, b1, b2);

    // compute maximal projection radius
    float r1 = e.dot(b1.cwiseAbs());
    float r2 = e.dot(b2.cwiseAbs());
    return std::sqrt(r1*r1 + r2*r2);
}

template<size_t DIM>
struct BoundingCone {
    // constructor
    BoundingCone(): axis(Vector<DIM>::Zero()), halfAngle(M_PI), radius(0.0f) {}

    // constructor
    BoundingCone(const Vector<DIM>& axis_, float halfAngle_, float radius_):
                 axis(axis_), halfAngle(halfAngle_), radius(radius_) {}

    // check for overlap between this cone and the "view" cone defined by the given
    // point o and bounding box b; the two cones overlap when there exist two vectors,
    // one in each cone, that are orthogonal to each other.
    // NOTE: Tighter view cone construction is available in Falcor, but doesn't seem to help:
    // https://github.com/NVIDIAGameWorks/Falcor/blob/master/Source/Falcor/Utils/Geometry/GeometryHelpers.slang
    bool overlap(const Vector<DIM>& o, const BoundingBox<DIM>& b, float distToBox,
                 float& minAngleRange, float& maxAngleRange) const {
        // initialize angle bounds
        minAngleRange = 0.0f;
        maxAngleRange = M_PI_2;

        // there's overlap if this cone's halfAngle is greater than 90 degrees, or
        // if the box contains the view cone origin (since the view cone is invalid)
        if (halfAngle >= M_PI_2 || distToBox < epsilon) return true;

        // compute the view cone axis
        Vector<DIM> c = b.centroid();
        Vector<DIM> viewConeAxis = c - o;
        float l = viewConeAxis.norm();
        viewConeAxis /= l;

        // check for overlap between the view cone axis and this cone
        float dAxisAngle = std::acos(std::max(-1.0f, std::min(1.0f, axis.dot(viewConeAxis)))); // [0, 180]
        if (inRange(M_PI_2, dAxisAngle - halfAngle, dAxisAngle + halfAngle)) return true;

        // check if the view cone origin lies outside this cone's bounding sphere;
        // if it does, compute the view cone halfAngle and check for overlap
        if (l > radius) {
            float viewConeHalfAngle = std::asin(radius/l);
            float halfAngleSum = halfAngle + viewConeHalfAngle;
            minAngleRange = dAxisAngle - halfAngleSum;
            maxAngleRange = dAxisAngle + halfAngleSum;
            return halfAngleSum >= M_PI_2 ? true : inRange(M_PI_2, minAngleRange, maxAngleRange);
        }

        // the view cone origin lies inside the box's bounding sphere, so check if
        // the plane defined by the view cone axis intersects the box; if it does, then
        // there's overlap since the view cone has a halfAngle greater than 90 degrees
        Vector<DIM> e = b.pMax - c;
        float d = e.dot(viewConeAxis.cwiseAbs()); // max projection length onto axis
        float s = l - d;
        if (s <= 0.0f) return true;

        // compute the view cone halfAngle by projecting the max extents of the box
        // onto the plane, and check for overlap
        d = projectToPlane<DIM>(viewConeAxis, e);
        float viewConeHalfAngle = std::atan2(d, s);
        float halfAngleSum = halfAngle + viewConeHalfAngle;
        minAngleRange = dAxisAngle - halfAngleSum;
        maxAngleRange = dAxisAngle + halfAngleSum;
        return halfAngleSum >= M_PI_2 ? true : inRange(M_PI_2, minAngleRange, maxAngleRange);
    }

    // returns whether cone is valid
    bool isValid() const {
        return halfAngle >= 0.0f;
    }

    // members
    Vector<DIM> axis;
    float halfAngle;
    float radius;
};

template<size_t DIM>
inline BoundingCone<DIM> mergeBoundingCones(const BoundingCone<DIM>& coneA,
                                            const BoundingCone<DIM>& coneB,
                                            const Vector<DIM>& originA,
                                            const Vector<DIM>& originB,
                                            const Vector<DIM>& newOrigin)
{
    BoundingCone<DIM> cone;
    if (coneA.isValid() && coneB.isValid()) {
        Vector<DIM> axisA = coneA.axis;
        Vector<DIM> axisB = coneB.axis;
        float halfAngleA = coneA.halfAngle;
        float halfAngleB = coneB.halfAngle;
        Vector<DIM> dOriginA = newOrigin - originA;
        Vector<DIM> dOriginB = newOrigin - originB;
        cone.radius = std::sqrt(std::max(coneA.radius*coneA.radius + dOriginA.squaredNorm(),
                                         coneB.radius*coneB.radius + dOriginB.squaredNorm()));

        if (halfAngleB > halfAngleA) {
            std::swap(axisA, axisB);
            std::swap(halfAngleA, halfAngleB);
        }

        float theta = std::acos(std::max(-1.0f, std::min(1.0f, axisA.dot(axisB))));
        if (std::min(theta + halfAngleB, static_cast<float>(M_PI)) <= halfAngleA) {
            // right cone is completely inside left cone
            cone.axis = axisA;
            cone.halfAngle = halfAngleA;
            return cone;
        }

        // merge cones by first computing the spread angle of the cone to cover both cones
        float oTheta = (halfAngleA + theta + halfAngleB)/2.0f;
        if (oTheta >= M_PI) {
            cone.axis = axisA;
            return cone;
        }

        float rTheta = oTheta - halfAngleA;
        cone.axis = rotate<DIM>(axisA, axisB, rTheta);
        cone.halfAngle = oTheta;

    } else if (coneA.isValid()) {
        cone = coneA;

    } else if (coneB.isValid()) {
        cone = coneB;

    } else {
        cone.halfAngle = -M_PI;
    }

    return cone;
}

} // namespace fcpw