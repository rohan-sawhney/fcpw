#pragma once

#include <fcpw/core/core.h>

namespace fcpw {

template<size_t DIM>
struct Ray {
    // constructor
    Ray(const Vector<DIM>& o_, const Vector<DIM>& d_, float tMax_=maxFloat):
        o(o_), d(d_), invD(d.cwiseInverse()), tMax(tMax_) {}

    // operator()
    Vector<DIM> operator()(float t) const {
        return o + d*t;
    }

    // computes transformed ray
    Ray<DIM> transform(const Transform<DIM>& t) const {
        Vector<DIM> to = t*o;
        Vector<DIM> td = t*(o + d*(tMax < maxFloat ? tMax : 1.0f)) - to;
        float tdNorm = td.norm();

        return Ray<DIM>(to, td/tdNorm, tMax < maxFloat ? tdNorm : maxFloat);
    }

    // members
    Vector<DIM> o, d, invD;
    float tMax;
};

// Robust ray data for conservative BVH traversal and triangle intersection;
// source: Woop, Benthin, Wald. Watertight Ray/Triangle Intersection. JCGT 2013.
template<size_t DIM>
struct RobustIntersectionData {
    // constructor
    RobustIntersectionData(const Ray<DIM>& r,
                           const Vector<DIM>& pMin,
                           const Vector<DIM>& pMax) {
        // do nothing
    }
};

template<>
struct RobustIntersectionData<3> {
    // constructor
    RobustIntersectionData(const Ray<3>& r,
                           const Vector3& pMin,
                           const Vector3& pMax) {
        // calculate dimension where the ray direction is maximal
        r.d.cwiseAbs().maxCoeff(&kz);
        kx = kz + 1; if (kx == 3) kx = 0;
        ky = kx + 1; if (ky == 3) ky = 0;

        // swap kx and ky dimensions to preserve winding direction of the triangle
        if (r.d[kz] < 0.0f) std::swap(kx, ky);

        // calculate shear constants
        Sx = r.d[kx]/r.d[kz];
        Sy = r.d[ky]/r.d[kz];
        Sz = 1.0f/r.d[kz];

        // setup near and far plane indices for a box stored as
        // {pMin.x, pMin.y, pMin.z, pMax.x, pMax.y, pMax.z}
        Vector3i nearId{0, 1, 2};
        Vector3i farId{3, 4, 5};
        nearX = nearId[kx], farX = farId[kx];
        nearY = nearId[ky], farY = farId[ky];
        nearZ = nearId[kz], farZ = farId[kz];
        if (r.d[kx] < 0.0f) std::swap(nearX, farX);
        if (r.d[ky] < 0.0f) std::swap(nearY, farY);
        if (r.d[kz] < 0.0f) std::swap(nearZ, farZ);

        // constants and helper lambdas for conservative rounding
        const float oneUlp = 1.1920929e-07f; // 2^(-23)
        const float p = 1.0f + oneUlp;
        const float m = 1.0f - oneUlp;
        auto up = [p, m](float a) { return a > 0.0f ? a*p : a*m; };
        auto dn = [p, m](float a) { return a > 0.0f ? a*m : a*p; };
        auto Up = [p](float a) { return a*p; };
        auto Dn = [m](float a) { return a*m; };

        // calculate corrected origin for near- and far-plane distance calculations
        const float halfUlp = 5.96046448e-08f; // 2^(-24)
        const float eps = 5.0f*halfUlp;
        Vector3 lower = (r.o - pMin).cwiseAbs()*m;
        Vector3 upper = (r.o - pMax).cwiseAbs()*p;
        float maxZ = std::max(lower[kz], upper[kz]);

        float errorNearX = Up(lower[kx] + maxZ);
        float errorNearY = Up(lower[ky] + maxZ);
        oNear[0] = up(r.o[kx] + Up(eps*errorNearX));
        oNear[1] = up(r.o[ky] + Up(eps*errorNearY));
        oNear[2] = r.o[kz];

        float errorFarX = Up(upper[kx] + maxZ);
        float errorFarY = Up(upper[ky] + maxZ);
        oFar[0] = dn(r.o[kx] - Up(eps*errorFarX));
        oFar[1] = dn(r.o[ky] - Up(eps*errorFarY));
        oFar[2] = r.o[kz];

        if (r.d[kx] < 0.0f) std::swap(oNear[0], oFar[0]);
        if (r.d[ky] < 0.0f) std::swap(oNear[1], oFar[1]);

        // calculate corrected inverse direction for near- and far-plane distance calculations
        invDNear[0] = Dn(Dn(r.invD[kx]));
        invDNear[1] = Dn(Dn(r.invD[ky]));
        invDNear[2] = Dn(Dn(r.invD[kz]));
        invDFar[0] = Up(Up(r.invD[kx]));
        invDFar[1] = Up(Up(r.invD[ky]));
        invDFar[2] = Up(Up(r.invD[kz]));
    }

    // members
    int kx, ky, kz; // axis permutation indices (kz is the dominant axis)
    float Sx, Sy, Sz; // shear constants for ray-triangle intersection
    int nearX, nearY, nearZ; // near axis indices for box intersection
    int farX, farY, farZ; // far axis indices for box intersection
    Vector3 oNear, oFar; // corrected ray origins for near/far plane calculations
    Vector3 invDNear, invDFar; // conservative inverse ray directions for near/far plane calculations
};

} // namespace fcpw