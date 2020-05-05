#pragma once

#include "bounding_volumes.h"

namespace fcpw {

// performs wide version of ray box intersection test
template <int WIDTH, int DIM>
inline MaskP<WIDTH> intersectWideBox(const Ray<DIM>& r,
									 const VectorP<WIDTH, DIM>& bMin,
									 const VectorP<WIDTH, DIM>& bMax,
									 FloatP<WIDTH>& tMin, FloatP<WIDTH>& tMax)
{
	VectorP<WIDTH, DIM> t0 = (bMin - r.o)*r.invD;
	VectorP<WIDTH, DIM> t1 = (bMax - r.o)*r.invD;
	VectorP<WIDTH, DIM> tNear = enoki::min(t0, t1);
	VectorP<WIDTH, DIM> tFar = enoki::max(t0, t1);

	tFar *= 1.0f + 2.0f*gamma(3);
	tMin = enoki::max(0.0f, enoki::hmax(tNear));
	tMax = enoki::min(r.tMax, enoki::hmin(tFar));

	return tMin <= tMax;
}

// performs wide version of sphere box overlap test
template <int WIDTH, int DIM>
inline MaskP<WIDTH> overlapWideBox(const BoundingSphere<DIM>& s,
								   const VectorP<WIDTH, DIM>& bMin,
								   const VectorP<WIDTH, DIM>& bMax,
								   FloatP<WIDTH>& d2Min, FloatP<WIDTH>& d2Max)
{
	VectorP<WIDTH, DIM> u = bMin - s.c;
	VectorP<WIDTH, DIM> v = s.c - bMax;
	d2Min = enoki::squared_norm(enoki::max(enoki::max(u, v), 0.0f));
	d2Max = enoki::squared_norm(enoki::min(u, v));

	return d2Min <= s.r2;
}

// performs wide version of ray triangle intersection test
template <int WIDTH>
inline MaskP<WIDTH> intersectWideTriangle(const Ray<3>& r, const Vector3P<WIDTH>& pa,
										  const Vector3P<WIDTH>& pb, const Vector3P<WIDTH>& pc,
										  FloatP<WIDTH>& d, Vector3P<WIDTH>& pt, Vector2P<WIDTH>& t)
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	Vector3P<WIDTH> v1 = pb - pa;
	Vector3P<WIDTH> v2 = pc - pa;
	Vector3P<WIDTH> p = enoki::cross(r.d, v2);
	FloatP<WIDTH> det = enoki::dot(v1, p);

	MaskP<WIDTH> active = enoki::abs(det) >= epsilon;
	FloatP<WIDTH> invDet = enoki::rcp(det);

	Vector3P<WIDTH> s = r.o - pa;
	FloatP<WIDTH> u = enoki::dot(s, p)*invDet;
	active &= u >= 0.0f && u <= 1.0f;

	Vector3 q = enoki::cross(s, v1);
	FloatP<WIDTH> v = enoki::dot(r.d, q)*invDet;
	active &= v >= 0.0f && u + v <= 1.0f;

	d = enoki::dot(v2, q)*invDet;
	active &= d > epsilon && d <= r.tMax;
	pt = r.o + r.d*Vector3P<WIDTH>(d);
	t = Vector2P<WIDTH>(u, v);

	return active;
}

// finds closest point on wide triangle to point
template <int WIDTH>
inline FloatP<WIDTH> findClosestPointWideTriangle(const Vector3P<WIDTH>& x, const Vector3P<WIDTH>& pa,
												  const Vector3P<WIDTH>& pb, const Vector3P<WIDTH>& pc,
												  Vector3P<WIDTH>& pt, Vector2P<WIDTH>& t,
												  IntP<WIDTH>& vIndex, IntP<WIDTH>& eIndex)
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	// TODO
	return FloatP<WIDTH>(0.0f);
}

} // namespace fcpw
