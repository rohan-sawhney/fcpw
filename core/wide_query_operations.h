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
	// vectorized slab test
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

	// vectorized Möller–Trumbore intersection algorithm
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
inline FloatP<WIDTH> findClosestPointWideTriangle(const Vector3& x, const Vector3P<WIDTH>& pa,
												  const Vector3P<WIDTH>& pb, const Vector3P<WIDTH>& pc,
												  Vector3P<WIDTH>& pt, Vector2P<WIDTH>& t,
												  IntP<WIDTH>& vIndex, IntP<WIDTH>& eIndex)
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	// check if x in vertex region outside pa
	Vector3P<WIDTH> ab = pb - pa;
	Vector3P<WIDTH> ac = pc - pa;
	Vector3P<WIDTH> ax = x - pa;
	FloatP<WIDTH> d1 = enoki::dot(ab, ax);
	FloatP<WIDTH> d2 = enoki::dot(ac, ax);
	MaskP<WIDTH> active1 = d1 <= 0.0f && d2 <= 0.0f;
	MaskP<WIDTH> active7 = active1;

	// barycentric coordinates (1, 0, 0)
	enoki::masked(pt, active1) = pa;
	enoki::masked(t[0], active1) = 1.0f;
	enoki::masked(t[1], active1) = 0.0f;
	enoki::masked(vIndex, active1) = 0;
	if (enoki::all(active7)) return enoki::norm(x - pt);

	// check if x in vertex region outside pb
	Vector3P<WIDTH> bx = x - pb;
	FloatP<WIDTH> d3 = enoki::dot(ab, bx);
	FloatP<WIDTH> d4 = enoki::dot(ac, bx);
	MaskP<WIDTH> active2 = d3 >= 0.0f && d4 <= d3;
	active7 |= active2;

	// barycentric coordinates (0, 1, 0)
	enoki::masked(pt, active2) = pb;
	enoki::masked(t[0], active2) = 0.0f;
	enoki::masked(t[1], active2) = 1.0f;
	enoki::masked(vIndex, active2) = 1;
	if (enoki::all(active7)) return enoki::norm(x - pt);

	// check if x in edge region of ab, if so return projection of x onto ab
	FloatP<WIDTH> vc = d1*d4 - d3*d2;
	MaskP<WIDTH> active3 = vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f;
	active7 |= active3;

	// barycentric coordinates (1 - v, v, 0)
	FloatP<WIDTH> v = d1/(d1 - d3);
	enoki::masked(pt, active3) = pa + ab*v;
	enoki::masked(t[0], active3) = 1.0f - v;
	enoki::masked(t[1], active3) = v;
	enoki::masked(eIndex, active3) = 0;
	if (enoki::all(active7)) return enoki::norm(x - pt);

	// check if x in vertex region outside pc
	Vector3P<WIDTH> cx = x - pc;
	FloatP<WIDTH> d5 = enoki::dot(ab, cx);
	FloatP<WIDTH> d6 = enoki::dot(ac, cx);
	MaskP<WIDTH> active4 = d6 >= 0.0f && d5 <= d6;
	active7 |= active4;

	// barycentric coordinates (0, 0, 1)
	enoki::masked(pt, active4) = pc;
	enoki::masked(t[0], active4) = 0.0f;
	enoki::masked(t[1], active4) = 0.0f;
	enoki::masked(vIndex, active4) = 2;
	if (enoki::all(active7)) return enoki::norm(x - pt);

	// check if x in edge region of ac, if so return projection of x onto ac
	FloatP<WIDTH> vb = d5*d2 - d1*d6;
	MaskP<WIDTH> active5 = vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f;
	active7 |= active5;

	// barycentric coordinates (1 - w, 0, w)
	FloatP<WIDTH> w = d2/(d2 - d6);
	enoki::masked(pt, active5) = pa + ac*w;
	enoki::masked(t[0], active5) = 1.0f - w;
	enoki::masked(t[1], active5) = 0.0f;
	enoki::masked(eIndex, active5) = 2;
	if (enoki::all(active7)) return enoki::norm(x - pt);

	// check if x in edge region of bc, if so return projection of x onto bc
	FloatP<WIDTH> va = d3*d6 - d5*d4;
	MaskP<WIDTH> active6 = va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f;
	active7 |= active6;

	// barycentric coordinates (0, 1 - w, w)
	w = (d4 - d3)/((d4 - d3) + (d5 - d6));
	enoki::masked(pt, active6) = pb + (pc - pb)*w;
	enoki::masked(t[0], active6) = 0.0f;
	enoki::masked(t[1], active6) = 1.0f - w;
	enoki::masked(eIndex, active6) = 1;
	if (enoki::all(active7)) return enoki::norm(x - pt);

	// x inside face region. Compute pt through its barycentric coordinates (u, v, w)
	FloatP<WIDTH> denom = 1.0f/(va + vb + vc);
	v = vb*denom;
	w = vc*denom;
	active7 = ~active7;

	enoki::masked(pt, active7) = pa + ab*v + ac*w; //= u*a + v*b + w*c, u = va*denom = 1.0f - v - w
	enoki::masked(t[0], active7) = 1.0f - v - w;
	enoki::masked(t[1], active7) = v;
	return enoki::norm(x - pt);
}

} // namespace fcpw
