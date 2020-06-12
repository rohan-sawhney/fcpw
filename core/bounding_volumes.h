#pragma once

#include "ray.h"

namespace fcpw {

template<size_t DIM>
struct BoundingSphere {
	// constructor
	BoundingSphere(const Vector<DIM>& c_, float r2_): c(c_), r2(r2_) {}

	// computes transformed sphere
	BoundingSphere<DIM> transform(const Transform<DIM>& t) const {
		Vector<DIM> tc = transformVector<DIM>(t, c);
		float tr2 = maxFloat;
		if (r2 < maxFloat) {
			Vector<DIM> direction = zeroVector<DIM>();
			direction[0] = 1;
			tr2 = squaredNorm<DIM>(transformVector<DIM>(t, c + direction*std::sqrt(r2)) - tc);
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
	BoundingBox(): pMin(constantVector<DIM>(maxFloat)),
				   pMax(constantVector<DIM>(minFloat)) {}

	// constructor
	BoundingBox(const Vector<DIM>& p): pMin(p), pMax(p) {}

	// expands volume to include point
	void expandToInclude(const Vector<DIM>& p) {
		pMin = cwiseMin<DIM>(pMin, p);
		pMax = cwiseMax<DIM>(pMax, p);
	}

	// expands volume to include box
	void expandToInclude(const BoundingBox<DIM>& b)	{
		pMin = cwiseMin<DIM>(pMin, b.pMin);
		pMax = cwiseMax<DIM>(pMax, b.pMax);
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
		d2Min = squaredNorm<DIM>(cwiseMax<DIM>(cwiseMax<DIM>(u, v), 0.0f));
		d2Max = squaredNorm<DIM>(cwiseMin<DIM>(u, v));
	}

	// checks whether box contains point
	bool contains(const Vector<DIM>& p) const {
		return allGeq<DIM>(p, pMin) && allLeq<DIM>(p, pMax);
	}

	// checks for overlap with sphere
	bool overlap(const BoundingSphere<DIM>& s, float& d2Min, float& d2Max) const {
		computeSquaredDistance(s.c, d2Min, d2Max);
		return d2Min <= s.r2;
	}

	// checks for overlap with bounding box
	bool overlap(const BoundingBox<DIM>& b) const {
		return allGeq<DIM>(b.pMax, pMin) && allLeq<DIM>(b.pMin, pMax);
	}

	// checks for ray intersection
	bool intersect(const Ray<DIM>& r, float& tMin, float& tMax) const {
		// slab test for ray box intersection
		// source: http://www.jcgt.org/published/0007/03/04/paper-lowres.pdf
		Vector<DIM> t0 = cwiseProduct<DIM>(pMin - r.o, r.invD);
		Vector<DIM> t1 = cwiseProduct<DIM>(pMax - r.o, r.invD);
		Vector<DIM> tNear = cwiseMin<DIM>(t0, t1);
		Vector<DIM> tFar = cwiseMax<DIM>(t0, t1);

		tFar *= 1.0f + 2.0f*gamma(3);
		float tNearMax = std::max(0.0f, maxCoeff<DIM>(tNear));
		float tFarMin = std::min(r.tMax, minCoeff<DIM>(tFar));
		if (tNearMax > tFarMin) return false;

		tMin = tNearMax;
		tMax = tFarMin;
		return true;
	}

	// checks whether bounding box is valid
	bool isValid() const {
		return allGeq<DIM>(pMax, pMin);
	}

	// returns max dimension
	int maxDimension() const {
		int index;
		float maxLength = maxCoeff<DIM>(pMax - pMin, index);

		return index;
	}

	// returns centroid
	Vector<DIM> centroid() const {
		return (pMin + pMax)*0.5f;
	}

	// returns surface area
	float surfaceArea() const {
		Vector<DIM> e = cwiseMax<DIM>(extent(), 1e-5); // the 1e-5 is to prevent division by zero
		return 2.0f*sum<DIM>(cwiseQuotient<DIM>(constantVector<DIM>(product<DIM>(e)), e));
	}

	// returns volume
	float volume() const {
		return product<DIM>(extent());
	}

	// computes transformed box
	BoundingBox<DIM> transform(const Transform<DIM>& t) const {
		BoundingBox<DIM> b;
		int nCorners = 1 << DIM;

		for (int i = 0; i < nCorners; i++) {
			Vector<DIM> p = zeroVector<DIM>();
			int temp = i;

			for (int j = 0; j < DIM; j++) {
				int idx = temp%2;
				p[j] = idx == 0 ? pMin[j] : pMax[j];
				temp /= 2;
			}

			b.expandToInclude(transformVector<DIM>(t, p));
		}

		return b;
	}

	// returns the intersection of two bounding boxes
	BoundingBox<DIM> intersect(const BoundingBox<DIM>& b) const {
		BoundingBox<DIM> bIntersect;
		bIntersect.pMin = cwiseMax<DIM>(pMin, b.pMin);
		bIntersect.pMax = cwiseMin<DIM>(pMax, b.pMax);

		return bIntersect;
	}

	// members
	Vector<DIM> pMin, pMax;
};

} // namespace fcpw
