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
		pMin = cwiseMin<DIM>(pMin, p - epsilonVector);
		pMax = cwiseMax<DIM>(pMax, p + epsilonVector);
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
		d2Min = cwiseMax<DIM>(cwiseMax<DIM>(u, v), 0.0f).squaredNorm();
		d2Max = cwiseMin<DIM>(u, v).squaredNorm();
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
		Vector<DIM> tNear = cwiseMin<DIM>(t0, t1);
		Vector<DIM> tFar = cwiseMax<DIM>(t0, t1);

		float tNearMax = std::max(0.0f, tNear.maxCoeff());
		float tFarMin = std::min(r.tMax, tFar.minCoeff());
		if (tNearMax > tFarMin) return false;

		tMin = tNearMax;
		tMax = tFarMin;
		return true;
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
		Vector<DIM> e = cwiseMax<DIM>(extent(), 1e-5); // the 1e-5 is to prevent division by zero
		return 2.0f*Vector<DIM>::Constant(e.prod()).cwiseQuotient(e).sum();
	}

	// returns volume
	float volume() const {
		return extent().prod();
	}

	// computes transformed box
	BoundingBox<DIM> transform(const Transform<DIM>& t) const {
		BoundingBox<DIM> b;
		int nCorners = 1 << DIM;

		for (int i = 0; i < nCorners; i++) {
			Vector<DIM> p = Vector<DIM>::Zero();
			int temp = i;

			for (int j = 0; j < DIM; j++) {
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
		bIntersect.pMin = cwiseMax<DIM>(pMin, b.pMin);
		bIntersect.pMax = cwiseMin<DIM>(pMax, b.pMax);

		return bIntersect;
	}

	// members
	Vector<DIM> pMin, pMax;
};

} // namespace fcpw
