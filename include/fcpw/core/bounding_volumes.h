#pragma once

#include <fcpw/core/core.h>
#include <fcpw/core/ray.h>

namespace fcpw {

template<size_t DIM>
struct BoundingBox;

template<size_t DIM>
struct BoundingSphere {

	BoundingSphere() : c(Vector<DIM>::Zero()), r2(-1.0f) {}

	// constructor
	BoundingSphere(const Vector<DIM>& c_, float r2_): c(c_), r2(r2_) {}

	// computes transformed sphere
	BoundingSphere<DIM> transform(const Transform<DIM>& t) const {
		Vector<DIM> tc = t*c;
		float tr2 = -1.0f;

		if (r2 >= 0.0f) {
			Vector<DIM> direction = Vector<DIM>::Zero();
			direction[0] = 1;
			tr2 = (t*(c + direction*std::sqrt(r2)) - tc).squaredNorm();
		}

		return BoundingSphere<DIM>(tc, tr2);
	}

	// checks for ray intersection
	bool intersect(const Ray<DIM>& r, float& tMin, float& tMax) const {
		
		Vector<DIM> rel_pos = r.o - c;
		float b = 2.0f * rel_pos.dot(r.d);
		float c = rel_pos.squaredNorm() - r2;
		float d = b * b - 4 * c;

		if(d <= 0.0f) return false;

		float sqd = std::sqrt(d);

		tMin = std::max((-b - sqd) / 2.0f, 0.0f);
		tMax = std::min((-b + sqd) / 2.0f, r.tMax);
		return true;
	}

	bool overlap(const BoundingSphere<DIM>& s, float& d2Min, float& d2Max) const {

		float center_dist = (s.c - c).norm();
		float r = std::sqrt(r2);
		float close = std::max(center_dist - r, 0.0f);
		float far = center_dist + r;
		d2Min = close * close;
		d2Max = far * far;
		return d2Min <= s.r2;
	}

	void expandToInclude(const Vector<DIM>& p) {
		if(r2 < 0.0f) {
			c = p;
			r2 = 0.0f;
		} else {
			r2 = std::max(r2, (p - c).squaredNorm());
		}
	}

	void expandToInclude(const BoundingSphere<DIM>& b)	{
		if(r2 < 0.0f) {
			*this = b;
		} else {
			float dist = (b.c - c).norm() + std::sqrt(b.r2);
			r2 = std::max(r2, dist * dist);
		}
	}

	BoundingBox<DIM> box() const;
	BoundingSphere sphere() const {
		return *this;
	}

	static constexpr float PI_F = 3.1415926535897f;
	
	float surfaceArea() const {
		return 4.0f * PI_F * r2;
	}
	float volume() const {
		return (4.0f / 3.0f) * PI_F * r2 * std::sqrt(r2);
	}

	bool isValid() const {
		return r2 >= 0.0f;
	}

	BoundingSphere<DIM> intersect(const BoundingSphere<DIM>& b) const {
		float P = (b.c - c).squaredNorm();
		float Q = (r2 - b.r2 + P) / (2.0f * P);
		Vector<DIM> B = c + Q * (b.c - c);
		float R = r2 - (B - c).squaredNorm();
		return BoundingSphere(B, R);
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

	BoundingBox(const Vector<DIM>& pMin, const Vector<DIM>& pMax) : pMin(pMin), pMax(pMax) {}

	BoundingBox box() const {
		return *this;
	}
	BoundingSphere<DIM> sphere() const {
		Vector<DIM> avg = 0.5f * (pMin + pMax);
		BoundingSphere<DIM> box(avg, 0.0f);
		box.expandToInclude(pMin);
		box.expandToInclude(pMax);
		return box;
	}

	// expands volume to include point
	void expandToInclude(const Vector<DIM>& p) {
		Vector<DIM> epsilonVector = Vector<DIM>::Constant(epsilon);
		pMin = pMin.cwiseMin(p - epsilonVector);
		pMax = pMax.cwiseMax(p + epsilonVector);
	}

	// expands volume to include box
	void expandToInclude(const BoundingBox<DIM>& b)	{
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
		Vector<DIM> e = extent().cwiseMax(1e-5f); // the 1e-5 is to prevent division by zero
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

template<size_t DIM>
BoundingBox<DIM> BoundingSphere<DIM>::box() const {
	float r = std::sqrt(r2);
	return BoundingBox<DIM>(c.array() - r, c.array() + r);
}

} // namespace fcpw
