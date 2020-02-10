#pragma once

#include "ray.h"

namespace fcpw {

template <int DIM>
struct BoundingSphere {
	// constructor
	BoundingSphere(const Vector<DIM>& c_, float r2_): c(c_), r2(r2_) {}

	// computes transformed sphere
	BoundingSphere<DIM> transform(const Transform<float, DIM, Affine>& t) const {
		Vector<DIM> tc = t*c;
		float tr2 = maxFloat;
		if (r2 < maxFloat) {
			Vector<DIM> direction = Vector<DIM>::Zero();
			direction(0) = 1;
			tr2 = (t*(c + std::sqrtf(r2)*direction) - tc).squaredNorm();
		}

		return BoundingSphere<DIM>(tc, tr2);
	}

	// members
	Vector<DIM> c;
	float r2;
};

template <int DIM>
class BoundingBox {
public:
	// constructor
	BoundingBox() {
		for (int i = 0; i < DIM; i++) {
			pMin(i) = maxFloat;
			pMax(i) = minFloat;
		}
	}

	// constructor
	BoundingBox(const Vector<DIM>& p): pMin(p), pMax(p) {}

	// expands volume to include point
	void expandToInclude(const Vector<DIM>& p) {
		for (int i = 0; i < DIM; i++) {
			if (pMin(i) > p(i)) pMin(i) = p(i);
			if (pMax(i) < p(i)) pMax(i) = p(i);
		}
	}

	// expands volume to include box
	void expandToInclude(const BoundingBox<DIM>& b)	{
		for (int i = 0; i < DIM; i++) {
			if (pMin(i) > b.pMin(i)) pMin(i) = b.pMin(i);
			if (pMax(i) < b.pMax(i)) pMax(i) = b.pMax(i);
		}
	}

	// returns box extent
	Vector<DIM> extent() const {
		return pMax - pMin;
	}

	// computes min and max squared distance to point;
	// min squared distance is 0 if point is inside box
	void computeSquaredDistance(const Vector<DIM>& p, float& d2Min, float& d2Max) const {
		d2Min = 0.0f;
		d2Max = 0.0f;

		for (int i = 0; i < DIM; i++) {
			float d = std::max({pMin(i) - p(i), 0.0f, p(i) - pMax(i)});
			d2Min += d*d;

			d = std::max(p(i) - pMin(i), pMax(i) - p(i));
			d2Max += d*d;
		}
	}

	// checks whether box contains point
	bool contains(const Vector<DIM>& p) const {
		for (int i = 0; i < DIM; i++) {
			if (pMin(i) > p(i) || pMax(i) < p(i)) return false;
		}

		return true;
	}

	// checks for overlap with sphere
	bool overlaps(const BoundingSphere<DIM>& s, float& d2Min, float& d2Max) const {
		computeSquaredDistance(s.c, d2Min, d2Max);
		return d2Min <= s.r2;
	}

	// checks for overlap with box
	bool overlaps(const BoundingBox<DIM>& b) const {
		for (int i = 0; i < DIM; i++) {
			bool doesOverlap = (pMax(i) >= b.pMin(i)) && (pMin(i) <= b.pMax(i));
			if (!doesOverlap) return false;
		}

		return true;
	}

	// checks for ray intersection
	bool intersect(const Ray<DIM>& r, float& tMin, float& tMax) const {
		float t0 = 0.0f;
		float t1 = r.tMax;
		const Vector<DIM>& o = r.o;
		const Vector<DIM>& invD = r.invD;

		for (int i = 0; i < DIM; i++) {
			// update interval for _i_th bounding box slab
			float tNear = (pMin(i) - o(i))*invD(i);
			float tFar = (pMax(i) - o(i))*invD(i);

			// update parametric interval from slab intersection $t$ values
			if (tNear > tFar) std::swap(tNear, tFar);

			// update _tFar_ to ensure robust ray--bounds intersection
			tFar *= 1.0f + 2.0f*gamma(3);
			t0 = tNear > t0 ? tNear : t0;
			t1 = tFar < t1 ? tFar : t1;
			if (t0 > t1) return false;
		}

		tMin = t0;
		tMax = t1;
		return true;
	}

	// returns max dimension
	int maxDimension() const {
		Vector<DIM> e = extent();

		for (int i = 0; i < DIM - 1; i++) {
			bool isMaxDimension = true;

			for (int j = i + 1; j < DIM; j++) {
				if (e(i) < e(j)) {
					isMaxDimension = false;
					break;
				}
			}

			if (isMaxDimension) return i;
		}

		return DIM - 1;
	}

	// returns centroid
	Vector<DIM> centroid() const {
		return (pMin + pMax)*0.5f;
	}

	// returns surface area
	float surfaceArea() const {
		LOG(FATAL) << "BoundingBox::surfaceArea(): not implemented for dimension: " << DIM;
		return 0.0f;
	}

	// returns volume
	float volume() const {
		Vector<DIM> e = extent();
		float v = 1.0f;
		for (int i = 0; i < DIM; i++) v *= e(i);

		return v;
	}

	// computes transformed box
	BoundingBox<DIM> transform(const Transform<float, DIM, Affine>& t) const {
		BoundingBox<DIM> b;
		int nCorners = 1 << DIM;

		for (int i = 0; i < nCorners; i++) {
			Vector<DIM> p = Vector<DIM>::Zero();
			int temp = i;

			for (int j = 0; j < DIM; j++) {
				int idx = temp%2;
				p(j) = idx == 0 ? pMin(j) : pMax(j);
				temp /= 2;
			}

			b.expandToInclude(t*p);
		}

		return b;
	}

	// members
	Vector<DIM> pMin, pMax;
};

template <>
inline float BoundingBox<2>::surfaceArea() const {
	Vector2f e = extent();
	return 2.0f*(e(0) + e(1));
}

template <>
inline float BoundingBox<3>::surfaceArea() const {
	Vector3f e = extent();
	return 2.0f*(e(0)*e(1) + e(0)*e(2) + e(1)*e(2));
}

} // namespace fcpw