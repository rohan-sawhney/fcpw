#pragma once

#include "core.h"

namespace fcpw {

enum class DistanceInfo {
	Exact,
	Bounded
};

template <int DIM>
struct Interaction {
	// constructor
	Interaction(): d(maxFloat), sign(0), distanceInfo(DistanceInfo::Exact), shape(nullptr) {
		p.setZero();
		n.setZero();
		uv.setZero();
	}

	// comparison operators
	bool operator==(const Interaction<DIM>& i) {
		bool distancesMatch = std::fabsf(d - i.d) < 1e-6;
		if (distanceInfo == DistanceInfo::Bounded) return distancesMatch;

		return distancesMatch && (p - i.p).squaredNorm() < 1e-6;
	}

	bool operator!=(const Interaction<DIM>& i) {
		return !(*this == i);
	}

	// returns signed distance
	float signedDistance(const Vector<DIM>& x) const {
		return sign == 0 ? ((x - p).dot(n) > 0.0f ? 1.0f : -1.0f)*d : sign*d;
	}

	// applies transform
	void applyTransform(const Transform<float, DIM, Affine>& t,
						const Transform<float, DIM, Affine>& tInv,
						const Vector<DIM>& query) {
		p = t*p;
		d = (p - query).norm();
		n = Transform<float, DIM, Affine>(tInv.matrix().transpose())*n;
		n.normalize();
	}

	// members
	float d;
	int sign;
	Vector<DIM> p, n;
	Vector<DIM - 1> uv;
	DistanceInfo distanceInfo;
	const Shape<DIM> *shape;
};

template <int DIM>
inline bool compareInteractions(const Interaction<DIM>& i, const Interaction<DIM>& j) {
	return i.d < j.d;
}

} // namespace fcpw
