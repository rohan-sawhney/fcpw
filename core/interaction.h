#pragma once

#include "core.h"

namespace fcpw {

enum class DistanceInfo {
	Exact,
	Bounded
};

template<size_t DIM>
struct Interaction {
	// constructor
	Interaction(): d(maxFloat), sign(0), nodeIndex(-1), distanceInfo(DistanceInfo::Exact),
				   primitive(nullptr), p(zeroVector<DIM>()), n(constantVector<DIM>(NAN)),
				   uv(zeroVector<DIM - 1>()) {}

	// comparison operators
	bool operator==(const Interaction<DIM>& i) const {
		bool distancesMatch = std::fabs(d - i.d) < 1e-6;
		if (distanceInfo == DistanceInfo::Bounded) return distancesMatch;

		return distancesMatch && squaredNorm<DIM>(p - i.p) < 1e-6;
	}

	bool operator!=(const Interaction<DIM>& i) const {
		return !(*this == i);
	}

	// returns signed distance
	float signedDistance(const Vector<DIM>& x) const {
		return sign == 0 ? (dot<DIM>(x - p, n) > 0.0f ? 1.0f : -1.0f)*d : sign*d;
	}

	// computes normal from geometric primitive if unspecified
	void computeNormal() {
		if (isNaN<DIM>(n)) n = primitive->normal(uv);
	}

	// applies transform
	void applyTransform(const Transform<DIM>& t,
						const Transform<DIM>& tInv,
						const Vector<DIM>& query) {
		p = transformVector<DIM>(t, p);
		d = norm<DIM>(p - query);
		n = transformVector<DIM>(Transform<DIM>(tInv.matrix().transpose()), n);
		n = unit<DIM>(n);
	}

	// members
	float d;
	int sign; // sign bit used for difference ops
	int nodeIndex; // index of aggregate node containing intersected or closest point
	Vector<DIM> p, n;
	Vector<DIM - 1> uv;
	DistanceInfo distanceInfo;
	const GeometricPrimitive<DIM> *primitive;
};

template<size_t DIM>
inline bool compareInteractions(const Interaction<DIM>& i, const Interaction<DIM>& j) {
	return i.d < j.d;
}

template<size_t DIM>
inline std::vector<Interaction<DIM>> removeDuplicates(const std::vector<Interaction<DIM>>& is) {
	int N = (int)is.size();
	std::vector<bool> isDuplicate(N, false);
	std::vector<Interaction<DIM>> cs;

	for (int i = 0; i < N - 1; i++) {
		if (is[i] == is[i + 1]) isDuplicate[i + 1] = true;
	}

	for (int i = 0; i < N; i++) {
		if (!isDuplicate[i]) cs.emplace_back(is[i]);
	}

	return cs;
}

} // namespace fcpw
