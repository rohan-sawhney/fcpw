#pragma once

#ifdef BUILD_ENOKI
	#include <enoki/array.h>
#endif
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace fcpw {

#ifdef BUILD_ENOKI
	template<size_t DIM>
	using Vector = enoki::Array<float, DIM>;
	template<size_t WIDTH>
	using IntP = enoki::Packet<int, WIDTH>;
	template<size_t WIDTH>
	using FloatP = enoki::Packet<float, WIDTH>;
	template<size_t WIDTH>
	using MaskP = enoki::mask_t<FloatP<WIDTH>>;
	template<size_t WIDTH, size_t DIM>
	using VectorP = enoki::Array<FloatP<WIDTH>, DIM>;
	template<size_t WIDTH>
	using Vector2P = VectorP<WIDTH, 2>;
	template<size_t WIDTH>
	using Vector3P = VectorP<WIDTH, 3>;
#else
	template<size_t DIM>
	using Vector = Eigen::Matrix<float, DIM, 1>;
#endif

using Vector2 = Vector<2>;
using Vector3 = Vector<3>;

template<size_t DIM>
using Transform = Eigen::Transform<float, DIM, Eigen::Affine>;

template<size_t DIM>
inline Vector<DIM> zeroVector()
{
#ifdef BUILD_ENOKI
	return enoki::zero<Vector<DIM>>();
#else
	return Vector<DIM>::Zero();
#endif
}

template<size_t DIM>
inline Vector<DIM> constantVector(float constant)
{
#ifdef BUILD_ENOKI
	return Vector<DIM>(constant);
#else
	return Vector<DIM>::Constant(constant);
#endif
}

template<size_t DIM>
inline Vector<DIM> unit(const Vector<DIM>& v)
{
#ifdef BUILD_ENOKI
	return enoki::normalize(v);
#else
	return v.normalized();
#endif
}

template<size_t DIM>
inline float norm(const Vector<DIM>& v)
{
#ifdef BUILD_ENOKI
	return enoki::norm(v);
#else
	return v.norm();
#endif
}

template<size_t DIM>
inline float squaredNorm(const Vector<DIM>& v)
{
#ifdef BUILD_ENOKI
	return enoki::squared_norm(v);
#else
	return v.squaredNorm();
#endif
}

template<size_t DIM>
inline float sum(const Vector<DIM>& v)
{
#ifdef BUILD_ENOKI
	return enoki::hsum(v);
#else
	return v.sum();
#endif
}

template<size_t DIM>
inline float product(const Vector<DIM>& v)
{
#ifdef BUILD_ENOKI
	return enoki::hprod(v);
#else
	return v.prod();
#endif
}

template<size_t DIM>
inline float minCoeff(const Vector<DIM>& v)
{
#ifdef BUILD_ENOKI
	return enoki::hmin(v);
#else
	return v.minCoeff();
#endif
}

template<size_t DIM>
inline float maxCoeff(const Vector<DIM>& v)
{
#ifdef BUILD_ENOKI
	return enoki::hmax(v);
#else
	return v.maxCoeff();
#endif
}

template<size_t DIM>
inline float minCoeff(const Vector<DIM>& v, int& index)
{
#ifdef BUILD_ENOKI
	index = 0;
	float value = v[0];

	for (int i = 1; i < DIM; i++) {
		auto mask = v[i] < value;
		enoki::masked(index, mask) = i;
		enoki::masked(value, mask) = v[i];
	}

	return index;
#else
	return v.minCoeff(&index);
#endif
}

template<size_t DIM>
inline float maxCoeff(const Vector<DIM>& v, int& index)
{
#ifdef BUILD_ENOKI
	index = 0;
	float value = v[0];

	for (int i = 1; i < DIM; i++) {
		auto mask = v[i] > value;
		enoki::masked(index, mask) = i;
		enoki::masked(value, mask) = v[i];
	}

	return index;
#else
	return v.maxCoeff(&index);
#endif
}

template<size_t DIM>
inline float dot(const Vector<DIM>& u, const Vector<DIM>& v)
{
#ifdef BUILD_ENOKI
	return enoki::dot(u, v);
#else
	return u.dot(v);
#endif
}

inline Vector3 cross(const Vector3& u, const Vector3& v)
{
#ifdef BUILD_ENOKI
	return enoki::cross(u, v);
#else
	return u.cross(v);
#endif
}

template<size_t DIM>
inline Vector<DIM> cwiseMin(const Vector<DIM>& u, const Vector<DIM>& v)
{
#ifdef BUILD_ENOKI
	return enoki::min(u, v);
#else
	return u.cwiseMin(v);
#endif
}

template<size_t DIM>
inline Vector<DIM> cwiseMax(const Vector<DIM>& u, const Vector<DIM>& v)
{
#ifdef BUILD_ENOKI
	return enoki::max(u, v);
#else
	return u.cwiseMax(v);
#endif
}

template<size_t DIM>
inline Vector<DIM> cwiseMin(const Vector<DIM>& v, float s)
{
#ifdef BUILD_ENOKI
	return enoki::min(v, s);
#else
	return v.cwiseMin(s);
#endif
}

template<size_t DIM>
inline Vector<DIM> cwiseMax(const Vector<DIM>& v, float s)
{
#ifdef BUILD_ENOKI
	return enoki::max(v, s);
#else
	return v.cwiseMax(s);
#endif
}

template<size_t DIM>
inline Vector<DIM> cwiseProduct(const Vector<DIM>& u, const Vector<DIM>& v)
{
#ifdef BUILD_ENOKI
	return u*v;
#else
	return u.cwiseProduct(v);
#endif
}

template<size_t DIM>
inline Vector<DIM> cwiseQuotient(const Vector<DIM>& u, const Vector<DIM>& v)
{
#ifdef BUILD_ENOKI
	return u/v;
#else
	return u.cwiseQuotient(v);
#endif
}

template<size_t DIM>
inline Vector<DIM> cwiseInverse(const Vector<DIM>& v)
{
#ifdef BUILD_ENOKI
	return enoki::rcp(v);
#else
	return v.cwiseInverse();
#endif
}

template<size_t DIM>
inline Vector<DIM> transformVector(const Transform<DIM>& t, const Vector<DIM>& v)
{
#ifdef BUILD_ENOKI
	// convert enoki array to eigen
	Eigen::Matrix<float, DIM, 1> u;
	enoki::Array<int, DIM> index = enoki::arange<enoki::Array<int, DIM>>();
	enoki::scatter(u.data(), v, index);

	// transform
	u = t*u;

	// convert u to enoki
	return enoki::gather<Vector<DIM>>(u.data(), index);
#else
	return t*v;
#endif
}

template<size_t DIM>
inline bool allLeq(const Vector<DIM>& u, const Vector<DIM>& v)
{
#ifdef BUILD_ENOKI
	return enoki::all(u <= v);
#else
	return (u.array() <= v.array()).all();
#endif
}

template<size_t DIM>
inline bool allGeq(const Vector<DIM>& u, const Vector<DIM>& v)
{
#ifdef BUILD_ENOKI
	return enoki::all(u >= v);
#else
	return (u.array() >= v.array()).all();
#endif
}

template<size_t DIM>
inline bool isNaN(const Vector<DIM>& v)
{
#ifdef BUILD_ENOKI
	return enoki::any(enoki::isnan(v));
#else
	return v.isNaN().any();
#endif
}

} // namespace fcpw
