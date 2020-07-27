#pragma once

#ifdef FCPW_USE_ENOKI
	#include <enoki/array.h>
#endif
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace fcpw {

#ifdef FCPW_USE_ENOKI
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
#endif

template<size_t DIM>
using Vector = Eigen::Matrix<float, DIM, 1>;
using Vector2 = Vector<2>;
using Vector3 = Vector<3>;

template<size_t DIM>
using Transform = Eigen::Transform<float, DIM, Eigen::Affine>;

template<size_t DIM>
inline float dot(const Vector<DIM>& u, const Vector<DIM>& v)
{
#ifdef FCPW_USE_ENOKI
	return enoki::dot(u, v);
#else
	return u.dot(v);
#endif
}

inline Vector3 cross(const Vector3& u, const Vector3& v)
{
#ifdef FCPW_USE_ENOKI
	return enoki::cross(u, v);
#else
	return u.cross(v);
#endif
}

template<size_t DIM>
inline Vector<DIM> cwiseMin(const Vector<DIM>& u, const Vector<DIM>& v)
{
#ifdef FCPW_USE_ENOKI
	return enoki::min(u, v);
#else
	return u.cwiseMin(v);
#endif
}

template<size_t DIM>
inline Vector<DIM> cwiseMax(const Vector<DIM>& u, const Vector<DIM>& v)
{
#ifdef FCPW_USE_ENOKI
	return enoki::max(u, v);
#else
	return u.cwiseMax(v);
#endif
}

template<size_t DIM>
inline Vector<DIM> cwiseMin(const Vector<DIM>& v, float s)
{
#ifdef FCPW_USE_ENOKI
	return enoki::min(v, s);
#else
	return v.cwiseMin(s);
#endif
}

template<size_t DIM>
inline Vector<DIM> cwiseMax(const Vector<DIM>& v, float s)
{
#ifdef FCPW_USE_ENOKI
	return enoki::max(v, s);
#else
	return v.cwiseMax(s);
#endif
}

template<size_t DIM>
inline Vector<DIM> cwiseProduct(const Vector<DIM>& u, const Vector<DIM>& v)
{
#ifdef FCPW_USE_ENOKI
	return u*v;
#else
	return u.cwiseProduct(v);
#endif
}

template<size_t DIM>
inline Vector<DIM> cwiseQuotient(const Vector<DIM>& u, const Vector<DIM>& v)
{
#ifdef FCPW_USE_ENOKI
	return u/v;
#else
	return u.cwiseQuotient(v);
#endif
}

template<size_t DIM>
inline Vector<DIM> cwiseInverse(const Vector<DIM>& v)
{
#ifdef FCPW_USE_ENOKI
	return enoki::rcp(v);
#else
	return v.cwiseInverse();
#endif
}

} // namespace fcpw
