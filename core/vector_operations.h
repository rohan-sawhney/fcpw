#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace fcpw {

using Vector2f = Eigen::Vector2f;
using Vector3f = Eigen::Vector3f;
template <int DIM>
using Vector = Eigen::Matrix<float, DIM, 1>;
template <int DIM>
using Transform = Eigen::Transform<float, DIM, Eigen::Affine>;

template <int DIM>
inline Vector<DIM> zeroVector()
{
	return Vector<DIM>::Zero();
}

template <int DIM>
inline Vector<DIM> constantVector(float constant)
{
	return Vector<DIM>::Constant(constant);
}

template <int DIM>
inline Vector<DIM> unit(const Vector<DIM>& v)
{
	return v.normalized();
}

template <int DIM>
inline float norm(const Vector<DIM>& v)
{
	return v.norm();
}

template <int DIM>
inline float squaredNorm(const Vector<DIM>& v)
{
	return v.squaredNorm();
}

template <int DIM>
inline float sum(const Vector<DIM>& v)
{
	return v.sum();
}

template <int DIM>
inline float product(const Vector<DIM>& v)
{
	return v.prod();
}

template <int DIM>
inline float minCoeff(const Vector<DIM>& v)
{
	return v.minCoeff();
}

template <int DIM>
inline float maxCoeff(const Vector<DIM>& v)
{
	return v.maxCoeff();
}

template <int DIM>
inline float minCoeff(const Vector<DIM>& v, int& index)
{
	return v.minCoeff(&index);
}

template <int DIM>
inline float maxCoeff(const Vector<DIM>& v, int& index)
{
	return v.maxCoeff(&index);
}

template <int DIM>
inline float dot(const Vector<DIM>& u, const Vector<DIM>& v)
{
	return u.dot(v);
}

inline Vector3f cross(const Vector3f& u, const Vector3f& v)
{
	return u.cross(v);
}

template <int DIM>
inline Vector<DIM> cwiseMin(const Vector3f& u, const Vector3f& v)
{
	return u.cwiseMin(v);
}

template <int DIM>
inline Vector<DIM> cwiseMax(const Vector3f& u, const Vector3f& v)
{
	return u.cwiseMax(v);
}

template <int DIM>
inline Vector<DIM> cwiseMin(const Vector3f& v, float s)
{
	return v.cwiseMin(s);
}

template <int DIM>
inline Vector<DIM> cwiseMax(const Vector3f& v, float s)
{
	return v.cwiseMax(s);
}

template <int DIM>
inline Vector<DIM> cwiseProduct(const Vector3f& u, const Vector3f& v)
{
	return u.cwiseProduct(v);
}

template <int DIM>
inline Vector<DIM> cwiseQuotient(const Vector3f& u, const Vector3f& v)
{
	return u.cwiseQuotient(v);
}

template <int DIM>
inline Vector<DIM> cwiseInverse(const Vector3f& v)
{
	return v.cwiseInverse();
}

template <int DIM>
inline Vector<DIM> transformVector(const Transform<DIM>& t, const Vector3f& v)
{
	return t*v;
}

template <int DIM>
inline bool allLeq(const Vector3f& u, const Vector3f& v)
{
	return (u.array() <= v.array()).all();
}

template <int DIM>
inline bool allGeq(const Vector3f& u, const Vector3f& v)
{
	return (u.array() >= v.array()).all();
}

template <int DIM>
inline bool anyLeq(const Vector3f& u, const Vector3f& v)
{
	return (u.array() <= v.array()).any();
}

template <int DIM>
inline bool anyGeq(const Vector3f& u, const Vector3f& v)
{
	return (u.array() >= v.array()).any();
}

} // namespace fcpw
