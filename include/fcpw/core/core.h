#pragma once

// global includes
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <chrono>
#include <random>
#include <type_traits>
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
	template<size_t DIM>
	using enokiVector = enoki::Array<float, DIM>;
	using enokiVector2 = enoki::Array<float, 2>;
	using enokiVector3 = enoki::Array<float, 3>;
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
struct Ray;
template<size_t DIM>
struct BoundingSphere;
template<size_t DIM>
struct BoundingBox;
template<size_t DIM>
struct BoundingCone;
template<size_t DIM>
struct Interaction;
template<size_t DIM>
class Primitive;
template<size_t DIM>
class GeometricPrimitive;
template<size_t DIM>
class SilhouettePrimitive;
template<size_t DIM>
class Aggregate;

static const float minFloat = std::numeric_limits<float>::lowest();
static const float maxFloat = std::numeric_limits<float>::max();
static const int minInt = std::numeric_limits<int>::min();
static const int maxInt = std::numeric_limits<int>::max();
static const float epsilon = std::numeric_limits<float>::epsilon();
static const float oneMinusEpsilon = 1.0f - epsilon;

template<typename T, typename U, typename V>
inline T clamp(T val, U low, V high) {
	if (val < low) return low;
	else if (val > high) return high;

	return val;
}

inline bool inRange(float val, float low, float high) {
	return val >= low && val <= high;
}

inline float uniformRealRandomNumber(float a=0.0f, float b=1.0f)
{
	thread_local std::mt19937 generator(std::random_device{}());
	std::uniform_real_distribution<float> distribution(a, b);

	return distribution(generator);
}

template<size_t DIM>
inline Vector<DIM> uniformRealRandomVector(float a=0.0f, float b=1.0f)
{
	Vector<DIM> v;
	for (size_t i = 0; i < DIM; i++) {
		v[i] = uniformRealRandomNumber(a, b);
	}

	return v;
}

} // namespace fcpw
