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
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <glog/logging.h>
#include <glog/raw_logging.h>
#ifdef PROFILE
	#include <profiler/Profiler.h>
#endif

namespace fcpw {

using namespace Eigen;

template <int DIM>
using Vector = Matrix<float, DIM, 1>;
template <int DIM>
class Ray;
template <int DIM>
struct BoundingSphere;
template <int DIM>
class BoundingBox;
template <int DIM>
class Interaction;
template <int DIM>
class Primitive;

static const float minFloat = std::numeric_limits<float>::lowest();
static const float maxFloat = std::numeric_limits<float>::max();
static const float epsilon = std::numeric_limits<float>::epsilon();
static const float inf = std::numeric_limits<float>::infinity();

template <typename T, typename U, typename V>
inline T clamp(T val, U low, V high) {
	if (val < low) return low;
	else if (val > high) return high;

	return val;
}

template <int DIM>
inline Vector<DIM> reflect(const Vector<DIM>& d, const Vector<DIM>& n) {
	// NOTE: d is assumed to be going out of the surface
	return -d + 2.0f*n.dot(d)*n;
}

inline int factorial(int n) {
	if (n < 2) return 1;

	int result = n;
	for (int i = n - 1; i > 1; i--) {
		result *= i;
	}

	return result;
}

inline float gamma(int n) {
	return (n*epsilon)/(1.0f - n*epsilon);
}

inline float radians(float deg) {
	return (M_PI/180.0f)*deg;
}

inline float degrees(float rad) {
	return (180.0f/M_PI)*rad;
}

inline float cross2d(const Vector2f& u, const Vector2f& v) {
	return u(0)*v(1) - u(1)*v(0);
}

inline float uniformRealRandomNumber(float a=0.0f, float b=1.0f)
{
	thread_local std::mt19937 generator(std::random_device{}());
	std::uniform_real_distribution<float> distribution(a, b);

	return distribution(generator);
}

template <int DIM>
inline Vector<DIM> uniformRealRandomVector(float a=0.0f, float b=1.0f)
{
	Vector<DIM> v;
	for (int i = 0; i < DIM; i++) {
		v(i) = uniformRealRandomNumber(a, b);
	}

	return v;
}

} // namespace fcpw
