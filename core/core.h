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
#include <glog/logging.h>
#include <glog/raw_logging.h>
#include "vector_operations.h"
#ifdef PROFILE
	#include <profiler/Profiler.h>
#endif

namespace fcpw {

template<size_t DIM>
struct Ray;
template<size_t DIM>
struct BoundingSphere;
template<size_t DIM>
struct BoundingBox;
template<size_t DIM>
struct Interaction;
template<size_t DIM>
class Primitive;
template<size_t DIM>
class GeometricPrimitive;
template<size_t DIM>
class Aggregate;
template<size_t DIM>
class TransformedAggregate;
template<size_t DIM, typename PrimitiveType=Primitive<DIM>>
class CsgNode;

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

inline float gamma(int n) {
	return (n*epsilon)/(1.0f - n*epsilon);
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
	for (int i = 0; i < DIM; i++) {
		v[i] = uniformRealRandomNumber(a, b);
	}

	return v;
}

} // namespace fcpw
