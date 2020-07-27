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

} // namespace fcpw
