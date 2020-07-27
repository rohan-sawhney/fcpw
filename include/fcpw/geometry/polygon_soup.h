#pragma once

#include <fcpw/core/primitive.h>

namespace fcpw {

template<size_t DIM>
struct PolygonSoup {
	// constructor
	PolygonSoup() {}

	// members
	std::vector<Vector<DIM>> positions;
	std::vector<Vector<DIM>> vNormals, eNormals; // normalized values
	std::vector<Vector<DIM - 1>> textureCoordinates;
	std::vector<int> indices /* a.k.a. vIndices */, eIndices, tIndices;
};

} // namespace fcpw
