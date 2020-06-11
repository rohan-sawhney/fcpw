#pragma once

#include "primitive.h"

namespace fcpw {

template <int DIM>
struct PolygonSoup {
	// constructor
	PolygonSoup() {}

	// constructor
 	PolygonSoup(const std::vector<int>& indices_,
				const std::vector<Vector<DIM>>& positions_):
				indices(indices_), positions(positions_) {}

	// members
	std::vector<int> indices /* a.k.a. vIndices */, eIndices, tIndices;
	std::vector<Vector<DIM>> positions;
	std::vector<Vector<DIM - 1>> textureCoordinates;
	std::vector<Vector<DIM>> vNormals, eNormals; // normalized values
};

////////////////////////////////////////////////////////////////////////////////
// helper class for loading polygons from obj files
struct Index {
	Index() {}

	Index(int v, int vt, int vn) : position(v), uv(vt), normal(vn) {}

	bool operator<(const Index& i) const {
		if (position < i.position) return true;
		if (position > i.position) return false;
		if (uv < i.uv) return true;
		if (uv > i.uv) return false;
		if (normal < i.normal) return true;
		if (normal > i.normal) return false;

		return false;
	}

	int position;
	int uv;
	int normal;
};

// helper function for loading polygons from obj files
inline Index parseFaceIndex(const std::string& token) {
	std::stringstream in(token);
	std::string indexString;
	int indices[3] = {1, 1, 1};

	int i = 0;
	while (std::getline(in, indexString, '/')) {
		if (indexString != "\\") {
			std::stringstream ss(indexString);
			ss >> indices[i++];
		}
	}

	// decrement since indices in OBJ files are 1-based
	return Index(indices[0] - 1, indices[1] - 1, indices[2] - 1);
}

} // namespace fcpw
