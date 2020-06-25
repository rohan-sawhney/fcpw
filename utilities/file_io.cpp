#include "file_io.h"
#include <map>

namespace fcpw {

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
Index parseFaceIndex(const std::string& token) {
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

void readLineSegmentSoupFromOBJFile(const std::string& filename, PolygonSoup<3>& soup)
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	// initialize
	std::ifstream in(filename);
	if (in.is_open() == false) {
		std::cerr << "Unable to open file: " << filename << std::endl;
		exit(EXIT_FAILURE);
	}

	// parse obj format
	std::string line;
	while (getline(in, line)) {
		std::stringstream ss(line);
		std::string token;
		ss >> token;

		if (token == "v") {
			float x, y, z;
			ss >> x >> y >> z;

			soup.positions.emplace_back(Vector3(x, y, z));

		} else if (token == "f" || token == "l") {
			bool tokenIsF = token == "f";
			std::vector<int> indices;

			while (ss >> token) {
				Index index = parseFaceIndex(token);

				if (index.position < 0) {
					getline(in, line);
					size_t i = line.find_first_not_of("\t\n\v\f\r ");
					index = parseFaceIndex(line.substr(i));
				}

				if (tokenIsF) indices.emplace_back(index.position);
				else soup.indices.emplace_back(index.position);
			}

			if (tokenIsF) {
				int F = (int)indices.size();
				for (int i = 0; i < F - 1; i++) {
					int j = (i + 1)%F;
					soup.indices.emplace_back(indices[i]);
					soup.indices.emplace_back(indices[j]);
				}
			}
		}
	}

	// close
	in.close();
}

void buildLineSegments(PolygonSoup<3>& soup, std::vector<LineSegment *>& lineSegments)
{
	// initialize line segments
	int N = (int)soup.indices.size();
	if (N%2 != 0) {
		std::cerr << "Soup has curve segments with more than 2 vertices" << std::endl;
		exit(EXIT_FAILURE);
	}

	N /= 2;
	lineSegments.resize(N, nullptr);
	LineSegment *contiguousLineSegments = new LineSegment[N];

	for (int i = 0; i < N; i++) {
		lineSegments[i] = &contiguousLineSegments[i];
		lineSegments[i]->soup = &soup;
		lineSegments[i]->index = 2*i;
	}

	// determine whether line segment soup is flat
	bool isFlat = true;
	for (int i = 0; i < (int)soup.positions.size(); i++) {
		if (std::fabs(soup.positions[i][2]) > epsilon) {
			isFlat = false;
			break;
		}
	}

	// swap indices if segments of closed curve are oriented in clockwise order
	if (isFlat && N > 0 && soup.indices[0] == soup.indices[2*(N - 1) + 1]) {
		float signedVolume = 0.0f;
		for (int i = 0; i < N; i++) {
			signedVolume += lineSegments[i]->signedVolume();
		}

		if (signedVolume < 0) {
			for (int i = 0; i < N; i++) {
				std::swap(soup.indices[2*i], soup.indices[2*i + 1]);
			}
		}
	}
}

void computeWeightedLineSegmentNormals(const std::vector<LineSegment *>& lineSegments, PolygonSoup<3>& soup)
{
	int N = (int)soup.indices.size()/2;
	int V = (int)soup.positions.size();
	soup.vNormals.resize(V, zeroVector<3>());

	for (int i = 0; i < N; i++) {
		Vector3 n = lineSegments[i]->normal(true);
		soup.vNormals[soup.indices[lineSegments[i]->index]] += n;
		soup.vNormals[soup.indices[lineSegments[i]->index + 1]] += n;
	}

	for (int i = 0; i < V; i++) {
		soup.vNormals[i] = unit<3>(soup.vNormals[i]);
	}
}

void readTriangleSoupFromOBJFile(const std::string& filename, PolygonSoup<3>& soup)
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	// initialize
	std::ifstream in(filename);
	if (in.is_open() == false) {
		std::cerr << "Unable to open file: " << filename << std::endl;
		exit(EXIT_FAILURE);
	}

	// parse obj format
	std::string line;
	while (getline(in, line)) {
		std::stringstream ss(line);
		std::string token;
		ss >> token;

		if (token == "v") {
			float x, y, z;
			ss >> x >> y >> z;

			soup.positions.emplace_back(Vector3(x, y, z));

		} else if (token == "vt") {
			float u, v;
			ss >> u >> v;

			soup.textureCoordinates.emplace_back(Vector2(u, v));

		} else if (token == "f") {
			while (ss >> token) {
				Index index = parseFaceIndex(token);

				if (index.position < 0) {
					getline(in, line);
					size_t i = line.find_first_not_of("\t\n\v\f\r ");
					index = parseFaceIndex(line.substr(i));
				}

				soup.indices.emplace_back(index.position);
				soup.tIndices.emplace_back(index.uv);
			}
		}
	}

	// close
	in.close();

	if (soup.textureCoordinates.size() == 0) {
		soup.tIndices.clear();
	}
}

void buildTriangles(const PolygonSoup<3>& soup, std::vector<Triangle *>& triangles)
{
	// initialize triangles
	int N = (int)soup.indices.size();
	if (N%3 != 0) {
		std::cerr << "Soup has non-triangular polygons" << std::endl;
		exit(EXIT_FAILURE);
	}

	N /= 3;
	triangles.resize(N, nullptr);
	Triangle *contiguousTriangles = new Triangle[N];

	for (int i = 0; i < N; i++) {
		triangles[i] = &contiguousTriangles[i];
		triangles[i]->soup = &soup;
		triangles[i]->index = 3*i;
	}
}

void computeWeightedTriangleNormals(const std::vector<Triangle *>& triangles, PolygonSoup<3>& soup)
{
	// set edge indices
	int E = 0;
	int N = (int)soup.indices.size()/3;
	std::map<std::pair<int, int>, int> indexMap;

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < 3; j++) {
			int k = (j + 1)%3;
			int I = soup.indices[3*i + j];
			int J = soup.indices[3*i + k];
			if (I > J) std::swap(I, J);
			std::pair<int, int> e(I, J);

			if (indexMap.find(e) == indexMap.end()) indexMap[e] = E++;
			soup.eIndices.emplace_back(indexMap[e]);
		}
	}

	// compute normals
	int V = (int)soup.positions.size();
	soup.vNormals.resize(V, zeroVector<3>());
	soup.eNormals.resize(E, zeroVector<3>());

	for (int i = 0; i < N; i++) {
		Vector3 n = triangles[i]->normal(true);

		for (int j = 0; j < 3; j++) {
			soup.vNormals[soup.indices[triangles[i]->index + j]] += n;
			soup.eNormals[soup.eIndices[triangles[i]->index + j]] += n;
		}
	}

	for (int i = 0; i < V; i++) soup.vNormals[i] = unit<3>(soup.vNormals[i]);
	for (int i = 0; i < E; i++) soup.eNormals[i] = unit<3>(soup.eNormals[i]);
}

void loadCsgTree(const std::string& filename,
				 std::unordered_map<int, CsgTreeNode>& csgTree)
{
	// load scene
	std::ifstream in(filename);
	if (in.is_open() == false) {
		std::cerr << "Unable to open file: " << filename << std::endl;
		exit(EXIT_FAILURE);
	}

	// parse obj format
	std::string line;
	while (getline(in, line)) {
		std::stringstream ss(line);
		int node;
		ss >> node;

		std::string operationStr, child1Str, child2Str;
		ss >> operationStr >> child1Str >> child2Str;

		std::size_t found1 = child1Str.find_last_of("_");
		std::size_t found2 = child2Str.find_last_of("_");
		csgTree[node].child1 = std::stoi(child1Str.substr(found1 + 1));
		csgTree[node].child2 = std::stoi(child2Str.substr(found2 + 1));
		csgTree[node].isLeafChild1 = child1Str.find("node_") == std::string::npos;
		csgTree[node].isLeafChild2 = child2Str.find("node_") == std::string::npos;
		csgTree[node].operation = operationStr == "Union" ? BooleanOperation::Union :
								 (operationStr == "Intersection" ? BooleanOperation::Intersection :
								 (operationStr == "Difference" ? BooleanOperation::Difference : BooleanOperation::None));
	}

	// close file
	in.close();
}

} // namespace fcpw
