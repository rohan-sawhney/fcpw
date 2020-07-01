#pragma once

#include "fcpw.h"
#include <fstream>
#include <sstream>

enum class LoadingOption {
	ObjLineSegments,
	ObjTriangles
};

std::vector<std::pair<std::string, LoadingOption>> files;
std::string instanceFilename = "";
std::string csgFilename = "";

namespace fcpw {

template<size_t DIM>
class SceneLoader {
public:
	// loads files
	void loadFiles(Scene<DIM>& scene, bool computeNormals);

private:
	// member
	Scene<DIM>& scene;
	bool computeNormals;
};

// reads a line segment soup from an obj file
void readLineSegmentSoupFromOBJFile(const std::string& filename, PolygonSoup<3>& soup);

// reads a triangle soup from an obj file
void readTriangleSoupFromOBJFile(const std::string& filename, PolygonSoup<3>& soup);

// loads instance transforms from file
template<size_t DIM>
void loadInstanceTransforms(const std::string& filename,
							std::vector<std::vector<Transform<DIM>>>& instanceTransforms);

// loads a csg tree from file
void loadCsgTree(const std::string& filename, std::unordered_map<int, CsgTreeNode>& csgTree);

} // namespace fcpw

#include "file_io.inl"
