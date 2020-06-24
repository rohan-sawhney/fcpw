#pragma once

#include "core/csg_node.h"
#include "geometry/line_segments.h"
#include "geometry/triangles.h"
#include <unordered_map>
#include <fstream>
#include <sstream>

namespace fcpw {

// computes weighted normals at vertices
void computeWeightedLineSegmentNormals(const std::vector<LineSegment *>& lineSegments, PolygonSoup<3>& soup);

// reads soup from obj file
void readLineSegmentSoupFromOBJFile(const std::string& filename, PolygonSoup<3>& soup, bool& isFlat);

// reads line segment soup from obj file
void readLineSegmentSoupFromOBJFile(const std::string& filename, PolygonSoup<3>& soup,
									std::vector<LineSegment *>& lineSegments,
									bool computeWeightedNormals=true);

// computes weighted normals at vertices and edges
void computeWeightedTriangleNormals(const std::vector<Triangle *>& triangles, PolygonSoup<3>& soup);

// reads soup from obj file
void readTriangleSoupFromOBJFile(const std::string& filename, PolygonSoup<3>& soup);

// reads triangle soup from obj file
void readTriangleSoupFromOBJFile(const std::string& filename, PolygonSoup<3>& soup,
								 std::vector<Triangle *>& triangles, bool computeWeightedNormals=true);

// loads instance transforms from file
template<size_t DIM>
void loadInstanceTransforms(const std::string& filename,
							std::vector<std::vector<Transform<DIM>>>& instanceTransforms);

struct CsgTreeNode {
	int child1, child2;
	bool isLeafChild1, isLeafChild2;
	BooleanOperation operation;
};

// loads csg tree from file
void loadCsgTree(const std::string& filename, std::unordered_map<int, CsgTreeNode>& csgTree);

} // namespace fcpw

#include "file_io.inl"
