#pragma once

#include "file_io.h"

namespace fcpw {

enum class LoadingOption {
	ObjLineSegments,
	ObjTriangles
};

enum class ObjectType {
	LineSegments,
	Triangles,
	Mixed
};

enum class AggregateType {
	Baseline = 0,
	Bvh_LongestAxisCenter = 1,
	Bvh_OverlapSurfaceArea = 2,
	Bvh_SurfaceArea = 3,
	Bvh_OverlapVolume = 4,
	Bvh_Volume = 5
};

std::vector<std::pair<std::string, LoadingOption>> files;
std::string instanceFilename = "";
std::string csgFilename = "";

template<size_t DIM>
class Scene {
public:
	// constructor
	Scene(bool computeNormals_);

	// destructor
	~Scene();

	// loads files
	void loadFiles();

	// clears aggregate
	void clearAggregate();

	// builds aggregate
	void buildAggregate(const AggregateType& aggregateType,
						bool printStats=false, bool vectorize=false);

	// members; NOTE: if initializing scene manually, populate soups, objects,
	// objectTypes, instanceTransforms & csgTree before calling buildAggregate;
	// see loadFiles for example
	std::vector<PolygonSoup<DIM>> soups;
	std::vector<std::vector<LineSegment *>> lineSegmentObjects;
	std::vector<std::vector<Triangle *>> triangleObjects;
	std::vector<std::vector<GeometricPrimitive<DIM> *>> mixedObjects;
	std::vector<ObjectType> objectTypes;
	std::vector<std::vector<Transform<DIM>>> instanceTransforms;
	std::unordered_map<int, CsgTreeNode> csgTree;
	Aggregate<DIM> *aggregate;

private:
	// clears data
	void clearData();

	// members
	std::vector<Aggregate<DIM> *> objectInstances;
	bool computeNormals;
};

} // namespace fcpw

#include "scene.inl"
