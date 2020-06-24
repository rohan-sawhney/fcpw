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
	Bvh_SurfaceArea = 2,
	Bvh_OverlapSurfaceArea = 3,
	Bvh_Volume = 4,
	Bvh_OverlapVolume = 5,
	Sbvh_SurfaceArea = 6,
	Sbvh_Volume = 7
};

std::vector<std::pair<std::string, LoadingOption>> files;
std::string instanceFilename = "";
std::string csgFilename = "";

template<size_t DIM>
class Scene {
public:
	// constructor
	Scene();

	// destructor
	~Scene();

	// loads files
	void loadFiles();

	// builds aggregate
	void buildAggregate(const AggregateType& aggregateType, bool vectorize=false);

#ifdef BENCHMARK_EMBREE
	// builds embree aggregate
	bool buildEmbreeAggregate();
#endif

	// members; NOTE: if initializing scene manually, populate soups, objects,
	// objectTypes, instanceTransforms & csgTree before calling buildAggregate;
	// see loadFiles for example
	std::vector<std::unique_ptr<PolygonSoup<DIM>>> soups;
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

	// clears aggregate
	void clearAggregate();

	// member
	std::vector<Aggregate<DIM> *> objectInstances;
};

} // namespace fcpw

#include "scene.inl"
