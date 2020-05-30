#pragma once

#include "csg_node.h"
#include "geometry/triangles.h"
#include <unordered_map>

namespace fcpw {

enum class LoadingOption {
	ObjTriangles
};

enum class ObjectType {
	Triangles
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

struct CsgTreeNode {
	int child1, child2;
	bool isLeafChild1, isLeafChild2;
	BooleanOperation operation;
};

std::vector<std::pair<std::string, LoadingOption>> files;
std::string instanceFilename = "";
std::string csgFilename = "";

template<int DIM>
class Scene {
public:
	// loads files
	void loadFiles(bool computeWeightedNormals=false);

	// builds aggregate
	void buildAggregate(const AggregateType& aggregateType, bool vectorize=false);

#ifdef BENCHMARK_EMBREE
	// builds embree aggregate
	void buildEmbreeAggregate();
#endif

	// members
	std::vector<std::shared_ptr<PolygonSoup<DIM>>> soups;
	std::vector<std::vector<std::shared_ptr<Primitive<DIM>>>> objects;
	std::vector<std::vector<Transform<DIM>>> instanceTransforms;
	std::vector<ObjectType> objectTypes;
	std::shared_ptr<Aggregate<DIM>> aggregate;

private:
	// members
	std::unordered_map<int, CsgTreeNode> csgTree;
	std::vector<std::shared_ptr<Primitive<DIM>>> objectInstances;
};

} // namespace fcpw

#include "scene.inl"
