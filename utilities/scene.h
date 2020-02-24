#pragma once

#include "primitive.h"
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
	Baseline,
	Bvh,
	Bvh_SAH,
	Bvh_Vol,
	Bvh_Overlap_SAH,
	Bvh_Overlap_Vol,
	Sbvh,
	SSEBvh,
	AVXBvh,
	AVX512Bvh
};

struct CsgTreeNode;
std::vector<std::pair<std::string, LoadingOption>> files;
std::string instanceFilename = "";
std::string csgFilename = "";

template <int DIM>
class Scene {
public:
	// loads files
	void loadFiles(bool computeWeightedNormals=false, bool randomizeObjectTransforms=false);

	// builds aggregate
	void buildAggregate(const AggregateType& aggregateType);

#ifdef BENCHMARK_EMBREE
	// builds embree aggregate
	void buildEmbreeAggregate();
#endif

	// members
	std::vector<std::shared_ptr<PolygonSoup<DIM>>> soups;
	std::vector<std::vector<std::shared_ptr<Primitive<DIM>>>> objects;
	std::vector<std::vector<Transform<float, DIM, Affine>>> instanceTransforms;
	std::vector<ObjectType> objectTypes;
	std::shared_ptr<Aggregate<DIM>> aggregate;

private:
	// members
	std::unordered_map<int, CsgTreeNode> csgTree;
	std::vector<std::shared_ptr<Primitive<DIM>>> objectInstances;
};

} // namespace fcpw

#include "scene.inl"
