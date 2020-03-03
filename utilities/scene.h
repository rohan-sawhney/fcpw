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
	Sbvh
};

enum class SimdClass{
	None,
	SSE,
	AVX,
	AVX512
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

	// sets BVH split method
	void setSplitMethod(uint splitMethod);

	// sets leaf size
	void setLeafSize(uint leafSize);

	// set simd vector type
	void setSimdType(uint simdType);

	// set bin size
	void setBinSize(uint binSize);

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
	SimdClass simdType = SimdClass::None;
	uint bvhSplitMethod = 0;
	uint leafSize = 4;
	uint bins = 32;
};

} // namespace fcpw

#include "scene.inl"
