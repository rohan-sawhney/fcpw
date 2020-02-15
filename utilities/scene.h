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
	Bvh
};

std::vector<std::pair<std::string, LoadingOption>> files;
std::string instanceFilename = "";
std::string csgFilename = "";

template <int DIM>
class Scene {
public:
	// loads files
	void loadFiles(bool computeWeightedNormals=false, bool randomizeObjectTransforms=false);

	// builds aggregate
	std::shared_ptr<Aggregate<DIM>> buildAggregate(const AggregateType& aggregateType);

	// builds embree aggregate
	std::shared_ptr<Aggregate<DIM>> buildEmbreeAggregate();

	// members
	std::vector<std::shared_ptr<PolygonSoup<DIM>>> soups;
	std::vector<std::vector<std::shared_ptr<Primitive<DIM>>>> objects;
	std::vector<std::vector<Transform<float, DIM, Affine>>> instanceTransforms;
	std::vector<ObjectType> objectTypes;

private:
	// builds csg aggregates
	std::shared_ptr<Aggregate<DIM>> buildCsgAggregate() const;

	// members
	std::vector<std::shared_ptr<Aggregate<DIM>>> objectAggregates;
	std::vector<std::shared_ptr<Primitive<DIM>>> objectInstances;
};

} // namespace fcpw

#include "scene.inl"
