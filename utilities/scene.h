#pragma once

#include "primitive.h"
#include "geometry/triangles.h"
#include <unordered_map>

namespace fcpw {

std::vector<std::pair<std::string, std::string>> files;
std::string instanceFilename = "";
std::string csgFilename = "";

template <int DIM>
class Scene {
public:
	// loads files
	void loadFiles(bool computeWeightedNormals=false, bool randomizeObjectTransforms=false);

	// builds aggregate
	std::shared_ptr<Aggregate<DIM>> buildAggregate(const std::string& aggregateType);

	// members
	std::vector<std::shared_ptr<PolygonSoup<DIM>>> soups;
	std::vector<std::vector<std::shared_ptr<Primitive<DIM>>>> objects;
	std::vector<std::vector<Transform<float, DIM, Affine>>> instanceTransforms;
	std::vector<std::string> objectTypes;

private:
	// builds csg aggregates
	std::shared_ptr<Aggregate<DIM>> buildCsgAggregate() const;

	// members
	std::vector<std::shared_ptr<Aggregate<DIM>>> objectAggregates;
	std::vector<std::shared_ptr<Primitive<DIM>>> objectInstances;
};

} // namespace fcpw

#include "scene.inl"