#pragma once

#include "core/csg_node.h"
#include "geometry/line_segments.h"
#include "geometry/triangles.h"
#include <unordered_map>

namespace fcpw {

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

struct CsgTreeNode {
	int child1, child2;
	bool isLeafChild1, isLeafChild2;
	BooleanOperation operation;
};

template<size_t DIM>
class SceneData {
public:
	// constructor
	SceneData();

	// clears object data
	void clearObjectData();

	// clears aggregate data
	void clearAggregateData();

	// members
	std::vector<PolygonSoup<DIM>> soups;
	std::unordered_map<int, std::vector<std::pair<ObjectType, int>>> soupToObjectsMap;

	std::vector<std::unique_ptr<std::vector<LineSegment>>> lineSegmentObjects;
	std::vector<std::unique_ptr<std::vector<Triangle>>> triangleObjects;
	std::vector<std::vector<Transform<DIM>>> instanceTransforms;
	std::unordered_map<int, CsgTreeNode> csgTree;

	std::unique_ptr<Aggregate<DIM>> aggregate;

private:
	// members
	std::vector<std::vector<LineSegment *>> lineSegmentObjectPtrs;
	std::vector<std::vector<Triangle *>> triangleObjectPtrs;
	std::vector<std::vector<GeometricPrimitive<DIM> *>> mixedObjectPtrs;

	std::vector<std::unique_ptr<Aggregate<DIM>>> aggregateInstances;
	std::vector<Aggregate<DIM> *> aggregateInstancePtrs;

	template<size_t U>
	friend class Scene;
};

} // namespace fcpw

#include "scene_data.inl"
