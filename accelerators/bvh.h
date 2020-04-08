#pragma once

#include "primitive.h"

namespace fcpw {
// source: https://github.com/brandonpelfrey/Fast-BVH
// TODO:
// - implement sbvh
// - implement mbvh with vectorization
// - build a spatial data structure on top of bvh
// - estimate closest point radius (i.e., conversative guess of spherical region containing query point)
// - implement "queueless" closest point traversal
// - try bottom up closest point traversal strategy

enum class CostHeuristic {
	LongestAxisCenter,
	SurfaceArea,
	OverlapSurfaceArea,
	Volume,
	OverlapVolume
};

template <int DIM>
struct BvhFlatNode {
	// constructor
	BvhFlatNode(): bbox(false), start(0), nPrimitives(0), rightOffset(0) {}

	// members
	BoundingBox<DIM> bbox;
	int start, nPrimitives, rightOffset;
};

template <int DIM>
class Bvh: public Aggregate<DIM> {
public:
	// constructor
	Bvh(std::vector<std::shared_ptr<Primitive<DIM>>>& primitives_,
		const CostHeuristic& costHeuristic_, int leafSize_=4);

	// returns bounding box
	BoundingBox<DIM> boundingBox() const;

	// returns centroid
	Vector<DIM> centroid() const;

	// returns surface area
	float surfaceArea() const;

	// returns signed volume
	float signedVolume() const;

	// intersects with ray
	int intersect(Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
				  bool checkOcclusion=false, bool countHits=false) const;

	// finds closest point to sphere center
	bool findClosestPoint(BoundingSphere<DIM>& s, Interaction<DIM>& i) const;

protected:
	// builds binary tree
	void build(const CostHeuristic& costHeuristic);

	// members
	int nNodes, nLeafs, leafSize;
	std::vector<std::shared_ptr<Primitive<DIM>>>& primitives;
	std::vector<BvhFlatNode<DIM>> flatTree;
};

} // namespace fcpw

#include "bvh.inl"
