#pragma once

#include "bvh_common.h"
#include "bvh_simd.h"

namespace fcpw {
// source: https://github.com/brandonpelfrey/Fast-BVH
// TODO:
// - implement more tree construction heuristics (sah, volume ratio, volume overlap)
// - implement sbvh
// - implement mbvh with vectorization
// - estimate closest point radius (i.e., conversative guess of spherical region containing query point)
// - implement "queueless" closest point traversal
// - try bottom up closest point traversal strategy

template <int DIM>
class Bvh: public Aggregate<DIM> {
public:
	// constructor
	Bvh(std::vector<std::shared_ptr<Primitive<DIM>>>& primitives_, int leafSize_=4, int splittingMethod_=0, int binCount_=32, bool packLeaves=false, bool makeBvh=true);

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

	// converts bvh into a SIMD-parallel bvh (mbvh)
	virtual void convert(const int simdWidth, std::shared_ptr<Aggregate<DIM>>& mbvh);

protected:
	// applies closest point to leaves
	virtual bool applyClosestPoint(BoundingSphere<DIM>& s, Interaction<DIM>& i, int pos) const;

	// builds binary tree
	void build(bool packLeaves=false);

	// members
	std::vector<std::shared_ptr<Primitive<DIM>>>& primitives;
	std::vector<BvhFlatNode<DIM>> flatTree;
	int nNodes, nLeaves, leafSize, splittingMethod, binCount, depth;
	float totalCost;
};

} // namespace fcpw

#include "bvh.inl"
