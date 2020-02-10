#pragma once

#include "shape.h"

namespace fcpw {
// source: https://github.com/brandonpelfrey/Fast-BVH

template <int DIM>
struct BvhFlatNode {
	BoundingBox<DIM> bbox;
	int start, nShapes, rightOffset;
};

template <int DIM>
class Bvh: public Aggregate<DIM> {
public:
	// constructor
	Bvh(std::vector<std::shared_ptr<Shape<DIM>>>& shapes_, int leafSize_=4);

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
				  bool checkOcclusion=false, bool countHits=false,
				  bool collectAll=false) const;

	// finds closest point to sphere center
	bool findClosestPoint(BoundingSphere<DIM>& s, Interaction<DIM>& i) const;

private:
	// builds binary tree
	void build();

	// members
	int nNodes, nLeafs, leafSize;
	std::vector<std::shared_ptr<Shape<DIM>>>& shapes;
	std::vector<BvhFlatNode<DIM>> flatTree;
};

} // namespace fcpw

#include "bvh.inl"
