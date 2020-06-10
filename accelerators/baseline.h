#pragma once

#include "primitive.h"

namespace fcpw {

template<int DIM>
class Baseline: public Aggregate<DIM> {
public:
	// constructor
	Baseline(const std::vector<std::shared_ptr<Primitive<DIM>>>& primitives_);

	// returns bounding box
	BoundingBox<DIM> boundingBox() const;

	// returns centroid
	Vector<DIM> centroid() const;

	// returns surface area
	float surfaceArea() const;

	// returns signed volume
	float signedVolume() const;

	// intersects with ray, starting the traversal at the specified node;
	// use this for spatially/temporally coherent queries
	// NOTE: interactions are invalid when checkOcclusion is enabled
	int intersectFromNode(Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
						  int nodeStartIndex, int& nodesVisited,
						  bool checkOcclusion=false, bool countHits=false) const;

	// finds closest point to sphere center, starting the traversal at the specified node;
	// use this for spatially/temporally coherent queries
	bool findClosestPointFromNode(BoundingSphere<DIM>& s, Interaction<DIM>& i,
								  int nodeStartIndex, int& nodesVisited) const;

protected:
	// members
	const std::vector<std::shared_ptr<Primitive<DIM>>>& primitives;
};

} // namespace fcpw

#include "baseline.inl"
