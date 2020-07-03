#pragma once

#include "core/primitive.h"

namespace fcpw {

template<size_t DIM, typename PrimitiveType=Primitive<DIM>>
class Baseline: public Aggregate<DIM> {
public:
	// constructor
	Baseline(const std::vector<PrimitiveType *>& primitives_);

	// returns bounding box
	BoundingBox<DIM> boundingBox() const;

	// returns centroid
	Vector<DIM> centroid() const;

	// returns surface area
	float surfaceArea() const;

	// returns signed volume
	float signedVolume() const;

	// intersects with ray, starting the traversal at the specified node in an aggregate;
	// use this for spatially/temporally coherent queries
	// NOTE: interactions are invalid when checkForOcclusion is enabled
	int intersectFromNode(Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
						  int nodeStartIndex, int aggregateIndex, int& nodesVisited,
						  bool checkForOcclusion=false, bool recordAllHits=false) const;

	// finds closest point to sphere center, starting the traversal at the specified node in an aggregate;
	// use this for spatially/temporally coherent queries
	bool findClosestPointFromNode(BoundingSphere<DIM>& s, Interaction<DIM>& i,
								  int nodeStartIndex, int aggregateIndex,
								  const Vector<DIM>& boundaryHint, int& nodesVisited) const;

protected:
	// members
	const std::vector<PrimitiveType *>& primitives;
	bool primitiveTypeIsAggregate;
};

} // namespace fcpw

#include "baseline.inl"
