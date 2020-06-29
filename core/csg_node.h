#pragma once

#include "primitive.h"

namespace fcpw {

enum class BooleanOperation {
	Union,
	Intersection,
	Difference,
	None
};

template<size_t DIM, typename PrimitiveTypeLeft=Primitive<DIM>, typename PrimitiveTypeRight=Primitive<DIM>>
class CsgNode: public Aggregate<DIM> {
public:
	// constructor
	CsgNode(std::unique_ptr<PrimitiveTypeLeft> left_,
			std::unique_ptr<PrimitiveTypeRight> right_,
			const BooleanOperation& operation_);

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
	// NOTE: interactions are invalid when checkOcclusion is enabled
	int intersectFromNode(Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
						  int nodeStartIndex, int aggregateIndex, int& nodesVisited,
						  bool checkOcclusion=false, bool countHits=false) const;

	// finds closest point to sphere center, starting the traversal at the specified node in an aggregate;
	// use this for spatially/temporally coherent queries
	bool findClosestPointFromNode(BoundingSphere<DIM>& s, Interaction<DIM>& i,
								  int nodeStartIndex, int aggregateIndex,
								  const Vector<DIM>& boundaryHint, int& nodesVisited) const;

private:
	// computes bounding box in world sparce
	void computeBoundingBox();

	// computes interactions for ray intersection
	void computeInteractions(const std::vector<Interaction<DIM>>& isLeft,
							 const std::vector<Interaction<DIM>>& isRight,
							 std::vector<Interaction<DIM>>& is) const;

	// members
	std::unique_ptr<PrimitiveTypeLeft> left;
	std::unique_ptr<PrimitiveTypeRight> right;
	BooleanOperation operation;
	BoundingBox<DIM> box;
	bool leftPrimitiveTypeIsAggregate, rightPrimitiveTypeIsAggregate;
};

} // namespace fcpw

#include "csg_node.inl"
