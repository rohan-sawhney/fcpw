#pragma once

#include "primitive.h"

namespace fcpw {

enum class BooleanOperation {
	Union,
	Intersection,
	Difference,
	None
};

template<size_t DIM, typename PrimitiveType=Primitive<DIM>>
class CsgNode: public Aggregate<DIM> {
public:
	// constructor
	CsgNode(const std::shared_ptr<PrimitiveType>& left_,
			const std::shared_ptr<PrimitiveType>& right_,
			const BooleanOperation& operation_);

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
								  int nodeStartIndex, const Vector<DIM>& boundaryHint,
								  int& nodesVisited) const;

private:
	// computes bounding box in world sparce
	void computeBoundingBox();

	// computes interactions for ray intersection
	void computeInteractions(const std::vector<Interaction<DIM>>& isLeft,
							 const std::vector<Interaction<DIM>>& isRight,
							 std::vector<Interaction<DIM>>& is) const;

	// members
	std::shared_ptr<PrimitiveType> left, right;
	BooleanOperation operation;
	BoundingBox<DIM> box;
	bool primitiveTypeIsAggregate;
};

} // namespace fcpw

#include "csg_node.inl"
