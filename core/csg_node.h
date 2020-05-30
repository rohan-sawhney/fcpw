#pragma once

#include "primitive.h"

namespace fcpw {

enum class BooleanOperation {
	Union,
	Intersection,
	Difference,
	None
};

template<int DIM>
class CsgNode: public Aggregate<DIM> {
public:
	// constructor
	CsgNode(const std::shared_ptr<Primitive<DIM>>& left_,
			const std::shared_ptr<Primitive<DIM>>& right_,
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
	int intersectFromNode(Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
						  int nodeStartIndex, int& nodesVisited,
						  bool checkOcclusion=false, bool countHits=false) const;

	// finds closest point to sphere center, starting the traversal at the specified node;
	// use this for spatially/temporally coherent queries
	bool findClosestPointFromNode(BoundingSphere<DIM>& s, Interaction<DIM>& i,
								  int nodeStartIndex, int& nodesVisited) const;

private:
	// computes bounding box in world sparce
	void computeBoundingBox();

	// computes interactions for ray intersection
	void computeInteractions(const std::vector<Interaction<DIM>>& isLeft,
							 const std::vector<Interaction<DIM>>& isRight,
							 std::vector<Interaction<DIM>>& is) const;

	// members
	std::shared_ptr<Primitive<DIM>> left;
	std::shared_ptr<Primitive<DIM>> right;
	BooleanOperation operation;
	BoundingBox<DIM> box;
};

} // namespace fcpw

#include "csg_node.inl"
