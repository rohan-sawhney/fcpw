#pragma once

#include "primitive.h"

namespace fcpw {

enum class BooleanOperation {
	Union,
	Intersection,
	Difference,
	None
};

template <int DIM>
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

	// intersects with ray
	int intersect(Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
				  bool checkOcclusion=false, bool countHits=false,
				  bool collectAll=false) const;

	// finds closest point to sphere center
	bool findClosestPoint(BoundingSphere<DIM>& s, Interaction<DIM>& i) const;

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

#include "node.inl"
