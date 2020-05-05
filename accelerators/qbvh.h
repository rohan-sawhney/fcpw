#pragma once

#include "sbvh.h"

namespace fcpw {

template <int WIDTH, int DIM>
struct QbvhNode {
	// constructor
	QbvhNode(): boxMin(FloatP<WIDTH>(maxFloat)),
				boxMax(FloatP<WIDTH>(minFloat)),
				child(-1), parent(-1) {
		for (int i = 0; i < DIM; i++) axis[i] = -1;
	}

	// members
	VectorP<WIDTH, DIM> boxMin, boxMax;
	IntP<WIDTH> child;
	int axis[DIM];
	int parent;
};

template <int WIDTH, int DIM>
class Qbvh: public Aggregate<DIM> {
public:
	// constructor
	Qbvh(const std::shared_ptr<Sbvh<DIM>>& sbvh_);

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
};

} // namespace fcpw

#include "qbvh.inl"
