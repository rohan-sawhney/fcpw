#pragma once

#include "sbvh.h"

namespace fcpw {

template <int WIDTH, int DIM>
struct MbvhNode {
	// constructor
	MbvhNode(): boxMin(FloatP<WIDTH>(maxFloat)),
				boxMax(FloatP<WIDTH>(minFloat)),
				child(-1), parent(-1) {
		for (int i = 0; i < DIM; i++) splitDim[i] = -1;
	}

	// members
	VectorP<WIDTH, DIM> boxMin, boxMax;
	IntP<WIDTH> child; // use sign to differentiate between inner and leaf nodes
	int splitDim[DIM];
	int parent;
};

template <int WIDTH, int DIM>
class Mbvh: public Aggregate<DIM> {
public:
	// constructor
	Mbvh(const std::shared_ptr<Sbvh<DIM>>& sbvh_);

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

protected:
	// collapses sbvh into a mbvh
	int collapseSbvh(const std::shared_ptr<Sbvh<DIM>>& sbvh,
					 int sbvhNodeIndex, int parent, int depth);

	// members
	int nNodes, nLeafs, maxDepth, maxLevel;
	const std::vector<std::shared_ptr<Primitive<DIM>>>& primitives;
	std::vector<MbvhNode<WIDTH, DIM>> nodes;
	std::vector<std::pair<int, int>> stackSbvhNodes;
	std::vector<int> references;
};

} // namespace fcpw

#include "mbvh.inl"
