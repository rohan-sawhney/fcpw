#pragma once

#include "sbvh.h"

namespace fcpw {

template<size_t WIDTH, size_t DIM>
struct MbvhNode {
	// constructor
	MbvhNode(): boxMin(FloatP<WIDTH>(maxFloat)),
				boxMax(FloatP<WIDTH>(minFloat)),
				child(maxInt), parent(-1), leafIndex(-1) {}

	// members
	VectorP<WIDTH, DIM> boxMin, boxMax;
	IntP<WIDTH> child; // use sign to differentiate between inner and leaf nodes
	int parent, leafIndex;
};

template<size_t WIDTH, size_t DIM, typename PrimitiveType=Primitive<DIM>>
class Mbvh: public Aggregate<DIM> {
public:
	// constructor
	Mbvh(const Sbvh<DIM, PrimitiveType> *sbvh_);

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

protected:
	// collapses sbvh into a mbvh
	int collapseSbvh(const Sbvh<DIM, PrimitiveType> *sbvh, int sbvhNodeIndex, int parent, int depth);

	// determines whether mbvh node is a leaf node
	bool isLeafNode(const MbvhNode<WIDTH, DIM>& node) const;

	// populates leaf node
	void populateLeafNode(const MbvhNode<WIDTH, DIM>& node, int leafIndex);

	// populates leaf nodes
	void populateLeafNodes();

	// performs vectorized ray intersection query to line segment
	int intersectLineSegment(const MbvhNode<WIDTH, DIM>& node, int nodeIndex,
							 Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
							 bool countHits) const;

	// performs vectorized ray intersection query to triangle
	int intersectTriangle(const MbvhNode<WIDTH, DIM>& node, int nodeIndex,
						  Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
						  bool countHits) const;

	// performs vectorized closest point query to line segment
	bool findClosestPointLineSegment(const MbvhNode<WIDTH, DIM>& node, int nodeIndex,
									 BoundingSphere<DIM>& s, Interaction<DIM>& i) const;

	// performs vectorized closest point query to triangle
	bool findClosestPointTriangle(const MbvhNode<WIDTH, DIM>& node, int nodeIndex,
								  BoundingSphere<DIM>& s, Interaction<DIM>& i) const;

	// members
	int nNodes, nLeafs, maxDepth, maxLevel;
	const std::vector<PrimitiveType *>& primitives;
	std::vector<MbvhNode<WIDTH, DIM>> flatTree;
	std::vector<VectorP<WIDTH, DIM>> leafNodes;
	std::vector<int> references;
	ObjectType vectorizedLeafType;
	bool primitiveTypeIsAggregate;
};

} // namespace fcpw

#include "mbvh.inl"
