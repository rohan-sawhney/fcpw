#pragma once

#include "sbvh.h"
#ifdef USE_EIGHT_WIDE_BRANCHING
	#define MBVH_BRANCHING_FACTOR 8
	#define MBVH_MAX_DEPTH 154
#else
	#define MBVH_BRANCHING_FACTOR 4
	#define MBVH_MAX_DEPTH 96
#endif

namespace fcpw {

template<size_t DIM>
struct MbvhNode {
	// constructor
	MbvhNode(): boxMin(FloatP<MBVH_BRANCHING_FACTOR>(maxFloat)),
				boxMax(FloatP<MBVH_BRANCHING_FACTOR>(minFloat)),
				child(maxInt) {}

	// members
	VectorP<MBVH_BRANCHING_FACTOR, DIM> boxMin, boxMax;
	IntP<MBVH_BRANCHING_FACTOR> child; // use sign to differentiate between inner and leaf nodes
};

template<size_t WIDTH, size_t DIM, typename PrimitiveType>
struct MbvhLeafNode {
	// members
	VectorP<WIDTH, DIM> positions[0];
	IntP<WIDTH> primitiveIndex;
};

template<size_t WIDTH, size_t DIM>
struct MbvhLeafNode<WIDTH, DIM, LineSegment> {
	// members
	VectorP<WIDTH, DIM> positions[2];
	IntP<WIDTH> primitiveIndex;
};

template<size_t WIDTH, size_t DIM>
struct MbvhLeafNode<WIDTH, DIM, Triangle> {
	// members
	VectorP<WIDTH, DIM> positions[3];
	IntP<WIDTH> primitiveIndex;
};

template<size_t WIDTH, size_t DIM, typename PrimitiveType=Primitive<DIM>>
class Mbvh: public Aggregate<DIM> {
public:
	// constructor
	Mbvh(const Sbvh<DIM, PrimitiveType> *sbvh_, bool printStats_=false);

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
	bool isLeafNode(const MbvhNode<DIM>& node) const;

	// populates leaf node
	void populateLeafNode(const MbvhNode<DIM>& node);

	// populates leaf nodes
	void populateLeafNodes();

	// performs vectorized ray intersection query to line segment
	int intersectLineSegment(const MbvhNode<DIM>& node, int nodeIndex,
							 Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
							 bool countHits) const;

	// performs vectorized ray intersection query to triangle
	int intersectTriangle(const MbvhNode<DIM>& node, int nodeIndex,
						  Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
						  bool countHits) const;

	// performs vectorized closest point query to line segment
	bool findClosestPointLineSegment(const MbvhNode<DIM>& node, int nodeIndex,
									 BoundingSphere<DIM>& s, Interaction<DIM>& i) const;

	// performs vectorized closest point query to triangle
	bool findClosestPointTriangle(const MbvhNode<DIM>& node, int nodeIndex,
								  BoundingSphere<DIM>& s, Interaction<DIM>& i) const;

	// members
	int nNodes, nLeafs, maxDepth, maxLevel;
	const std::vector<PrimitiveType *>& primitives;
	std::vector<MbvhNode<DIM>> flatTree;
	std::vector<MbvhLeafNode<WIDTH, DIM, PrimitiveType>> leafNodes;
	ObjectType vectorizedLeafType;
	bool primitiveTypeIsAggregate;
};

} // namespace fcpw

#include "mbvh.inl"
