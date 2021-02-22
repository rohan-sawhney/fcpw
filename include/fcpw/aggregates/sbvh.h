#pragma once

#include <fcpw/core/primitive.h>
#include <tuple>
#define FCPW_SBVH_MAX_DEPTH 64

namespace fcpw {
// modified version of https://github.com/brandonpelfrey/Fast-BVH and
// https://github.com/straaljager/GPU-path-tracing-with-CUDA-tutorial-4

enum class CostHeuristic {
	LongestAxisCenter,
	SurfaceArea,
	OverlapSurfaceArea,
	Volume,
	OverlapVolume
};

template<size_t DIM>
struct SbvhNode {
	// members
	BoundingBox<DIM> box;
	union {
		int referenceOffset;
		int secondChildOffset;
	};
	int nReferences;
};

struct BvhTraversal {
	// constructors
	BvhTraversal(): node(-1), distance(0.0f) {}
	BvhTraversal(int node_, float distance_): node(node_), distance(distance_) {}

	// members
	int node; // node index
	float distance; // minimum distance (parametric, squared, ...) to this node
};

template<size_t DIM, typename PrimitiveType>
using SortPositionsFunc = std::function<void(const std::vector<SbvhNode<DIM>>&, std::vector<PrimitiveType *>&)>;

template<size_t DIM, typename PrimitiveType=Primitive<DIM>>
class Sbvh: public Aggregate<DIM> {
public:
	// constructor
	Sbvh(const CostHeuristic& costHeuristic_,
		 std::vector<PrimitiveType *>& primitives_,
		 SortPositionsFunc<DIM, PrimitiveType> sortPositions_={},
		 bool printStats_=false, bool packLeaves_=false, int leafSize_=4, int nBuckets_=8);

	// returns bounding box
	BoundingBox<DIM> boundingBox() const;

	// returns centroid
	Vector<DIM> centroid() const;

	// returns surface area
	float surfaceArea() const;

	// returns signed volume
	float signedVolume() const;

	// intersects with ray, starting the traversal at the specified node in an aggregate
	// NOTE: interactions are invalid when checkForOcclusion is enabled
	int intersectFromNode(Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
						  int nodeStartIndex, int aggregateIndex, int& nodesVisited,
						  bool checkForOcclusion=false, bool recordAllHits=false) const;

	int intersectFromNodeTimed(Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
						  int nodeStartIndex, int aggregateIndex, int& nodesVisited, uint64_t& ticks,
						  bool checkForOcclusion=false, bool recordAllHits=false) const;

	// finds closest point to sphere center, starting the traversal at the specified node in an aggregate
	bool findClosestPointFromNode(BoundingSphere<DIM>& s, Interaction<DIM>& i,
								  int nodeStartIndex, int aggregateIndex,
								  const Vector<DIM>& boundaryHint, int& nodesVisited) const;

	bool findClosestPointFromNodeTimed(BoundingSphere<DIM>& s, Interaction<DIM>& i,
									   int nodeStartIndex, int aggregateIndex,
									   const Vector<DIM>& boundaryHint, int& nodesVisited, uint64_t& ticks) const;

protected:
	// computes split cost based on heuristic
	float computeSplitCost(const BoundingBox<DIM>& boxLeft,
						   const BoundingBox<DIM>& boxRight,
						   float parentSurfaceArea, float parentVolume,
						   int nReferencesLeft, int nReferencesRight,
						   int depth) const;

	// computes object split
	float computeObjectSplit(const BoundingBox<DIM>& nodeBoundingBox,
							 const BoundingBox<DIM>& nodeCentroidBox,
							 const std::vector<BoundingBox<DIM>>& referenceBoxes,
							 const std::vector<Vector<DIM>>& referenceCentroids,
							 int depth, int nodeStart, int nodeEnd, int& splitDim,
							 float& splitCoord, BoundingBox<DIM>& boxIntersected);

	// performs object split
	int performObjectSplit(int nodeStart, int nodeEnd, int splitDim, float splitCoord,
						   std::vector<BoundingBox<DIM>>& referenceBoxes,
						   std::vector<Vector<DIM>>& referenceCentroids);

	// helper function to build binary tree
	void buildRecursive(std::vector<BoundingBox<DIM>>& referenceBoxes,
						std::vector<Vector<DIM>>& referenceCentroids,
						std::vector<SbvhNode<DIM>>& buildNodes,
						int parent, int start, int end, int depth);

	// builds binary tree
	void build();

	// processes subtree for intersection
	bool processSubtreeForIntersection(Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
									   int nodeStartIndex, int aggregateIndex, bool checkForOcclusion,
									   bool recordAllHits, BvhTraversal *subtree,
									   float *boxHits, int& hits, int& nodesVisited) const;

	bool processSubtreeForIntersectionTimed(Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
									   int nodeStartIndex, int aggregateIndex, bool checkForOcclusion,
									   bool recordAllHits, BvhTraversal *subtree,
									   float *boxHits, int& hits, int& nodesVisited, uint64_t& ticks) const;

	// processes subtree for closest point
	void processSubtreeForClosestPoint(BoundingSphere<DIM>& s, Interaction<DIM>& i,
									   int nodeStartIndex, int aggregateIndex,
									   const Vector<DIM>& boundaryHint,
									   BvhTraversal *subtree, float *boxHits,
									   bool& notFound, int& nodesVisited) const;

	void processSubtreeForClosestPointTimed(BoundingSphere<DIM>& s, Interaction<DIM>& i,
									   int nodeStartIndex, int aggregateIndex,
									   const Vector<DIM>& boundaryHint,
									   BvhTraversal *subtree, float *boxHits,
									   bool& notFound, int& nodesVisited, uint64_t& ticks) const;

	// members
	CostHeuristic costHeuristic;
	int nNodes, nLeafs, leafSize, nBuckets, maxDepth, depthGuess;
	std::vector<std::pair<BoundingBox<DIM>, int>> buckets, rightBucketBoxes;
	std::vector<PrimitiveType *>& primitives;
	std::vector<SbvhNode<DIM>> flatTree;
	bool packLeaves, primitiveTypeIsAggregate;

	template<size_t U, size_t V, typename W>
	friend class Mbvh;
};

} // namespace fcpw

#include "sbvh.inl"
