#pragma once

#include "primitive.h"
#include <tuple>
#include <deque>

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

template<int DIM>
struct SbvhNode {
	// constructor
	SbvhNode(): parent(-1), start(-1), nReferences(-1), rightOffset(-1) {}

	// members
	BoundingBox<DIM> box;
	int parent, start, nReferences, rightOffset;
};

struct BvhTraversal {
	// constructors
	BvhTraversal(): node(-1), distance(0.0f) {}
	BvhTraversal(int node_, float distance_): node(node_), distance(distance_) {}

	// members
	int node; // node index
	float distance; // minimum distance (parametric, squared, ...) to this node
};

template<int DIM>
class Sbvh: public Aggregate<DIM> {
public:
	// constructor
	Sbvh(std::vector<std::shared_ptr<Primitive<DIM>>>& primitives_,
		 const CostHeuristic& costHeuristic_, float splitAlpha_,
		 bool packLeaves_=false, int leafSize_=4, int nBuckets_=8, int nBins_=8);

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
								  int nodeStartIndex, const Vector<DIM>& dirGuess,
								  int& nodesVisited) const;

protected:
	// determines object type
	void determineObjectType();

	// computes split cost based on heuristic
	float computeSplitCost(const BoundingBox<DIM>& boxLeft,
						   const BoundingBox<DIM>& boxRight,
						   float parentSurfaceArea, float parentVolume,
						   int nReferencesLeft, int nReferencesRight,
						   int depth) const;

	// computes unsplitting costs based on heuristic
	void computeUnsplittingCosts(const BoundingBox<DIM>& boxLeft,
								 const BoundingBox<DIM>& boxRight,
								 const BoundingBox<DIM>& boxReference,
								 const BoundingBox<DIM>& boxRefLeft,
								 const BoundingBox<DIM>& boxRefRight,
								 int nReferencesLeft, int nReferencesRight,
								 float& costDuplicate, float& costUnsplitLeft,
								 float& costUnsplitRight) const;

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

	// splits primitive
	void splitPrimitive(const std::shared_ptr<Primitive<DIM>>& primitive, int dim,
						float splitCoord, const BoundingBox<DIM>& boxReference,
						BoundingBox<DIM>& boxLeft, BoundingBox<DIM>& boxRight) const;

	// computes spatial split
	float computeSpatialSplit(const BoundingBox<DIM>& nodeBoundingBox,
							  const std::vector<BoundingBox<DIM>>& referenceBoxes,
							  int depth, int nodeStart, int nodeEnd, int splitDim,
							  float& splitCoord, BoundingBox<DIM>& boxLeft,
							  BoundingBox<DIM>& boxRight);

	// performs spatial split
	int performSpatialSplit(const BoundingBox<DIM>& boxLeft, const BoundingBox<DIM>& boxRight,
							int splitDim, float splitCoord, int nodeStart, int& nodeEnd,
							int& nReferencesAdded, int& nTotalReferences,
							std::vector<BoundingBox<DIM>>& referenceBoxes,
							std::vector<Vector<DIM>>& referenceCentroids);

	// helper function to build binary tree
	int buildRecursive(std::vector<BoundingBox<DIM>>& referenceBoxes,
					   std::vector<Vector<DIM>>& referenceCentroids,
					   std::vector<SbvhNode<DIM>>& buildNodes,
					   int parent, int start, int end, int depth,
					   int& nTotalReferences);

	// builds binary tree
	void build();

	// processes subtree for intersection
	bool processSubtreeForIntersection(Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
									   bool checkOcclusion, bool countHits,
									   std::vector<BvhTraversal>& subtree,
									   float *boxHits, int& hits, int& nodesVisited) const;

	// processes subtree for closest point
	void processSubtreeForClosestPoint(BoundingSphere<DIM>& s, Interaction<DIM>& i,
									   const Vector<DIM>& dirGuess, std::vector<BvhTraversal>& subtree,
									   float *boxHits, bool& notFound, int& nodesVisited) const;

	// members
	CostHeuristic costHeuristic;
	float splitAlpha, rootSurfaceArea, rootVolume;
	int nNodes, nLeafs, leafSize, nBuckets, nBins, memoryBudget, maxDepth, depthGuess;
	std::vector<std::pair<BoundingBox<DIM>, int>> buckets, rightBucketBoxes, rightBinBoxes;
	std::vector<std::tuple<BoundingBox<DIM>, int, int>> bins;
	const std::vector<std::shared_ptr<Primitive<DIM>>>& primitives;
	std::vector<SbvhNode<DIM>> flatTree;
	std::vector<int> references, referencesToAdd;
	std::vector<BoundingBox<DIM>> referenceBoxesToAdd;
	std::vector<Vector<DIM>> referenceCentroidsToAdd;
	ObjectType objectType;
	bool packLeaves;

	template<int U, int V>
	friend class Mbvh;
};

} // namespace fcpw

#include "sbvh.inl"
