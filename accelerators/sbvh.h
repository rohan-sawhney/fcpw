#pragma once

#include "primitive.h"
#include <tuple>
#include <stack>
#include <queue>

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

template <int DIM>
struct SbvhFlatNode {
	// constructor
	SbvhFlatNode(): parent(-1), start(-1), nReferences(-1),
					rightOffset(-1), overlapsSibling(false) {}

	// members
	BoundingBox<DIM> bbox;
	int parent, start, nReferences, rightOffset;
	bool overlapsSibling;
};

struct SbvhTraversal {
	// constructor
	SbvhTraversal(int i_, float d_): i(i_), d(d_) {}

	// members
	int i; // node index
	float d; // minimum distance (parametric, squared, ...) to this node
};

template <int DIM>
class Sbvh: public Aggregate<DIM> {
public:
	// constructor
	Sbvh(std::vector<std::shared_ptr<Primitive<DIM>>>& primitives_,
		 const CostHeuristic& costHeuristic_, float splitAlpha_,
		 int leafSize_=4, int nBuckets_=8, int nBins_=8);

	// returns bounding box
	BoundingBox<DIM> boundingBox() const;

	// returns centroid
	Vector<DIM> centroid() const;

	// returns surface area
	float surfaceArea() const;

	// returns signed volume
	float signedVolume() const;

	// intersects with ray, starting the traversal at the specified node
	int intersectFromNode(Ray<DIM>& r, std::vector<Interaction<DIM>>& is, int startNodeIndex,
						  bool checkOcclusion=false, bool countHits=false) const;

	// intersects with ray
	int intersect(Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
				  bool checkOcclusion=false, bool countHits=false) const;

	// finds closest point to sphere center, starting the traversal at the specified node
	bool findClosestPointFromNode(BoundingSphere<DIM>& s, Interaction<DIM>& i,
						int startNodeIndex) const;

	// finds closest point to sphere center
	bool findClosestPoint(BoundingSphere<DIM>& s, Interaction<DIM>& i) const;

protected:
	// computes split cost based on heuristic
	float computeSplitCost(const BoundingBox<DIM>& bboxLeft,
						   const BoundingBox<DIM>& bboxRight,
						   float parentSurfaceArea, float parentVolume,
						   int nReferencesLeft, int nReferencesRight) const;

	// computes unsplitting costs based on heuristic
	void computeUnsplittingCosts(const BoundingBox<DIM>& bboxLeft,
								 const BoundingBox<DIM>& bboxRight,
								 const BoundingBox<DIM>& bboxReference,
								 const BoundingBox<DIM>& bboxRefLeft,
								 const BoundingBox<DIM>& bboxRefRight,
								 int nReferencesLeft, int nReferencesRight,
								 float& costDuplicate, float& costUnsplitLeft,
								 float& costUnsplitRight) const;

	// computes object split
	float computeObjectSplit(const BoundingBox<DIM>& nodeBoundingBox,
							 const BoundingBox<DIM>& nodeCentroidBox,
							 const std::vector<BoundingBox<DIM>>& referenceBoxes,
							 const std::vector<Vector<DIM>>& referenceCentroids,
							 int nodeStart, int nodeEnd, int& splitDim,
							 float& splitCoord, BoundingBox<DIM>& bboxIntersected);

	// performs object split
	int performObjectSplit(int nodeStart, int nodeEnd, int splitDim, float splitCoord,
						   std::vector<BoundingBox<DIM>>& referenceBoxes,
						   std::vector<Vector<DIM>>& referenceCentroids);

	// splits primitive
	void splitPrimitive(const std::shared_ptr<Primitive<DIM>>& primitive, int dim,
						float splitCoord, const BoundingBox<DIM>& bboxReference,
						BoundingBox<DIM>& bboxLeft, BoundingBox<DIM>& bboxRight) const;

	// computes spatial split
	float computeSpatialSplit(const BoundingBox<DIM>& nodeBoundingBox,
							  const std::vector<BoundingBox<DIM>>& referenceBoxes,
							  int nodeStart, int nodeEnd, int splitDim, float& splitCoord,
							  BoundingBox<DIM>& bboxLeft, BoundingBox<DIM>& bboxRight);

	// performs spatial split
	int performSpatialSplit(const BoundingBox<DIM>& bboxLeft, const BoundingBox<DIM>& bboxRight,
							int splitDim, float splitCoord, int nodeStart, int& nodeEnd,
							int& nReferencesAdded, int& nTotalReferences,
							std::vector<BoundingBox<DIM>>& referenceBoxes,
							std::vector<Vector<DIM>>& referenceCentroids);

	// helper function to build binary tree
	int buildRecursive(std::vector<BoundingBox<DIM>>& referenceBoxes,
					   std::vector<Vector<DIM>>& referenceCentroids,
					   std::vector<SbvhFlatNode<DIM>>& buildNodes,
					   int parent, int start, int end, int& nTotalReferences);

	// builds binary tree
	void build();

	// processes subtree for intersection
	bool processSubtreeForIntersection(Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
									   bool checkOcclusion, bool countHits,
									   std::stack<SbvhTraversal>& subtree,
									   float *bboxHits, int& hits) const;

	// processes subtree for closest point
	void processSubtreeForClosestPoint(BoundingSphere<DIM>& s, Interaction<DIM>& i,
									   std::queue<SbvhTraversal>& subtree,
									   float *bboxHits, bool& notFound) const;

	// members
	CostHeuristic costHeuristic;
	float splitAlpha, rootSurfaceArea, rootVolume;
	int nNodes, nLeafs, leafSize, nBuckets, nBins, memoryBudget;
	std::vector<std::pair<BoundingBox<DIM>, int>> buckets, rightBucketBoxes, rightBinBoxes;
	std::vector<std::tuple<BoundingBox<DIM>, int, int>> bins;
	const std::vector<std::shared_ptr<Primitive<DIM>>>& primitives;
	std::vector<SbvhFlatNode<DIM>> flatTree;
	std::vector<int> references, referencesToAdd;
	std::vector<BoundingBox<DIM>> referenceBoxesToAdd;
	std::vector<Vector<DIM>> referenceCentroidsToAdd;

	template <int T>
	friend class Qbvh;
};

} // namespace fcpw

#include "sbvh.inl"
