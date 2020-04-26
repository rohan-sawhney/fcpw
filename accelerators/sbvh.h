#pragma once

#include "primitive.h"
#include <tuple>

namespace fcpw {
// modified versions of https://github.com/brandonpelfrey/Fast-BVH and
// https://github.com/straaljager/GPU-path-tracing-with-CUDA-tutorial-4
// TODO:
// - enoki + implement mbvh/qbvh
// - for closest point queries:
// -- (for spatio-temporially related queries) warm start traversal from node of
//    previous query, traversing hierarchy bottom up ; distance from previous query can
//    be use to determine whether this is a good idea (it is a good idea when the radius
//    is small compared to the model size)
// -- alternatively, build spatial data structure that stores pointers to nodes
//    (for non-spatio-temporially related queries)
// -- implement "queueless" closest point traversal
// - for intersections:
// -- smarter traversal for incoherent rays (?)
// -- implement "stackless" ray intersection traversal
// - oriented bounding boxes/RSS
// - add support for more geometries: line segments (3d), beziers (3d), implicits, nurbs

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
	SbvhFlatNode(): start(0), nReferences(0), rightOffset(0) {}

	// members
	BoundingBox<DIM> bbox;
	int start, nReferences, rightOffset;
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

	// intersects with ray
	int intersect(Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
				  bool checkOcclusion=false, bool countHits=false) const;

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
};

} // namespace fcpw

#include "sbvh.inl"
