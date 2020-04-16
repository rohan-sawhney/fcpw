#pragma once

#include "primitive.h"
#include <tuple>

namespace fcpw {
// modified version of https://github.com/brandonpelfrey/Fast-BVH and
// https://github.com/straaljager/GPU-path-tracing-with-CUDA-tutorial-4
// TODO:
// - implement mbvh/qbvh with vectorization (try enoki?)
// - Oriented bounding boxes/RSS
// - build a spatial data structure on top of bvh
// - estimate closest point radius (i.e., conversative guess of spherical region containing query point)
// - implement "queueless" closest point traversal
// - try bottom up closest point traversal strategy

// TODO:
// - check speedup with just maxDimension as splitDim
// - cap max tree depth

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
	// computes object split
	float computeObjectSplit(const BoundingBox<DIM>& nodeBoundingBox,
							 const BoundingBox<DIM>& nodeCentroidBox,
							 const std::vector<BoundingBox<DIM>>& referenceBoxes,
							 const std::vector<Vector<DIM>>& referenceCentroids,
							 int nodeStart, int nodeEnd, int& splitDim, float& splitCoord,
							 BoundingBox<DIM>& bboxIntersected);

	// performs object split
	int performObjectSplit(std::vector<BoundingBox<DIM>>& referenceBoxes,
						   std::vector<Vector<DIM>>& referenceCentroids,
						   int nodeStart, int nodeEnd, int splitDim, float splitCoord);

	// splits reference
	void splitReference(int referenceIndex, int dim, float splitCoord,
						const BoundingBox<DIM>& bboxReference,
						BoundingBox<DIM>& bboxLeft, BoundingBox<DIM>& bboxRight) const;

	// computes spatial split
	float computeSpatialSplit(const BoundingBox<DIM>& nodeBoundingBox,
							  const std::vector<BoundingBox<DIM>>& referenceBoxes,
							  int nodeStart, int nodeEnd, int& splitDim, float& splitCoord,
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
