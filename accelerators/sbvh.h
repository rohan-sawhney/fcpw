#pragma once

#include "primitive.h"

namespace fcpw {
// modified version of https://github.com/brandonpelfrey/Fast-BVH
// TODO:
// - implement mbvh/qbvh with vectorization (try enoki?)
// - build a spatial data structure on top of bvh
// - estimate closest point radius (i.e., conversative guess of spherical region containing query point)
// - implement "queueless" closest point traversal
// - try bottom up closest point traversal strategy

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
	SbvhFlatNode(): bbox(false), start(0), nReferences(0), rightOffset(0) {}

	// members
	BoundingBox<DIM> bbox;
	int start, nReferences, rightOffset;
};

template <int DIM>
class Sbvh: public Aggregate<DIM> {
public:
	// constructor
	Sbvh(std::vector<std::shared_ptr<Primitive<DIM>>>& primitives_,
		 const CostHeuristic& costHeuristic_, float splitAlpha_=1e-5,
		 int leafSize_=4, int nBuckets_=8);

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
							 int nodeStart, int nodeEnd, int& splitDim, float& splitCoord);

	// helper function to build binary tree
	void buildRecursive(std::vector<BoundingBox<DIM>>& referenceBoxes,
						std::vector<Vector<DIM>>& referenceCentroids,
						std::vector<SbvhFlatNode<DIM>>& buildNodes,
						int parent, int start, int end);

	// builds binary tree
	void build();

	// members
	CostHeuristic costHeuristic;
	float splitAlpha;
	int nNodes, nLeafs, leafSize, nBuckets;
	std::vector<std::pair<BoundingBox<DIM>, int>> buckets, rightBucketBoxes;
	const std::vector<std::shared_ptr<Primitive<DIM>>>& primitives;
	std::vector<SbvhFlatNode<DIM>> flatTree;
	std::vector<int> references;
};

} // namespace fcpw

#include "sbvh.inl"
