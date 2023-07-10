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

template<size_t DIM, bool CONEDATA>
struct SbvhNode {
	SbvhNode() {
		std::cerr << "SbvhNode(): DIM: " << DIM << ", CONEDATA: " << CONEDATA << " not supported" << std::endl;
		exit(EXIT_FAILURE);
	}
};

template<size_t DIM>
struct SbvhNode<DIM, false> {
	// constructor
	SbvhNode(): nReferences(0) {}

	// members
	BoundingBox<DIM> box;
	union {
		int referenceOffset;
		int secondChildOffset;
	};
	int nReferences;
};

template<size_t DIM>
struct SbvhNode<DIM, true> {
	// constructor
	SbvhNode(): nReferences(0), nSilhouetteReferences(0) {}

	// members
	BoundingBox<DIM> box;
	BoundingCone<DIM> cone;
	union {
		int referenceOffset;
		int secondChildOffset;
	};
	int silhouetteReferenceOffset;
	int nReferences;
	int nSilhouetteReferences;
};

struct BvhTraversal {
	// constructors
	BvhTraversal(): node(-1), distance(0.0f) {}
	BvhTraversal(int node_, float distance_): node(node_), distance(distance_) {}

	// members
	int node; // node index
	float distance; // minimum distance (parametric, squared, ...) to this node
};

template<size_t DIM, bool CONEDATA, typename PrimitiveType, typename SilhouetteType>
using SortPositionsFunc = std::function<void(const std::vector<SbvhNode<DIM, CONEDATA>>&, std::vector<PrimitiveType *>&, std::vector<SilhouetteType *>&)>;

template<size_t DIM, bool CONEDATA=false, typename PrimitiveType=Primitive<DIM>, typename SilhouetteType=SilhouettePrimitive<DIM>>
class Sbvh: public Aggregate<DIM> {
public:
	// constructor
	Sbvh(const CostHeuristic& costHeuristic_,
		 std::vector<PrimitiveType *>& primitives_,
		 std::vector<SilhouetteType *>& silhouettes_,
		 SortPositionsFunc<DIM, CONEDATA, PrimitiveType, SilhouetteType> sortPositions_={},
		 const std::function<bool(float, int)>& ignoreSilhouette_={},
		 bool printStats_=false, bool packLeaves_=false,
		 int leafSize_=4, int nBuckets_=8);

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

	// intersects with sphere, starting the traversal at the specified node in an aggregate
	// NOTE: interactions contain primitive index
	int intersectFromNode(const BoundingSphere<DIM>& s,
						  std::vector<Interaction<DIM>>& is,
						  int nodeStartIndex, int aggregateIndex,
						  int& nodesVisited, bool recordOneHit=false,
						  const std::function<float(float)>& primitiveWeight={}) const;

	// intersects with sphere, starting the traversal at the specified node in an aggregate
	// NOTE: interactions contain primitive index
	int intersectStochasticFromNode(const BoundingSphere<DIM>& s,
									std::vector<Interaction<DIM>>& is, float *randNums,
									int nodeStartIndex, int aggregateIndex, int& nodesVisited,
									const std::function<float(float)>& traversalWeight={},
									const std::function<float(float)>& primitiveWeight={}) const;

	// finds closest point to sphere center, starting the traversal at the specified node in an aggregate
	bool findClosestPointFromNode(BoundingSphere<DIM>& s, Interaction<DIM>& i,
								  int nodeStartIndex, int aggregateIndex,
								  int& nodesVisited, bool recordNormal=false) const;

	// finds closest silhouette point to sphere center, starting the traversal at the specified node in an aggregate
	bool findClosestSilhouettePointFromNode(BoundingSphere<DIM>& s, Interaction<DIM>& i,
											int nodeStartIndex, int aggregateIndex,
											int& nodesVisited, bool flipNormalOrientation=false,
											float squaredMinRadius=0.0f, float precision=1e-3f,
											bool recordNormal=false) const;

protected:
	// computes split cost based on heuristic
	float computeSplitCost(const BoundingBox<DIM>& boxLeft,
						   const BoundingBox<DIM>& boxRight,
						   int nReferencesLeft, int nReferencesRight,
						   int depth) const;

	// computes object split
	float computeObjectSplit(const BoundingBox<DIM>& nodeBoundingBox,
							 const BoundingBox<DIM>& nodeCentroidBox,
							 const std::vector<BoundingBox<DIM>>& referenceBoxes,
							 const std::vector<Vector<DIM>>& referenceCentroids,
							 int depth, int nodeStart, int nodeEnd,
							 int& splitDim, float& splitCoord);

	// performs object split
	int performObjectSplit(int nodeStart, int nodeEnd, int splitDim, float splitCoord,
						   std::vector<BoundingBox<DIM>>& referenceBoxes,
						   std::vector<Vector<DIM>>& referenceCentroids);

	// helper function to build binary tree
	void buildRecursive(std::vector<BoundingBox<DIM>>& referenceBoxes,
						std::vector<Vector<DIM>>& referenceCentroids,
						std::vector<SbvhNode<DIM, CONEDATA>>& buildNodes,
						int parent, int start, int end, int depth);

	// builds binary tree
	void build();

	// processes subtree for intersection
	bool processSubtreeForIntersection(Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
									   int nodeStartIndex, int aggregateIndex, bool checkForOcclusion,
									   bool recordAllHits, BvhTraversal *subtree,
									   float *boxHits, int& hits, int& nodesVisited) const;

	// processes subtree for intersection
	float processSubtreeForIntersection(const BoundingSphere<DIM>& s, std::vector<Interaction<DIM>>& is,
									    int nodeStartIndex, int aggregateIndex, bool recordOneHit,
									    const std::function<float(float)>& primitiveWeight,
										BvhTraversal *subtree, float *boxHits, int& hits, int& nodesVisited) const;

	// processes subtree for intersection
	void processSubtreeForIntersection(const BoundingSphere<DIM>& s, std::vector<Interaction<DIM>>& is,
									   float *randNums, int nodeStartIndex, int aggregateIndex,
									   const std::function<float(float)>& traversalWeight,
									   const std::function<float(float)>& primitiveWeight,
									   int nodeIndex, float traversalPdf, float *boxHits,
									   int& hits, int& nodesVisited) const;

	// processes subtree for closest point
	void processSubtreeForClosestPoint(BoundingSphere<DIM>& s, Interaction<DIM>& i,
									   int nodeStartIndex, int aggregateIndex,
									   bool recordNormal, BvhTraversal *subtree,
									   float *boxHits, bool& notFound, int& nodesVisited) const;

	// members
	CostHeuristic costHeuristic;
	int nNodes, nLeafs, leafSize, nBuckets, maxDepth, depthGuess;
	std::vector<std::pair<BoundingBox<DIM>, int>> buckets, rightBuckets;
	std::vector<PrimitiveType *>& primitives;
	std::vector<SilhouetteType *>& silhouettes;
	std::vector<SilhouetteType *> silhouetteRefs;
	std::vector<SbvhNode<DIM, CONEDATA>> flatTree;
	bool packLeaves, primitiveTypeIsAggregate;

	template<size_t U, size_t V, bool W, typename X, typename Y>
	friend class Mbvh;
};

} // namespace fcpw

#include "sbvh.inl"
