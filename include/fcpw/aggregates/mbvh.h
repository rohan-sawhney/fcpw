#pragma once

#include <fcpw/aggregates/sbvh.h>
#ifdef FCPW_USE_EIGHT_WIDE_BRANCHING
	#define FCPW_MBVH_BRANCHING_FACTOR 8
	#define FCPW_MBVH_MAX_DEPTH 154
#else
	#define FCPW_MBVH_BRANCHING_FACTOR 4
	#define FCPW_MBVH_MAX_DEPTH 96
#endif

namespace fcpw {

template<size_t DIM, bool CONEDATA>
struct MbvhNode {
	MbvhNode() {
		std::cerr << "MbvhNode(): DIM: " << DIM << ", CONEDATA: " << CONEDATA << " not supported" << std::endl;
		exit(EXIT_FAILURE);
	}
};

template<size_t DIM>
struct MbvhNode<DIM, false> {
	// constructor
	MbvhNode(): boxMin(FloatP<FCPW_MBVH_BRANCHING_FACTOR>(maxFloat)),
				boxMax(FloatP<FCPW_MBVH_BRANCHING_FACTOR>(minFloat)),
				child(maxInt) {}

	// members
	VectorP<FCPW_MBVH_BRANCHING_FACTOR, DIM> boxMin, boxMax;
	IntP<FCPW_MBVH_BRANCHING_FACTOR> child; // use sign to differentiate between inner and leaf nodes
};

template<size_t DIM>
struct MbvhNode<DIM, true> {
	// constructor
	MbvhNode(): boxMin(FloatP<FCPW_MBVH_BRANCHING_FACTOR>(maxFloat)),
				boxMax(FloatP<FCPW_MBVH_BRANCHING_FACTOR>(minFloat)),
				coneAxis(FloatP<FCPW_MBVH_BRANCHING_FACTOR>(0.0f)),
				coneHalfAngle(M_PI), child(maxInt), silhouetteChild(maxInt) {}

	// members;
	VectorP<FCPW_MBVH_BRANCHING_FACTOR, DIM> boxMin, boxMax;
	VectorP<FCPW_MBVH_BRANCHING_FACTOR, DIM> coneAxis;
	FloatP<FCPW_MBVH_BRANCHING_FACTOR> coneHalfAngle;
	IntP<FCPW_MBVH_BRANCHING_FACTOR> child; // use sign to differentiate between inner and leaf nodes
	IntP<FCPW_MBVH_BRANCHING_FACTOR> silhouetteChild; // use sign to differentiate between inner and silhouette leaf nodes
};

template<size_t WIDTH, size_t DIM, typename PrimitiveType>
struct MbvhLeafNode {
	MbvhLeafNode() {
		std::cerr << "MbvhLeafNode(): WIDTH: " << WIDTH << ", DIM: " << DIM << " not supported" << std::endl;
		exit(EXIT_FAILURE);
	}
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

template<size_t WIDTH, size_t DIM, typename SilhouetteType>
struct MbvhSilhouetteLeafNode {
	MbvhSilhouetteLeafNode() {
		std::cerr << "MbvhSilhouetteLeafNode(): WIDTH: " << WIDTH << ", DIM: " << DIM << " not supported" << std::endl;
		exit(EXIT_FAILURE);
	}
};

template<size_t WIDTH, size_t DIM>
struct MbvhSilhouetteLeafNode<WIDTH, DIM, SilhouetteVertex> {
	// members
	VectorP<WIDTH, DIM> positions[3];
	IntP<WIDTH> primitiveIndex;
	MaskP<WIDTH> missingFace;
};

template<size_t WIDTH, size_t DIM>
struct MbvhSilhouetteLeafNode<WIDTH, DIM, SilhouetteEdge> {
	// members
	VectorP<WIDTH, DIM> positions[4];
	IntP<WIDTH> primitiveIndex;
	MaskP<WIDTH> missingFace;
};

template<size_t WIDTH, size_t DIM, bool CONEDATA=false, typename PrimitiveType=Primitive<DIM>, typename SilhouetteType=SilhouettePrimitive<DIM>>
class Mbvh: public Aggregate<DIM> {
public:
	// constructor
	Mbvh(const Sbvh<DIM, CONEDATA, PrimitiveType, SilhouetteType> *sbvh_, bool printStats_=false);

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
	// collapses sbvh into a mbvh
	int collapseSbvh(const Sbvh<DIM, CONEDATA, PrimitiveType, SilhouetteType> *sbvh, int sbvhNodeIndex, int parent, int depth);

	// determines whether mbvh node is a leaf node
	bool isLeafNode(const MbvhNode<DIM, CONEDATA>& node) const;

	// populates leaf nodes
	void populateLeafNodes();

	// populates leaf nodes
	void populateSilhouetteLeafNodes();

	// members
	int nNodes, nLeafs, nSilhouetteLeafs, maxDepth;
	float area, volume;
	Vector<DIM> aggregateCentroid;
	const std::vector<PrimitiveType *>& primitives;
	const std::vector<SilhouetteType *>& silhouettes;
	std::vector<SilhouetteType *> silhouetteRefs;
	std::vector<MbvhNode<DIM, CONEDATA>> flatTree;
	std::vector<MbvhLeafNode<WIDTH, DIM, PrimitiveType>> leafNodes;
	std::vector<MbvhSilhouetteLeafNode<WIDTH, DIM, SilhouetteType>> silhouetteLeafNodes;
	bool primitiveTypeIsAggregate;
	enoki::Array<int, DIM> range;
};

} // namespace fcpw

#include "mbvh.inl"
