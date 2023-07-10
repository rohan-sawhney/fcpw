#include <fcpw/core/wide_query_operations.h>

namespace fcpw {

template<size_t DIM>
inline void assignBoundingCone(const SbvhNode<DIM, false>& sbvhNode, MbvhNode<DIM, false>& mbvhNode, int index)
{
	// do nothing
}

template<size_t DIM>
inline void assignBoundingCone(const SbvhNode<DIM, true>& sbvhNode, MbvhNode<DIM, true>& mbvhNode, int index)
{
	for (size_t j = 0; j < DIM; j++) {
		mbvhNode.coneAxis[j][index] = sbvhNode.cone.axis[j];
	}

	mbvhNode.coneHalfAngle[index] = sbvhNode.cone.halfAngle;
}

template<size_t WIDTH, size_t DIM>
inline void assignSilhouetteLeafRange(const SbvhNode<DIM, false>& sbvhNode, MbvhNode<DIM, false>& mbvhNode, int& nSilhouetteLeafs)
{
	// do nothing
}

template<size_t WIDTH, size_t DIM>
inline void assignSilhouetteLeafRange(const SbvhNode<DIM, true>& sbvhNode, MbvhNode<DIM, true>& mbvhNode, int& nSilhouetteLeafs)
{
	if (sbvhNode.nSilhouetteReferences > 0) {
		mbvhNode.silhouetteChild[0] = -(nSilhouetteLeafs + 1); // negative value indicates that node is a leaf
		mbvhNode.silhouetteChild[1] = sbvhNode.nSilhouetteReferences/WIDTH;
		if (sbvhNode.nSilhouetteReferences%WIDTH != 0) mbvhNode.silhouetteChild[1] += 1;
		mbvhNode.silhouetteChild[2] = sbvhNode.silhouetteReferenceOffset;
		mbvhNode.silhouetteChild[3] = sbvhNode.nSilhouetteReferences;
		nSilhouetteLeafs += mbvhNode.silhouetteChild[1];

	} else {
		mbvhNode.silhouetteChild[0] = 0;
		mbvhNode.silhouetteChild[1] = 0;
		mbvhNode.silhouetteChild[2] = 0;
		mbvhNode.silhouetteChild[3] = 0;
	}
}

template<size_t WIDTH, size_t DIM, bool CONEDATA, typename PrimitiveType, typename SilhouetteType>
inline int Mbvh<WIDTH, DIM, CONEDATA, PrimitiveType, SilhouetteType>::collapseSbvh(const Sbvh<DIM, CONEDATA, PrimitiveType, SilhouetteType> *sbvh,
																				   int sbvhNodeIndex, int parent, int depth)
{
	const SbvhNode<DIM, CONEDATA>& sbvhNode = sbvh->flatTree[sbvhNodeIndex];
	maxDepth = std::max(depth, maxDepth);

	// create mbvh node
	MbvhNode<DIM, CONEDATA> mbvhNode;
	int mbvhNodeIndex = nNodes;

	nNodes++;
	flatTree.emplace_back(mbvhNode);

	if (sbvhNode.nReferences > 0) {
		// sbvh node is a leaf node; assign mbvh node its reference indices
		MbvhNode<DIM, CONEDATA>& mbvhNode = flatTree[mbvhNodeIndex];
		mbvhNode.child[0] = -(nLeafs + 1); // negative value indicates that node is a leaf
		mbvhNode.child[1] = sbvhNode.nReferences/WIDTH;
		if (sbvhNode.nReferences%WIDTH != 0) mbvhNode.child[1] += 1;
		mbvhNode.child[2] = sbvhNode.referenceOffset;
		mbvhNode.child[3] = sbvhNode.nReferences;
		nLeafs += mbvhNode.child[1];
		assignSilhouetteLeafRange<WIDTH, DIM>(sbvhNode, mbvhNode, nSilhouetteLeafs);

	} else {
		// sbvh node is an inner node, flatten it
		int nNodesToCollapse = 2;
		int nodesToCollapse[FCPW_MBVH_BRANCHING_FACTOR];
		nodesToCollapse[0] = sbvhNodeIndex + sbvhNode.secondChildOffset;
		nodesToCollapse[1] = sbvhNodeIndex + 1;
		bool noMoreNodesToCollapse = false;

		while (nNodesToCollapse < FCPW_MBVH_BRANCHING_FACTOR && !noMoreNodesToCollapse) {
			// find the (non-leaf) node entry with the largest surface area
			float maxSurfaceArea = minFloat;
			int maxIndex = -1;

			for (int i = 0; i < nNodesToCollapse; i++) {
				int sbvhNodeIndex = nodesToCollapse[i];
				const SbvhNode<DIM, CONEDATA>& sbvhNode = sbvh->flatTree[sbvhNodeIndex];

				if (sbvhNode.nReferences == 0) {
					float surfaceArea = sbvhNode.box.surfaceArea();

					if (maxSurfaceArea < surfaceArea) {
						maxSurfaceArea = surfaceArea;
						maxIndex = i;
					}
				}
			}

			if (maxIndex == -1) {
				// no more nodes to collapse
				noMoreNodesToCollapse = true;

			} else {
				// remove the selected node from the list, and add its two children
				int sbvhNodeIndex = nodesToCollapse[maxIndex];
				const SbvhNode<DIM, CONEDATA>& sbvhNode = sbvh->flatTree[sbvhNodeIndex];

				nodesToCollapse[maxIndex] = sbvhNodeIndex + sbvhNode.secondChildOffset;
				nodesToCollapse[nNodesToCollapse] = sbvhNodeIndex + 1;
				nNodesToCollapse++;
			}
		}

		// collapse the nodes
		std::sort(nodesToCollapse, nodesToCollapse + nNodesToCollapse);
		for (int i = 0; i < nNodesToCollapse; i++) {
			int sbvhNodeIndex = nodesToCollapse[i];
			const SbvhNode<DIM, CONEDATA>& sbvhNode = sbvh->flatTree[sbvhNodeIndex];

			// assign mbvh node this sbvh node's bounding box and index
			for (size_t j = 0; j < DIM; j++) {
				flatTree[mbvhNodeIndex].boxMin[j][i] = sbvhNode.box.pMin[j];
				flatTree[mbvhNodeIndex].boxMax[j][i] = sbvhNode.box.pMax[j];
			}

			// assign mbvh node this sbvh node's cone data
			assignBoundingCone<DIM>(sbvhNode, flatTree[mbvhNodeIndex], i);

			flatTree[mbvhNodeIndex].child[i] = collapseSbvh(sbvh, sbvhNodeIndex, mbvhNodeIndex, depth + 1);
		}
	}

	return mbvhNodeIndex;
}

template<size_t WIDTH, size_t DIM, bool CONEDATA, typename PrimitiveType, typename SilhouetteType>
inline bool Mbvh<WIDTH, DIM, CONEDATA, PrimitiveType, SilhouetteType>::isLeafNode(const MbvhNode<DIM, CONEDATA>& node) const
{
	return node.child[0] < 0;
}

template<size_t WIDTH, size_t DIM, bool CONEDATA, typename PrimitiveType>
inline void populateLeafNode(const MbvhNode<DIM, CONEDATA>& node, const std::vector<PrimitiveType *>& primitives,
							 std::vector<MbvhLeafNode<WIDTH, DIM, PrimitiveType>>& leafNodes)
{
	std::cerr << "populateLeafNode(): WIDTH: " << WIDTH << ", DIM: " << DIM << " not supported" << std::endl;
	exit(EXIT_FAILURE);
}

template<size_t WIDTH, bool CONEDATA>
inline void populateLeafNode(const MbvhNode<3, CONEDATA>& node, const std::vector<LineSegment *>& primitives,
							 std::vector<MbvhLeafNode<WIDTH, 3, LineSegment>>& leafNodes)
{
	int leafOffset = -node.child[0] - 1;
	int referenceOffset = node.child[2];
	int nReferences = node.child[3];

	// populate leaf node with line segments
	for (int p = 0; p < nReferences; p++) {
		int referenceIndex = referenceOffset + p;
		int leafIndex = leafOffset + p/WIDTH;
		int w = p%WIDTH;

		const LineSegment *lineSegment = primitives[referenceIndex];
		const Vector3& pa = lineSegment->soup->positions[lineSegment->indices[0]];
		const Vector3& pb = lineSegment->soup->positions[lineSegment->indices[1]];

		leafNodes[leafIndex].primitiveIndex[w] = lineSegment->pIndex;
		for (int i = 0; i < 3; i++) {
			leafNodes[leafIndex].positions[0][i][w] = pa[i];
			leafNodes[leafIndex].positions[1][i][w] = pb[i];
		}
	}
}

template<size_t WIDTH, bool CONEDATA>
inline void populateLeafNode(const MbvhNode<3, CONEDATA>& node, const std::vector<Triangle *>& primitives,
							 std::vector<MbvhLeafNode<WIDTH, 3, Triangle>>& leafNodes)
{
	int leafOffset = -node.child[0] - 1;
	int referenceOffset = node.child[2];
	int nReferences = node.child[3];

	// populate leaf node with triangles
	for (int p = 0; p < nReferences; p++) {
		int referenceIndex = referenceOffset + p;
		int leafIndex = leafOffset + p/WIDTH;
		int w = p%WIDTH;

		const Triangle *triangle = primitives[referenceIndex];
		const Vector3& pa = triangle->soup->positions[triangle->indices[0]];
		const Vector3& pb = triangle->soup->positions[triangle->indices[1]];
		const Vector3& pc = triangle->soup->positions[triangle->indices[2]];

		leafNodes[leafIndex].primitiveIndex[w] = triangle->pIndex;
		for (int i = 0; i < 3; i++) {
			leafNodes[leafIndex].positions[0][i][w] = pa[i];
			leafNodes[leafIndex].positions[1][i][w] = pb[i];
			leafNodes[leafIndex].positions[2][i][w] = pc[i];
		}
	}
}

template<size_t WIDTH, size_t DIM, bool CONEDATA, typename PrimitiveType, typename SilhouetteType>
inline void Mbvh<WIDTH, DIM, CONEDATA, PrimitiveType, SilhouetteType>::populateLeafNodes()
{
	if (std::is_same<PrimitiveType, LineSegment>::value ||
		std::is_same<PrimitiveType, Triangle>::value) {
		leafNodes.resize(nLeafs);

		for (int i = 0; i < nNodes; i++) {
			const MbvhNode<DIM, CONEDATA>& node = flatTree[i];
			if (isLeafNode(node)) populateLeafNode(node, primitives, leafNodes);
		}
	}
}

template<size_t WIDTH, size_t DIM, bool CONEDATA, typename SilhouetteType>
inline void populateSilhouetteLeafNode(const MbvhNode<DIM, CONEDATA>& node, const std::vector<SilhouetteType *>& silhouettes,
									   std::vector<MbvhSilhouetteLeafNode<WIDTH, DIM, SilhouetteType>>& silhouetteLeafNodes)
{
	std::cerr << "populateSilhouetteLeafNode(): WIDTH: " << WIDTH << ", DIM: " << DIM << " not supported" << std::endl;
	exit(EXIT_FAILURE);
}

template<size_t WIDTH>
inline void populateSilhouetteLeafNode(const MbvhNode<3, true>& node, const std::vector<SilhouetteVertex *>& silhouettes,
									   std::vector<MbvhSilhouetteLeafNode<WIDTH, 3, SilhouetteVertex>>& silhouetteLeafNodes)
{
	int silhouetteLeafOffset = -node.silhouetteChild[0] - 1;
	int silhouetteReferenceOffset = node.silhouetteChild[2];
	int nSilhouetteReferences = node.silhouetteChild[3];

	// populate silhouette leaf node with silhouette vertices
	for (int p = 0; p < nSilhouetteReferences; p++) {
		int referenceIndex = silhouetteReferenceOffset + p;
		int leafIndex = silhouetteLeafOffset + p/WIDTH;
		int w = p%WIDTH;

		const SilhouetteVertex *silhouetteVertex = silhouettes[referenceIndex];
		MbvhSilhouetteLeafNode<WIDTH, 3, SilhouetteVertex>& silhouetteLeafNode = silhouetteLeafNodes[leafIndex];
		silhouetteLeafNode.primitiveIndex[w] = silhouetteVertex->pIndex;
		silhouetteLeafNode.missingFace[w] = !silhouetteVertex->hasFace(0) || !silhouetteVertex->hasFace(1);

		const Vector3& pb = silhouetteVertex->soup->positions[silhouetteVertex->indices[1]];
		for (int i = 0; i < 3; i++) {
			silhouetteLeafNode.positions[1][i][w] = pb[i];
		}

		if (silhouetteVertex->hasFace(0)) {
			Vector3 n0 = silhouetteVertex->normal(0);
			for (int i = 0; i < 3; i++) {
				silhouetteLeafNode.positions[0][i][w] = n0[i];
			}
		}

		if (silhouetteVertex->hasFace(1)) {
			Vector3 n1 = silhouetteVertex->normal(1);
			for (int i = 0; i < 3; i++) {
				silhouetteLeafNode.positions[2][i][w] = n1[i];
			}
		}
	}
}

template<size_t WIDTH>
inline void populateSilhouetteLeafNode(const MbvhNode<3, true>& node, const std::vector<SilhouetteEdge *>& silhouettes,
									   std::vector<MbvhSilhouetteLeafNode<WIDTH, 3, SilhouetteEdge>>& silhouetteLeafNodes)
{
	int silhouetteLeafOffset = -node.silhouetteChild[0] - 1;
	int silhouetteReferenceOffset = node.silhouetteChild[2];
	int nSilhouetteReferences = node.silhouetteChild[3];

	// populate silhouette leaf node with silhouette edges
	for (int p = 0; p < nSilhouetteReferences; p++) {
		int referenceIndex = silhouetteReferenceOffset + p;
		int leafIndex = silhouetteLeafOffset + p/WIDTH;
		int w = p%WIDTH;

		const SilhouetteEdge *silhouetteEdge = silhouettes[referenceIndex];
		MbvhSilhouetteLeafNode<WIDTH, 3, SilhouetteEdge>& silhouetteLeafNode = silhouetteLeafNodes[leafIndex];
		silhouetteLeafNode.primitiveIndex[w] = silhouetteEdge->pIndex;
		silhouetteLeafNode.missingFace[w] = !silhouetteEdge->hasFace(0) || !silhouetteEdge->hasFace(1);

		const Vector3& pb = silhouetteEdge->soup->positions[silhouetteEdge->indices[1]];
		const Vector3& pc = silhouetteEdge->soup->positions[silhouetteEdge->indices[2]];
		for (int i = 0; i < 3; i++) {
			silhouetteLeafNode.positions[1][i][w] = pb[i];
			silhouetteLeafNode.positions[2][i][w] = pc[i];
		}

		if (silhouetteEdge->hasFace(0)) {
			Vector3 n0 = silhouetteEdge->normal(0);
			for (int i = 0; i < 3; i++) {
				silhouetteLeafNode.positions[0][i][w] = n0[i];
			}
		}

		if (silhouetteEdge->hasFace(1)) {
			Vector3 n1 = silhouetteEdge->normal(1);
			for (int i = 0; i < 3; i++) {
				silhouetteLeafNode.positions[3][i][w] = n1[i];
			}
		}
	}
}

template<size_t WIDTH, size_t DIM, bool CONEDATA, typename PrimitiveType, typename SilhouetteType>
inline void Mbvh<WIDTH, DIM, CONEDATA, PrimitiveType, SilhouetteType>::populateSilhouetteLeafNodes()
{
	if (std::is_same<SilhouetteType, SilhouetteVertex>::value ||
		std::is_same<SilhouetteType, SilhouetteEdge>::value) {
		silhouetteLeafNodes.resize(nSilhouetteLeafs);

		for (int i = 0; i < nNodes; i++) {
			const MbvhNode<DIM, CONEDATA>& node = flatTree[i];
			if (isLeafNode(node)) populateSilhouetteLeafNode(node, silhouetteRefs, silhouetteLeafNodes);
		}
	}
}

template<size_t WIDTH, size_t DIM>
inline void updateSilhouetteLeafInfo(const MbvhNode<DIM, false>& mbvhNode, float& nSilhouetteLeafsNotFull)
{
	// do nothing
}

template<size_t WIDTH, size_t DIM>
inline void updateSilhouetteLeafInfo(const MbvhNode<DIM, true>& mbvhNode, float& nSilhouetteLeafsNotFull)
{
	if (mbvhNode.silhouetteChild[3]%WIDTH != 0) {
		nSilhouetteLeafsNotFull += 1.0f;
	}
}

template<size_t WIDTH, size_t DIM, bool CONEDATA, typename PrimitiveType, typename SilhouetteType>
inline Mbvh<WIDTH, DIM, CONEDATA, PrimitiveType, SilhouetteType>::Mbvh(const Sbvh<DIM, CONEDATA, PrimitiveType, SilhouetteType> *sbvh_, bool printStats_):
nNodes(0),
nLeafs(0),
nSilhouetteLeafs(0),
maxDepth(0),
area(0.0f),
volume(0.0f),
primitives(sbvh_->primitives),
silhouettes(sbvh_->silhouettes),
silhouetteRefs(sbvh_->silhouetteRefs),
primitiveTypeIsAggregate(std::is_base_of<Aggregate<DIM>, PrimitiveType>::value),
range(enoki::arange<enoki::Array<int, DIM>>())
{
	static_assert(FCPW_MBVH_BRANCHING_FACTOR == 4 || FCPW_MBVH_BRANCHING_FACTOR == 8,
				  "Branching factor must be atleast 4");

	using namespace std::chrono;
	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	// collapse sbvh
	collapseSbvh(sbvh_, 0, 0xfffffffc, 0);

	// populate leaf nodes if primitive type is supported
	populateLeafNodes();

	// populate silhouette leaf nodes if primitive type is supported
	populateSilhouetteLeafNodes();

	// precompute surface area and signed volume
	int nPrimitives = (int)primitives.size();
	aggregateCentroid = Vector<DIM>::Zero();

	for (int p = 0; p < nPrimitives; p++) {
		aggregateCentroid += primitives[p]->centroid();
		area += primitives[p]->surfaceArea();
		volume += primitives[p]->signedVolume();
	}

	aggregateCentroid /= nPrimitives;

	// print stats
	if (printStats_) {
		// count not-full nodes
		float nLeafsNotFull = 0.0f;
		float nSilhouetteLeafsNotFull = 0.0f;
		float nNodesNotFull = 0.0f;
		int nInnerNodes = 0;

		for (int i = 0; i < nNodes; i++) {
			MbvhNode<DIM, CONEDATA>& node = flatTree[i];

			if (isLeafNode(node)) {
				if (node.child[3]%WIDTH != 0) {
					nLeafsNotFull += 1.0f;
				}

				updateSilhouetteLeafInfo<WIDTH, DIM>(node, nSilhouetteLeafsNotFull);

			} else {
				nInnerNodes++;
				for (int w = 0; w < FCPW_MBVH_BRANCHING_FACTOR; w++) {
					if (node.child[w] == maxInt) {
						nNodesNotFull += 1.0f;
						break;
					}
				}
			}
		}

		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		duration<double> timeSpan = duration_cast<duration<double>>(t2 - t1);
		std::cout << "Built " << FCPW_MBVH_BRANCHING_FACTOR << "-bvh with "
				  << nNodes << " nodes, "
				  << nLeafs << " leaves, "
				  << nSilhouetteLeafs << " silhouette leaves, "
				  << (nNodesNotFull*100/nInnerNodes) << "% nodes, "
				  << (nLeafsNotFull*100/nLeafs) << "% leaves not full & "
				  << (nSilhouetteLeafsNotFull*100/nSilhouetteLeafs) << "% silhouette leaves not full, "
				  << maxDepth << " max depth in "
				  << timeSpan.count() << " seconds" << std::endl;
	}
}

template<size_t WIDTH, size_t DIM, bool CONEDATA, typename PrimitiveType, typename SilhouetteType>
inline BoundingBox<DIM> Mbvh<WIDTH, DIM, CONEDATA, PrimitiveType, SilhouetteType>::boundingBox() const
{
	BoundingBox<DIM> box;
	if (flatTree.size() == 0) return box;

	enoki::scatter(box.pMin.data(), enoki::hmin_inner(flatTree[0].boxMin), range);
	enoki::scatter(box.pMax.data(), enoki::hmax_inner(flatTree[0].boxMax), range);

	return box;
}

template<size_t WIDTH, size_t DIM, bool CONEDATA, typename PrimitiveType, typename SilhouetteType>
inline Vector<DIM> Mbvh<WIDTH, DIM, CONEDATA, PrimitiveType, SilhouetteType>::centroid() const
{
	return aggregateCentroid;
}

template<size_t WIDTH, size_t DIM, bool CONEDATA, typename PrimitiveType, typename SilhouetteType>
inline float Mbvh<WIDTH, DIM, CONEDATA, PrimitiveType, SilhouetteType>::surfaceArea() const
{
	return area;
}

template<size_t WIDTH, size_t DIM, bool CONEDATA, typename PrimitiveType, typename SilhouetteType>
inline float Mbvh<WIDTH, DIM, CONEDATA, PrimitiveType, SilhouetteType>::signedVolume() const
{
	return volume;
}

template<size_t WIDTH, size_t DIM, bool CONEDATA>
inline void enqueueNodes(const MbvhNode<DIM, CONEDATA>& node, const FloatP<WIDTH>& tMin,
						 const FloatP<WIDTH>& tMax, const MaskP<WIDTH>& mask, float minDist,
						 float& tMaxMin, int& stackPtr, BvhTraversal *subtree)
{
	// enqueue nodes
	int closestIndex = -1;
	for (int w = 0; w < WIDTH; w++) {
		if (mask[w]) {
			stackPtr++;
			subtree[stackPtr].node = node.child[w];
			subtree[stackPtr].distance = tMin[w];
			tMaxMin = std::min(tMaxMin, tMax[w]);

			if (tMin[w] < minDist) {
				closestIndex = stackPtr;
				minDist = tMin[w];
			}
		}
	}

	// put closest node first
	if (closestIndex != -1) {
		std::swap(subtree[stackPtr], subtree[closestIndex]);
	}
}

inline void sortOrder4(const FloatP<4>& t, int& a, int& b, int& c, int& d)
{
	// source: https://stackoverflow.com/questions/25070577/sort-4-numbers-without-array
	int tmp;
	if (t[a] < t[b]) { tmp = a; a = b; b = tmp; }
	if (t[c] < t[d]) { tmp = c; c = d; d = tmp; }
	if (t[a] < t[c]) { tmp = a; a = c; c = tmp; }
	if (t[b] < t[d]) { tmp = b; b = d; d = tmp; }
	if (t[b] < t[c]) { tmp = b; b = c; c = tmp; }
}

template<size_t DIM, bool CONEDATA>
inline void enqueueNodes(const MbvhNode<DIM, CONEDATA>& node, const FloatP<4>& tMin,
						 const FloatP<4>& tMax, const MaskP<4>& mask, float minDist,
						 float& tMaxMin, int& stackPtr, BvhTraversal *subtree)
{
	// sort nodes
	int order[4] = {0, 1, 2, 3};
	sortOrder4(tMin, order[0], order[1], order[2], order[3]);

	// enqueue overlapping nodes in sorted order
	for (int w = 0; w < 4; w++) {
		int W = order[w];

		if (mask[W]) {
			stackPtr++;
			subtree[stackPtr].node = node.child[W];
			subtree[stackPtr].distance = tMin[W];
			tMaxMin = std::min(tMaxMin, tMax[W]);
		}
	}
}

template<size_t WIDTH, size_t DIM, bool CONEDATA, typename PrimitiveType>
inline int intersectRayPrimitives(const MbvhNode<DIM, CONEDATA>& node,
								  const std::vector<MbvhLeafNode<WIDTH, DIM, PrimitiveType>>& leafNodes,
								  int nodeIndex, int aggregateIndex, const enokiVector<DIM>& ro, const enokiVector<DIM>& rd,
								  float& rtMax, std::vector<Interaction<DIM>>& is, bool checkForOcclusion, bool recordAllHits)
{
	std::cerr << "intersectRayPrimitives(): WIDTH: " << WIDTH << ", DIM: " << DIM << " not supported" << std::endl;
	exit(EXIT_FAILURE);

	return 0;
}

template<size_t WIDTH, bool CONEDATA>
inline int intersectRayPrimitives(const MbvhNode<3, CONEDATA>& node,
								  const std::vector<MbvhLeafNode<WIDTH, 3, LineSegment>>& leafNodes,
								  int nodeIndex, int aggregateIndex, const enokiVector3& ro, const enokiVector3& rd,
								  float& rtMax, std::vector<Interaction<3>>& is, bool checkForOcclusion, bool recordAllHits)
{
	int leafOffset = -node.child[0] - 1;
	int nLeafs = node.child[1];
	int referenceOffset = node.child[2];
	int nReferences = node.child[3];
	int startReference = 0;
	int hits = 0;

	for (int l = 0; l < nLeafs; l++) {
		// perform vectorized intersection query
		FloatP<WIDTH> d;
		Vector3P<WIDTH> pt, n;
		FloatP<WIDTH> t;
		int leafIndex = leafOffset + l;
		const Vector3P<WIDTH>& pa = leafNodes[leafIndex].positions[0];
		const Vector3P<WIDTH>& pb = leafNodes[leafIndex].positions[1];
		const IntP<WIDTH>& primitiveIndex = leafNodes[leafIndex].primitiveIndex;
		MaskP<WIDTH> mask = intersectWideLineSegment<WIDTH>(pa, pb, ro, rd, rtMax, d,
															pt, n, t, checkForOcclusion);

		if (recordAllHits) {
			// record interactions
			int endReference = startReference + WIDTH;
			if (endReference > nReferences) endReference = nReferences;

			for (int p = startReference; p < endReference; p++) {
				int w = p - startReference;

				if (mask[w]) {
					hits++;
					auto it = is.emplace(is.end(), Interaction<3>());
					it->d = d[w];
					it->p[0] = pt[0][w];
					it->p[1] = pt[1][w];
					it->p[2] = pt[2][w];
					it->n[0] = n[0][w];
					it->n[1] = n[1][w];
					it->n[2] = n[2][w];
					it->uv[0] = t[w];
					it->uv[1] = -1;
					it->primitiveIndex = primitiveIndex[w];
					it->nodeIndex = nodeIndex;
					it->referenceIndex = referenceOffset + p;
					it->objectIndex = aggregateIndex;
				}
			}

		} else {
			// determine closest index
			int closestIndex = -1;
			int W = std::min((int)WIDTH, nReferences - startReference);

			for (int w = 0; w < W; w++) {
				if (mask[w] && d[w] <= rtMax) {
					if (checkForOcclusion) return 1;
					closestIndex = w;
					rtMax = d[w];
				}
			}

			// update interaction
			if (closestIndex != -1) {
				hits = 1;
				is[0].d = d[closestIndex];
				is[0].p[0] = pt[0][closestIndex];
				is[0].p[1] = pt[1][closestIndex];
				is[0].p[2] = pt[2][closestIndex];
				is[0].n[0] = n[0][closestIndex];
				is[0].n[1] = n[1][closestIndex];
				is[0].n[2] = n[2][closestIndex];
				is[0].uv[0] = t[closestIndex];
				is[0].uv[1] = -1;
				is[0].primitiveIndex = primitiveIndex[closestIndex];
				is[0].nodeIndex = nodeIndex;
				is[0].referenceIndex = referenceOffset + startReference + closestIndex;
				is[0].objectIndex = aggregateIndex;
			}
		}

		startReference += WIDTH;
	}

	return hits;
}

template<size_t WIDTH, bool CONEDATA>
inline int intersectRayPrimitives(const MbvhNode<3, CONEDATA>& node,
								  const std::vector<MbvhLeafNode<WIDTH, 3, Triangle>>& leafNodes,
								  int nodeIndex, int aggregateIndex, const enokiVector3& ro, const enokiVector3& rd,
								  float& rtMax, std::vector<Interaction<3>>& is, bool checkForOcclusion, bool recordAllHits)
{
	int leafOffset = -node.child[0] - 1;
	int nLeafs = node.child[1];
	int referenceOffset = node.child[2];
	int nReferences = node.child[3];
	int startReference = 0;
	int hits = 0;

	for (int l = 0; l < nLeafs; l++) {
		// perform vectorized intersection query
		FloatP<WIDTH> d;
		Vector3P<WIDTH> pt, n;
		Vector2P<WIDTH> t;
		int leafIndex = leafOffset + l;
		const Vector3P<WIDTH>& pa = leafNodes[leafIndex].positions[0];
		const Vector3P<WIDTH>& pb = leafNodes[leafIndex].positions[1];
		const Vector3P<WIDTH>& pc = leafNodes[leafIndex].positions[2];
		const IntP<WIDTH>& primitiveIndex = leafNodes[leafIndex].primitiveIndex;
		MaskP<WIDTH> mask = intersectWideTriangle<WIDTH>(pa, pb, pc, ro, rd, rtMax, d,
														 pt, n, t, checkForOcclusion);

		if (recordAllHits) {
			// record interactions
			int endReference = startReference + WIDTH;
			if (endReference > nReferences) endReference = nReferences;

			for (int p = startReference; p < endReference; p++) {
				int w = p - startReference;

				if (mask[w]) {
					hits++;
					auto it = is.emplace(is.end(), Interaction<3>());
					it->d = d[w];
					it->p[0] = pt[0][w];
					it->p[1] = pt[1][w];
					it->p[2] = pt[2][w];
					it->n[0] = n[0][w];
					it->n[1] = n[1][w];
					it->n[2] = n[2][w];
					it->uv[0] = t[0][w];
					it->uv[1] = t[1][w];
					it->primitiveIndex = primitiveIndex[w];
					it->nodeIndex = nodeIndex;
					it->referenceIndex = referenceOffset + p;
					it->objectIndex = aggregateIndex;
				}
			}

		} else {
			// determine closest index
			int closestIndex = -1;
			int W = std::min((int)WIDTH, nReferences - startReference);

			for (int w = 0; w < W; w++) {
				if (mask[w] && d[w] <= rtMax) {
					if (checkForOcclusion) return 1;
					closestIndex = w;
					rtMax = d[w];
				}
			}

			// update interaction
			if (closestIndex != -1) {
				hits = 1;
				is[0].d = d[closestIndex];
				is[0].p[0] = pt[0][closestIndex];
				is[0].p[1] = pt[1][closestIndex];
				is[0].p[2] = pt[2][closestIndex];
				is[0].n[0] = n[0][closestIndex];
				is[0].n[1] = n[1][closestIndex];
				is[0].n[2] = n[2][closestIndex];
				is[0].uv[0] = t[0][closestIndex];
				is[0].uv[1] = t[1][closestIndex];
				is[0].primitiveIndex = primitiveIndex[closestIndex];
				is[0].nodeIndex = nodeIndex;
				is[0].referenceIndex = referenceOffset + startReference + closestIndex;
				is[0].objectIndex = aggregateIndex;
			}
		}

		startReference += WIDTH;
	}

	return hits;
}

template<size_t WIDTH, size_t DIM, bool CONEDATA, typename PrimitiveType, typename SilhouetteType>
inline int Mbvh<WIDTH, DIM, CONEDATA, PrimitiveType, SilhouetteType>::intersectFromNode(Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
																						int nodeStartIndex, int aggregateIndex, int& nodesVisited,
																						bool checkForOcclusion, bool recordAllHits) const
{
	int hits = 0;
	if (!recordAllHits) is.resize(1);
	BvhTraversal subtree[FCPW_MBVH_MAX_DEPTH];
	FloatP<FCPW_MBVH_BRANCHING_FACTOR> tMin, tMax;
	enokiVector<DIM> ro = enoki::gather<enokiVector<DIM>>(r.o.data(), range);
	enokiVector<DIM> rd = enoki::gather<enokiVector<DIM>>(r.d.data(), range);
	enokiVector<DIM> rinvD = enoki::gather<enokiVector<DIM>>(r.invD.data(), range);

	// push root node
	int rootIndex = aggregateIndex == this->index ? nodeStartIndex : 0;
	subtree[0].node = rootIndex;
	subtree[0].distance = minFloat;
	int stackPtr = 0;

	while (stackPtr >= 0) {
		// pop off the next node to work on
		int nodeIndex = subtree[stackPtr].node;
		float currentDist = subtree[stackPtr].distance;
		stackPtr--;

		// if this node is further than the closest found intersection, continue
		if (!recordAllHits && currentDist > r.tMax) continue;
		const MbvhNode<DIM, CONEDATA>& node(flatTree[nodeIndex]);

		if (isLeafNode(node)) {
			if (std::is_same<PrimitiveType, LineSegment>::value ||
				std::is_same<PrimitiveType, Triangle>::value) {
				// perform vectorized intersection query
				hits += intersectRayPrimitives(node, leafNodes, nodeIndex, this->index, ro, rd,
											   r.tMax, is, checkForOcclusion, recordAllHits);
				nodesVisited++;

				if (hits > 0 && checkForOcclusion) {
					return 1;
				}

			} else {
				// primitive type does not support vectorized intersection query,
				// perform query to each primitive one by one
				int referenceOffset = node.child[2];
				int nReferences = node.child[3];

				for (int p = 0; p < nReferences; p++) {
					int referenceIndex = referenceOffset + p;
					const PrimitiveType *prim = primitives[referenceIndex];
					nodesVisited++;

					int hit = 0;
					std::vector<Interaction<DIM>> cs;
					if (primitiveTypeIsAggregate) {
						const Aggregate<DIM> *aggregate = reinterpret_cast<const Aggregate<DIM> *>(prim);
						hit = aggregate->intersectFromNode(r, cs, nodeStartIndex, aggregateIndex,
														   nodesVisited, checkForOcclusion, recordAllHits);

					} else {
						hit = prim->intersect(r, cs, checkForOcclusion, recordAllHits);
						for (int i = 0; i < (int)cs.size(); i++) {
							cs[i].nodeIndex = nodeIndex;
							cs[i].referenceIndex = referenceIndex;
							cs[i].objectIndex = this->index;
						}
					}

					// keep the closest intersection only
					if (hit > 0) {
						if (checkForOcclusion) {
							return 1;
						}

						hits += hit;
						if (recordAllHits) {
							is.insert(is.end(), cs.begin(), cs.end());

						} else {
							r.tMax = std::min(r.tMax, cs[0].d);
							is[0] = cs[0];
						}
					}
				}
			}

		} else {
			// intersect ray with boxes
			MaskP<FCPW_MBVH_BRANCHING_FACTOR> mask = intersectWideBox<FCPW_MBVH_BRANCHING_FACTOR, DIM>(
														node.boxMin, node.boxMax, ro, rinvD, r.tMax, tMin, tMax);

			// enqueue intersecting boxes in sorted order
			nodesVisited++;
			mask &= enoki::neq(node.child, maxInt);
			if (enoki::any(mask)) {
				float stub = 0.0f;
				enqueueNodes(node, tMin, tMax, mask, r.tMax, stub, stackPtr, subtree);
			}
		}
	}

	if (hits > 0) {
		// sort by distance and remove duplicates
		if (recordAllHits) {
			std::sort(is.begin(), is.end(), compareInteractions<DIM>);
			is = removeDuplicates<DIM>(is);
			hits = (int)is.size();

		} else {
			hits = 1;
		}

		return hits;
	}

	return 0;
}

template<size_t WIDTH, size_t DIM, bool CONEDATA, typename PrimitiveType>
inline int intersectSpherePrimitives(const MbvhNode<DIM, CONEDATA>& node,
									 const std::vector<MbvhLeafNode<WIDTH, DIM, PrimitiveType>>& leafNodes,
									 const std::function<float(float)>& primitiveWeight,
									 int nodeIndex, int aggregateIndex, const enokiVector<DIM>& sc,
									 float sr2, float u, std::vector<Interaction<DIM>>& is,
									 float& totalPrimitiveWeight, bool recordOneHit,
									 bool isNodeInsideSphere=false)
{
	std::cerr << "intersectSpherePrimitives(): WIDTH: " << WIDTH << ", DIM: " << DIM << " not supported" << std::endl;
	exit(EXIT_FAILURE);

	return 0;
}

template<size_t WIDTH, bool CONEDATA>
inline int intersectSpherePrimitives(const MbvhNode<3, CONEDATA>& node,
									 const std::vector<MbvhLeafNode<WIDTH, 3, LineSegment>>& leafNodes,
									 const std::function<float(float)>& primitiveWeight,
									 int nodeIndex, int aggregateIndex, const enokiVector3& sc,
									 float sr2, float u, std::vector<Interaction<3>>& is,
									 float& totalPrimitiveWeight, bool recordOneHit,
									 bool isNodeInsideSphere=false)
{
	Vector3 queryPt(sc[0], sc[1], sc[2]);
	int leafOffset = -node.child[0] - 1;
	int nLeafs = node.child[1];
	int referenceOffset = node.child[2];
	int nReferences = node.child[3];
	int startReference = 0;
	int hits = 0;

	for (int l = 0; l < nLeafs; l++) {
		// perform vectorized closest point query
		Vector3P<WIDTH> pt;
		FloatP<WIDTH> t;
		int leafIndex = leafOffset + l;
		const Vector3P<WIDTH>& pa = leafNodes[leafIndex].positions[0];
		const Vector3P<WIDTH>& pb = leafNodes[leafIndex].positions[1];
		const IntP<WIDTH>& primitiveIndex = leafNodes[leafIndex].primitiveIndex;
		FloatP<WIDTH> surfaceArea = enoki::norm(pb - pa);
		FloatP<WIDTH> d = isNodeInsideSphere ? 0.0f : findClosestPointWideLineSegment<WIDTH>(pa, pb, sc, pt, t);
		FloatP<WIDTH> d2 = d*d;

		// record interactions
		int endReference = startReference + WIDTH;
		if (endReference > nReferences) endReference = nReferences;

		for (int p = startReference; p < endReference; p++) {
			int w = p - startReference;

			if (d2[w] <= sr2) {
				hits++;

				if (recordOneHit) {
					Vector3 closestPt(pt[0][w], pt[1][w], pt[2][w]);
					float weight = surfaceArea[w];
					if (primitiveWeight) weight *= primitiveWeight((queryPt - closestPt).squaredNorm());
					totalPrimitiveWeight += weight;
					float selectionProb = weight/totalPrimitiveWeight;

					if (u < selectionProb) {
						u = u/selectionProb; // rescale to [0,1)
						is[0].d = weight;
						is[0].primitiveIndex = primitiveIndex[w];
						is[0].nodeIndex = nodeIndex;
						is[0].referenceIndex = referenceOffset + p;
						is[0].objectIndex = aggregateIndex;

					} else {
						u = (u - selectionProb)/(1.0f - selectionProb);
					}

				} else {
					auto it = is.emplace(is.end(), Interaction<3>());
					it->d = 1.0f;
					it->primitiveIndex = primitiveIndex[w];
					it->nodeIndex = nodeIndex;
					it->referenceIndex = referenceOffset + p;
					it->objectIndex = aggregateIndex;
				}
			}
		}

		startReference += WIDTH;
	}

	return hits;
}

template<size_t WIDTH, bool CONEDATA>
inline int intersectSpherePrimitives(const MbvhNode<3, CONEDATA>& node,
									 const std::vector<MbvhLeafNode<WIDTH, 3, Triangle>>& leafNodes,
									 const std::function<float(float)>& primitiveWeight,
									 int nodeIndex, int aggregateIndex, const enokiVector3& sc,
									 float sr2, float u, std::vector<Interaction<3>>& is,
									 float& totalPrimitiveWeight, bool recordOneHit,
									 bool isNodeInsideSphere=false)
{
	Vector3 queryPt(sc[0], sc[1], sc[2]);
	int leafOffset = -node.child[0] - 1;
	int nLeafs = node.child[1];
	int referenceOffset = node.child[2];
	int nReferences = node.child[3];
	int startReference = 0;
	int hits = 0;

	for (int l = 0; l < nLeafs; l++) {
		// perform vectorized closest point query
		Vector3P<WIDTH> pt;
		Vector2P<WIDTH> t;
		int leafIndex = leafOffset + l;
		const Vector3P<WIDTH>& pa = leafNodes[leafIndex].positions[0];
		const Vector3P<WIDTH>& pb = leafNodes[leafIndex].positions[1];
		const Vector3P<WIDTH>& pc = leafNodes[leafIndex].positions[2];
		const IntP<WIDTH>& primitiveIndex = leafNodes[leafIndex].primitiveIndex;
		FloatP<WIDTH> surfaceArea = 0.5f*enoki::norm(enoki::cross(pb - pa, pc - pa));
		FloatP<WIDTH> d = isNodeInsideSphere ? 0.0f : findClosestPointWideTriangle<WIDTH>(pa, pb, pc, sc, pt, t);
		FloatP<WIDTH> d2 = d*d;

		// record interactions
		int endReference = startReference + WIDTH;
		if (endReference > nReferences) endReference = nReferences;

		for (int p = startReference; p < endReference; p++) {
			int w = p - startReference;

			if (d2[w] <= sr2) {
				hits++;

				if (recordOneHit) {
					Vector3 closestPt(pt[0][w], pt[1][w], pt[2][w]);
					float weight = surfaceArea[w];
					if (primitiveWeight) weight *= primitiveWeight((queryPt - closestPt).squaredNorm());
					totalPrimitiveWeight += weight;
					float selectionProb = weight/totalPrimitiveWeight;

					if (u < selectionProb) {
						u = u/selectionProb; // rescale to [0,1)
						is[0].d = weight;
						is[0].primitiveIndex = primitiveIndex[w];
						is[0].nodeIndex = nodeIndex;
						is[0].referenceIndex = referenceOffset + p;
						is[0].objectIndex = aggregateIndex;

					} else {
						u = (u - selectionProb)/(1.0f - selectionProb);
					}

				} else {
					auto it = is.emplace(is.end(), Interaction<3>());
					it->d = 1.0f;
					it->primitiveIndex = primitiveIndex[w];
					it->nodeIndex = nodeIndex;
					it->referenceIndex = referenceOffset + p;
					it->objectIndex = aggregateIndex;
				}
			}
		}

		startReference += WIDTH;
	}

	return hits;
}

template<size_t WIDTH, size_t DIM, bool CONEDATA, typename PrimitiveType, typename SilhouetteType>
inline int Mbvh<WIDTH, DIM, CONEDATA, PrimitiveType, SilhouetteType>::intersectFromNode(const BoundingSphere<DIM>& s,
																						std::vector<Interaction<DIM>>& is,
																						int nodeStartIndex, int aggregateIndex,
																						int& nodesVisited, bool recordOneHit,
																						const std::function<float(float)>& primitiveWeight) const
{
	int hits = 0;
	float totalPrimitiveWeight = 0.0f;
	if (recordOneHit && !primitiveTypeIsAggregate) is.resize(1);
	BvhTraversal subtree[FCPW_MBVH_MAX_DEPTH];
	FloatP<FCPW_MBVH_BRANCHING_FACTOR> d2Min, d2Max;
	enokiVector<DIM> sc = enoki::gather<enokiVector<DIM>>(s.c.data(), range);

	// push root node
	int rootIndex = aggregateIndex == this->index ? nodeStartIndex : 0;
	subtree[0].node = rootIndex;
	subtree[0].distance = s.r2;
	int stackPtr = 0;

	while (stackPtr >= 0) {
		// pop off the next node to work on
		int nodeIndex = subtree[stackPtr].node;
		const MbvhNode<DIM, CONEDATA>& node(flatTree[nodeIndex]);
		stackPtr--;

		if (isLeafNode(node)) {
			if (std::is_same<PrimitiveType, LineSegment>::value ||
				std::is_same<PrimitiveType, Triangle>::value) {
				// perform vectorized intersection query
				float u = uniformRealRandomNumber();
				hits += intersectSpherePrimitives(node, leafNodes, primitiveWeight, nodeIndex, this->index,
												  sc, s.r2, u, is, totalPrimitiveWeight, recordOneHit);
				nodesVisited++;

			} else {
				// primitive type does not support vectorized intersection query,
				// perform query to each primitive one by one
				int referenceOffset = node.child[2];
				int nReferences = node.child[3];

				for (int p = 0; p < nReferences; p++) {
					int referenceIndex = referenceOffset + p;
					const PrimitiveType *prim = primitives[referenceIndex];
					nodesVisited++;

					int hit = 0;
					std::vector<Interaction<DIM>> cs;
					if (primitiveTypeIsAggregate) {
						const Aggregate<DIM> *aggregate = reinterpret_cast<const Aggregate<DIM> *>(prim);
						hit = aggregate->intersectFromNode(s, cs, nodeStartIndex, aggregateIndex,
														   nodesVisited, recordOneHit, primitiveWeight);

					} else {
						hit = prim->intersect(s, cs, recordOneHit, primitiveWeight);
						for (int i = 0; i < (int)cs.size(); i++) {
							cs[i].nodeIndex = nodeIndex;
							cs[i].referenceIndex = referenceIndex;
							cs[i].objectIndex = this->index;
						}
					}

					if (hit > 0) {
						hits += hit;
						if (recordOneHit && !primitiveTypeIsAggregate) {
							totalPrimitiveWeight += cs[0].d;
							if (uniformRealRandomNumber()*totalPrimitiveWeight < cs[0].d) {
								is[0] = cs[0];
							}

						} else {
							is.insert(is.end(), cs.begin(), cs.end());
						}
					}
				}
			}

		} else {
			// overlap sphere with boxes
			MaskP<FCPW_MBVH_BRANCHING_FACTOR> mask = overlapWideBox<FCPW_MBVH_BRANCHING_FACTOR, DIM>(
																node.boxMin, node.boxMax, sc, s.r2, d2Min, d2Max);

			// enqueue overlapping boxes
			nodesVisited++;
			mask &= enoki::neq(node.child, maxInt);
			if (enoki::any(mask)) {
				for (int w = 0; w < FCPW_MBVH_BRANCHING_FACTOR; w++) {
					if (mask[w]) {
						stackPtr++;
						subtree[stackPtr].node = node.child[w];
					}
				}
			}
		}
	}

	if (hits > 0) {
		if (recordOneHit && !primitiveTypeIsAggregate) {
			if (is[0].primitiveIndex == -1) {
				hits = 0;
				is.clear();

			} else if (totalPrimitiveWeight > 0.0f) {
				is[0].d /= totalPrimitiveWeight;
			}
		}

		return hits;
	}

	return 0;
}

template<size_t WIDTH, size_t DIM, bool CONEDATA, typename PrimitiveType, typename SilhouetteType>
inline int Mbvh<WIDTH, DIM, CONEDATA, PrimitiveType, SilhouetteType>::intersectStochasticFromNode(const BoundingSphere<DIM>& s,
																								  std::vector<Interaction<DIM>>& is, float *randNums,
																								  int nodeStartIndex, int aggregateIndex, int& nodesVisited,
																								  const std::function<float(float)>& traversalWeight,
																								  const std::function<float(float)>& primitiveWeight) const
{
	int hits = 0;
	if (!primitiveTypeIsAggregate) is.resize(1);
	FloatP<FCPW_MBVH_BRANCHING_FACTOR> d2Min, d2Max;
	float d2NodeMax = maxFloat;
	float u = randNums[0];
	enokiVector<DIM> sc = enoki::gather<enokiVector<DIM>>(s.c.data(), range);

	// push root node
	int nodeIndex = aggregateIndex == this->index ? nodeStartIndex : 0;
	float traversalPdf = 1.0f;
	int stackPtr = 0;

	while (stackPtr >= 0) {
		// pop off the next node to work on
		const MbvhNode<DIM, CONEDATA>& node(flatTree[nodeIndex]);
		stackPtr--;

		if (isLeafNode(node)) {
			float totalPrimitiveWeight = 0.0f;
			if (std::is_same<PrimitiveType, LineSegment>::value ||
				std::is_same<PrimitiveType, Triangle>::value) {
				// perform vectorized intersection query
				int nInteractions = (int)is.size();
				int hit = intersectSpherePrimitives(node, leafNodes, primitiveWeight, nodeIndex, this->index,
													sc, s.r2, u, is, totalPrimitiveWeight, true, d2NodeMax <= s.r2);
				nodesVisited++;

				if (hit > 0) {
					hits += hit;
					if (!primitiveTypeIsAggregate) {
						is[0].d *= traversalPdf;

					} else {
						for (int i = nInteractions; i < (int)is.size(); i++) {
							is[i].d *= traversalPdf;
						}
					}
				}

			} else {
				// primitive type does not support vectorized intersection query,
				// perform query to each primitive one by one
				int referenceOffset = node.child[2];
				int nReferences = node.child[3];

				for (int p = 0; p < nReferences; p++) {
					int referenceIndex = referenceOffset + p;
					const PrimitiveType *prim = primitives[referenceIndex];
					nodesVisited++;

					int hit = 0;
					std::vector<Interaction<DIM>> cs;
					if (primitiveTypeIsAggregate) {
						float modifiedRandNums[DIM];
						modifiedRandNums[0] = u;
						for (int i = 1; i < DIM; i++) modifiedRandNums[i] = randNums[i];
						const Aggregate<DIM> *aggregate = reinterpret_cast<const Aggregate<DIM> *>(prim);
						hit = aggregate->intersectStochasticFromNode(s, cs, modifiedRandNums, nodeStartIndex, aggregateIndex,
																	 nodesVisited, traversalWeight, primitiveWeight);

					} else {
						if (d2NodeMax <= s.r2) {
							hit = 1;
							auto it = cs.emplace(cs.end(), Interaction<3>());
							it->primitiveIndex = reinterpret_cast<const GeometricPrimitive<DIM> *>(prim)->pIndex;
							it->d = prim->surfaceArea();
							if (primitiveWeight) {
								float d2 = (s.c - prim->centroid()).squaredNorm();
								it->d *= primitiveWeight(d2);
							}

						} else {
							hit = prim->intersect(s, cs, true, primitiveWeight);
						}

						for (int i = 0; i < (int)cs.size(); i++) {
							cs[i].nodeIndex = nodeIndex;
							cs[i].referenceIndex = referenceIndex;
							cs[i].objectIndex = this->index;
						}
					}

					if (hit > 0) {
						hits += hit;
						if (!primitiveTypeIsAggregate) {
							totalPrimitiveWeight += cs[0].d;
							float selectionProb = cs[0].d/totalPrimitiveWeight;

							if (u < selectionProb) {
								u = u/selectionProb; // rescale to [0,1)
								is[0] = cs[0];
								is[0].d *= traversalPdf;

							} else {
								u = (u - selectionProb)/(1.0f - selectionProb);
							}

						} else {
							int nInteractions = (int)is.size();
							is.insert(is.end(), cs.begin(), cs.end());
							for (int i = nInteractions; i < (int)is.size(); i++) {
								is[i].d *= traversalPdf;
							}
						}
					}
				}
			}

			if (!primitiveTypeIsAggregate) {
				if (totalPrimitiveWeight > 0.0f) {
					is[0].d /= totalPrimitiveWeight;
				}
			}

		} else {
			// overlap sphere with boxes
			MaskP<FCPW_MBVH_BRANCHING_FACTOR> mask = overlapWideBox<FCPW_MBVH_BRANCHING_FACTOR, DIM>(
																node.boxMin, node.boxMax, sc, s.r2, d2Min, d2Max);

			// enqueue overlapping boxes
			nodesVisited++;
			mask &= enoki::neq(node.child, maxInt);
			if (enoki::any(mask)) {
				int selectedIndex = -1;
				float selectedWeight = 0.0f;
				float totalTraversalWeight = 0.0f;
				FloatP<FCPW_MBVH_BRANCHING_FACTOR> r2;
				if (traversalWeight) {
					VectorP<FCPW_MBVH_BRANCHING_FACTOR, DIM> boxCenter = (node.boxMin + node.boxMax)*0.5f;
					r2 = enoki::squared_norm(sc - boxCenter);
				}

				for (int w = 0; w < FCPW_MBVH_BRANCHING_FACTOR; w++) {
					if (mask[w]) {
						float weight = traversalWeight ? traversalWeight(r2[w]) : 1.0f;
						totalTraversalWeight += weight;
						float prob = weight/totalTraversalWeight;

						if (u < prob) {
							selectedIndex = w;
							selectedWeight = weight;
							u = u/prob;

						} else {
							u = (u - prob)/(1.0f - prob);
						}
					}
				}

				if (selectedIndex != -1) {
					stackPtr++;
					nodeIndex = node.child[selectedIndex];
					traversalPdf *= selectedWeight/totalTraversalWeight;
					d2NodeMax = d2Max[selectedIndex];
				}
			}
		}
	}

	if (hits > 0) {
		if (!primitiveTypeIsAggregate) {
			if (is[0].primitiveIndex == -1) {
				hits = 0;
				is.clear();

			} else {
				// sample a point on the selected geometric primitive
				const PrimitiveType *prim = primitives[is[0].referenceIndex];
				float pdf = is[0].samplePoint(prim, randNums);
				is[0].d *= pdf;
			}
		}

		return hits;
	}

	return 0;
}

template<size_t WIDTH, size_t DIM, bool CONEDATA, typename PrimitiveType>
inline bool findClosestPointPrimitives(const MbvhNode<DIM, CONEDATA>& node,
									   const std::vector<MbvhLeafNode<WIDTH, DIM, PrimitiveType>>& leafNodes,
									   int nodeIndex, int aggregateIndex, const enokiVector<DIM>& sc,
									   float& sr2, Interaction<DIM>& i)
{
	std::cerr << "findClosestPointPrimitives(): WIDTH: " << WIDTH << ", DIM: " << DIM << " not supported" << std::endl;
	exit(EXIT_FAILURE);

	return false;
}

template<size_t WIDTH, bool CONEDATA>
inline bool findClosestPointPrimitives(const MbvhNode<3, CONEDATA>& node,
									   const std::vector<MbvhLeafNode<WIDTH, 3, LineSegment>>& leafNodes,
									   int nodeIndex, int aggregateIndex, const enokiVector3& sc,
									   float& sr2, Interaction<3>& i)
{
	int leafOffset = -node.child[0] - 1;
	int nLeafs = node.child[1];
	int referenceOffset = node.child[2];
	int nReferences = node.child[3];
	int startReference = 0;
	bool found = false;

	for (int l = 0; l < nLeafs; l++) {
		// perform vectorized closest point query
		Vector3P<WIDTH> pt;
		FloatP<WIDTH> t;
		int leafIndex = leafOffset + l;
		const Vector3P<WIDTH>& pa = leafNodes[leafIndex].positions[0];
		const Vector3P<WIDTH>& pb = leafNodes[leafIndex].positions[1];
		const IntP<WIDTH>& primitiveIndex = leafNodes[leafIndex].primitiveIndex;
		FloatP<WIDTH> d = findClosestPointWideLineSegment<WIDTH>(pa, pb, sc, pt, t);
		FloatP<WIDTH> d2 = d*d;

		// determine closest index
		int closestIndex = -1;
		int W = std::min((int)WIDTH, nReferences - startReference);

		for (int w = 0; w < W; w++) {
			if (d2[w] <= sr2) {
				closestIndex = w;
				sr2 = d2[w];
			}
		}

		// update interaction
		if (closestIndex != -1) {
			i.d = d[closestIndex];
			i.p[0] = pt[0][closestIndex];
			i.p[1] = pt[1][closestIndex];
			i.p[2] = pt[2][closestIndex];
			i.uv[0] = t[closestIndex];
			i.uv[1] = -1;
			i.primitiveIndex = primitiveIndex[closestIndex];
			i.nodeIndex = nodeIndex;
			i.referenceIndex = referenceOffset + startReference + closestIndex;
			i.objectIndex = aggregateIndex;
			found = true;
		}

		startReference += WIDTH;
	}

	return found;
}

template<size_t WIDTH, bool CONEDATA>
inline bool findClosestPointPrimitives(const MbvhNode<3, CONEDATA>& node,
									   const std::vector<MbvhLeafNode<WIDTH, 3, Triangle>>& leafNodes,
									   int nodeIndex, int aggregateIndex, const enokiVector3& sc,
									   float& sr2, Interaction<3>& i)
{
	int leafOffset = -node.child[0] - 1;
	int nLeafs = node.child[1];
	int referenceOffset = node.child[2];
	int nReferences = node.child[3];
	int startReference = 0;
	bool found = false;

	for (int l = 0; l < nLeafs; l++) {
		// perform vectorized closest point query
		Vector3P<WIDTH> pt;
		Vector2P<WIDTH> t;
		int leafIndex = leafOffset + l;
		const Vector3P<WIDTH>& pa = leafNodes[leafIndex].positions[0];
		const Vector3P<WIDTH>& pb = leafNodes[leafIndex].positions[1];
		const Vector3P<WIDTH>& pc = leafNodes[leafIndex].positions[2];
		const IntP<WIDTH>& primitiveIndex = leafNodes[leafIndex].primitiveIndex;
		FloatP<WIDTH> d = findClosestPointWideTriangle<WIDTH>(pa, pb, pc, sc, pt, t);
		FloatP<WIDTH> d2 = d*d;

		// determine closest index
		int closestIndex = -1;
		int W = std::min((int)WIDTH, nReferences - startReference);

		for (int w = 0; w < W; w++) {
			if (d2[w] <= sr2) {
				closestIndex = w;
				sr2 = d2[w];
			}
		}

		// update interaction
		if (closestIndex != -1) {
			i.d = d[closestIndex];
			i.p[0] = pt[0][closestIndex];
			i.p[1] = pt[1][closestIndex];
			i.p[2] = pt[2][closestIndex];
			i.uv[0] = t[0][closestIndex];
			i.uv[1] = t[1][closestIndex];
			i.primitiveIndex = primitiveIndex[closestIndex];
			i.nodeIndex = nodeIndex;
			i.referenceIndex = referenceOffset + startReference + closestIndex;
			i.objectIndex = aggregateIndex;
			found = true;
		}

		startReference += WIDTH;
	}

	return found;
}

template<size_t WIDTH, size_t DIM, bool CONEDATA, typename PrimitiveType, typename SilhouetteType>
inline bool Mbvh<WIDTH, DIM, CONEDATA, PrimitiveType, SilhouetteType>::findClosestPointFromNode(BoundingSphere<DIM>& s, Interaction<DIM>& i,
																								int nodeStartIndex, int aggregateIndex,
																								int& nodesVisited, bool recordNormal) const
{
	bool notFound = true;
	BvhTraversal subtree[FCPW_MBVH_MAX_DEPTH];
	FloatP<FCPW_MBVH_BRANCHING_FACTOR> d2Min, d2Max;
	enokiVector<DIM> sc = enoki::gather<enokiVector<DIM>>(s.c.data(), range);

	// push root node
	int rootIndex = aggregateIndex == this->index ? nodeStartIndex : 0;
	subtree[0].node = rootIndex;
	subtree[0].distance = minFloat;
	int stackPtr = 0;

	while (stackPtr >= 0) {
		// pop off the next node to work on
		int nodeIndex = subtree[stackPtr].node;
		float currentDist = subtree[stackPtr].distance;
		stackPtr--;

		// if this node is further than the closest found primitive, continue
		if (currentDist > s.r2) continue;
		const MbvhNode<DIM, CONEDATA>& node(flatTree[nodeIndex]);

		if (isLeafNode(node)) {
			if (std::is_same<PrimitiveType, LineSegment>::value ||
				std::is_same<PrimitiveType, Triangle>::value) {
				// perform vectorized closest point query
				bool found = findClosestPointPrimitives(node, leafNodes, nodeIndex, this->index, sc, s.r2, i);
				if (found) notFound = false;
				nodesVisited++;

			} else {
				// primitive type does not support vectorized closest point query,
				// perform query to each primitive one by one
				int referenceOffset = node.child[2];
				int nReferences = node.child[3];

				for (int p = 0; p < nReferences; p++) {
					int referenceIndex = referenceOffset + p;
					const PrimitiveType *prim = primitives[referenceIndex];
					nodesVisited++;

					bool found = false;
					Interaction<DIM> c;

					if (primitiveTypeIsAggregate) {
						const Aggregate<DIM> *aggregate = reinterpret_cast<const Aggregate<DIM> *>(prim);
						found = aggregate->findClosestPointFromNode(s, c, nodeStartIndex, aggregateIndex,
																	nodesVisited, recordNormal);

					} else {
						found = prim->findClosestPoint(s, c, recordNormal);
						c.nodeIndex = nodeIndex;
						c.referenceIndex = referenceIndex;
						c.objectIndex = this->index;
					}

					// keep the closest point only
					if (found) {
						notFound = false;
						s.r2 = std::min(s.r2, c.d*c.d);
						i = c;
					}
				}
			}

		} else {
			// overlap sphere with boxes
			MaskP<FCPW_MBVH_BRANCHING_FACTOR> mask = overlapWideBox<FCPW_MBVH_BRANCHING_FACTOR, DIM>(
																node.boxMin, node.boxMax, sc, s.r2, d2Min, d2Max);

			// enqueue overlapping boxes in sorted order
			nodesVisited++;
			mask &= enoki::neq(node.child, maxInt);
			if (enoki::any(mask)) {
				enqueueNodes(node, d2Min, d2Max, mask, s.r2, s.r2, stackPtr, subtree);
			}
		}
	}

	if (!notFound) {
		// compute normal
		if (recordNormal && !primitiveTypeIsAggregate) {
			i.computeNormal(primitives[i.referenceIndex]);
		}

		return true;
	}

	return false;
}

template<size_t WIDTH, size_t DIM, typename SilhouetteType>
inline bool findClosestSilhouettes(const MbvhNode<DIM, true>& node,
								   const std::vector<MbvhSilhouetteLeafNode<WIDTH, DIM, SilhouetteType>>& silhouetteLeafNodes,
								   int nodeIndex, int aggregateIndex, const enokiVector<DIM>& sc, float& sr2,
								   Interaction<DIM>& i, bool flipNormalOrientation, float squaredMinRadius, float precision)
{
	std::cerr << "findClosestSilhouettes(): WIDTH: " << WIDTH << ", DIM: " << DIM << " not supported" << std::endl;
	exit(EXIT_FAILURE);

	return false;
}

template<size_t WIDTH>
inline bool findClosestSilhouettes(const MbvhNode<3, true>& node,
								   const std::vector<MbvhSilhouetteLeafNode<WIDTH, 3, SilhouetteVertex>>& silhouetteLeafNodes,
								   int nodeIndex, int aggregateIndex, const enokiVector3& sc, float& sr2,
								   Interaction<3>& i, bool flipNormalOrientation, float squaredMinRadius, float precision)
{
	if (squaredMinRadius >= sr2) return false;

	int silhouetteLeafOffset = -node.silhouetteChild[0] - 1;
	int nSilhouetteLeafs = node.silhouetteChild[1];
	int silhouetteReferenceOffset = node.silhouetteChild[2];
	int nSilhouetteReferences = node.silhouetteChild[3];
	int startReference = 0;
	bool found = false;

	for (int l = 0; l < nSilhouetteLeafs; l++) {
		// perform vectorized closest silhouette query
		int leafIndex = silhouetteLeafOffset + l;
		const Vector3P<WIDTH>& pb = silhouetteLeafNodes[leafIndex].positions[1];
		const Vector3P<WIDTH>& n0 = silhouetteLeafNodes[leafIndex].positions[0];
		const Vector3P<WIDTH>& n1 = silhouetteLeafNodes[leafIndex].positions[2];
		const IntP<WIDTH>& primitiveIndex = silhouetteLeafNodes[leafIndex].primitiveIndex;
		const MaskP<WIDTH>& missingFace = silhouetteLeafNodes[leafIndex].missingFace;
		Vector3P<WIDTH> viewDir = sc - pb;
		FloatP<WIDTH> d = enoki::norm(viewDir);
		FloatP<WIDTH> d2 = d*d;
		if (enoki::all(d2 > sr2)) continue;
		MaskP<WIDTH> isSilhouette = missingFace;
		enoki::masked(isSilhouette, ~missingFace) = isWideSilhouetteVertex(n0, n1, viewDir, d, flipNormalOrientation, precision);

		// determine closest index
		int closestIndex = -1;
		int W = std::min((int)WIDTH, nSilhouetteReferences - startReference);

		for (int w = 0; w < W; w++) {
			if (isSilhouette[w] && d2[w] <= sr2) {
				closestIndex = w;
				sr2 = d2[w];
			}
		}

		// update interaction
		if (closestIndex != -1) {
			i.d = d[closestIndex];
			i.p[0] = pb[0][closestIndex];
			i.p[1] = pb[1][closestIndex];
			i.p[2] = pb[2][closestIndex];
			i.uv[0] = -1;
			i.uv[1] = -1;
			i.primitiveIndex = primitiveIndex[closestIndex];
			i.nodeIndex = nodeIndex;
			i.referenceIndex = silhouetteReferenceOffset + startReference + closestIndex;
			i.objectIndex = aggregateIndex;
			found = true;
		}

		startReference += WIDTH;
	}

	return found;
}

template<size_t WIDTH>
inline bool findClosestSilhouettes(const MbvhNode<3, true>& node,
								   const std::vector<MbvhSilhouetteLeafNode<WIDTH, 3, SilhouetteEdge>>& silhouetteLeafNodes,
								   int nodeIndex, int aggregateIndex, const enokiVector3& sc, float& sr2,
								   Interaction<3>& i, bool flipNormalOrientation, float squaredMinRadius, float precision)
{
	if (squaredMinRadius >= sr2) return false;

	int silhouetteLeafOffset = -node.silhouetteChild[0] - 1;
	int nSilhouetteLeafs = node.silhouetteChild[1];
	int silhouetteReferenceOffset = node.silhouetteChild[2];
	int nSilhouetteReferences = node.silhouetteChild[3];
	int startReference = 0;
	bool found = false;

	for (int l = 0; l < nSilhouetteLeafs; l++) {
		// perform vectorized closest silhouette query
		Vector3P<WIDTH> pt;
		FloatP<WIDTH> t;
		int leafIndex = silhouetteLeafOffset + l;
		const Vector3P<WIDTH>& pb = silhouetteLeafNodes[leafIndex].positions[1];
		const Vector3P<WIDTH>& pc = silhouetteLeafNodes[leafIndex].positions[2];
		const Vector3P<WIDTH>& n0 = silhouetteLeafNodes[leafIndex].positions[0];
		const Vector3P<WIDTH>& n1 = silhouetteLeafNodes[leafIndex].positions[3];
		const IntP<WIDTH>& primitiveIndex = silhouetteLeafNodes[leafIndex].primitiveIndex;
		const MaskP<WIDTH>& missingFace = silhouetteLeafNodes[leafIndex].missingFace;
		FloatP<WIDTH> d = findClosestPointWideLineSegment<WIDTH>(pb, pc, sc, pt, t);
		FloatP<WIDTH> d2 = d*d;
		if (enoki::all(d2 > sr2)) continue;
		Vector3P<WIDTH> viewDir = sc - pt;
		MaskP<WIDTH> isSilhouette = missingFace;
		enoki::masked(isSilhouette, ~missingFace) = isWideSilhouetteEdge(pb, pc, n0, n1, viewDir, d, flipNormalOrientation, precision);

		// determine closest index
		int closestIndex = -1;
		int W = std::min((int)WIDTH, nSilhouetteReferences - startReference);

		for (int w = 0; w < W; w++) {
			if (isSilhouette[w] && d2[w] <= sr2) {
				closestIndex = w;
				sr2 = d2[w];
			}
		}

		// update interaction
		if (closestIndex != -1) {
			i.d = d[closestIndex];
			i.p[0] = pt[0][closestIndex];
			i.p[1] = pt[1][closestIndex];
			i.p[2] = pt[2][closestIndex];
			i.uv[0] = t[closestIndex];
			i.uv[1] = -1;
			i.primitiveIndex = primitiveIndex[closestIndex];
			i.nodeIndex = nodeIndex;
			i.referenceIndex = silhouetteReferenceOffset + startReference + closestIndex;
			i.objectIndex = aggregateIndex;
			found = true;
		}

		startReference += WIDTH;
	}

	return found;
}

template<size_t WIDTH, size_t DIM, typename PrimitiveType, typename SilhouetteType>
inline void processSubtreeForClosestSilhouettePoint(const std::vector<MbvhNode<DIM, false>>& flatTree,
													const std::vector<PrimitiveType *>& primitives,
													const std::vector<SilhouetteType *>& silhouetteRefs,
													const std::vector<MbvhSilhouetteLeafNode<WIDTH, DIM, SilhouetteType>>& silhouetteLeafNodes,
													const enokiVector<DIM>& sc,
													BoundingSphere<DIM>& s, Interaction<DIM>& i,
													int nodeStartIndex, int aggregateIndex, int objectIndex,
													bool primitiveTypeIsAggregate, bool flipNormalOrientation,
													float squaredMinRadius, float precision, bool recordNormal,
													BvhTraversal *subtree, FloatP<FCPW_MBVH_BRANCHING_FACTOR>& d2Min,
													bool& notFound, int& nodesVisited)
{
	std::cerr << "Mbvh::processSubtreeForClosestSilhouettePoint() not supported without cone data" << std::endl;
	exit(EXIT_FAILURE);
}

template<size_t WIDTH, size_t DIM, typename PrimitiveType, typename SilhouetteType>
inline void processSubtreeForClosestSilhouettePoint(const std::vector<MbvhNode<DIM, true>>& flatTree,
													const std::vector<PrimitiveType *>& primitives,
													const std::vector<SilhouetteType *>& silhouetteRefs,
													const std::vector<MbvhSilhouetteLeafNode<WIDTH, DIM, SilhouetteType>>& silhouetteLeafNodes,
													const enokiVector<DIM>& sc,
													BoundingSphere<DIM>& s, Interaction<DIM>& i,
													int nodeStartIndex, int aggregateIndex, int objectIndex,
													bool primitiveTypeIsAggregate, bool flipNormalOrientation,
													float squaredMinRadius, float precision, bool recordNormal,
													BvhTraversal *subtree, FloatP<FCPW_MBVH_BRANCHING_FACTOR>& d2Min,
													bool& notFound, int& nodesVisited)
{
	int stackPtr = 0;
	while (stackPtr >= 0) {
		// pop off the next node to work on
		int nodeIndex = subtree[stackPtr].node;
		float currentDist = subtree[stackPtr].distance;
		stackPtr--;

		// if this node is further than the closest found primitive, continue
		if (currentDist > s.r2) continue;
		const MbvhNode<DIM, true>& node(flatTree[nodeIndex]);

		if (node.child[0] < 0) { // is leaf
			if (primitiveTypeIsAggregate) {
				int referenceOffset = node.child[2];
				int nReferences = node.child[3];

				for (int p = 0; p < nReferences; p++) {
					int referenceIndex = referenceOffset + p;
					const PrimitiveType *prim = primitives[referenceIndex];
					nodesVisited++;

					Interaction<DIM> c;
					const Aggregate<DIM> *aggregate = reinterpret_cast<const Aggregate<DIM> *>(prim);
					bool found = aggregate->findClosestSilhouettePointFromNode(s, c, nodeStartIndex, aggregateIndex,
																			   nodesVisited, flipNormalOrientation,
																			   squaredMinRadius, precision, recordNormal);

					// keep the closest silhouette point
					if (found) {
						notFound = false;
						s.r2 = std::min(s.r2, c.d*c.d);
						i = c;

						if (squaredMinRadius >= s.r2) {
							break;
						}
					}
				}

			} else {
				if (std::is_same<SilhouetteType, SilhouetteVertex>::value ||
					std::is_same<SilhouetteType, SilhouetteEdge>::value) {
					// perform vectorized closest silhouette query
					nodesVisited++;
					bool found = findClosestSilhouettes(node, silhouetteLeafNodes, nodeIndex, objectIndex, sc, s.r2,
														i, flipNormalOrientation, squaredMinRadius, precision);
					if (found) {
						notFound = false;
						if (squaredMinRadius >= s.r2) break;
					}

				} else {
					// silhouette type does not support vectorized closest silhouette
					// query, perform query to each silhouette one by one
					int silhouetteReferenceOffset = node.silhouetteChild[2];
					int nSilhouetteReferences = node.silhouetteChild[3];

					for (int p = 0; p < nSilhouetteReferences; p++) {
						int referenceIndex = silhouetteReferenceOffset + p;
						const SilhouetteType *silhouette = silhouetteRefs[referenceIndex];

						// skip query if silhouette index is the same as i.primitiveIndex (and object indices match)
						int primitiveIndex = static_cast<const SilhouettePrimitive<DIM> *>(silhouette)->pIndex;
						if (primitiveIndex == i.primitiveIndex && objectIndex == i.objectIndex) continue;
						nodesVisited++;

						Interaction<DIM> c;
						bool found = silhouette->findClosestSilhouettePoint(s, c, flipNormalOrientation, squaredMinRadius,
																			precision, recordNormal);

						// keep the closest silhouette point
						if (found) {
							notFound = false;
							s.r2 = std::min(s.r2, c.d*c.d);
							i = c;
							i.nodeIndex = nodeIndex;
							i.referenceIndex = referenceIndex;
							i.objectIndex = objectIndex;

							if (squaredMinRadius >= s.r2) {
								break;
							}
						}
					}
				}
			}

		} else { // not a leaf
			// overlap sphere with boxes, and normal and view cones
			MaskP<FCPW_MBVH_BRANCHING_FACTOR> mask = enoki::neq(node.child, maxInt) && node.coneHalfAngle >= 0.0f;
			mask &= overlapWideBox<FCPW_MBVH_BRANCHING_FACTOR, DIM>(node.boxMin, node.boxMax, sc, s.r2, d2Min);
			overlapWideCone<FCPW_MBVH_BRANCHING_FACTOR, DIM>(node.coneAxis, node.coneHalfAngle, sc, node.boxMin, node.boxMax, d2Min, mask);

			// enqueue overlapping boxes in sorted order
			nodesVisited++;
			if (enoki::any(mask)) {
				float stub = 0.0f;
				enqueueNodes(node, d2Min, 0.0f, mask, s.r2, stub, stackPtr, subtree);
			}
		}
	}
}

template<size_t WIDTH, size_t DIM, bool CONEDATA, typename PrimitiveType, typename SilhouetteType>
inline bool Mbvh<WIDTH, DIM, CONEDATA, PrimitiveType, SilhouetteType>::findClosestSilhouettePointFromNode(BoundingSphere<DIM>& s, Interaction<DIM>& i,
																										  int nodeStartIndex, int aggregateIndex,
																										  int& nodesVisited, bool flipNormalOrientation,
																										  float squaredMinRadius, float precision,
																										  bool recordNormal) const
{
	if (squaredMinRadius >= s.r2) return false;

	bool notFound = true;
	FloatP<FCPW_MBVH_BRANCHING_FACTOR> d2Min;
	enokiVector<DIM> sc = enoki::gather<enokiVector<DIM>>(s.c.data(), range);
	BvhTraversal subtree[FCPW_MBVH_MAX_DEPTH];
	int rootIndex = aggregateIndex == this->index ? nodeStartIndex : 0;
	subtree[0].node = rootIndex;
	subtree[0].distance = minFloat;

	processSubtreeForClosestSilhouettePoint<WIDTH, DIM, PrimitiveType, SilhouetteType>(flatTree, primitives, silhouetteRefs, silhouetteLeafNodes,
																					   sc, s, i, nodeStartIndex, aggregateIndex, this->index,
																					   primitiveTypeIsAggregate, flipNormalOrientation,
																					   squaredMinRadius, precision, recordNormal,
																					   subtree, d2Min, notFound, nodesVisited);

	if (!notFound) {
		// compute normal
		if (recordNormal && !primitiveTypeIsAggregate) {
			i.computeSilhouetteNormal(silhouetteRefs[i.referenceIndex]);
		}

		return true;
	}

	return false;
}

} // namespace fcpw
