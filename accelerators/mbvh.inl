#include "core/wide_query_operations.h"

namespace fcpw {

template<size_t WIDTH, size_t DIM, typename PrimitiveType>
inline int Mbvh<WIDTH, DIM, PrimitiveType>::collapseSbvh(const Sbvh<DIM, PrimitiveType> *sbvh,
														 int sbvhNodeIndex, int parent, int depth)
{
	const SbvhNode<DIM>& sbvhNode = sbvh->flatTree[sbvhNodeIndex];
	maxDepth = std::max(depth, maxDepth);

	// create mbvh node
	MbvhNode<DIM> mbvhNode;
	int mbvhNodeIndex = nNodes;

	nNodes++;
	flatTree.emplace_back(mbvhNode);

	if (sbvhNode.nReferences > 0) {
		// sbvh node is a leaf node; assign mbvh node its reference indices
		MbvhNode<DIM>& mbvhNode = flatTree[mbvhNodeIndex];
		mbvhNode.child[0] = -(nLeafs + 1); // negative value indicates that node is a leaf
		mbvhNode.child[1] = sbvhNode.nReferences/WIDTH;
		if (sbvhNode.nReferences%WIDTH != 0) mbvhNode.child[1] += 1;
		mbvhNode.child[2] = sbvhNode.referenceOffset;
		mbvhNode.child[3] = sbvhNode.nReferences;
		nLeafs += mbvhNode.child[1];

	} else {
		// sbvh node is an inner node, flatten it
		int stackPtr = 0;
		int nNodesCollapsed = 0;
		int stackSbvhNodes[MBVH_BRANCHING_FACTOR][2];
		stackSbvhNodes[stackPtr][0] = sbvhNodeIndex;
		stackSbvhNodes[stackPtr][1] = 0;

		while (stackPtr >= 0) {
			int sbvhNodeIndex = stackSbvhNodes[stackPtr][0];
			int level = stackSbvhNodes[stackPtr][1];
			stackPtr--;

			const SbvhNode<DIM>& sbvhNode = sbvh->flatTree[sbvhNodeIndex];
			if (level < maxLevel && sbvhNode.nReferences == 0) {
				// enqueue sbvh children nodes till max level or leaf node is reached
				stackPtr++;
				stackSbvhNodes[stackPtr][0] = sbvhNodeIndex + sbvhNode.secondChildOffset;
				stackSbvhNodes[stackPtr][1] = level + 1;

				stackPtr++;
				stackSbvhNodes[stackPtr][0] = sbvhNodeIndex + 1;
				stackSbvhNodes[stackPtr][1] = level + 1;

			} else {
				// assign mbvh node this sbvh node's bounding box and index
				for (int i = 0; i < DIM; i++) {
					flatTree[mbvhNodeIndex].boxMin[i][nNodesCollapsed] = sbvhNode.box.pMin[i];
					flatTree[mbvhNodeIndex].boxMax[i][nNodesCollapsed] = sbvhNode.box.pMax[i];
				}

				flatTree[mbvhNodeIndex].child[nNodesCollapsed] = collapseSbvh(sbvh,
											sbvhNodeIndex, mbvhNodeIndex, depth + 1);
				nNodesCollapsed++;
			}
		}
	}

	return mbvhNodeIndex;
}

template<size_t WIDTH, size_t DIM, typename PrimitiveType>
inline bool Mbvh<WIDTH, DIM, PrimitiveType>::isLeafNode(const MbvhNode<DIM>& node) const
{
	return node.child[0] < 0;
}

template<size_t WIDTH, size_t DIM, typename PrimitiveType>
inline void populateLeafNode(const MbvhNode<DIM>& node, const std::vector<PrimitiveType *>& primitives,
							 std::vector<MbvhLeafNode<WIDTH, DIM, PrimitiveType>>& leafNodes)
{
	std::cerr << "populateLeafNode(): WIDTH: " << WIDTH << ", DIM: " << DIM << " not supported" << std::endl;
	exit(EXIT_FAILURE);
}

template<size_t WIDTH>
inline void populateLeafNode(const MbvhNode<3>& node, const std::vector<LineSegment *>& primitives,
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

template<size_t WIDTH>
inline void populateLeafNode(const MbvhNode<3>& node, const std::vector<Triangle *>& primitives,
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

template<size_t WIDTH, size_t DIM, typename PrimitiveType>
inline void Mbvh<WIDTH, DIM, PrimitiveType>::populateLeafNodes()
{
	if (vectorizedLeafType == ObjectType::LineSegments ||
		vectorizedLeafType == ObjectType::Triangles) {
		leafNodes.resize(nLeafs);

		for (int i = 0; i < nNodes; i++) {
			MbvhNode<DIM>& node = flatTree[i];
			if (isLeafNode(node)) populateLeafNode(node, primitives, leafNodes);
		}
	}
}

template<size_t WIDTH, size_t DIM, typename PrimitiveType>
inline Mbvh<WIDTH, DIM, PrimitiveType>::Mbvh(const Sbvh<DIM, PrimitiveType> *sbvh_, bool printStats_):
primitives(sbvh_->primitives),
nNodes(0),
nLeafs(0),
maxDepth(0),
maxLevel(std::log2(MBVH_BRANCHING_FACTOR)),
primitiveTypeIsAggregate(std::is_base_of<Aggregate<DIM>, PrimitiveType>::value)
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	static_assert(MBVH_BRANCHING_FACTOR == 4 || MBVH_BRANCHING_FACTOR == 8,
				  "Branching factor must be atleast 4");

	using namespace std::chrono;
	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	// collapse sbvh
	collapseSbvh(sbvh_, 0, 0xfffffffc, 0);

	// determine object type
	vectorizedLeafType = std::is_same<PrimitiveType, LineSegment>::value ? ObjectType::LineSegments :
						 std::is_same<PrimitiveType, Triangle>::value ? ObjectType::Triangles :
						 ObjectType::Mixed;

	// populate leaf nodes if primitive type is supported
	populateLeafNodes();

	// don't compute normals by default
	this->computeNormals = false;

	// print stats
	if (printStats_) {
		// count not-full leaves
		float nLeafsNotFull = 0;
		for (int i = 0; i < nNodes; i++) {
			MbvhNode<DIM>& node = flatTree[i];
			if (isLeafNode(node) && node.child[3]%WIDTH != 0) nLeafsNotFull++;
		}

		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		duration<double> timeSpan = duration_cast<duration<double>>(t2 - t1);
		std::cout << "Built " << MBVH_BRANCHING_FACTOR << "-bvh with "
				  << nNodes << " nodes, "
				  << (nLeafsNotFull*100/nLeafs) << "% leaves not full, "
				  << nLeafs << " leaves, "
				  << maxDepth << " max depth, "
				  << primitives.size() << " primitives in "
				  << timeSpan.count() << " seconds" << std::endl;
	}
}

template<size_t WIDTH, size_t DIM, typename PrimitiveType>
inline BoundingBox<DIM> Mbvh<WIDTH, DIM, PrimitiveType>::boundingBox() const
{
	BoundingBox<DIM> box;
	if (flatTree.size() == 0) return box;

	box.pMin = enoki::hmin_inner(flatTree[0].boxMin);
	box.pMax = enoki::hmax_inner(flatTree[0].boxMax);
	return box;
}

template<size_t WIDTH, size_t DIM, typename PrimitiveType>
inline Vector<DIM> Mbvh<WIDTH, DIM, PrimitiveType>::centroid() const
{
	Vector<DIM> c = zeroVector<DIM>();
	int nPrimitives = (int)primitives.size();

	for (int p = 0; p < nPrimitives; p++) {
		c += primitives[p]->centroid();
	}

	return c/nPrimitives;
}

template<size_t WIDTH, size_t DIM, typename PrimitiveType>
inline float Mbvh<WIDTH, DIM, PrimitiveType>::surfaceArea() const
{
	float area = 0.0f;
	for (int p = 0; p < (int)primitives.size(); p++) {
		area += primitives[p]->surfaceArea();
	}

	return area;
}

template<size_t WIDTH, size_t DIM, typename PrimitiveType>
inline float Mbvh<WIDTH, DIM, PrimitiveType>::signedVolume() const
{
	float volume = 0.0f;
	for (int p = 0; p < (int)primitives.size(); p++) {
		volume += primitives[p]->signedVolume();
	}

	return volume;
}

template<size_t WIDTH, size_t DIM, typename PrimitiveType>
inline int intersectPrimitives(const MbvhNode<DIM>& node,
							   const std::vector<MbvhLeafNode<WIDTH, DIM, PrimitiveType>>& leafNodes,
							   int nodeIndex, Ray<DIM>& r, std::vector<Interaction<DIM>>& is, bool countHits)
{
	std::cerr << "intersectPrimitives(): WIDTH: " << WIDTH << ", DIM: " << DIM << " not supported" << std::endl;
	exit(EXIT_FAILURE);

	return 0;
}

template<size_t WIDTH>
inline int intersectPrimitives(const MbvhNode<3>& node,
							   const std::vector<MbvhLeafNode<WIDTH, 3, LineSegment>>& leafNodes,
							   int nodeIndex, Ray<3>& r, std::vector<Interaction<3>>& is, bool countHits)
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	int leafOffset = -node.child[0] - 1;
	int nLeafs = node.child[1];
	int referenceOffset = node.child[2];
	int nReferences = node.child[3];
	int startReference = 0;
	int hits = 0;

	for (int l = 0; l < nLeafs; l++) {
		// perform vectorized intersection query
		FloatP<WIDTH> d;
		Vector3P<WIDTH> pt;
		FloatP<WIDTH> t;
		int leafIndex = leafOffset + l;
		const Vector3P<WIDTH>& pa = leafNodes[leafIndex].positions[0];
		const Vector3P<WIDTH>& pb = leafNodes[leafIndex].positions[1];
		const IntP<WIDTH>& primitiveIndex = leafNodes[leafIndex].primitiveIndex;
		MaskP<WIDTH> mask = intersectWideLineSegment<WIDTH>(r, pa, pb, d, pt, t);

		// record interactions
		int endReference = startReference + WIDTH;
		if (endReference > nReferences) endReference = nReferences;

		for (int p = startReference; p < endReference; p++) {
			int w = p - startReference;

			if (countHits) {
				if (mask[w]) {
					hits++;
					auto it = is.emplace(is.end(), Interaction<3>());
					it->d = d[w];
					it->p[0] = pt[0][w];
					it->p[1] = pt[1][w];
					it->p[2] = pt[2][w];
					it->uv[0] = t[w];
					it->uv[1] = -1;
					it->primitiveIndex = primitiveIndex[w];
					it->nodeIndex = nodeIndex;
					it->referenceIndex = referenceOffset + p;
				}

			} else {
				if (mask[w] && d[w] <= r.tMax) {
					hits = 1;
					r.tMax = d[w];
					is[0].d = d[w];
					is[0].p[0] = pt[0][w];
					is[0].p[1] = pt[1][w];
					is[0].p[2] = pt[2][w];
					is[0].uv[0] = t[w];
					is[0].uv[1] = -1;
					is[0].primitiveIndex = primitiveIndex[w];
					is[0].nodeIndex = nodeIndex;
					is[0].referenceIndex = referenceOffset + p;
				}
			}
		}

		startReference += WIDTH;
	}

	return hits;
}

template<size_t WIDTH>
inline int intersectPrimitives(const MbvhNode<3>& node,
							   const std::vector<MbvhLeafNode<WIDTH, 3, Triangle>>& leafNodes,
							   int nodeIndex, Ray<3>& r, std::vector<Interaction<3>>& is, bool countHits)
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	int leafOffset = -node.child[0] - 1;
	int nLeafs = node.child[1];
	int referenceOffset = node.child[2];
	int nReferences = node.child[3];
	int startReference = 0;
	int hits = 0;

	for (int l = 0; l < nLeafs; l++) {
		// perform vectorized intersection query
		FloatP<WIDTH> d;
		Vector3P<WIDTH> pt;
		Vector2P<WIDTH> t;
		int leafIndex = leafOffset + l;
		const Vector3P<WIDTH>& pa = leafNodes[leafIndex].positions[0];
		const Vector3P<WIDTH>& pb = leafNodes[leafIndex].positions[1];
		const Vector3P<WIDTH>& pc = leafNodes[leafIndex].positions[2];
		const IntP<WIDTH>& primitiveIndex = leafNodes[leafIndex].primitiveIndex;
		MaskP<WIDTH> mask = intersectWideTriangle<WIDTH>(r, pa, pb, pc, d, pt, t);

		// record interactions
		int endReference = startReference + WIDTH;
		if (endReference > nReferences) endReference = nReferences;

		for (int p = startReference; p < endReference; p++) {
			int w = p - startReference;

			if (countHits) {
				if (mask[w]) {
					hits++;
					auto it = is.emplace(is.end(), Interaction<3>());
					it->d = d[w];
					it->p[0] = pt[0][w];
					it->p[1] = pt[1][w];
					it->p[2] = pt[2][w];
					it->uv[0] = t[0][w];
					it->uv[1] = t[1][w];
					it->primitiveIndex = primitiveIndex[w];
					it->nodeIndex = nodeIndex;
					it->referenceIndex = referenceOffset + p;
				}

			} else {
				if (mask[w] && d[w] <= r.tMax) {
					hits = 1;
					r.tMax = d[w];
					is[0].d = d[w];
					is[0].p[0] = pt[0][w];
					is[0].p[1] = pt[1][w];
					is[0].p[2] = pt[2][w];
					is[0].uv[0] = t[0][w];
					is[0].uv[1] = t[1][w];
					is[0].primitiveIndex = primitiveIndex[w];
					is[0].nodeIndex = nodeIndex;
					is[0].referenceIndex = referenceOffset + p;
				}
			}
		}

		startReference += WIDTH;
	}

	return hits;
}

template<size_t WIDTH, size_t DIM, typename PrimitiveType>
inline int Mbvh<WIDTH, DIM, PrimitiveType>::intersectFromNode(Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
															  int nodeStartIndex, int& nodesVisited,
															  bool checkOcclusion, bool countHits) const
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	// TODO: start from nodeStartIndex
	int hits = 0;
	if (!countHits) is.resize(1);
	BvhTraversal subtree[MBVH_MAX_DEPTH];
	FloatP<MBVH_BRANCHING_FACTOR> tMin, tMax;

	// push root node
	subtree[0].node = 0;
	subtree[0].distance = minFloat;
	int stackPtr = 0;

	while (stackPtr >= 0) {
		// pop off the next node to work on
		int nodeIndex = subtree[stackPtr].node;
		float near = subtree[stackPtr].distance;
		stackPtr--;

		// if this node is further than the closest found intersection, continue
		if (!countHits && near > r.tMax) continue;
		const MbvhNode<DIM>& node(flatTree[nodeIndex]);

		if (isLeafNode(node)) {
			if (vectorizedLeafType == ObjectType::LineSegments ||
				vectorizedLeafType == ObjectType::Triangles) {
				// perform vectorized intersection query
				hits += intersectPrimitives(node, leafNodes, nodeIndex, r, is, countHits);
				nodesVisited++;
				if (hits > 0 && checkOcclusion) return 1;

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
						hit = aggregate->intersectFromNode(r, cs, nodeStartIndex, nodesVisited, checkOcclusion, countHits);

					} else {
						hit = prim->intersect(r, cs, checkOcclusion, countHits);
						for (int i = 0; i < (int)cs.size(); i++) {
							cs[i].nodeIndex = nodeIndex;
							cs[i].referenceIndex = referenceIndex;
						}
					}

					// keep the closest intersection only
					if (hit > 0) {
						hits += hit;
						if (countHits) {
							is.insert(is.end(), cs.begin(), cs.end());

						} else {
							r.tMax = std::min(r.tMax, cs[0].d);
							is[0] = cs[0];
						}

						if (checkOcclusion) return 1;
					}
				}
			}

		} else {
			// intersect ray with boxes
			MaskP<MBVH_BRANCHING_FACTOR> mask = intersectWideBox<MBVH_BRANCHING_FACTOR, DIM>(r,
														  node.boxMin, node.boxMax, tMin, tMax);

			// find closest intersecting node
			int closestIndex = -1;
			float minHit = r.tMax;

			for (int w = 0; w < MBVH_BRANCHING_FACTOR; w++) {
				if (mask[w] && tMin[w] < minHit && node.child[w] != maxInt) {
					closestIndex = w;
					minHit = tMin[w];
				}
			}

			// enqueue remaining intersecting nodes first
			for (int w = 0; w < MBVH_BRANCHING_FACTOR; w++) {
				if (mask[w] && w != closestIndex && node.child[w] != maxInt) {
					stackPtr++;
					subtree[stackPtr].node = node.child[w];
					subtree[stackPtr].distance = tMin[w];
				}
			}

			// enqueue closest intersecting node
			if (closestIndex != -1) {
				stackPtr++;
				subtree[stackPtr].node = node.child[closestIndex];
				subtree[stackPtr].distance = minHit;
			}

			nodesVisited++;
		}
	}

	if (hits > 0) {
		// sort by distance and remove duplicates
		if (countHits) {
			std::sort(is.begin(), is.end(), compareInteractions<DIM>);
			is = removeDuplicates<DIM>(is);
			hits = (int)is.size();
		}

		// compute normals
		if (this->computeNormals && !primitiveTypeIsAggregate) {
			for (int i = 0; i < (int)is.size(); i++) {
				is[i].computeNormal(primitives[is[i].referenceIndex]);
			}
		}

		return hits;
	}

	return 0;
}

template<size_t WIDTH, size_t DIM, typename PrimitiveType>
inline bool findClosestPointPrimitives(const MbvhNode<DIM>& node,
									   const std::vector<MbvhLeafNode<WIDTH, DIM, PrimitiveType>>& leafNodes,
									   int nodeIndex, BoundingSphere<DIM>& s, Interaction<DIM>& i)
{
	std::cerr << "findClosestPointPrimitives(): WIDTH: " << WIDTH << ", DIM: " << DIM << " not supported" << std::endl;
	exit(EXIT_FAILURE);

	return false;
}

template<size_t WIDTH>
inline bool findClosestPointPrimitives(const MbvhNode<3>& node,
									   const std::vector<MbvhLeafNode<WIDTH, 3, LineSegment>>& leafNodes,
									   int nodeIndex, BoundingSphere<3>& s, Interaction<3>& i)
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	int leafOffset = -node.child[0] - 1;
	int nLeafs = node.child[1];
	int referenceOffset = node.child[2];
	int nReferences = node.child[3];
	int closestIndex = -1;
	int startReference = 0;

	for (int l = 0; l < nLeafs; l++) {
		// perform vectorized closest point query
		Vector3P<WIDTH> pt;
		FloatP<WIDTH> t;
		int leafIndex = leafOffset + l;
		const Vector3P<WIDTH>& pa = leafNodes[leafIndex].positions[0];
		const Vector3P<WIDTH>& pb = leafNodes[leafIndex].positions[1];
		const IntP<WIDTH>& primitiveIndex = leafNodes[leafIndex].primitiveIndex;
		FloatP<WIDTH> d = findClosestPointWideLineSegment<WIDTH>(s.c, pa, pb, pt, t);
		FloatP<WIDTH> d2 = d*d;

		// determine closest primitive
		int endReference = startReference + WIDTH;
		if (endReference > nReferences) endReference = nReferences;

		for (int p = startReference; p < endReference; p++) {
			int w = p - startReference;

			if (d2[w] <= s.r2) {
				s.r2 = d2[w];
				i.d = d[w];
				i.p[0] = pt[0][w];
				i.p[1] = pt[1][w];
				i.p[2] = pt[2][w];
				i.uv[0] = t[w];
				i.uv[1] = -1;
				i.primitiveIndex = primitiveIndex[w];
				i.nodeIndex = nodeIndex;
				i.referenceIndex = referenceOffset + p;
				closestIndex = i.referenceIndex;
			}
		}

		startReference += WIDTH;
	}

	return closestIndex != -1;
}

template<size_t WIDTH>
inline bool findClosestPointPrimitives(const MbvhNode<3>& node,
									   const std::vector<MbvhLeafNode<WIDTH, 3, Triangle>>& leafNodes,
									   int nodeIndex, BoundingSphere<3>& s, Interaction<3>& i)
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	int leafOffset = -node.child[0] - 1;
	int nLeafs = node.child[1];
	int referenceOffset = node.child[2];
	int nReferences = node.child[3];
	int closestIndex = -1;
	int startReference = 0;

	for (int l = 0; l < nLeafs; l++) {
		// perform vectorized closest point query
		Vector3P<WIDTH> pt;
		Vector2P<WIDTH> t;
		int leafIndex = leafOffset + l;
		const Vector3P<WIDTH>& pa = leafNodes[leafIndex].positions[0];
		const Vector3P<WIDTH>& pb = leafNodes[leafIndex].positions[1];
		const Vector3P<WIDTH>& pc = leafNodes[leafIndex].positions[2];
		const IntP<WIDTH>& primitiveIndex = leafNodes[leafIndex].primitiveIndex;
		FloatP<WIDTH> d = findClosestPointWideTriangle<WIDTH>(s.c, pa, pb, pc, pt, t);
		FloatP<WIDTH> d2 = d*d;

		// determine closest primitive
		int endReference = startReference + WIDTH;
		if (endReference > nReferences) endReference = nReferences;

		for (int p = startReference; p < endReference; p++) {
			int w = p - startReference;

			if (d2[w] <= s.r2) {
				s.r2 = d2[w];
				i.d = d[w];
				i.p[0] = pt[0][w];
				i.p[1] = pt[1][w];
				i.p[2] = pt[2][w];
				i.uv[0] = t[0][w];
				i.uv[1] = t[1][w];
				i.primitiveIndex = primitiveIndex[w];
				i.nodeIndex = nodeIndex;
				i.referenceIndex = referenceOffset + p;
				closestIndex = i.referenceIndex;
			}
		}

		startReference += WIDTH;
	}

	return closestIndex != -1;
}

template<size_t WIDTH, size_t DIM, typename PrimitiveType>
inline bool Mbvh<WIDTH, DIM, PrimitiveType>::findClosestPointFromNode(BoundingSphere<DIM>& s, Interaction<DIM>& i,
																	  int nodeStartIndex, const Vector<DIM>& boundaryHint,
																	  int& nodesVisited) const
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	// TODO: start from nodeStartIndex & use direction to boundary guess
	bool notFound = true;
	BvhTraversal subtree[MBVH_MAX_DEPTH];
	FloatP<MBVH_BRANCHING_FACTOR> d2Min, d2Max;

	// push root node
	subtree[0].node = 0;
	subtree[0].distance = minFloat;
	int stackPtr = 0;

	while (stackPtr >= 0) {
		// pop off the next node to work on
		int nodeIndex = subtree[stackPtr].node;
		float near = subtree[stackPtr].distance;
		stackPtr--;

		// if this node is further than the closest found primitive, continue
		if (near > s.r2) continue;
		const MbvhNode<DIM>& node(flatTree[nodeIndex]);

		if (isLeafNode(node)) {
			if (vectorizedLeafType == ObjectType::LineSegments ||
				vectorizedLeafType == ObjectType::Triangles) {
				// perform vectorized closest point query to triangle
				bool found = findClosestPointPrimitives(node, leafNodes, nodeIndex, s, i);
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
						found = aggregate->findClosestPointFromNode(s, c, nodeStartIndex, boundaryHint, nodesVisited);

					} else {
						found = prim->findClosestPoint(s, c);
						c.nodeIndex = nodeIndex;
						c.referenceIndex = referenceIndex;
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
			MaskP<MBVH_BRANCHING_FACTOR> mask = overlapWideBox<MBVH_BRANCHING_FACTOR, DIM>(s,
													  node.boxMin, node.boxMax, d2Min, d2Max);

			// find closest overlapping node
			int closestIndex = -1;
			float minDist = s.r2;

			for (int w = 0; w < MBVH_BRANCHING_FACTOR; w++) {
				if (mask[w] && d2Min[w] < minDist && node.child[w] != maxInt) {
					closestIndex = w;
					minDist = d2Min[w];
				}
			}

			// enqueue remaining overlapping nodes first
			for (int w = 0; w < MBVH_BRANCHING_FACTOR; w++) {
				if (mask[w] && w != closestIndex && node.child[w] != maxInt) {
					s.r2 = std::min(s.r2, d2Max[w]);
					stackPtr++;
					subtree[stackPtr].node = node.child[w];
					subtree[stackPtr].distance = d2Min[w];
				}
			}

			// enqueue closest intersecting node
			if (closestIndex != -1) {
				s.r2 = std::min(s.r2, d2Max[closestIndex]);
				stackPtr++;
				subtree[stackPtr].node = node.child[closestIndex];
				subtree[stackPtr].distance = minDist;
			}

			nodesVisited++;
		}
	}

	if (!notFound) {
		// compute normal
		if (this->computeNormals && !primitiveTypeIsAggregate) {
			i.computeNormal(primitives[i.referenceIndex]);
		}

		return true;
	}

	return false;
}

} // namespace fcpw
