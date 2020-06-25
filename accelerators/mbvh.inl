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
				stackSbvhNodes[stackPtr][0] = sbvhNodeIndex + 1;
				stackSbvhNodes[stackPtr][1] = level + 1;

				stackPtr++;
				stackSbvhNodes[stackPtr][0] = sbvhNodeIndex + sbvhNode.secondChildOffset;
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
inline void Mbvh<WIDTH, DIM, PrimitiveType>::populateLeafNode(const MbvhNode<DIM>& node)
{
	int leafOffset = -node.child[0] - 1;
	int referenceOffset = node.child[2];
	int nReferences = node.child[3];

	if (vectorizedLeafType == ObjectType::LineSegments) {
		// populate leaf node with line segments
		for (int p = 0; p < nReferences; p++) {
			int index = references[referenceOffset + p];
			int leafIndex = 2*(leafOffset + p/WIDTH);
			int w = p%WIDTH;

			const LineSegment *lineSegment = reinterpret_cast<const LineSegment *>(primitives[index]);
			int paIndex = lineSegment->soup->indices[lineSegment->index + 0];
			int pbIndex = lineSegment->soup->indices[lineSegment->index + 1];
			const Vector3& pa = lineSegment->soup->positions[paIndex];
			const Vector3& pb = lineSegment->soup->positions[pbIndex];

			for (int i = 0; i < DIM; i++) {
				leafNodes[leafIndex + 0][i][w] = pa[i];
				leafNodes[leafIndex + 1][i][w] = pb[i];
			}
		}

	} else if (vectorizedLeafType == ObjectType::Triangles) {
		// populate leaf node with triangles
		for (int p = 0; p < nReferences; p++) {
			int index = references[referenceOffset + p];
			int leafIndex = 3*(leafOffset + p/WIDTH);
			int w = p%WIDTH;

			const Triangle *triangle = reinterpret_cast<const Triangle *>(primitives[index]);
			int paIndex = triangle->soup->indices[triangle->index + 0];
			int pbIndex = triangle->soup->indices[triangle->index + 1];
			int pcIndex = triangle->soup->indices[triangle->index + 2];
			const Vector3& pa = triangle->soup->positions[paIndex];
			const Vector3& pb = triangle->soup->positions[pbIndex];
			const Vector3& pc = triangle->soup->positions[pcIndex];

			for (int i = 0; i < DIM; i++) {
				leafNodes[leafIndex + 0][i][w] = pa[i];
				leafNodes[leafIndex + 1][i][w] = pb[i];
				leafNodes[leafIndex + 2][i][w] = pc[i];
			}
		}
	}
}

template<size_t WIDTH, size_t DIM, typename PrimitiveType>
inline void Mbvh<WIDTH, DIM, PrimitiveType>::populateLeafNodes()
{
	if (vectorizedLeafType == ObjectType::LineSegments ||
		vectorizedLeafType == ObjectType::Triangles) {
		int shift = vectorizedLeafType == ObjectType::LineSegments ? 2 : 3;
		leafNodes.resize(nLeafs*shift);

		for (int i = 0; i < nNodes; i++) {
			MbvhNode<DIM>& node = flatTree[i];
			if (isLeafNode(node)) populateLeafNode(node);
		}
	}
}

template<size_t WIDTH, size_t DIM, typename PrimitiveType>
inline Mbvh<WIDTH, DIM, PrimitiveType>::Mbvh(const Sbvh<DIM, PrimitiveType> *sbvh_, bool printStats_):
primitives(sbvh_->primitives),
references(std::move(sbvh_->references)),
nNodes(0),
nLeafs(0),
maxDepth(0),
maxLevel(std::log2(MBVH_BRANCHING_FACTOR)),
primitiveTypeIsAggregate(std::is_base_of<Aggregate<DIM>, PrimitiveType>::value)
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	if (MBVH_BRANCHING_FACTOR < 4) {
		std::cerr << "Branching factor must be atleast 4" << std::endl;
		exit(EXIT_FAILURE);
	}

	using namespace std::chrono;
	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	// collapse sbvh
	collapseSbvh(sbvh_, 0, 0xfffffffc, 0);

	// determine object type
	vectorizedLeafType = std::is_same<PrimitiveType, Triangle>::value ? ObjectType::Triangles :
						 std::is_same<PrimitiveType, LineSegment>::value ? ObjectType::LineSegments :
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
				  << primitives.size() << " primitives, "
				  << references.size() << " references in "
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
inline int Mbvh<WIDTH, DIM, PrimitiveType>::intersectLineSegment(const MbvhNode<DIM>& node, int nodeIndex,
																 Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
																 bool countHits) const
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
		VectorP<WIDTH, DIM> pt;
		FloatP<WIDTH> t;
		int leafIndex = 2*(leafOffset + l);
		const VectorP<WIDTH, DIM>& pa = leafNodes[leafIndex + 0];
		const VectorP<WIDTH, DIM>& pb = leafNodes[leafIndex + 1];
		MaskP<WIDTH> mask = intersectWideLineSegment(r, pa, pb, d, pt, t);

		// record interactions
		int endReference = startReference + WIDTH;
		if (endReference > nReferences) endReference = nReferences;

		for (int p = startReference; p < endReference; p++) {
			int w = p - startReference;

			if (countHits) {
				if (mask[w]) {
					int index = references[referenceOffset + p];
					const PrimitiveType *prim = primitives[index];

					const LineSegment *lineSegment = reinterpret_cast<const LineSegment *>(prim);
					hits++;
					auto it = is.emplace(is.end(), Interaction<DIM>());
					it->d = d[w];
					it->p[0] = pt[0][w];
					it->p[1] = pt[1][w];
					it->p[2] = pt[2][w];
					it->uv[0] = t[w];
					it->uv[1] = -1;
					it->nodeIndex = nodeIndex;
					it->primitive = lineSegment;
				}

			} else {
				if (mask[w] && d[w] <= r.tMax) {
					int index = references[referenceOffset + p];
					const PrimitiveType *prim = primitives[index];

					const LineSegment *lineSegment = reinterpret_cast<const LineSegment *>(prim);
					hits = 1;
					r.tMax = d[w];
					is[0].d = d[w];
					is[0].p[0] = pt[0][w];
					is[0].p[1] = pt[1][w];
					is[0].p[2] = pt[2][w];
					is[0].uv[0] = t[w];
					is[0].uv[1] = -1;
					is[0].nodeIndex = nodeIndex;
					is[0].primitive = lineSegment;
				}
			}
		}

		startReference += WIDTH;
	}

	return hits;
}

template<size_t WIDTH, size_t DIM, typename PrimitiveType>
inline int Mbvh<WIDTH, DIM, PrimitiveType>::intersectTriangle(const MbvhNode<DIM>& node, int nodeIndex,
															  Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
															  bool countHits) const
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
		VectorP<WIDTH, DIM> pt;
		VectorP<WIDTH, DIM - 1> t;
		int leafIndex = 3*(leafOffset + l);
		const VectorP<WIDTH, DIM>& pa = leafNodes[leafIndex + 0];
		const VectorP<WIDTH, DIM>& pb = leafNodes[leafIndex + 1];
		const VectorP<WIDTH, DIM>& pc = leafNodes[leafIndex + 2];
		MaskP<WIDTH> mask = intersectWideTriangle(r, pa, pb, pc, d, pt, t);

		// record interactions
		int endReference = startReference + WIDTH;
		if (endReference > nReferences) endReference = nReferences;

		for (int p = startReference; p < endReference; p++) {
			int w = p - startReference;

			if (countHits) {
				if (mask[w]) {
					int index = references[referenceOffset + p];
					const PrimitiveType *prim = primitives[index];

					const Triangle *triangle = reinterpret_cast<const Triangle *>(prim);
					hits++;
					auto it = is.emplace(is.end(), Interaction<DIM>());
					it->d = d[w];
					it->p[0] = pt[0][w];
					it->p[1] = pt[1][w];
					it->p[2] = pt[2][w];
					it->uv[0] = t[0][w];
					it->uv[1] = t[1][w];
					it->nodeIndex = nodeIndex;
					it->primitive = triangle;
				}

			} else {
				if (mask[w] && d[w] <= r.tMax) {
					int index = references[referenceOffset + p];
					const PrimitiveType *prim = primitives[index];

					const Triangle *triangle = reinterpret_cast<const Triangle *>(prim);
					hits = 1;
					r.tMax = d[w];
					is[0].d = d[w];
					is[0].p[0] = pt[0][w];
					is[0].p[1] = pt[1][w];
					is[0].p[2] = pt[2][w];
					is[0].uv[0] = t[0][w];
					is[0].uv[1] = t[1][w];
					is[0].nodeIndex = nodeIndex;
					is[0].primitive = triangle;
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
				hits += vectorizedLeafType == ObjectType::LineSegments ?
						intersectLineSegment(node, nodeIndex, r, is, countHits) :
						intersectTriangle(node, nodeIndex, r, is, countHits);
				nodesVisited++;
				if (hits > 0 && checkOcclusion) return 1;

			} else {
				// primitive type does not support vectorized intersection query,
				// perform query to each primitive one by one
				int referenceOffset = node.child[2];
				int nReferences = node.child[3];

				for (int p = 0; p < nReferences; p++) {
					int index = references[referenceOffset + p];
					const PrimitiveType *prim = primitives[index];

					// check if primitive has already been seen
					bool seenPrim = false;
					int nInteractions = (int)is.size();
					for (int sp = 0; sp < nInteractions; sp++) {
						if (prim == is[sp].primitive) {
							seenPrim = true;
							break;
						}
					}

					if (!seenPrim) {
						nodesVisited++;

						int hit = 0;
						std::vector<Interaction<DIM>> cs;
						if (primitiveTypeIsAggregate) {
							const Aggregate<DIM> *aggregate = reinterpret_cast<const Aggregate<DIM> *>(prim);
							hit = aggregate->intersectFromNode(r, cs, nodeStartIndex, nodesVisited, checkOcclusion, countHits);

						} else {
							hit = prim->intersect(r, cs, checkOcclusion, countHits);
						}

						// keep the closest intersection only
						if (hit > 0) {
							hits += hit;
							if (countHits) {
								is.insert(is.end(), cs.begin(), cs.end());
								for (int sp = nInteractions; sp < (int)is.size(); sp++) {
									is[sp].nodeIndex = nodeIndex;
								}

							} else {
								r.tMax = std::min(r.tMax, cs[0].d);
								is[0] = cs[0];
								is[0].nodeIndex = nodeIndex;
							}

							if (checkOcclusion) return 1;
						}
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
		if (this->computeNormals) {
			for (int i = 0; i < (int)is.size(); i++) {
				is[i].computeNormal();
			}
		}

		return hits;
	}

	return 0;
}

template<size_t WIDTH, size_t DIM, typename PrimitiveType>
inline bool Mbvh<WIDTH, DIM, PrimitiveType>::findClosestPointLineSegment(const MbvhNode<DIM>& node,
																		 int nodeIndex, BoundingSphere<DIM>& s,
																		 Interaction<DIM>& i) const
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
		VectorP<WIDTH, DIM> pt;
		FloatP<WIDTH> t;
		int leafIndex = 2*(leafOffset + l);
		const VectorP<WIDTH, DIM>& pa = leafNodes[leafIndex + 0];
		const VectorP<WIDTH, DIM>& pb = leafNodes[leafIndex + 1];
		FloatP<WIDTH> d = findClosestPointWideLineSegment(s.c, pa, pb, pt, t);
		FloatP<WIDTH> d2 = d*d;

		// determine closest primitive
		int endReference = startReference + WIDTH;
		if (endReference > nReferences) endReference = nReferences;

		for (int p = startReference; p < endReference; p++) {
			int w = p - startReference;

			if (d2[w] <= s.r2) {
				int index = references[referenceOffset + p];
				const PrimitiveType *prim = primitives[index];

				const LineSegment *lineSegment = reinterpret_cast<const LineSegment *>(prim);
				closestIndex = index;
				s.r2 = d2[w];
				i.d = d[w];
				i.p[0] = pt[0][w];
				i.p[1] = pt[1][w];
				i.p[2] = pt[2][w];
				i.uv[0] = t[w];
				i.uv[1] = -1;
				i.nodeIndex = nodeIndex;
				i.primitive = lineSegment;
			}
		}

		startReference += WIDTH;
	}

	return closestIndex != -1;
}

template<size_t WIDTH, size_t DIM, typename PrimitiveType>
inline bool Mbvh<WIDTH, DIM, PrimitiveType>::findClosestPointTriangle(const MbvhNode<DIM>& node,
																	  int nodeIndex, BoundingSphere<DIM>& s,
																	  Interaction<DIM>& i) const
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
		VectorP<WIDTH, DIM> pt;
		VectorP<WIDTH, DIM - 1> t;
		int leafIndex = 3*(leafOffset + l);
		const VectorP<WIDTH, DIM>& pa = leafNodes[leafIndex + 0];
		const VectorP<WIDTH, DIM>& pb = leafNodes[leafIndex + 1];
		const VectorP<WIDTH, DIM>& pc = leafNodes[leafIndex + 2];
		FloatP<WIDTH> d = findClosestPointWideTriangle(s.c, pa, pb, pc, pt, t);
		FloatP<WIDTH> d2 = d*d;

		// determine closest primitive
		int endReference = startReference + WIDTH;
		if (endReference > nReferences) endReference = nReferences;

		for (int p = startReference; p < endReference; p++) {
			int w = p - startReference;

			if (d2[w] <= s.r2) {
				int index = references[referenceOffset + p];
				const PrimitiveType *prim = primitives[index];

				const Triangle *triangle = reinterpret_cast<const Triangle *>(prim);
				closestIndex = index;
				s.r2 = d2[w];
				i.d = d[w];
				i.p[0] = pt[0][w];
				i.p[1] = pt[1][w];
				i.p[2] = pt[2][w];
				i.uv[0] = t[0][w];
				i.uv[1] = t[1][w];
				i.nodeIndex = nodeIndex;
				i.primitive = triangle;
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
				bool found = vectorizedLeafType == ObjectType::LineSegments ?
							 findClosestPointLineSegment(node, nodeIndex, s, i) :
							 findClosestPointTriangle(node, nodeIndex, s, i);
				if (found) notFound = false;
				nodesVisited++;

			} else {
				// primitive type does not support vectorized closest point query,
				// perform query to each primitive one by one
				int referenceOffset = node.child[2];
				int nReferences = node.child[3];

				for (int p = 0; p < nReferences; p++) {
					int index = references[referenceOffset + p];
					const PrimitiveType *prim = primitives[index];

					if (prim != i.primitive) {
						nodesVisited++;

						bool found = false;
						Interaction<DIM> c;
						if (primitiveTypeIsAggregate) {
							const Aggregate<DIM> *aggregate = reinterpret_cast<const Aggregate<DIM> *>(prim);
							found = aggregate->findClosestPointFromNode(s, c, nodeStartIndex, boundaryHint, nodesVisited);

						} else {
							found = prim->findClosestPoint(s, c);
						}

						// keep the closest point only
						if (found) {
							notFound = false;
							s.r2 = std::min(s.r2, c.d*c.d);
							i = c;
							i.nodeIndex = nodeIndex;
						}
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
		if (this->computeNormals) {
			i.computeNormal();
		}

		return true;
	}

	return false;
}

} // namespace fcpw
