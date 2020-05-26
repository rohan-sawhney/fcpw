#include "wide_query_operations.h"

namespace fcpw {

template <int WIDTH, int DIM>
inline int Mbvh<WIDTH, DIM>::collapseSbvh(const std::shared_ptr<Sbvh<DIM>>& sbvh,
										  int sbvhNodeIndex, int parent, int depth)
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	const SbvhNode<DIM>& sbvhNode = sbvh->flatTree[sbvhNodeIndex];
	maxDepth = std::max(depth, maxDepth);

	// create mbvh node
	MbvhNode<WIDTH, DIM> mbvhNode;
	int mbvhNodeIndex = nNodes;

	nNodes++;
	mbvhNode.parent = parent;
	flatTree.emplace_back(mbvhNode);

	if (sbvhNode.rightOffset == 0) {
		// sbvh node is a leaf node; assign mbvh node its reference indices
		for (int p = 0; p < sbvhNode.nReferences; p++) {
			flatTree[mbvhNodeIndex].child[p] = -(references[sbvhNode.start + p] + 1);
		}

		nLeafs++;

	} else {
		// sbvh node is an inner node, flatten it
		int nNodesCollapsed = 0;
		int stackPtr = 0;
		stackSbvhNodes[stackPtr].first = sbvhNodeIndex;
		stackSbvhNodes[stackPtr].second = 0;

		while (stackPtr >= 0) {
			int sbvhNodeIndex = stackSbvhNodes[stackPtr].first;
			int level = stackSbvhNodes[stackPtr].second;
			stackPtr--;

			const SbvhNode<DIM>& sbvhNode = sbvh->flatTree[sbvhNodeIndex];
			if (level < maxLevel && sbvhNode.rightOffset != 0) {
				// enqueue sbvh children nodes till max level or leaf node is reached
				stackPtr++;
				stackSbvhNodes[stackPtr].first = sbvhNodeIndex + 1;
				stackSbvhNodes[stackPtr].second = level + 1;

				stackPtr++;
				stackSbvhNodes[stackPtr].first = sbvhNodeIndex + sbvhNode.rightOffset;
				stackSbvhNodes[stackPtr].second = level + 1;

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

template <int WIDTH, int DIM>
inline bool Mbvh<WIDTH, DIM>::isLeafNode(const MbvhNode<WIDTH, DIM>& node) const
{
	return node.child[0] < 0;
}

template <int WIDTH, int DIM>
inline void Mbvh<WIDTH, DIM>::populateLeafNode(const MbvhNode<WIDTH, DIM>& node,
											   std::vector<VectorP<WIDTH, DIM>>& leafNode)
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	leafNode.resize(3);

	for (int w = 0; w < WIDTH; w++) {
		if (node.child[w] != maxInt) {
			int index = -node.child[w] - 1;
			const Triangle *triangle = dynamic_cast<const Triangle *>(primitives[index].get());
			const Vector3& pa = triangle->soup->positions[triangle->indices[0]];
			const Vector3& pb = triangle->soup->positions[triangle->indices[1]];
			const Vector3& pc = triangle->soup->positions[triangle->indices[2]];

			for (int i = 0; i < DIM; i++) {
				leafNode[0][i][w] = pa[i];
				leafNode[1][i][w] = pb[i];
				leafNode[2][i][w] = pc[i];
			}
		}
	}
}

template <int WIDTH, int DIM>
inline void Mbvh<WIDTH, DIM>::populateLeafNodes()
{
	// check if primitive type is supported
	for (int p = 0; p < (int)primitives.size(); p++) {
		const Triangle *triangle = dynamic_cast<const Triangle *>(primitives[p].get());

		if (triangle) {
			primitiveType = 1;

		} else {
			primitiveType = 0;
			break;
		}
	}

	if (primitiveType > 0) {
		// populate leaf nodes
		int leafIndex = 0;
		leafNodes.resize(nLeafs);

		for (int i = 0; i < nNodes; i++) {
			MbvhNode<WIDTH, DIM>& node = flatTree[i];

			if (isLeafNode(node)) {
				populateLeafNode(node, leafNodes[leafIndex]);
				node.leafIndex = leafIndex++;
			}
		}
	}
}

template <int WIDTH, int DIM>
inline Mbvh<WIDTH, DIM>::Mbvh(const std::shared_ptr<Sbvh<DIM>>& sbvh_):
primitives(sbvh_->primitives),
references(sbvh_->references),
stackSbvhNodes(WIDTH, std::make_pair(-1, -1)),
nNodes(0),
nLeafs(0),
maxDepth(0),
maxLevel(std::log2(WIDTH)),
primitiveType(0)
{
	LOG_IF(FATAL, sbvh_->leafSize != WIDTH) << "Sbvh leaf size must equal mbvh width";

	using namespace std::chrono;
	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	// collapse sbvh
	collapseSbvh(sbvh_, 0, 0xfffffffc, 0);
	stackSbvhNodes.clear();

	// populate leaf nodes if primitive type is supported
	populateLeafNodes();

	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	duration<double> timeSpan = duration_cast<duration<double>>(t2 - t1);
	std::cout << "Built " << WIDTH << "-bvh with "
			  << nNodes << " nodes, "
			  << nLeafs << " leaves, "
			  << maxDepth << " max depth, "
			  << primitives.size() << " primitives, "
			  << references.size() << " references in "
			  << timeSpan.count() << " seconds" << std::endl;
}

template <int WIDTH, int DIM>
inline BoundingBox<DIM> Mbvh<WIDTH, DIM>::boundingBox() const
{
	BoundingBox<DIM> box;
	if (flatTree.size() == 0) return box;

	box.pMin = enoki::hmin_inner(flatTree[0].boxMin);
	box.pMax = enoki::hmax_inner(flatTree[0].boxMax);
	return box;
}

template <int WIDTH, int DIM>
inline Vector<DIM> Mbvh<WIDTH, DIM>::centroid() const
{
	Vector<DIM> c = zeroVector<DIM>();
	int nPrimitives = (int)primitives.size();

	for (int p = 0; p < nPrimitives; p++) {
		c += primitives[p]->centroid();
	}

	return c/nPrimitives;
}

template <int WIDTH, int DIM>
inline float Mbvh<WIDTH, DIM>::surfaceArea() const
{
	float area = 0.0f;
	for (int p = 0; p < (int)primitives.size(); p++) {
		area += primitives[p]->surfaceArea();
	}

	return area;
}

template <int WIDTH, int DIM>
inline float Mbvh<WIDTH, DIM>::signedVolume() const
{
	float volume = 0.0f;
	for (int p = 0; p < (int)primitives.size(); p++) {
		volume += primitives[p]->signedVolume();
	}

	return volume;
}

template <int WIDTH, int DIM>
inline int Mbvh<WIDTH, DIM>::intersectFromNode(Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
											   int nodeStartIndex, int& nodesVisited,
											   bool checkOcclusion, bool countHits) const
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	LOG_IF(FATAL, nodeStartIndex < 0 || nodeStartIndex >= nNodes) << "Start node index: "
								 << nodeStartIndex << " out of range [0, " << nNodes << ")";
	int hits = 0;
	if (!countHits) is.resize(1);
	BvhTraversal subtree[maxDepth + 1];

	// TODO:
	// push root
	// while stack on empty
	// - dequeue node
	// - continue if near > d
	// - if leaf node
	// -- process primitives
	// - else
	// -- intersect boxes
	// -- sort boxes
	// -- enqueue

	if (countHits) {
		std::sort(is.begin(), is.end(), compareInteractions<DIM>);
		is = removeDuplicates<DIM>(is);
		hits = (int)is.size();
	}

	return hits;
}

template <int WIDTH, int DIM>
inline bool Mbvh<WIDTH, DIM>::findClosestPointTriangle(const MbvhNode<WIDTH, DIM>& node, int nodeIndex,
													   BoundingSphere<DIM>& s, Interaction<DIM>& i) const
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	// perform vectorized closest point query
	Vector3P<WIDTH> pt;
	Vector2P<WIDTH> t;
	IntP<WIDTH> vIndex, eIndex;
	const std::vector<VectorP<WIDTH, DIM>>& leafNode = leafNodes[node.leafIndex];
	FloatP<WIDTH> d = findClosestPointWideTriangle<WIDTH>(s.c, leafNode[0], leafNode[1],
													leafNode[2], pt, t, vIndex, eIndex);
	FloatP<WIDTH> d2 = d*d;

	// update interaction with the closest seen primitive
	int W = maxInt;
	for (int w = 0; w < WIDTH; w++) {
		if (node.child[w] != maxInt && d2[w] <= s.r2) {
			s.r2 = d2[w];
			W = w;
		}
	}

	// update interaction
	if (W != maxInt) {
		int index = -node.child[W] - 1;
		const Triangle *triangle = dynamic_cast<const Triangle *>(primitives[index].get());
		i.d = d[W];
		i.p[0] = pt[0][W];
		i.p[1] = pt[1][W];
		i.p[2] = pt[2][W];
		i.uv[0] = t[0][W];
		i.uv[1] = t[1][W];
		i.n = triangle->normal(vIndex[W], eIndex[W]);
		i.nodeIndex = nodeIndex;
		i.primitive = triangle;

		return true;
	}

	return false;
}

template <int WIDTH, int DIM>
inline bool Mbvh<WIDTH, DIM>::findClosestPointFromNode(BoundingSphere<DIM>& s, Interaction<DIM>& i,
													   int nodeStartIndex, int& nodesVisited) const
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	LOG_IF(FATAL, nodeStartIndex < 0 || nodeStartIndex >= nNodes) << "Start node index: "
								 << nodeStartIndex << " out of range [0, " << nNodes << ")";
	bool notFound = true;
	std::deque<BvhTraversal> subtree;
	FloatP<WIDTH> d2Min, d2Max;

	// push root node
	subtree.emplace_back(BvhTraversal(0, minFloat));

	while (!subtree.empty()) {
		// pop off the next node to work on
		BvhTraversal traversal = subtree.front();
		subtree.pop_front();

		int nodeIndex = traversal.node;
		float near = traversal.distance;

		// if this node is further than the closest found primitive, continue
		if (near > s.r2) continue;
		const MbvhNode<WIDTH, DIM>& node(flatTree[nodeIndex]);

		if (isLeafNode(node)) {
			if (primitiveType > 0) {
				// perform vectorized closest point query to triangle
				if (findClosestPointTriangle(node, nodeIndex, s, i)) notFound = false;
				nodesVisited += WIDTH;

			} else {
				// primitive type does not support vectorized closest point query,
				// perform query to each primitive one by one
				for (int w = 0; w < WIDTH; w++) {
					if (node.child[w] != maxInt) {
						int index = -node.child[w] - 1;
						const std::shared_ptr<Primitive<DIM>>& prim = primitives[index];

						if (prim.get() != i.primitive) {
							Interaction<DIM> c;
							bool found = prim->findClosestPoint(s, c);
							nodesVisited++;

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
			}

		} else {
			// overlap sphere with boxes
			MaskP<WIDTH> mask = overlapWideBox<WIDTH, DIM>(s, node.boxMin,
												node.boxMax, d2Min, d2Max);
			s.r2 = std::min(s.r2, enoki::hmin(d2Max));

			// TODO: sort node indices according to d2Min

			// enqueue masked nodes
			for (int w = 0; w < WIDTH; w++) {
				int index = w;
				if (node.child[index] != maxInt && mask[index]) {
					subtree.emplace_back(BvhTraversal(node.child[index], d2Min[index]));
				}
			}

			nodesVisited++;
		}
	}

	return !notFound;
}

} // namespace fcpw
