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
	nodes.emplace_back(mbvhNode);

	if (sbvhNode.rightOffset == 0) {
		// sbvh node is a leaf node; assign mbvh node its reference indices
		for (int p = 0; p < sbvhNode.nReferences; p++) {
			nodes[mbvhNodeIndex].child[p] = -(references[sbvhNode.start + p] + 1);
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
					nodes[mbvhNodeIndex].boxMin[i][nNodesCollapsed] = sbvhNode.box.pMin[i];
					nodes[mbvhNodeIndex].boxMax[i][nNodesCollapsed] = sbvhNode.box.pMax[i];
				}

				nodes[mbvhNodeIndex].child[nNodesCollapsed] = collapseSbvh(sbvh,
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

	for (int i = 0; i < WIDTH; i++) {
		if (node.child[i] != maxInt) {
			int index = -node.child[i] - 1;
			const Triangle *triangle = dynamic_cast<const Triangle *>(primitives[index].get());
			const Vector3& pa = triangle->soup->positions[triangle->indices[0]];
			const Vector3& pb = triangle->soup->positions[triangle->indices[1]];
			const Vector3& pc = triangle->soup->positions[triangle->indices[2]];

			for (int j = 0; j < DIM; j++) {
				leafNode[0][j][i] = pa[j];
				leafNode[1][j][i] = pb[j];
				leafNode[2][j][i] = pc[j];
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
			MbvhNode<WIDTH, DIM>& node = nodes[i];

			if (isLeafNode(node)) {
				populateLeafNode(nodes[i], leafNodes[leafIndex]);
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
	if (nodes.size() == 0) return box;

	box.pMin = enoki::hmin_inner(nodes[0].boxMin);
	box.pMax = enoki::hmax_inner(nodes[0].boxMax);
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

	// push root
	// while stack on empty
	// - dequeue node
	// - continue if near > d
	// - if leaf node
	// -- process primitives
	// - else
	// -- overlap boxes
	// -- enqueue

	// TODO
	return 0;
}

template <int WIDTH, int DIM>
inline bool Mbvh<WIDTH, DIM>::findClosestPointFromNode(BoundingSphere<DIM>& s, Interaction<DIM>& i,
													   int nodeStartIndex, int& nodesVisited) const
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	// push root
	// while stack on empty
	// - dequeue node
	// - continue if near > d
	// - if leaf node
	// -- process primitives
	// - else
	// -- overlap boxes
	// -- enqueue

	// TODO
	return false;
}

} // namespace fcpw
