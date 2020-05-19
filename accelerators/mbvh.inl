#include "wide_query_operations.h"

namespace fcpw {

template <int WIDTH, int DIM>
inline int Mbvh<WIDTH, DIM>::collapseSbvh(const std::shared_ptr<Sbvh<DIM>>& sbvh,
										  int sbvhNodeIndex, int parent, int depth)
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	const SbvhFlatNode<DIM>& sbvhNode = sbvh->flatTree[sbvhNodeIndex];
	maxDepth = std::max(depth, maxDepth);

	// create mbvh node
	MbvhNode<WIDTH, DIM> mbvhNode;
	int mbvhNodeIndex = nNodes;

	nNodes++;
	mbvhNode.splitDim[0] = sbvhNode.splitDim;
	mbvhNode.splitDim[1] = 0;
	mbvhNode.splitDim[2] = 0;
	mbvhNode.parent = parent;
	nodes.emplace_back(mbvhNode);

	if (sbvhNode.rightOffset == 0) {
		// sbvh node is a leaf node; assign mbvh node its reference indices
		for (int p = 0; p < sbvhNode.nReferences; p++) {
			nodes[mbvhNodeIndex].child[p] = -references[sbvhNode.start + p];
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

			const SbvhFlatNode<DIM>& sbvhNode = sbvh->flatTree[sbvhNodeIndex];
			if (level < maxLevel && sbvhNode.rightOffset != 0) {
				// enqueue sbvh children nodes till max level or leaf node is reached
				stackPtr++;
				stackSbvhNodes[stackPtr].first = sbvhNodeIndex + 1;
				stackSbvhNodes[stackPtr].second = level + 1;

				stackPtr++;
				stackSbvhNodes[stackPtr].first = sbvhNodeIndex + sbvhNode.rightOffset;
				stackSbvhNodes[stackPtr].second = level + 1;

				if (level == 0) {
					nodes[mbvhNodeIndex].splitDim[1] = sbvh->flatTree[sbvhNodeIndex].splitDim;
					nodes[mbvhNodeIndex].splitDim[2] = sbvh->flatTree[sbvhNodeIndex +
														   sbvhNode.rightOffset].splitDim;
				}

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
inline Mbvh<WIDTH, DIM>::Mbvh(const std::shared_ptr<Sbvh<DIM>>& sbvh_):
primitives(sbvh_->primitives),
references(sbvh_->references),
stackSbvhNodes(WIDTH, std::make_pair(-1, -1)),
nNodes(0),
nLeafs(0),
maxDepth(0),
maxLevel(std::log2(WIDTH))
{
	LOG_IF(FATAL, sbvh_->leafSize != WIDTH) << "Sbvh leaf size must equal mbvh width";

	using namespace std::chrono;
	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	// collapse sbvh
	collapseSbvh(sbvh_, 0, 0xfffffffc, 0);
	maxDepth = std::pow(2, std::ceil(std::log2(maxDepth)));
	stackSbvhNodes.clear();

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

	// TODO
	return false;
}

} // namespace fcpw
