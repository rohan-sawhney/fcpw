#include "wide_query_operations.h"

namespace fcpw {

template <int WIDTH, int DIM>
inline int Mbvh<WIDTH, DIM>::collapseSbvh(const std::shared_ptr<Sbvh<DIM>>& sbvh,
										  int sbvhNodeIndex, int parent, int depth)
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	// TODO: set axis
	const SbvhFlatNode<DIM>& sbvhNode = sbvh->flatTree[sbvhNodeIndex];
	maxDepth = std::max(depth, maxDepth);

	// create mbvh node
	MbvhNode<WIDTH, DIM> node;
	int nodeIndex = nNodes++;
	node.parent = parent;
	nodes.emplace_back(node);

	if (sbvhNode.rightOffset == 0) {
		// sbvh node is a leaf node
		for (int p = 0; p < sbvhNode.nReferences; p++) {
			int index = sbvhNode.start + p;
			for (int i = 0; i < DIM; i++) {
				nodes[nodeIndex].boxMin[i][p] = sbvh->referenceBoxes[index][i];
				nodes[nodeIndex].boxMax[i][p] = sbvh->referenceBoxes[index][i];
			}

			nodes[nodeIndex].child[p] = -sbvh->references[index];
		}

		nLeafs++;

	} else {
		// sbvh node is an inner node

		// TODO: set boxMin, boxMax, child
		// - children = {sbvhNodeIndex + 1, sbvhNodeIndex + sbvh.flatTree[sbvhNodeIndex].rightOffset}
		// - grandchildren = {children[0] + 1, children[0] + sbvh.flatTree[children[0]].rightOffset,
		//                    children[1] + 1, children[1] + sbvh.flatTree[children[1]].rightOffset}
		// - populate MbvhNode
		// -- cases: sbvh.flatTree[children[0]].rightOffset != 0 && sbvh.flatTree[children[1]].rightOffset != 0,
		//			 sbvh.flatTree[children[0]].rightOffset != 0 && sbvh.flatTree[children[1]].rightOffset == 0,
		//			 sbvh.flatTree[children[0]].rightOffset == 0 && sbvh.flatTree[children[1]].rightOffset != 0,
		//			 sbvh.flatTree[children[0]].rightOffset == 0 && sbvh.flatTree[children[1]].rightOffset == 0
	}

	return nodeIndex;
}

template <int WIDTH, int DIM>
inline Mbvh<WIDTH, DIM>::Mbvh(const std::shared_ptr<Sbvh<DIM>>& sbvh_):
primitives(sbvh_->primitives),
references(std::move(sbvh_->references)),
nNodes(0),
nLeafs(0),
maxDepth(0)
{
	LOG_IF(FATAL, sbvh_->leafSize != WIDTH) << "Sbvh leaf size must equal mbvh width";

	using namespace std::chrono;
	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	// collapse sbvh
	collapseSbvh(sbvh_, 0, 0xfffffffc, 0);
	maxDepth = std::pow(2, std::ceil(std::log2(maxDepth)));

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
