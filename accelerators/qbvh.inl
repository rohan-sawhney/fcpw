#include "wide_query_operations.h"

namespace fcpw {

template <int WIDTH, int DIM>
inline void Qbvh<WIDTH, DIM>::collapseSbvh(const std::shared_ptr<Sbvh<DIM>>& sbvh,
										   int grandParent, int depth)
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	// TODO:
	// collapse(sbvh, grandparent, maxDepth):
	// - if sbvh.flatTree[grandparent].rightOffset == 0
	// -- TODO: process leaf
	// - children = {grandparent + 1, grandparent + sbvh.flatTree[grandparent].rightOffset}
	// - grandchildren = {children[0] + 1, children[0] + sbvh.flatTree[children[0]].rightOffset,
	//                    children[1] + 1, children[1] + sbvh.flatTree[children[1]].rightOffset}
	// - populate QbvhNode
	// -- cases: sbvh.flatTree[children[0]].rightOffset != 0 && sbvh.flatTree[children[1]].rightOffset != 0,
	//			 sbvh.flatTree[children[0]].rightOffset != 0 && sbvh.flatTree[children[1]].rightOffset == 0,
	//			 sbvh.flatTree[children[0]].rightOffset == 0 && sbvh.flatTree[children[1]].rightOffset != 0,
	//			 sbvh.flatTree[children[0]].rightOffset == 0 && sbvh.flatTree[children[1]].rightOffset == 0
}

template <int WIDTH, int DIM>
inline Qbvh<WIDTH, DIM>::Qbvh(const std::shared_ptr<Sbvh<DIM>>& sbvh_):
primitives(sbvh_->primitives),
references(std::move(sbvh_->references)),
nNodes(0),
nLeafs(0),
maxDepth(0)
{
	LOG_IF(FATAL, sbvh_->leafSize != WIDTH) << "Sbvh leaf size must equal bvh width";

	using namespace std::chrono;
	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	// collapse sbvh
	collapseSbvh(sbvh_, 0, 0);

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
inline BoundingBox<DIM> Qbvh<WIDTH, DIM>::boundingBox() const
{
	// TODO
	return BoundingBox<DIM>();
}

template <int WIDTH, int DIM>
inline Vector<DIM> Qbvh<WIDTH, DIM>::centroid() const
{
	// TODO
	return zeroVector<DIM>();
}

template <int WIDTH, int DIM>
inline float Qbvh<WIDTH, DIM>::surfaceArea() const
{
	// TODO
	return 0.0f;
}

template <int WIDTH, int DIM>
inline float Qbvh<WIDTH, DIM>::signedVolume() const
{
	// TODO
	return 0.0f;
}

template <int WIDTH, int DIM>
inline int Qbvh<WIDTH, DIM>::intersectFromNode(Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
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
inline bool Qbvh<WIDTH, DIM>::findClosestPointFromNode(BoundingSphere<DIM>& s, Interaction<DIM>& i,
													   int nodeStartIndex, int& nodesVisited) const
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	// TODO
	return false;
}

} // namespace fcpw
