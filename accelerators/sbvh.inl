#include <stack>
#include <queue>

namespace fcpw {

template <int DIM>
inline Sbvh<DIM>::Sbvh(std::vector<std::shared_ptr<Primitive<DIM>>>& primitives_,
					   const CostHeuristic& costHeuristic_, int leafSize_):
nNodes(0),
nLeafs(0),
leafSize(leafSize_),
primitives(primitives_)
{
	using namespace std::chrono;
	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	build(costHeuristic_);

	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	duration<double> timeSpan = duration_cast<duration<double>>(t2 - t1);
	std::cout << "Built Sbvh with "
			  << nNodes << " nodes, "
			  << nLeafs << " leaves, "
			  << primitives.size() << " primitives in "
			  << timeSpan.count() << " seconds" << std::endl;
}

template <int DIM>
inline void Sbvh<DIM>::build(const CostHeuristic& costHeuristic)
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	// TODO
}

template <int DIM>
inline BoundingBox<DIM> Sbvh<DIM>::boundingBox() const
{
	// TODO
	return BoundingBox<DIM>(false);
}

template <int DIM>
inline Vector<DIM> Sbvh<DIM>::centroid() const
{
	return boundingBox().centroid();
}

template <int DIM>
inline float Sbvh<DIM>::surfaceArea() const
{
	float area = 0.0f;
	for (int p = 0; p < (int)primitives.size(); p++) {
		area += primitives[p]->surfaceArea();
	}

	return area;
}

template <int DIM>
inline float Sbvh<DIM>::signedVolume() const
{
	float volume = 0.0f;
	for (int p = 0; p < (int)primitives.size(); p++) {
		volume += primitives[p]->signedVolume();
	}

	return volume;
}

template <int DIM>
inline int Sbvh<DIM>::intersect(Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
								bool checkOcclusion, bool countHits) const
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	// TODO
	return 0;
}

template <int DIM>
inline bool Sbvh<DIM>::findClosestPoint(BoundingSphere<DIM>& s, Interaction<DIM>& i) const
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	// TODO
	return false;
}

} // namespace fcpw
