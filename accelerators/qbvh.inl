namespace fcpw {

template <int DIM>
inline Qbvh<DIM>::Qbvh(const std::shared_ptr<Sbvh<DIM>>& sbvh_)
{
	// TODO
}

template <int DIM>
inline BoundingBox<DIM> Qbvh<DIM>::boundingBox() const
{
	// TODO
	return BoundingBox<DIM>();
}

template <int DIM>
inline Vector<DIM> Qbvh<DIM>::centroid() const
{
	// TODO
	return zeroVector<DIM>();
}

template <int DIM>
inline float Qbvh<DIM>::surfaceArea() const
{
	// TODO
	return 0.0f;
}

template <int DIM>
inline float Qbvh<DIM>::signedVolume() const
{
	// TODO
	return 0.0f;
}

template <int DIM>
inline int Qbvh<DIM>::intersectFromNode(Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
										int nodeStartIndex, int& nodesVisited,
										bool checkOcclusion, bool countHits) const
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	// TODO
	return 0;
}

template <int DIM>
inline bool Qbvh<DIM>::findClosestPointFromNode(BoundingSphere<DIM>& s, Interaction<DIM>& i,
												int nodeStartIndex, int& nodesVisited) const
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	// TODO
	return false;
}

} // namespace fcpw