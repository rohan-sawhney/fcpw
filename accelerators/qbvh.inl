namespace fcpw {

template <int DIM>
inline Qbvh<DIM>::Qbvh(const std::shared_ptr<Sbvh<DIM>>& sbvh_):
sbvh(sbvh_)
{
	// TODO
}

template <int DIM>
inline BoundingBox<DIM> Qbvh<DIM>::boundingBox() const
{
	return sbvh->boundingBox();
}

template <int DIM>
inline Vector<DIM> Qbvh<DIM>::centroid() const
{
	return sbvh->centroid();
}

template <int DIM>
inline float Qbvh<DIM>::surfaceArea() const
{
	return sbvh->surfaceArea();
}

template <int DIM>
inline float Qbvh<DIM>::signedVolume() const
{
	return sbvh->signedVolume();
}

template <int DIM>
inline int Qbvh<DIM>::intersect(Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
								bool checkOcclusion, bool countHits) const
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	// TODO
	return 0;
}

template <int DIM>
inline bool Qbvh<DIM>::findClosestPoint(BoundingSphere<DIM>& s, Interaction<DIM>& i) const
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	// TODO
	return false;
}

} // namespace fcpw
