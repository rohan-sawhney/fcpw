namespace fcpw {

template <int WIDTH, int DIM>
inline Qbvh<WIDTH, DIM>::Qbvh(const std::shared_ptr<Sbvh<DIM>>& sbvh_)
{
	// TODO
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
