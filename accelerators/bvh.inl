namespace fcpw {

template <int DIM>
inline Bvh<DIM>::Bvh(std::vector<std::shared_ptr<Primitive<DIM>>>& primitives_,
					 const CostHeuristic& costHeuristic_, int leafSize_):
Sbvh<DIM>(primitives_, costHeuristic_, 1.0f, leafSize_)
{

}

template <int DIM>
inline BoundingBox<DIM> Bvh<DIM>::boundingBox() const
{
	return Sbvh<DIM>::boundingBox();
}

template <int DIM>
inline Vector<DIM> Bvh<DIM>::centroid() const
{
	return Sbvh<DIM>::centroid();
}

template <int DIM>
inline float Bvh<DIM>::surfaceArea() const
{
	return Sbvh<DIM>::surfaceArea();
}

template <int DIM>
inline float Bvh<DIM>::signedVolume() const
{
	return Sbvh<DIM>::signedVolume();
}

template <int DIM>
inline int Bvh<DIM>::intersect(Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
							   bool checkOcclusion, bool countHits) const
{
	return Sbvh<DIM>::intersect(r, is, checkOcclusion, countHits);
}

template <int DIM>
inline bool Bvh<DIM>::findClosestPoint(BoundingSphere<DIM>& s, Interaction<DIM>& i) const
{
	return Sbvh<DIM>::findClosestPoint(s, i);
}

} // namespace fcpw
