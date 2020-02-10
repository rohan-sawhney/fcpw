namespace fcpw {

template <int DIM>
inline Baseline<DIM>::Baseline(const std::vector<std::shared_ptr<Shape<DIM>>>& shapes_):
shapes(shapes_)
{

}

template <int DIM>
inline BoundingBox<DIM> Baseline<DIM>::boundingBox() const
{
	BoundingBox<DIM> bb;
	for (int p = 0; p < (int)shapes.size(); p++) {
		bb.expandToInclude(shapes[p]->boundingBox());
	}

	return bb;
}

template <int DIM>
inline Vector<DIM> Baseline<DIM>::centroid() const
{
	return boundingBox().centroid();
}

template <int DIM>
inline float Baseline<DIM>::surfaceArea() const
{
	float area = 0.0f;
	for (int p = 0; p < (int)shapes.size(); p++) {
		area += shapes[p]->surfaceArea();
	}

	return area;
}

template <int DIM>
inline float Baseline<DIM>::signedVolume() const
{
	float volume = 0.0f;
	for (int p = 0; p < (int)shapes.size(); p++) {
		volume += shapes[p]->signedVolume();
	}

	return volume;
}

template <int DIM>
inline int Baseline<DIM>::intersect(Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
									bool checkOcclusion, bool countHits, bool collectAll) const
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	int hits = 0;
	if (!collectAll) is.resize(1);

	for (int p = 0; p < (int)shapes.size(); p++) {
		std::vector<Interaction<DIM>> cs;
		int hit = shapes[p]->intersect(r, cs, checkOcclusion, countHits, collectAll);

		if (hit > 0) {
			hits += hit;
			if (!countHits && !collectAll) r.tMax = cs[0].d;
			if (collectAll) is.insert(is.end(), cs.begin(), cs.end());
			else is[0] = cs[0];

			if (checkOcclusion) return 1;
		}
	}

	if (collectAll) std::sort(is.begin(), is.end(), compareInteractions<DIM>);
	return hits;
}

template <int DIM>
inline bool Baseline<DIM>::findClosestPoint(BoundingSphere<DIM>& s,
											Interaction<DIM>& i) const
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	bool notFound = true;
	for (int p = 0; p < (int)shapes.size(); p++) {
		Interaction<DIM> c;
		bool found = shapes[p]->findClosestPoint(s, c);

		// keep the closest point only
		if (found) {
			notFound = false;
			s.r2 = std::min(s.r2, c.d*c.d);
			i = c;
		}
	}

	return !notFound;
}

} // namespace fcpw
