namespace fcpw {

template<size_t DIM>
inline Baseline<DIM>::Baseline(const std::vector<std::shared_ptr<Primitive<DIM>>>& primitives_):
primitives(primitives_)
{

}

template<size_t DIM>
inline BoundingBox<DIM> Baseline<DIM>::boundingBox() const
{
	BoundingBox<DIM> bb;
	for (int p = 0; p < (int)primitives.size(); p++) {
		bb.expandToInclude(primitives[p]->boundingBox());
	}

	return bb;
}

template<size_t DIM>
inline Vector<DIM> Baseline<DIM>::centroid() const
{
	Vector<DIM> c = zeroVector<DIM>();
	int nPrimitives = (int)primitives.size();

	for (int p = 0; p < nPrimitives; p++) {
		c += primitives[p]->centroid();
	}

	return c/nPrimitives;
}

template<size_t DIM>
inline float Baseline<DIM>::surfaceArea() const
{
	float area = 0.0f;
	for (int p = 0; p < (int)primitives.size(); p++) {
		area += primitives[p]->surfaceArea();
	}

	return area;
}

template<size_t DIM>
inline float Baseline<DIM>::signedVolume() const
{
	float volume = 0.0f;
	for (int p = 0; p < (int)primitives.size(); p++) {
		volume += primitives[p]->signedVolume();
	}

	return volume;
}

template<size_t DIM>
inline int Baseline<DIM>::intersectFromNode(Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
											int nodeStartIndex, int& nodesVisited,
											bool checkOcclusion, bool countHits) const
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	int hits = 0;
	if (!countHits) is.resize(1);

	// find closest hit
	for (int p = 0; p < (int)primitives.size(); p++) {
		if (this->ignorePrimitive(primitives[p].get())) continue;
		nodesVisited++;

		std::vector<Interaction<DIM>> cs;
		const Aggregate<DIM> *aggregate = dynamic_cast<const Aggregate<DIM> *>(primitives[p].get());
		int hit = aggregate ? aggregate->intersectFromNode(r, cs, nodeStartIndex, nodesVisited,
														   checkOcclusion, countHits) :
							  primitives[p]->intersect(r, cs, checkOcclusion, countHits);

		if (hit > 0) {
			hits += hit;
			if (countHits) {
				is.insert(is.end(), cs.begin(), cs.end());

			} else {
				r.tMax = std::min(r.tMax, cs[0].d);
				is[0] = cs[0];
			}

			if (checkOcclusion) return 1;
		}
	}

	if (hits > 0) {
		// sort by distance and remove duplicates
		if (countHits) {
			std::sort(is.begin(), is.end(), compareInteractions<DIM>);
			is = removeDuplicates<DIM>(is);
			hits = (int)is.size();
		}

		// set normals
		for (int i = 0; i < (int)is.size(); i++) {
			is[i].computeNormal();
		}

		return hits;
	}

	return 0;
}

template<size_t DIM>
inline bool Baseline<DIM>::findClosestPointFromNode(BoundingSphere<DIM>& s, Interaction<DIM>& i,
													int nodeStartIndex, const Vector<DIM>& boundaryHint,
													int& nodesVisited) const
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	// find closest point
	bool notFound = true;
	for (int p = 0; p < (int)primitives.size(); p++) {
		if (this->ignorePrimitive(primitives[p].get())) continue;
		nodesVisited++;

		Interaction<DIM> c;
		const Aggregate<DIM> *aggregate = dynamic_cast<const Aggregate<DIM> *>(primitives[p].get());
		bool found = aggregate ? aggregate->findClosestPointFromNode(s, c, nodeStartIndex,
															   boundaryHint, nodesVisited) :
								 primitives[p]->findClosestPoint(s, c);

		// keep the closest point only
		if (found) {
			notFound = false;
			s.r2 = std::min(s.r2, c.d*c.d);
			i = c;
		}
	}

	if (!notFound) {
		// set normal
		i.computeNormal();

		return true;
	}

	return false;
}

} // namespace fcpw
