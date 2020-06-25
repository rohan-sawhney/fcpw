namespace fcpw {

template<size_t DIM, typename PrimitiveType>
inline Baseline<DIM, PrimitiveType>::Baseline(const std::vector<PrimitiveType *>& primitives_):
primitives(primitives_),
primitiveTypeIsAggregate(std::is_base_of<Aggregate<DIM>, PrimitiveType>::value)
{
	// don't compute normals by default
	this->computeNormals = false;
}

template<size_t DIM, typename PrimitiveType>
inline BoundingBox<DIM> Baseline<DIM, PrimitiveType>::boundingBox() const
{
	BoundingBox<DIM> bb;
	for (int p = 0; p < (int)primitives.size(); p++) {
		bb.expandToInclude(primitives[p]->boundingBox());
	}

	return bb;
}

template<size_t DIM, typename PrimitiveType>
inline Vector<DIM> Baseline<DIM, PrimitiveType>::centroid() const
{
	Vector<DIM> c = zeroVector<DIM>();
	int nPrimitives = (int)primitives.size();

	for (int p = 0; p < nPrimitives; p++) {
		c += primitives[p]->centroid();
	}

	return c/nPrimitives;
}

template<size_t DIM, typename PrimitiveType>
inline float Baseline<DIM, PrimitiveType>::surfaceArea() const
{
	float area = 0.0f;
	for (int p = 0; p < (int)primitives.size(); p++) {
		area += primitives[p]->surfaceArea();
	}

	return area;
}

template<size_t DIM, typename PrimitiveType>
inline float Baseline<DIM, PrimitiveType>::signedVolume() const
{
	float volume = 0.0f;
	for (int p = 0; p < (int)primitives.size(); p++) {
		volume += primitives[p]->signedVolume();
	}

	return volume;
}

template<size_t DIM, typename PrimitiveType>
inline int Baseline<DIM, PrimitiveType>::intersectFromNode(Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
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
		nodesVisited++;

		int hit = 0;
		std::vector<Interaction<DIM>> cs;
		if (primitiveTypeIsAggregate) {
			const Aggregate<DIM> *aggregate = reinterpret_cast<const Aggregate<DIM> *>(primitives[p]);
			hit = aggregate->intersectFromNode(r, cs, nodeStartIndex, nodesVisited, checkOcclusion, countHits);

		} else {
			hit = primitives[p]->intersect(r, cs, checkOcclusion, countHits);
		}

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

		// compute normals
		if (this->computeNormals) {
			for (int i = 0; i < (int)is.size(); i++) {
				is[i].computeNormal();
			}
		}

		return hits;
	}

	return 0;
}

template<size_t DIM, typename PrimitiveType>
inline bool Baseline<DIM, PrimitiveType>::findClosestPointFromNode(BoundingSphere<DIM>& s, Interaction<DIM>& i,
																   int nodeStartIndex, const Vector<DIM>& boundaryHint,
																   int& nodesVisited) const
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	// find closest point
	bool notFound = true;
	for (int p = 0; p < (int)primitives.size(); p++) {
		nodesVisited++;

		bool found = false;
		Interaction<DIM> c;
		if (primitiveTypeIsAggregate) {
			const Aggregate<DIM> *aggregate = reinterpret_cast<const Aggregate<DIM> *>(primitives[p]);
			found = aggregate->findClosestPointFromNode(s, c, nodeStartIndex, boundaryHint, nodesVisited);

		} else {
			found = primitives[p]->findClosestPoint(s, c);
		}

		// keep the closest point only
		if (found) {
			notFound = false;
			s.r2 = std::min(s.r2, c.d*c.d);
			i = c;
		}
	}

	if (!notFound) {
		// compute normal
		if (this->computeNormals) {
			i.computeNormal();
		}

		return true;
	}

	return false;
}

} // namespace fcpw
