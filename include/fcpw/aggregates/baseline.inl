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
inline BoundingSphere<DIM> Baseline<DIM, PrimitiveType>::boundingSphere() const
{
	BoundingBox<DIM> bb;
	for (int p = 0; p < (int)primitives.size(); p++) {
		bb.expandToInclude(primitives[p]->boundingBox());
	}

	return bb.sphere();
}

template<size_t DIM, typename PrimitiveType>
inline Vector<DIM> Baseline<DIM, PrimitiveType>::centroid() const
{
	Vector<DIM> c = Vector<DIM>::Zero();
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
														   int nodeStartIndex, int aggregateIndex, int& nodesVisited,
														   bool checkForOcclusion, bool recordAllHits) const
{
	int hits = 0;
	if (!recordAllHits) is.resize(1);

	// find closest hit
	for (int p = 0; p < (int)primitives.size(); p++) {
		nodesVisited++;

		int hit = 0;
		std::vector<Interaction<DIM>> cs;
		if (primitiveTypeIsAggregate) {
			const Aggregate<DIM> *aggregate = reinterpret_cast<const Aggregate<DIM> *>(primitives[p]);
			hit = aggregate->intersectFromNode(r, cs, nodeStartIndex, aggregateIndex,
											   nodesVisited, checkForOcclusion, recordAllHits);

		} else {
			hit = primitives[p]->intersect(r, cs, checkForOcclusion, recordAllHits);
			for (int i = 0; i < (int)cs.size(); i++) {
				cs[i].referenceIndex = p;
				cs[i].objectIndex = this->index;
			}
		}

		if (hit > 0) {
			if (checkForOcclusion) {
				is.clear();
				return 1;
			}

			hits += hit;
			if (recordAllHits) {
				is.insert(is.end(), cs.begin(), cs.end());

			} else {
				r.tMax = std::min(r.tMax, cs[0].d);
				is[0] = cs[0];
			}
		}
	}

	if (hits > 0) {
		// sort by distance and remove duplicates
		if (recordAllHits) {
			std::sort(is.begin(), is.end(), compareInteractions<DIM>);
			is = removeDuplicates<DIM>(is);
			hits = (int)is.size();

		} else {
			hits = 1;
		}

		// compute normals
		if (this->computeNormals && !primitiveTypeIsAggregate) {
			for (int i = 0; i < (int)is.size(); i++) {
				is[i].computeNormal(primitives[is[i].referenceIndex]);
			}
		}

		return hits;
	}

	return 0;
}

template<size_t DIM, typename PrimitiveType>
inline int Baseline<DIM, PrimitiveType>::intersectFromNodeTimed(Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
														   int nodeStartIndex, int aggregateIndex, int& nodesVisited, uint64_t& ticks,
														   bool checkForOcclusion, bool recordAllHits) const
{
	int hits = 0;
	if (!recordAllHits) is.resize(1);

	// find closest hit
	for (int p = 0; p < (int)primitives.size(); p++) {
		nodesVisited++;

		int hit = 0;
		std::vector<Interaction<DIM>> cs;
		if (primitiveTypeIsAggregate) {
			const Aggregate<DIM> *aggregate = reinterpret_cast<const Aggregate<DIM> *>(primitives[p]);
			hit = aggregate->intersectFromNodeTimed(r, cs, nodeStartIndex, aggregateIndex,
											   		nodesVisited, ticks, checkForOcclusion, recordAllHits);

		} else {
			auto t1 = std::chrono::high_resolution_clock::now();
			hit = primitives[p]->intersect(r, cs, checkForOcclusion, recordAllHits);
			for (int i = 0; i < (int)cs.size(); i++) {
				cs[i].referenceIndex = p;
				cs[i].objectIndex = this->index;
			}
			auto t2 = std::chrono::high_resolution_clock::now();
			ticks += (t2 - t1).count();
		}

		if (hit > 0) {
			if (checkForOcclusion) {
				is.clear();
				return 1;
			}

			hits += hit;
			if (recordAllHits) {
				is.insert(is.end(), cs.begin(), cs.end());

			} else {
				r.tMax = std::min(r.tMax, cs[0].d);
				is[0] = cs[0];
			}
		}
	}

	if (hits > 0) {
		// sort by distance and remove duplicates
		if (recordAllHits) {
			std::sort(is.begin(), is.end(), compareInteractions<DIM>);
			is = removeDuplicates<DIM>(is);
			hits = (int)is.size();

		} else {
			hits = 1;
		}

		// compute normals
		if (this->computeNormals && !primitiveTypeIsAggregate) {
			for (int i = 0; i < (int)is.size(); i++) {
				is[i].computeNormal(primitives[is[i].referenceIndex]);
			}
		}

		return hits;
	}

	return 0;
}

template<size_t DIM, typename PrimitiveType>
inline bool Baseline<DIM, PrimitiveType>::findClosestPointFromNode(BoundingSphere<DIM>& s, Interaction<DIM>& i,
																   int nodeStartIndex, int aggregateIndex,
																   const Vector<DIM>& boundaryHint, int& nodesVisited) const
{
	// find closest point
	bool notFound = true;
	for (int p = 0; p < (int)primitives.size(); p++) {
		nodesVisited++;

		bool found = false;
		Interaction<DIM> c;
		if (primitiveTypeIsAggregate) {
			const Aggregate<DIM> *aggregate = reinterpret_cast<const Aggregate<DIM> *>(primitives[p]);
			found = aggregate->findClosestPointFromNode(s, c, nodeStartIndex, aggregateIndex,
														boundaryHint, nodesVisited);

		} else {
			found = primitives[p]->findClosestPoint(s, c);
			c.referenceIndex = p;
			c.objectIndex = this->index;
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
		if (this->computeNormals && !primitiveTypeIsAggregate) {
			i.computeNormal(primitives[i.referenceIndex]);
		}

		return true;
	}

	return false;
}


template<size_t DIM, typename PrimitiveType>
inline bool Baseline<DIM, PrimitiveType>::findClosestPointFromNodeTimed(BoundingSphere<DIM>& s, Interaction<DIM>& i,
																   int nodeStartIndex, int aggregateIndex,
																   const Vector<DIM>& boundaryHint, int& nodesVisited, uint64_t& ticks) const
{
	// find closest point
	bool notFound = true;
	for (int p = 0; p < (int)primitives.size(); p++) {
		nodesVisited++;

		bool found = false;
		Interaction<DIM> c;
		if (primitiveTypeIsAggregate) {
			const Aggregate<DIM> *aggregate = reinterpret_cast<const Aggregate<DIM> *>(primitives[p]);
			found = aggregate->findClosestPointFromNodeTimed(s, c, nodeStartIndex, aggregateIndex,
														boundaryHint, nodesVisited, ticks);

		} else {
			auto t1 = std::chrono::high_resolution_clock::now();
			found = primitives[p]->findClosestPoint(s, c);
			auto t2 = std::chrono::high_resolution_clock::now();
			ticks += (t2 - t1).count();
			c.referenceIndex = p;
			c.objectIndex = this->index;
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
		if (this->computeNormals && !primitiveTypeIsAggregate) {
			i.computeNormal(primitives[i.referenceIndex]);
		}

		return true;
	}

	return false;
}

} // namespace fcpw
