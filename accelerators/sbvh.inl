namespace fcpw {

template<size_t DIM, typename PrimitiveType>
inline Sbvh<DIM, PrimitiveType>::Sbvh(const std::vector<PrimitiveType *>& primitives_,
									  const CostHeuristic& costHeuristic_, bool printStats_,
									  bool packLeaves_, int leafSize_, int nBuckets_):
primitives(primitives_),
costHeuristic(costHeuristic_),
nNodes(0),
nLeafs(0),
leafSize(leafSize_),
nBuckets(nBuckets_),
maxDepth(0),
depthGuess(std::log2(primitives_.size())),
buckets(nBuckets, std::make_pair(BoundingBox<DIM>(), 0)),
rightBucketBoxes(nBuckets, std::make_pair(BoundingBox<DIM>(), 0)),
packLeaves(packLeaves_),
primitiveTypeIsAggregate(std::is_base_of<Aggregate<DIM>, PrimitiveType>::value)
{
	using namespace std::chrono;
	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	// build sbvh
	build();

	// don't compute normals by default
	this->computeNormals = false;

	// print stats
	if (printStats_) {
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		duration<double> timeSpan = duration_cast<duration<double>>(t2 - t1);
		std::cout << "Built bvh with "
				  << nNodes << " nodes, "
				  << nLeafs << " leaves, "
				  << maxDepth << " max depth, "
				  << primitives.size() << " primitives, "
				  << references.size() << " references in "
				  << timeSpan.count() << " seconds" << std::endl;
	}
}

template<size_t DIM, typename PrimitiveType>
inline float Sbvh<DIM, PrimitiveType>::computeSplitCost(const BoundingBox<DIM>& boxLeft,
														const BoundingBox<DIM>& boxRight,
														float parentSurfaceArea, float parentVolume,
														int nReferencesLeft, int nReferencesRight,
														int depth) const
{
	float cost = maxFloat;

#ifdef BUILD_ENOKI
	if (packLeaves && depth > 0 && (depthGuess/depth) < 2 &&
		nReferencesLeft%leafSize != 0 && nReferencesRight%leafSize != 0) {
		return cost;
	}
#endif

	if (costHeuristic == CostHeuristic::SurfaceArea) {
		cost = (nReferencesLeft*boxLeft.surfaceArea() +
				nReferencesRight*boxRight.surfaceArea())/parentSurfaceArea;

	} else if (costHeuristic == CostHeuristic::OverlapSurfaceArea) {
		// set the cost to be negative if the left and right boxes don't overlap at all
		BoundingBox<DIM> boxIntersected = boxLeft.intersect(boxRight);
		cost = (nReferencesLeft/boxRight.surfaceArea() +
				nReferencesRight/boxLeft.surfaceArea())*std::fabs(boxIntersected.surfaceArea());
		if (!boxIntersected.isValid()) cost *= -1;

	} else if (costHeuristic == CostHeuristic::Volume) {
		cost = (nReferencesLeft*boxLeft.volume() +
				nReferencesRight*boxRight.volume())/parentVolume;

	} else if (costHeuristic == CostHeuristic::OverlapVolume) {
		// set the cost to be negative if the left and right boxes don't overlap at all
		BoundingBox<DIM> boxIntersected = boxLeft.intersect(boxRight);
		cost = (nReferencesLeft/boxRight.volume() +
				nReferencesRight/boxLeft.volume())*std::fabs(boxIntersected.volume());
		if (!boxIntersected.isValid()) cost *= -1;
	}

	return cost;
}

template<size_t DIM, typename PrimitiveType>
inline float Sbvh<DIM, PrimitiveType>::computeObjectSplit(const BoundingBox<DIM>& nodeBoundingBox,
														  const BoundingBox<DIM>& nodeCentroidBox,
														  const std::vector<BoundingBox<DIM>>& referenceBoxes,
														  const std::vector<Vector<DIM>>& referenceCentroids,
														  int depth, int nodeStart, int nodeEnd, int& splitDim,
														  float& splitCoord, BoundingBox<DIM>& boxIntersected)
{
	float splitCost = maxFloat;
	splitDim = -1;
	splitCoord = 0.0f;
	boxIntersected = BoundingBox<DIM>();

	if (costHeuristic != CostHeuristic::LongestAxisCenter) {
		Vector<DIM> extent = nodeBoundingBox.extent();
		float surfaceArea = nodeBoundingBox.surfaceArea();
		float volume = nodeBoundingBox.volume();

		// find the best split across all dimensions
		for (int dim = 0; dim < DIM; dim++) {
			// ignore flat dimension
			if (extent[dim] < 1e-6) continue;

			// bin references into buckets
			float bucketWidth = extent[dim]/nBuckets;
			for (int b = 0; b < nBuckets; b++) {
				buckets[b].first = BoundingBox<DIM>();
				buckets[b].second = 0;
			}

			for (int p = nodeStart; p < nodeEnd; p++) {
				int bucketIndex = (int)((referenceCentroids[p][dim] - nodeBoundingBox.pMin[dim])/bucketWidth);
				bucketIndex = clamp(bucketIndex, 0, nBuckets - 1);
				buckets[bucketIndex].first.expandToInclude(referenceBoxes[p]);
				buckets[bucketIndex].second += 1;
			}

			// sweep right to left to build right bucket bounding boxes
			BoundingBox<DIM> boxRefRight;
			for (int b = nBuckets - 1; b > 0; b--) {
				boxRefRight.expandToInclude(buckets[b].first);
				rightBucketBoxes[b].first = boxRefRight;
				rightBucketBoxes[b].second = buckets[b].second;
				if (b != nBuckets - 1) rightBucketBoxes[b].second += rightBucketBoxes[b + 1].second;
			}

			// evaluate bucket split costs
			BoundingBox<DIM> boxRefLeft;
			int nReferencesLeft = 0;
			for (int b = 1; b < nBuckets; b++) {
				boxRefLeft.expandToInclude(buckets[b - 1].first);
				nReferencesLeft += buckets[b - 1].second;

				if (nReferencesLeft > 0 && rightBucketBoxes[b].second > 0) {
					float cost = computeSplitCost(boxRefLeft, rightBucketBoxes[b].first,
												  surfaceArea, volume, nReferencesLeft,
												  rightBucketBoxes[b].second, depth);

					if (cost < splitCost) {
						splitCost = cost;
						splitDim = dim;
						splitCoord = nodeBoundingBox.pMin[dim] + b*bucketWidth;
						boxIntersected = boxRefLeft.intersect(rightBucketBoxes[b].first);
					}
				}
			}
		}

		// set split dim to max dimension when packing leaves
		if (packLeaves && splitDim == -1) splitDim = nodeCentroidBox.maxDimension();
	}

	// if no split dimension was chosen, fallback to LongestAxisCenter heuristic
	if (splitDim == -1) {
		splitDim = nodeCentroidBox.maxDimension();
		splitCoord = (nodeCentroidBox.pMin[splitDim] + nodeCentroidBox.pMax[splitDim])*0.5f;
	}

	return splitCost;
}

template<size_t DIM, typename PrimitiveType>
inline int Sbvh<DIM, PrimitiveType>::performObjectSplit(int nodeStart, int nodeEnd, int splitDim, float splitCoord,
														std::vector<BoundingBox<DIM>>& referenceBoxes,
														std::vector<Vector<DIM>>& referenceCentroids)
{
	int mid = nodeStart;
	for (int i = nodeStart; i < nodeEnd; i++) {
		if (referenceCentroids[i][splitDim] < splitCoord) {
			std::swap(references[i], references[mid]);
			std::swap(referenceBoxes[i], referenceBoxes[mid]);
			std::swap(referenceCentroids[i], referenceCentroids[mid]);
			mid++;
		}
	}

	// if we get a bad split, just choose the center...
	if (mid == nodeStart || mid == nodeEnd) {
		mid = nodeStart + (nodeEnd - nodeStart)/2;
	}

	return mid;
}

template<size_t DIM, typename PrimitiveType>
inline void Sbvh<DIM, PrimitiveType>::buildRecursive(std::vector<BoundingBox<DIM>>& referenceBoxes,
													 std::vector<Vector<DIM>>& referenceCentroids,
													 std::vector<SbvhNode<DIM>>& buildNodes,
													 int parent, int start, int end, int depth)
{
	const int Untouched    = 0xffffffff;
	const int TouchedTwice = 0xfffffffd;
	maxDepth = std::max(depth, maxDepth);

	// add node to tree
	SbvhNode<DIM> node;
	int currentNodeIndex = nNodes;
	int nReferences = end - start;

	nNodes++;

	// calculate the bounding box for this node
	BoundingBox<DIM> bb, bc;
	for (int p = start; p < end; p++) {
		bb.expandToInclude(referenceBoxes[p]);
		bc.expandToInclude(referenceCentroids[p]);
	}

	node.box = bb;

	// if the number of references at this point is less than the leaf
	// size, then this will become a leaf
	if (nReferences <= leafSize || depth == SBVH_MAX_DEPTH - 2) {
		node.referenceOffset = start;
		node.nReferences = nReferences;
		nLeafs++;

	} else {
		node.secondChildOffset = Untouched;
		node.nReferences = 0;
	}

	buildNodes.emplace_back(node);

	// child touches parent...
	// special case: don't do this for the root
	if (parent != 0xfffffffc) {
		buildNodes[parent].secondChildOffset--;

		// when this is the second touch, this is the right child;
		// the right child sets up the offset for the flat tree
		if (buildNodes[parent].secondChildOffset == TouchedTwice) {
			buildNodes[parent].secondChildOffset = nNodes - 1 - parent;
		}
	}

	// if this is a leaf, no need to subdivide
	if (node.nReferences > 0) return;

	// compute object split
	int splitDim;
	float splitCoord;
	BoundingBox<DIM> boxLeft, boxRight, boxIntersected;
	float splitCost = computeObjectSplit(bb, bc, referenceBoxes, referenceCentroids, depth,
										 start, end, splitDim, splitCoord, boxIntersected);

	// partition the list of references on split
	int nReferencesAdded = 0;
	int mid = performObjectSplit(start, end, splitDim, splitCoord, referenceBoxes, referenceCentroids);

	// push left and right children
	buildRecursive(referenceBoxes, referenceCentroids, buildNodes, currentNodeIndex, start, mid, depth + 1);
	buildRecursive(referenceBoxes, referenceCentroids, buildNodes, currentNodeIndex, mid, end, depth + 1);
}

template<size_t DIM, typename PrimitiveType>
inline void Sbvh<DIM, PrimitiveType>::build()
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	// precompute bounding boxes and centroids
	int nReferences = (int)primitives.size();
	std::vector<BoundingBox<DIM>> referenceBoxes;
	std::vector<Vector<DIM>> referenceCentroids;

	int memoryBudget = nReferences*2;
	references.resize(memoryBudget, -1);
	referenceBoxes.resize(memoryBudget);
	referenceCentroids.resize(memoryBudget);
	flatTree.reserve(nReferences*2);
	BoundingBox<DIM> boxRoot;

	for (int i = 0; i < nReferences; i++) {
		references[i] = i;
		referenceBoxes[i] = primitives[i]->boundingBox();
		referenceCentroids[i] = primitives[i]->centroid();
		boxRoot.expandToInclude(referenceBoxes[i]);
	}

	// build tree recursively
	buildRecursive(referenceBoxes, referenceCentroids, flatTree, 0xfffffffc, 0, nReferences, 0);

	// resize references vector and clear working set
	references.resize(nReferences);
	buckets.clear();
	rightBucketBoxes.clear();
}

template<size_t DIM, typename PrimitiveType>
inline BoundingBox<DIM> Sbvh<DIM, PrimitiveType>::boundingBox() const
{
	return flatTree.size() > 0 ? flatTree[0].box : BoundingBox<DIM>();
}

template<size_t DIM, typename PrimitiveType>
inline Vector<DIM> Sbvh<DIM, PrimitiveType>::centroid() const
{
	Vector<DIM> c = zeroVector<DIM>();
	int nPrimitives = (int)primitives.size();

	for (int p = 0; p < nPrimitives; p++) {
		c += primitives[p]->centroid();
	}

	return c/nPrimitives;
}

template<size_t DIM, typename PrimitiveType>
inline float Sbvh<DIM, PrimitiveType>::surfaceArea() const
{
	float area = 0.0f;
	for (int p = 0; p < (int)primitives.size(); p++) {
		area += primitives[p]->surfaceArea();
	}

	return area;
}

template<size_t DIM, typename PrimitiveType>
inline float Sbvh<DIM, PrimitiveType>::signedVolume() const
{
	float volume = 0.0f;
	for (int p = 0; p < (int)primitives.size(); p++) {
		volume += primitives[p]->signedVolume();
	}

	return volume;
}

template<size_t DIM, typename PrimitiveType>
inline bool Sbvh<DIM, PrimitiveType>::processSubtreeForIntersection(Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
																	int nodeStartIndex, bool checkOcclusion,
																	bool countHits, BvhTraversal *subtree,
																	float *boxHits, int& hits, int& nodesVisited) const
{
	int stackPtr = 0;
	while (stackPtr >= 0) {
		// pop off the next node to work on
		int nodeIndex = subtree[stackPtr].node;
		float near = subtree[stackPtr].distance;
		stackPtr--;

		// if this node is further than the closest found intersection, continue
		if (!countHits && near > r.tMax) continue;
		const SbvhNode<DIM>& node(flatTree[nodeIndex]);

		// is leaf -> intersect
		if (node.nReferences > 0) {
			for (int p = 0; p < node.nReferences; p++) {
				int index = references[node.referenceOffset + p];
				const PrimitiveType *prim = primitives[index];
				nodesVisited++;

				int hit = 0;
				int nInteractions = (int)is.size();
				std::vector<Interaction<DIM>> cs;
				if (primitiveTypeIsAggregate) {
					const Aggregate<DIM> *aggregate = reinterpret_cast<const Aggregate<DIM> *>(prim);
					hit = aggregate->intersectFromNode(r, cs, nodeStartIndex, nodesVisited, checkOcclusion, countHits);

				} else {
					hit = prim->intersect(r, cs, checkOcclusion, countHits);
				}

				// keep the closest intersection only
				if (hit > 0) {
					hits += hit;
					if (countHits) {
						is.insert(is.end(), cs.begin(), cs.end());
						for (int sp = nInteractions; sp < (int)is.size(); sp++) {
							is[sp].nodeIndex = nodeIndex;
						}

					} else {
						r.tMax = std::min(r.tMax, cs[0].d);
						is[0] = cs[0];
						is[0].nodeIndex = nodeIndex;
					}

					if (checkOcclusion) return true;
				}
			}

		} else { // not a leaf
			bool hit0 = flatTree[nodeIndex + 1].box.intersect(r, boxHits[0], boxHits[1]);
			bool hit1 = flatTree[nodeIndex + node.secondChildOffset].box.intersect(r, boxHits[2], boxHits[3]);

			// did we hit both nodes?
			if (hit0 && hit1) {
				// we assume that the left child is a closer hit...
				int closer = nodeIndex + 1;
				int other = nodeIndex + node.secondChildOffset;

				// ... if the right child was actually closer, swap the relavent values
				if (boxHits[2] < boxHits[0]) {
					std::swap(boxHits[0], boxHits[2]);
					std::swap(boxHits[1], boxHits[3]);
					std::swap(closer, other);
				}

				// it's possible that the nearest object is still in the other side, but we'll
				// check the farther-away node later...

				// push the farther first, then the closer
				stackPtr++;
				subtree[stackPtr].node = other;
				subtree[stackPtr].distance = boxHits[2];

				stackPtr++;
				subtree[stackPtr].node = closer;
				subtree[stackPtr].distance = boxHits[0];

			} else if (hit0) {
				stackPtr++;
				subtree[stackPtr].node = nodeIndex + 1;
				subtree[stackPtr].distance = boxHits[0];

			} else if (hit1) {
				stackPtr++;
				subtree[stackPtr].node = nodeIndex + node.secondChildOffset;
				subtree[stackPtr].distance = boxHits[2];
			}

			nodesVisited++;
		}
	}

	return false;
}

template<size_t DIM, typename PrimitiveType>
inline int Sbvh<DIM, PrimitiveType>::intersectFromNode(Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
													   int nodeStartIndex, int& nodesVisited,
													   bool checkOcclusion, bool countHits) const
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	// TODO: start from nodeStartIndex
	int hits = 0;
	if (!countHits) is.resize(1);
	BvhTraversal subtree[SBVH_MAX_DEPTH];
	float boxHits[4];

	if (flatTree[0].box.intersect(r, boxHits[0], boxHits[1])) {
		subtree[0].node = 0;
		subtree[0].distance = boxHits[0];
		bool occluded = processSubtreeForIntersection(r, is, nodeStartIndex, checkOcclusion,
													  countHits, subtree, boxHits, hits,
													  nodesVisited);
		if (occluded) return 1;
	}

	if (hits > 0) {
		// sort by distance and remove duplicates
		if (countHits) {
			std::sort(is.begin(), is.end(), compareInteractions<DIM>);
			is = removeDuplicates<DIM>(is);
			hits = (int)is.size();
		}

		// compute normals
		if (this->computeNormals && !primitiveTypeIsAggregate) {
			for (int i = 0; i < (int)is.size(); i++) {
				is[i].computeNormal(primitives[is[i].primitiveIndex]);
			}
		}

		return hits;
	}

	return 0;
}

template<size_t DIM, typename PrimitiveType>
inline void Sbvh<DIM, PrimitiveType>::processSubtreeForClosestPoint(BoundingSphere<DIM>& s, Interaction<DIM>& i,
																	int nodeStartIndex, const Vector<DIM>& boundaryHint,
																	BvhTraversal *subtree, float *boxHits,
																	bool& notFound, int& nodesVisited) const
{
	// TODO: use direction to boundary guess
	int stackPtr = 0;
	while (stackPtr >= 0) {
		// pop off the next node to work on
		int nodeIndex = subtree[stackPtr].node;
		float near = subtree[stackPtr].distance;
		stackPtr--;

		// if this node is further than the closest found primitive, continue
		if (near > s.r2) continue;
		const SbvhNode<DIM>& node(flatTree[nodeIndex]);

		// is leaf -> compute squared distance
		if (node.nReferences > 0) {
			for (int p = 0; p < node.nReferences; p++) {
				int index = references[node.referenceOffset + p];
				const PrimitiveType *prim = primitives[index];
				nodesVisited++;

				bool found = false;
				Interaction<DIM> c;
				if (primitiveTypeIsAggregate) {
					const Aggregate<DIM> *aggregate = reinterpret_cast<const Aggregate<DIM> *>(prim);
					found = aggregate->findClosestPointFromNode(s, c, nodeStartIndex, boundaryHint, nodesVisited);

				} else {
					found = prim->findClosestPoint(s, c);
				}

				// keep the closest point only
				if (found) {
					notFound = false;
					s.r2 = std::min(s.r2, c.d*c.d);
					i = c;
					i.nodeIndex = nodeIndex;
				}
			}

		} else { // not a leaf
			bool hit0 = flatTree[nodeIndex + 1].box.overlap(s, boxHits[0], boxHits[1]);
			s.r2 = std::min(s.r2, boxHits[1]);

			bool hit1 = flatTree[nodeIndex + node.secondChildOffset].box.overlap(s, boxHits[2], boxHits[3]);
			s.r2 = std::min(s.r2, boxHits[3]);

			// is there overlap with both nodes?
			if (hit0 && hit1) {
				// we assume that the left child is a closer hit...
				int closer = nodeIndex + 1;
				int other = nodeIndex + node.secondChildOffset;

				// ... if the right child was actually closer, swap the relavent values
				if (boxHits[0] == 0.0f && boxHits[2] == 0.0f) {
					if (boxHits[3] < boxHits[1]) {
						std::swap(closer, other);
					}

				} else if (boxHits[2] < boxHits[0]) {
					std::swap(boxHits[0], boxHits[2]);
					std::swap(closer, other);
				}

				// it's possible that the nearest object is still in the other side, but we'll
				// check the farther-away node later...

				// push the farther first, then the closer
				stackPtr++;
				subtree[stackPtr].node = other;
				subtree[stackPtr].distance = boxHits[2];

				stackPtr++;
				subtree[stackPtr].node = closer;
				subtree[stackPtr].distance = boxHits[0];

			} else if (hit0) {
				stackPtr++;
				subtree[stackPtr].node = nodeIndex + 1;
				subtree[stackPtr].distance = boxHits[0];

			} else if (hit1) {
				stackPtr++;
				subtree[stackPtr].node = nodeIndex + node.secondChildOffset;
				subtree[stackPtr].distance = boxHits[2];
			}

			nodesVisited++;
		}
	}
}

template<size_t DIM, typename PrimitiveType>
inline bool Sbvh<DIM, PrimitiveType>::findClosestPointFromNode(BoundingSphere<DIM>& s, Interaction<DIM>& i,
															   int nodeStartIndex, const Vector<DIM>& boundaryHint,
															   int& nodesVisited) const
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	// TODO: start from nodeStartIndex & use direction to boundary guess
	bool notFound = true;
	BvhTraversal subtree[SBVH_MAX_DEPTH];
	float boxHits[4];

	if (flatTree[0].box.overlap(s, boxHits[0], boxHits[1])) {
		s.r2 = std::min(s.r2, boxHits[1]);
		subtree[0].node = 0;
		subtree[0].distance = boxHits[0];
		processSubtreeForClosestPoint(s, i, nodeStartIndex, boundaryHint, subtree,
									  boxHits, notFound, nodesVisited);
	}

	if (!notFound) {
		// compute normal
		if (this->computeNormals && !primitiveTypeIsAggregate) {
			i.computeNormal(primitives[i.primitiveIndex]);
		}

		return true;
	}

	return false;
}

} // namespace fcpw
