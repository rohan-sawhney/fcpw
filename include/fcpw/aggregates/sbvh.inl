namespace fcpw {

template<size_t DIM, bool CONEDATA, typename PrimitiveType, typename SilhouetteType>
inline float Sbvh<DIM, CONEDATA, PrimitiveType, SilhouetteType>::computeSplitCost(const BoundingBox<DIM>& boxLeft,
																				  const BoundingBox<DIM>& boxRight,
																				  int nReferencesLeft, int nReferencesRight,
																				  int depth) const
{
	float cost = maxFloat;
	if (packLeaves && depth > 0 && ((float)depthGuess/depth) < 1.5f &&
		nReferencesLeft%leafSize != 0 && nReferencesRight%leafSize != 0) {
		return cost;
	}

	if (costHeuristic == CostHeuristic::SurfaceArea) {
		cost = nReferencesLeft*boxLeft.surfaceArea() + nReferencesRight*boxRight.surfaceArea();

	} else if (costHeuristic == CostHeuristic::OverlapSurfaceArea) {
		// set the cost to be negative if the left and right boxes don't overlap at all
		BoundingBox<DIM> boxIntersected = boxLeft.intersect(boxRight);
		cost = (nReferencesLeft/boxRight.surfaceArea() +
				nReferencesRight/boxLeft.surfaceArea())*std::fabs(boxIntersected.surfaceArea());
		if (!boxIntersected.isValid()) cost *= -1;

	} else if (costHeuristic == CostHeuristic::Volume) {
		cost = nReferencesLeft*boxLeft.volume() + nReferencesRight*boxRight.volume();

	} else if (costHeuristic == CostHeuristic::OverlapVolume) {
		// set the cost to be negative if the left and right boxes don't overlap at all
		BoundingBox<DIM> boxIntersected = boxLeft.intersect(boxRight);
		cost = (nReferencesLeft/boxRight.volume() +
				nReferencesRight/boxLeft.volume())*std::fabs(boxIntersected.volume());
		if (!boxIntersected.isValid()) cost *= -1;
	}

	return cost;
}

template<size_t DIM, bool CONEDATA, typename PrimitiveType, typename SilhouetteType>
inline float Sbvh<DIM, CONEDATA, PrimitiveType, SilhouetteType>::computeObjectSplit(const BoundingBox<DIM>& nodeBoundingBox,
																					const BoundingBox<DIM>& nodeCentroidBox,
																					const std::vector<BoundingBox<DIM>>& referenceBoxes,
																					const std::vector<Vector<DIM>>& referenceCentroids,
																					int depth, int nodeStart, int nodeEnd,
																					int& splitDim, float& splitCoord)
{
	float splitCost = maxFloat;
	splitDim = -1;
	splitCoord = 0.0f;

	if (costHeuristic != CostHeuristic::LongestAxisCenter) {
		Vector<DIM> extent = nodeBoundingBox.extent();

		// find the best split across all dimensions
		for (size_t dim = 0; dim < DIM; dim++) {
			// ignore flat dimension
			if (extent[dim] < 1e-6f) continue;

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
				rightBuckets[b].first = boxRefRight;
				rightBuckets[b].second = buckets[b].second;
				if (b != nBuckets - 1) rightBuckets[b].second += rightBuckets[b + 1].second;
			}

			// evaluate bucket split costs
			BoundingBox<DIM> boxRefLeft;
			int nReferencesLeft = 0;
			for (int b = 1; b < nBuckets; b++) {
				boxRefLeft.expandToInclude(buckets[b - 1].first);
				nReferencesLeft += buckets[b - 1].second;

				if (nReferencesLeft > 0 && rightBuckets[b].second > 0) {
					float cost = computeSplitCost(boxRefLeft, rightBuckets[b].first,
												  nReferencesLeft, rightBuckets[b].second,
												  depth);

					if (cost < splitCost) {
						splitCost = cost;
						splitDim = dim;
						splitCoord = nodeBoundingBox.pMin[dim] + b*bucketWidth;
					}
				}
			}
		}
	}

	// if no split dimension was chosen, fallback to LongestAxisCenter heuristic
	if (splitDim == -1) {
		splitDim = nodeCentroidBox.maxDimension();
		splitCoord = (nodeCentroidBox.pMin[splitDim] + nodeCentroidBox.pMax[splitDim])*0.5f;
	}

	return splitCost;
}

template<size_t DIM, bool CONEDATA, typename PrimitiveType, typename SilhouetteType>
inline int Sbvh<DIM, CONEDATA, PrimitiveType, SilhouetteType>::performObjectSplit(int nodeStart, int nodeEnd, int splitDim, float splitCoord,
																				  std::vector<BoundingBox<DIM>>& referenceBoxes,
																				  std::vector<Vector<DIM>>& referenceCentroids)
{
	int mid = nodeStart;
	for (int i = nodeStart; i < nodeEnd; i++) {
		if (referenceCentroids[i][splitDim] < splitCoord) {
			std::swap(primitives[i], primitives[mid]);
			std::swap(referenceBoxes[i], referenceBoxes[mid]);
			std::swap(referenceCentroids[i], referenceCentroids[mid]);
			mid++;
		}
	}

	// if we get a bad split, just choose the center...
	if (mid == nodeStart || mid == nodeEnd) {
		mid = nodeStart + (nodeEnd - nodeStart)/2;

		// ensure the number of primitives in one branch is a multiple of the leaf size
		if (packLeaves) {
			while ((mid - nodeStart)%leafSize != 0 && mid < nodeEnd) mid++;
			if (mid == nodeEnd) mid = nodeStart + (nodeEnd - nodeStart)/2;
		}
	}

	return mid;
}

template<size_t DIM, bool CONEDATA, typename PrimitiveType, typename SilhouetteType>
inline void Sbvh<DIM, CONEDATA, PrimitiveType, SilhouetteType>::buildRecursive(std::vector<BoundingBox<DIM>>& referenceBoxes,
																			   std::vector<Vector<DIM>>& referenceCentroids,
																			   std::vector<SbvhNode<DIM, CONEDATA>>& buildNodes,
																			   int parent, int start, int end, int depth)
{
	const int Untouched    = 0xffffffff;
	const int TouchedTwice = 0xfffffffd;
	maxDepth = std::max(depth, maxDepth);

	// add node to tree
	SbvhNode<DIM, CONEDATA> node;
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
	if (nReferences <= leafSize || depth == FCPW_SBVH_MAX_DEPTH - 2) {
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
	float splitCost = computeObjectSplit(bb, bc, referenceBoxes, referenceCentroids,
										 depth, start, end, splitDim, splitCoord);

	// partition the list of references on split
	int mid = performObjectSplit(start, end, splitDim, splitCoord, referenceBoxes, referenceCentroids);

	// push left and right children
	buildRecursive(referenceBoxes, referenceCentroids, buildNodes, currentNodeIndex, start, mid, depth + 1);
	buildRecursive(referenceBoxes, referenceCentroids, buildNodes, currentNodeIndex, mid, end, depth + 1);
}

template<size_t DIM, bool CONEDATA, typename PrimitiveType, typename SilhouetteType>
inline void Sbvh<DIM, CONEDATA, PrimitiveType, SilhouetteType>::build()
{
	// precompute bounding boxes and centroids
	int nReferences = (int)primitives.size();
	std::vector<BoundingBox<DIM>> referenceBoxes;
	std::vector<Vector<DIM>> referenceCentroids;

	referenceBoxes.resize(nReferences);
	referenceCentroids.resize(nReferences);
	flatTree.reserve(nReferences*2);

	for (int i = 0; i < nReferences; i++) {
		referenceBoxes[i] = primitives[i]->boundingBox();
		referenceCentroids[i] = primitives[i]->centroid();
	}

	// build tree recursively
	buildRecursive(referenceBoxes, referenceCentroids, flatTree, 0xfffffffc, 0, nReferences, 0);

	// clear working set
	buckets.clear();
	rightBuckets.clear();
}

template<size_t DIM, typename SilhouetteType>
inline void computeBoundingConesRecursive(const std::vector<SilhouetteType *>& silhouetteRefs,
										  const std::vector<Vector<DIM>>& silhouetteNormals,
										  const std::vector<Vector<DIM>>& silhouetteFaceNormals,
										  std::vector<SbvhNode<DIM, true>>& flatTree, int start, int end)
{
	BoundingCone<DIM> cone;
	SbvhNode<DIM, true>& node(flatTree[start]);

	// compute bounding cone axis
	bool anySilhouetteRefs = false;
	for (int i = start; i < end; i++) {
		SbvhNode<DIM, true>& childNode(flatTree[i]);

		for (int j = 0; j < childNode.nSilhouetteReferences; j++) { // is leaf if nSilhouetteReferences > 0
			int referenceIndex = childNode.silhouetteReferenceOffset + j;
			cone.axis += silhouetteNormals[referenceIndex];
			anySilhouetteRefs = true;
		}
	}

	// compute bounding cone angle
	if (!anySilhouetteRefs) {
 		node.cone.halfAngle = -M_PI;

	} else {
		float axisNorm = cone.axis.norm();
		if (axisNorm > epsilon) {
			cone.axis /= axisNorm;
			cone.halfAngle = 0.0f;

			for (int i = start; i < end; i++) {
				SbvhNode<DIM, true>& childNode(flatTree[i]);

				for (int j = 0; j < childNode.nSilhouetteReferences; j++) { // is leaf if nSilhouetteReferences > 0
					int referenceIndex = childNode.silhouetteReferenceOffset + j;
					const SilhouetteType *silhouette = silhouetteRefs[referenceIndex];

					for (int k = 0; k < 2; k++) {
						if (silhouette->hasFace(k)) {
							const Vector<DIM>& n = silhouetteFaceNormals[2*referenceIndex + k];
							float angle = std::acos(std::max(-1.0f, std::min(1.0f, cone.axis.dot(n))));
							cone.halfAngle = std::max(cone.halfAngle, angle);
						}
					}
				}
			}

			node.cone = cone;
		}
	}

	// recurse on children
	if (node.nReferences == 0) { // not a leaf
		computeBoundingConesRecursive<DIM, SilhouetteType>(silhouetteRefs, silhouetteNormals, silhouetteFaceNormals,
														   flatTree, start + 1, start + node.secondChildOffset);
		computeBoundingConesRecursive<DIM, SilhouetteType>(silhouetteRefs, silhouetteNormals, silhouetteFaceNormals,
														   flatTree, start + node.secondChildOffset, end);
	}
}

template<size_t DIM, bool CONEDATA, typename PrimitiveType, typename SilhouetteType>
inline void assignSilhouettesToNodes(const std::vector<PrimitiveType *>& primitives,
									 const std::vector<SilhouetteType *>& silhouettes,
									 const std::function<bool(float, int)>& ignoreSilhouette,
									 std::vector<SbvhNode<DIM, CONEDATA>>& flatTree,
									 std::vector<SilhouetteType *>& silhouetteRefs)
{
	// do nothing
}

template<>
inline void assignSilhouettesToNodes<3, true, LineSegment, SilhouetteVertex>(const std::vector<LineSegment *>& lineSegments,
																			 const std::vector<SilhouetteVertex *>& silhouetteVertices,
																			 const std::function<bool(float, int)>& ignoreSilhouette,
																			 std::vector<SbvhNode<3, true>>& flatTree,
																			 std::vector<SilhouetteVertex *>& silhouetteVertexRefs)
{
	// collect silhouette references
	Vector3 zero = Vector3::Zero();
	std::vector<Vector3> silhouetteRefNormals, silhouetteRefFaceNormals;

	for (int i = 0; i < (int)flatTree.size(); i++) {
		SbvhNode<3, true>& node(flatTree[i]);
		std::unordered_map<int, bool> seenVertex;
		int start = (int)silhouetteVertexRefs.size();

		for (int j = 0; j < node.nReferences; j++) { // leaf node if nReferences > 0
			int referenceIndex = node.referenceOffset + j;
			LineSegment *lineSegment = lineSegments[referenceIndex];

			for (int k = 0; k < 2; k++) {
				int vIndex = lineSegment->indices[k];
				SilhouetteVertex *silhouetteVertex = silhouetteVertices[vIndex];

				if (seenVertex.find(vIndex) == seenVertex.end()) {
					seenVertex[vIndex] = true;
					Vector3 n = silhouetteVertex->normal();
					Vector3 n0 = silhouetteVertex->hasFace(0) ? silhouetteVertex->normal(0, true) : zero;
					Vector3 n1 = silhouetteVertex->hasFace(1) ? silhouetteVertex->normal(1, true) : zero;

					bool ignore = false;
	 				if (ignoreSilhouette && silhouetteVertex->hasFace(0) && silhouetteVertex->hasFace(1)) {
						float det = n0.x()*n1.y() - n1.x()*n0.y();
						ignore = ignoreSilhouette(det, lineSegment->pIndex);
					}

					if (!ignore) {
						silhouetteVertexRefs.emplace_back(silhouetteVertex);
						silhouetteRefNormals.emplace_back(n);
						silhouetteRefFaceNormals.emplace_back(n0);
						silhouetteRefFaceNormals.emplace_back(n1);
					}
				}
			}
		}

		int end = (int)silhouetteVertexRefs.size();
		node.silhouetteReferenceOffset = start;
		node.nSilhouetteReferences = end - start;
	}

	// compute bounding cones recursively
	computeBoundingConesRecursive<3, SilhouetteVertex>(silhouetteVertexRefs, silhouetteRefNormals,
													   silhouetteRefFaceNormals, flatTree,
													   0, (int)flatTree.size());
}

template<>
inline void assignSilhouettesToNodes<3, true, Triangle, SilhouetteEdge>(const std::vector<Triangle *>& triangles,
																		const std::vector<SilhouetteEdge *>& silhouetteEdges,
																		const std::function<bool(float, int)>& ignoreSilhouette,
																		std::vector<SbvhNode<3, true>>& flatTree,
																		std::vector<SilhouetteEdge *>& silhouetteEdgeRefs)
{
	// collect silhouette references
	Vector3 zero = Vector3::Zero();
	std::vector<Vector3> silhouetteRefNormals, silhouetteRefFaceNormals;

	for (int i = 0; i < (int)flatTree.size(); i++) {
		SbvhNode<3, true>& node(flatTree[i]);
		std::unordered_map<int, bool> seenEdge;
		int start = (int)silhouetteEdgeRefs.size();

		for (int j = 0; j < node.nReferences; j++) { // leaf node if nReferences > 0
			int referenceIndex = node.referenceOffset + j;
			Triangle *triangle = triangles[referenceIndex];

			for (int k = 0; k < 3; k++) {
				int eIndex = triangle->soup->eIndices[3*triangle->pIndex + k];
				SilhouetteEdge *silhouetteEdge = silhouetteEdges[eIndex];

				if (seenEdge.find(eIndex) == seenEdge.end()) {
					seenEdge[eIndex] = true;
					Vector3 n = silhouetteEdge->normal();
					Vector3 n0 = silhouetteEdge->hasFace(0) ? silhouetteEdge->normal(0, true) : zero;
					Vector3 n1 = silhouetteEdge->hasFace(1) ? silhouetteEdge->normal(1, true) : zero;

					bool ignore = false;
	 				if (ignoreSilhouette && silhouetteEdge->hasFace(0) && silhouetteEdge->hasFace(1)) {
						const Vector3& pa = silhouetteEdge->soup->positions[silhouetteEdge->indices[1]];
	 					const Vector3& pb = silhouetteEdge->soup->positions[silhouetteEdge->indices[2]];
	 					Vector3 edgeDir = (pb - pa).normalized();
						float dihedralAngle = std::atan2(edgeDir.dot(n0.cross(n1)), n0.dot(n1));
						ignore = ignoreSilhouette(dihedralAngle, triangle->pIndex);
					}

					if (!ignore) {
						silhouetteEdgeRefs.emplace_back(silhouetteEdge);
						silhouetteRefNormals.emplace_back(n);
						silhouetteRefFaceNormals.emplace_back(n0);
						silhouetteRefFaceNormals.emplace_back(n1);
					}
				}
			}
		}

		int end = (int)silhouetteEdgeRefs.size();
		node.silhouetteReferenceOffset = start;
		node.nSilhouetteReferences = end - start;
	}

	// compute bounding cones recursively
	computeBoundingConesRecursive<3, SilhouetteEdge>(silhouetteEdgeRefs, silhouetteRefNormals,
													 silhouetteRefFaceNormals, flatTree,
													 0, (int)flatTree.size());
}

template<size_t DIM, bool CONEDATA, typename PrimitiveType, typename SilhouetteType>
inline Sbvh<DIM, CONEDATA, PrimitiveType, SilhouetteType>::Sbvh(const CostHeuristic& costHeuristic_,
																std::vector<PrimitiveType *>& primitives_,
																std::vector<SilhouetteType *>& silhouettes_,
																SortPositionsFunc<DIM, CONEDATA, PrimitiveType, SilhouetteType> sortPositions_,
																const std::function<bool(float, int)>& ignoreSilhouette_,
																bool printStats_, bool packLeaves_, int leafSize_, int nBuckets_):
costHeuristic(costHeuristic_),
nNodes(0),
nLeafs(0),
leafSize(leafSize_),
nBuckets(nBuckets_),
maxDepth(0),
depthGuess(std::log2(primitives_.size())),
buckets(nBuckets, std::make_pair(BoundingBox<DIM>(), 0)),
rightBuckets(nBuckets, std::make_pair(BoundingBox<DIM>(), 0)),
primitives(primitives_),
silhouettes(silhouettes_),
packLeaves(packLeaves_),
primitiveTypeIsAggregate(std::is_base_of<Aggregate<DIM>, PrimitiveType>::value)
{
	using namespace std::chrono;
	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	// build sbvh
	build();

	// sort positions
	if (sortPositions_) {
		sortPositions_(flatTree, primitives, silhouettes);
	}

	// assign silhouettes to nodes
	assignSilhouettesToNodes<DIM, CONEDATA, PrimitiveType, SilhouetteType>(primitives, silhouettes, ignoreSilhouette_,
																		   flatTree, silhouetteRefs);

	// print stats
	if (printStats_) {
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		duration<double> timeSpan = duration_cast<duration<double>>(t2 - t1);
		std::cout << "Built bvh with "
				  << nNodes << " nodes, "
				  << nLeafs << " leaves, "
				  << maxDepth << " max depth, "
				  << primitives.size() << " primitives, "
				  << silhouettes.size() << " silhouettes and "
				  << silhouetteRefs.size() << " silhouetteRefs in "
				  << timeSpan.count() << " seconds" << std::endl;
	}
}

template<size_t DIM, bool CONEDATA, typename PrimitiveType, typename SilhouetteType>
inline BoundingBox<DIM> Sbvh<DIM, CONEDATA, PrimitiveType, SilhouetteType>::boundingBox() const
{
	return flatTree.size() > 0 ? flatTree[0].box : BoundingBox<DIM>();
}

template<size_t DIM, bool CONEDATA, typename PrimitiveType, typename SilhouetteType>
inline Vector<DIM> Sbvh<DIM, CONEDATA, PrimitiveType, SilhouetteType>::centroid() const
{
	Vector<DIM> c = Vector<DIM>::Zero();
	int nPrimitives = (int)primitives.size();

	for (int p = 0; p < nPrimitives; p++) {
		c += primitives[p]->centroid();
	}

	return c/nPrimitives;
}

template<size_t DIM, bool CONEDATA, typename PrimitiveType, typename SilhouetteType>
inline float Sbvh<DIM, CONEDATA, PrimitiveType, SilhouetteType>::surfaceArea() const
{
	float area = 0.0f;
	for (int p = 0; p < (int)primitives.size(); p++) {
		area += primitives[p]->surfaceArea();
	}

	return area;
}

template<size_t DIM, bool CONEDATA, typename PrimitiveType, typename SilhouetteType>
inline float Sbvh<DIM, CONEDATA, PrimitiveType, SilhouetteType>::signedVolume() const
{
	float volume = 0.0f;
	for (int p = 0; p < (int)primitives.size(); p++) {
		volume += primitives[p]->signedVolume();
	}

	return volume;
}

template<size_t DIM, bool CONEDATA, typename PrimitiveType, typename SilhouetteType>
inline bool Sbvh<DIM, CONEDATA, PrimitiveType, SilhouetteType>::processSubtreeForIntersection(Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
																							  int nodeStartIndex, int aggregateIndex, bool checkForOcclusion,
																							  bool recordAllHits, BvhTraversal *subtree,
																							  float *boxHits, int& hits, int& nodesVisited) const
{
	int stackPtr = 0;
	while (stackPtr >= 0) {
		// pop off the next node to work on
		int nodeIndex = subtree[stackPtr].node;
		float currentDist = subtree[stackPtr].distance;
		stackPtr--;

		// if this node is further than the closest found intersection, continue
		if (!recordAllHits && currentDist > r.tMax) continue;
		const SbvhNode<DIM, CONEDATA>& node(flatTree[nodeIndex]);

		// is leaf -> intersect
		if (node.nReferences > 0) {
			for (int p = 0; p < node.nReferences; p++) {
				int referenceIndex = node.referenceOffset + p;
				const PrimitiveType *prim = primitives[referenceIndex];
				nodesVisited++;

				int hit = 0;
				std::vector<Interaction<DIM>> cs;
				if (primitiveTypeIsAggregate) {
					const Aggregate<DIM> *aggregate = reinterpret_cast<const Aggregate<DIM> *>(prim);
					hit = aggregate->intersectFromNode(r, cs, nodeStartIndex, aggregateIndex,
													   nodesVisited, checkForOcclusion, recordAllHits);

				} else {
					hit = prim->intersect(r, cs, checkForOcclusion, recordAllHits);
					for (int i = 0; i < (int)cs.size(); i++) {
						cs[i].nodeIndex = nodeIndex;
						cs[i].referenceIndex = referenceIndex;
						cs[i].objectIndex = this->index;
					}
				}

				// keep the closest intersection only
				if (hit > 0) {
					if (checkForOcclusion) {
						return true;
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

template<size_t DIM, bool CONEDATA, typename PrimitiveType, typename SilhouetteType>
inline int Sbvh<DIM, CONEDATA, PrimitiveType, SilhouetteType>::intersectFromNode(Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
																				 int nodeStartIndex, int aggregateIndex, int& nodesVisited,
																				 bool checkForOcclusion, bool recordAllHits) const
{
	int hits = 0;
	if (!recordAllHits) is.resize(1);
	BvhTraversal subtree[FCPW_SBVH_MAX_DEPTH];
	float boxHits[4];

	int rootIndex = aggregateIndex == this->index ? nodeStartIndex : 0;
	if (flatTree[rootIndex].box.intersect(r, boxHits[0], boxHits[1])) {
		subtree[0].node = rootIndex;
		subtree[0].distance = boxHits[0];
		bool occluded = processSubtreeForIntersection(r, is, nodeStartIndex, aggregateIndex, checkForOcclusion,
													  recordAllHits, subtree, boxHits, hits, nodesVisited);
		if (occluded) return 1;
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

		return hits;
	}

	return 0;
}

template<size_t DIM, bool CONEDATA, typename PrimitiveType, typename SilhouetteType>
inline float Sbvh<DIM, CONEDATA, PrimitiveType, SilhouetteType>::processSubtreeForIntersection(const BoundingSphere<DIM>& s, std::vector<Interaction<DIM>>& is,
																							   int nodeStartIndex, int aggregateIndex, bool recordOneHit,
																							   const std::function<float(float)>& primitiveWeight,
																							   BvhTraversal *subtree, float *boxHits, int& hits, int& nodesVisited) const
{
	float totalPrimitiveWeight = 0.0f;
	int stackPtr = 0;

	while (stackPtr >= 0) {
		// pop off the next node to work on
		int nodeIndex = subtree[stackPtr].node;
		const SbvhNode<DIM, CONEDATA>& node(flatTree[nodeIndex]);
		stackPtr--;

		// is leaf -> intersect
		if (node.nReferences > 0) {
			for (int p = 0; p < node.nReferences; p++) {
				int referenceIndex = node.referenceOffset + p;
				const PrimitiveType *prim = primitives[referenceIndex];
				nodesVisited++;

				int hit = 0;
				std::vector<Interaction<DIM>> cs;
				if (primitiveTypeIsAggregate) {
					const Aggregate<DIM> *aggregate = reinterpret_cast<const Aggregate<DIM> *>(prim);
					hit = aggregate->intersectFromNode(s, cs, nodeStartIndex, aggregateIndex,
													   nodesVisited, recordOneHit, primitiveWeight);

				} else {
					hit = prim->intersect(s, cs, recordOneHit, primitiveWeight);
					for (int i = 0; i < (int)cs.size(); i++) {
						cs[i].nodeIndex = nodeIndex;
						cs[i].referenceIndex = referenceIndex;
						cs[i].objectIndex = this->index;
					}
				}

				if (hit > 0) {
					hits += hit;
					if (recordOneHit && !primitiveTypeIsAggregate) {
						totalPrimitiveWeight += cs[0].d;
						if (uniformRealRandomNumber()*totalPrimitiveWeight < cs[0].d) {
							is[0] = cs[0];
						}

					} else {
						is.insert(is.end(), cs.begin(), cs.end());
					}
				}
			}

		} else { // not a leaf
			bool hit0 = flatTree[nodeIndex + 1].box.overlap(s, boxHits[0], boxHits[1]);
			bool hit1 = flatTree[nodeIndex + node.secondChildOffset].box.overlap(s, boxHits[2], boxHits[3]);

			if (hit0) {
				stackPtr++;
				subtree[stackPtr].node = nodeIndex + 1;
			}

			if (hit1) {
				stackPtr++;
				subtree[stackPtr].node = nodeIndex + node.secondChildOffset;
			}

			nodesVisited++;
		}
	}

	return totalPrimitiveWeight;
}

template<size_t DIM, bool CONEDATA, typename PrimitiveType, typename SilhouetteType>
inline int Sbvh<DIM, CONEDATA, PrimitiveType, SilhouetteType>::intersectFromNode(const BoundingSphere<DIM>& s,
																				 std::vector<Interaction<DIM>>& is,
																				 int nodeStartIndex, int aggregateIndex,
																				 int& nodesVisited, bool recordOneHit,
																				 const std::function<float(float)>& primitiveWeight) const
{
	int hits = 0;
	float totalPrimitiveWeight = 0.0f;
	if (recordOneHit && !primitiveTypeIsAggregate) is.resize(1);
	BvhTraversal subtree[FCPW_SBVH_MAX_DEPTH];
	float boxHits[4];

	int rootIndex = aggregateIndex == this->index ? nodeStartIndex : 0;
	if (flatTree[rootIndex].box.overlap(s, boxHits[0], boxHits[1])) {
		subtree[0].node = rootIndex;
		subtree[0].distance = s.r2;
		totalPrimitiveWeight = processSubtreeForIntersection(s, is, nodeStartIndex, aggregateIndex, recordOneHit,
															 primitiveWeight, subtree, boxHits, hits, nodesVisited);
	}

	if (hits > 0) {
		if (recordOneHit && !primitiveTypeIsAggregate) {
			if (is[0].primitiveIndex == -1) {
				hits = 0;
				is.clear();

			} else if (totalPrimitiveWeight > 0.0f) {
				is[0].d /= totalPrimitiveWeight;
			}
		}

		return hits;
	}

	return 0;
}

template<size_t DIM, bool CONEDATA, typename PrimitiveType, typename SilhouetteType>
inline void Sbvh<DIM, CONEDATA, PrimitiveType, SilhouetteType>::processSubtreeForIntersection(const BoundingSphere<DIM>& s, std::vector<Interaction<DIM>>& is,
																							  float *randNums, int nodeStartIndex, int aggregateIndex,
																							  const std::function<float(float)>& traversalWeight,
																							  const std::function<float(float)>& primitiveWeight,
																							  int nodeIndex, float traversalPdf, float *boxHits,
																							  int& hits, int& nodesVisited) const
{
	int stackPtr = 0;
	float d2NodeMax = boxHits[1];
	float u = randNums[0];

	while (stackPtr >= 0) {
		// pop off the next node to work on
		const SbvhNode<DIM, CONEDATA>& node(flatTree[nodeIndex]);
		stackPtr--;

		// is leaf -> intersect
		if (node.nReferences > 0) {
			float totalPrimitiveWeight = 0.0f;
			for (int p = 0; p < node.nReferences; p++) {
				int referenceIndex = node.referenceOffset + p;
				const PrimitiveType *prim = primitives[referenceIndex];
				nodesVisited++;

				int hit = 0;
				std::vector<Interaction<DIM>> cs;
				if (primitiveTypeIsAggregate) {
					float modifiedRandNums[DIM];
					modifiedRandNums[0] = u;
					for (int i = 1; i < DIM; i++) modifiedRandNums[i] = randNums[i];
					const Aggregate<DIM> *aggregate = reinterpret_cast<const Aggregate<DIM> *>(prim);
					hit = aggregate->intersectStochasticFromNode(s, cs, modifiedRandNums, nodeStartIndex, aggregateIndex,
																 nodesVisited, traversalWeight, primitiveWeight);

				} else {
					if (d2NodeMax <= s.r2) {
						hit = 1;
						auto it = cs.emplace(cs.end(), Interaction<3>());
						it->primitiveIndex = reinterpret_cast<const GeometricPrimitive<DIM> *>(prim)->pIndex;
						it->d = prim->surfaceArea();
						if (primitiveWeight) {
							float d2 = (s.c - prim->centroid()).squaredNorm();
							it->d *= primitiveWeight(d2);
						}

					} else {
						hit = prim->intersect(s, cs, true, primitiveWeight);
					}

					for (int i = 0; i < (int)cs.size(); i++) {
						cs[i].nodeIndex = nodeIndex;
						cs[i].referenceIndex = referenceIndex;
						cs[i].objectIndex = this->index;
					}
				}

				if (hit > 0) {
					hits += hit;
					if (!primitiveTypeIsAggregate) {
						totalPrimitiveWeight += cs[0].d;
						float selectionProb = cs[0].d/totalPrimitiveWeight;

						if (u < selectionProb) {
							u = u/selectionProb; // rescale to [0,1)
							is[0] = cs[0];
							is[0].d *= traversalPdf;

						} else {
							u = (u - selectionProb)/(1.0f - selectionProb);
						}

					} else {
						int nInteractions = (int)is.size();
						is.insert(is.end(), cs.begin(), cs.end());
						for (int i = nInteractions; i < (int)is.size(); i++) {
							is[i].d *= traversalPdf;
						}
					}
				}
			}

			if (!primitiveTypeIsAggregate) {
				if (totalPrimitiveWeight > 0.0f) {
					is[0].d /= totalPrimitiveWeight;
				}
			}

		} else { // not a leaf
			const BoundingBox<DIM>& box0(flatTree[nodeIndex + 1].box);
			float weight0 = box0.overlap(s, boxHits[0], boxHits[1]) ? 1.0f : 0.0f;

			const BoundingBox<DIM>& box1(flatTree[nodeIndex + node.secondChildOffset].box);
			float weight1 = box1.overlap(s, boxHits[2], boxHits[3]) ? 1.0f : 0.0f;

			if (traversalWeight) {
				if (weight0 > 0.0f) weight0 *= traversalWeight((s.c - box0.centroid()).squaredNorm());
				if (weight1 > 0.0f) weight1 *= traversalWeight((s.c - box1.centroid()).squaredNorm());
			}

			float totalTraversalWeight = weight0 + weight1;
			if (totalTraversalWeight > 0.0f) {
				stackPtr++;
				float traversalProb0 = weight0/totalTraversalWeight;
				float traversalProb1 = 1.0f - traversalProb0;

				if (u < traversalProb0) {
					u = u/traversalProb0; // rescale to [0,1)
					nodeIndex = nodeIndex + 1;
					traversalPdf *= traversalProb0;
					d2NodeMax = boxHits[1];

				} else {
					u = (u - traversalProb0)/traversalProb1; // rescale to [0,1)
					nodeIndex = nodeIndex + node.secondChildOffset;
					traversalPdf *= traversalProb1;
					d2NodeMax = boxHits[3];
				}
			}

			nodesVisited++;
		}
	}
}

template<size_t DIM, bool CONEDATA, typename PrimitiveType, typename SilhouetteType>
inline int Sbvh<DIM, CONEDATA, PrimitiveType, SilhouetteType>::intersectStochasticFromNode(const BoundingSphere<DIM>& s,
																						   std::vector<Interaction<DIM>>& is, float *randNums,
																						   int nodeStartIndex, int aggregateIndex, int& nodesVisited,
																						   const std::function<float(float)>& traversalWeight,
																						   const std::function<float(float)>& primitiveWeight) const
{
	int hits = 0;
	if (!primitiveTypeIsAggregate) is.resize(1);
	float boxHits[4];

	int rootIndex = aggregateIndex == this->index ? nodeStartIndex : 0;
	if (flatTree[rootIndex].box.overlap(s, boxHits[0], boxHits[1])) {
		processSubtreeForIntersection(s, is, randNums, nodeStartIndex, aggregateIndex, traversalWeight,
									  primitiveWeight, rootIndex, 1.0f, boxHits, hits, nodesVisited);
	}

	if (hits > 0) {
		if (!primitiveTypeIsAggregate) {
			if (is[0].primitiveIndex == -1) {
				hits = 0;
				is.clear();

			} else {
				// sample a point on the selected geometric primitive
				const PrimitiveType *prim = primitives[is[0].referenceIndex];
				float pdf = is[0].samplePoint(prim, randNums);
				is[0].d *= pdf;
			}
		}

		return hits;
	}

	return 0;
}

template<size_t DIM, bool CONEDATA, typename PrimitiveType, typename SilhouetteType>
inline void Sbvh<DIM, CONEDATA, PrimitiveType, SilhouetteType>::processSubtreeForClosestPoint(BoundingSphere<DIM>& s, Interaction<DIM>& i,
																							  int nodeStartIndex, int aggregateIndex,
																							  bool recordNormal, BvhTraversal *subtree,
																							  float *boxHits, bool& notFound, int& nodesVisited) const
{
	int stackPtr = 0;
	while (stackPtr >= 0) {
		// pop off the next node to work on
		int nodeIndex = subtree[stackPtr].node;
		float currentDist = subtree[stackPtr].distance;
		stackPtr--;

		// if this node is further than the closest found primitive, continue
		if (currentDist > s.r2) continue;
		const SbvhNode<DIM, CONEDATA>& node(flatTree[nodeIndex]);

		// is leaf -> compute squared distance
		if (node.nReferences > 0) {
			for (int p = 0; p < node.nReferences; p++) {
				int referenceIndex = node.referenceOffset + p;
				const PrimitiveType *prim = primitives[referenceIndex];
				nodesVisited++;

				bool found = false;
				Interaction<DIM> c;

				if (primitiveTypeIsAggregate) {
					const Aggregate<DIM> *aggregate = reinterpret_cast<const Aggregate<DIM> *>(prim);
					found = aggregate->findClosestPointFromNode(s, c, nodeStartIndex, aggregateIndex,
																nodesVisited, recordNormal);

				} else {
					found = prim->findClosestPoint(s, c, recordNormal);
					c.nodeIndex = nodeIndex;
					c.referenceIndex = referenceIndex;
					c.objectIndex = this->index;
				}

				// keep the closest point only
				if (found) {
					notFound = false;
					s.r2 = std::min(s.r2, c.d*c.d);
					i = c;
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

template<size_t DIM, bool CONEDATA, typename PrimitiveType, typename SilhouetteType>
inline bool Sbvh<DIM, CONEDATA, PrimitiveType, SilhouetteType>::findClosestPointFromNode(BoundingSphere<DIM>& s, Interaction<DIM>& i,
																						 int nodeStartIndex, int aggregateIndex,
																						 int& nodesVisited, bool recordNormal) const
{
	bool notFound = true;
	BvhTraversal subtree[FCPW_SBVH_MAX_DEPTH];
	float boxHits[4];

	int rootIndex = aggregateIndex == this->index ? nodeStartIndex : 0;
	if (flatTree[rootIndex].box.overlap(s, boxHits[0], boxHits[1])) {
		s.r2 = std::min(s.r2, boxHits[1]);
		subtree[0].node = rootIndex;
		subtree[0].distance = boxHits[0];
		processSubtreeForClosestPoint(s, i, nodeStartIndex, aggregateIndex, recordNormal,
									  subtree, boxHits, notFound, nodesVisited);
	}

	if (!notFound) {
		// compute normal
		if (recordNormal && !primitiveTypeIsAggregate) {
			i.computeNormal(primitives[i.referenceIndex]);
		}

		return true;
	}

	return false;
}

template<size_t DIM, typename PrimitiveType, typename SilhouetteType>
inline void processSubtreeForClosestSilhouettePoint(const std::vector<SbvhNode<DIM, false>>& flatTree,
													const std::vector<PrimitiveType *>& primitives,
													const std::vector<SilhouetteType *>& silhouetteRefs,
													BoundingSphere<DIM>& s, Interaction<DIM>& i,
													int nodeStartIndex, int aggregateIndex, int objectIndex,
													bool primitiveTypeIsAggregate, bool flipNormalOrientation,
													float squaredMinRadius, float precision, bool recordNormal,
													BvhTraversal *subtree, float *boxHits, bool& notFound,
													int& nodesVisited)
{
	std::cerr << "Sbvh::processSubtreeForClosestSilhouettePoint() not supported without cone data" << std::endl;
	exit(EXIT_FAILURE);
}

template<size_t DIM, typename PrimitiveType, typename SilhouetteType>
inline void processSubtreeForClosestSilhouettePoint(const std::vector<SbvhNode<DIM, true>>& flatTree,
													const std::vector<PrimitiveType *>& primitives,
													const std::vector<SilhouetteType *>& silhouetteRefs,
													BoundingSphere<DIM>& s, Interaction<DIM>& i,
													int nodeStartIndex, int aggregateIndex, int objectIndex,
													bool primitiveTypeIsAggregate, bool flipNormalOrientation,
													float squaredMinRadius, float precision, bool recordNormal,
													BvhTraversal *subtree, float *boxHits, bool& notFound,
													int& nodesVisited)
{
	int stackPtr = 0;
	while (stackPtr >= 0) {
		// pop off the next node to work on
		int nodeIndex = subtree[stackPtr].node;
		float currentDist = subtree[stackPtr].distance;
		stackPtr--;

		// if this node is further than the closest found silhouette, continue
		if (currentDist > s.r2) continue;
		const SbvhNode<DIM, true>& node(flatTree[nodeIndex]);

		// is leaf -> compute silhouette distance
		if (node.nReferences > 0) {
			if (primitiveTypeIsAggregate) {
				for (int p = 0; p < node.nReferences; p++) {
					int referenceIndex = node.referenceOffset + p;
					const PrimitiveType *prim = primitives[referenceIndex];
					nodesVisited++;

					Interaction<DIM> c;
					const Aggregate<DIM> *aggregate = reinterpret_cast<const Aggregate<DIM> *>(prim);
					bool found = aggregate->findClosestSilhouettePointFromNode(s, c, nodeStartIndex, aggregateIndex,
																			   nodesVisited, flipNormalOrientation,
																			   squaredMinRadius, precision, recordNormal);

					// keep the closest silhouette point
					if (found) {
						notFound = false;
						s.r2 = std::min(s.r2, c.d*c.d);
						i = c;

						if (squaredMinRadius >= s.r2) {
							break;
						}
					}
				}

			} else {
				for (int p = 0; p < node.nSilhouetteReferences; p++) {
					int referenceIndex = node.silhouetteReferenceOffset + p;
					const SilhouetteType *silhouette = silhouetteRefs[referenceIndex];

					// skip query if silhouette index is the same as i.primitiveIndex (and object indices match)
					int primitiveIndex = static_cast<const SilhouettePrimitive<DIM> *>(silhouette)->pIndex;
					if (primitiveIndex == i.primitiveIndex && objectIndex == i.objectIndex) continue;
					nodesVisited++;

					Interaction<DIM> c;
					bool found = silhouette->findClosestSilhouettePoint(s, c, flipNormalOrientation, squaredMinRadius,
																		precision, recordNormal);

					// keep the closest silhouette point
					if (found) {
						notFound = false;
						s.r2 = std::min(s.r2, c.d*c.d);
						i = c;
						i.nodeIndex = nodeIndex;
						i.referenceIndex = referenceIndex;
						i.objectIndex = objectIndex;

						if (squaredMinRadius >= s.r2) {
							break;
						}
					}
				}
			}

		} else { // not a leaf
			const SbvhNode<DIM, true>& node0(flatTree[nodeIndex + 1]);
			bool hit0 = node0.cone.isValid() && node0.box.overlap(s, boxHits[0]) &&
						node0.cone.overlap(s.c, node0.box, boxHits[0]);

			const SbvhNode<DIM, true>& node1(flatTree[nodeIndex + node.secondChildOffset]);
			bool hit1 = node1.cone.isValid() && node1.box.overlap(s, boxHits[1]) &&
						node1.cone.overlap(s.c, node1.box, boxHits[1]);

			// is there overlap with both nodes?
			if (hit0 && hit1) {
				// we assume that the left child is a closer hit...
				int closer = nodeIndex + 1;
				int other = nodeIndex + node.secondChildOffset;

				// ... if the right child was actually closer, swap the relavent values
				if (boxHits[1] < boxHits[0]) {
					std::swap(boxHits[0], boxHits[1]);
					std::swap(closer, other);
				}

				// it's possible that the nearest object is still in the other side, but we'll
				// check the farther-away node later...

				// push the farther first, then the closer
				stackPtr++;
				subtree[stackPtr].node = other;
				subtree[stackPtr].distance = boxHits[1];

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
				subtree[stackPtr].distance = boxHits[1];
			}

			nodesVisited++;
		}
	}
}

template<size_t DIM, bool CONEDATA, typename PrimitiveType, typename SilhouetteType>
inline bool Sbvh<DIM, CONEDATA, PrimitiveType, SilhouetteType>::findClosestSilhouettePointFromNode(BoundingSphere<DIM>& s, Interaction<DIM>& i,
																								   int nodeStartIndex, int aggregateIndex,
																								   int& nodesVisited, bool flipNormalOrientation,
																								   float squaredMinRadius, float precision,
																								   bool recordNormal) const
{
	if (squaredMinRadius >= s.r2) return false;

	bool notFound = true;
	BvhTraversal subtree[FCPW_SBVH_MAX_DEPTH];
	float boxHits[2];

	int rootIndex = aggregateIndex == this->index ? nodeStartIndex : 0;
	if (flatTree[rootIndex].box.overlap(s, boxHits[0])) {
		subtree[0].node = rootIndex;
		subtree[0].distance = boxHits[0];
		processSubtreeForClosestSilhouettePoint<DIM, PrimitiveType, SilhouetteType>(flatTree, primitives, silhouetteRefs, s, i,
																					nodeStartIndex, aggregateIndex, this->index,
																					primitiveTypeIsAggregate, flipNormalOrientation,
																					squaredMinRadius, precision, recordNormal,
																					subtree, boxHits, notFound, nodesVisited);
	}

	if (!notFound) {
		// compute normal
		if (recordNormal && !primitiveTypeIsAggregate) {
			i.computeSilhouetteNormal(silhouetteRefs[i.referenceIndex]);
		}

		return true;
	}

	return false;
}

} // namespace fcpw
