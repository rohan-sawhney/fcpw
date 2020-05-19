namespace fcpw {

template <int DIM>
inline Sbvh<DIM>::Sbvh(std::vector<std::shared_ptr<Primitive<DIM>>>& primitives_,
					   const CostHeuristic& costHeuristic_, float splitAlpha_,
					   int leafSize_, int nBuckets_, int nBins_):
primitives(primitives_),
costHeuristic(costHeuristic_),
splitAlpha(splitAlpha_),
rootSurfaceArea(0.0f),
rootVolume(0.0f),
nNodes(0),
nLeafs(0),
leafSize(leafSize_),
nBuckets(nBuckets_),
nBins(nBins_),
memoryBudget(0),
maxDepth(0),
buckets(nBuckets, std::make_pair(BoundingBox<DIM>(), 0)),
rightBucketBoxes(nBuckets, std::make_pair(BoundingBox<DIM>(), 0)),
rightBinBoxes(nBins, std::make_pair(BoundingBox<DIM>(), 0)),
bins(nBins, std::make_tuple(BoundingBox<DIM>(), 0, 0))
{
	using namespace std::chrono;
	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	build();

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

template <int DIM>
inline float Sbvh<DIM>::computeSplitCost(const BoundingBox<DIM>& boxLeft,
										 const BoundingBox<DIM>& boxRight,
										 float parentSurfaceArea, float parentVolume,
										 int nReferencesLeft, int nReferencesRight) const
{
	float cost = maxFloat;
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

template <int DIM>
inline void Sbvh<DIM>::computeUnsplittingCosts(const BoundingBox<DIM>& boxLeft,
											   const BoundingBox<DIM>& boxRight,
											   const BoundingBox<DIM>& boxReference,
											   const BoundingBox<DIM>& boxRefLeft,
											   const BoundingBox<DIM>& boxRefRight,
											   int nReferencesLeft, int nReferencesRight,
											   float& costDuplicate, float& costUnsplitLeft,
											   float& costUnsplitRight) const
{
	BoundingBox<DIM> boxLeftUnsplit = boxLeft;
	BoundingBox<DIM> boxRightUnsplit = boxRight;
	BoundingBox<DIM> boxLeftDuplicate = boxLeft;
	BoundingBox<DIM> boxRightDuplicate = boxRight;
	boxLeftUnsplit.expandToInclude(boxReference);
	boxRightUnsplit.expandToInclude(boxReference);
	boxLeftDuplicate.expandToInclude(boxRefLeft);
	boxRightDuplicate.expandToInclude(boxRefRight);

	if (costHeuristic == CostHeuristic::SurfaceArea) {
		costDuplicate = boxLeftDuplicate.surfaceArea()*(nReferencesLeft + 1) +
						boxRightDuplicate.surfaceArea()*(nReferencesRight + 1);
		costUnsplitLeft = boxLeftUnsplit.surfaceArea()*(nReferencesLeft + 1) +
						  boxRight.surfaceArea()*nReferencesRight;
		costUnsplitRight = boxLeft.surfaceArea()*nReferencesLeft +
						   boxRightUnsplit.surfaceArea()*(nReferencesRight + 1);

	} else {
		costDuplicate = boxLeftDuplicate.volume()*(nReferencesLeft + 1) +
						boxRightDuplicate.volume()*(nReferencesRight + 1);
		costUnsplitLeft = boxLeftUnsplit.volume()*(nReferencesLeft + 1) +
						  boxRight.volume()*nReferencesRight;
		costUnsplitRight = boxLeft.volume()*nReferencesLeft +
						   boxRightUnsplit.volume()*(nReferencesRight + 1);
	}
}

template <int DIM>
inline float Sbvh<DIM>::computeObjectSplit(const BoundingBox<DIM>& nodeBoundingBox,
										   const BoundingBox<DIM>& nodeCentroidBox,
										   const std::vector<BoundingBox<DIM>>& referenceBoxes,
										   const std::vector<Vector<DIM>>& referenceCentroids,
										   int nodeStart, int nodeEnd, int& splitDim,
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
					float cost = computeSplitCost(boxRefLeft, rightBucketBoxes[b].first, surfaceArea,
												  volume, nReferencesLeft, rightBucketBoxes[b].second);

					if (cost < splitCost) {
						splitCost = cost;
						splitDim = dim;
						splitCoord = nodeBoundingBox.pMin[dim] + b*bucketWidth;
						boxIntersected = boxRefLeft.intersect(rightBucketBoxes[b].first);
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

template <int DIM>
inline int Sbvh<DIM>::performObjectSplit(int nodeStart, int nodeEnd, int splitDim, float splitCoord,
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

template <int DIM>
inline void Sbvh<DIM>::splitPrimitive(const std::shared_ptr<Primitive<DIM>>& primitive, int dim,
									  float splitCoord, const BoundingBox<DIM>& boxReference,
									  BoundingBox<DIM>& boxLeft, BoundingBox<DIM>& boxRight) const
{
	// split primitive along the provided coordinate and axis
	primitive->split(dim, splitCoord, boxLeft, boxRight);

	// intersect with bounds
	boxLeft = boxLeft.intersect(boxReference);
	boxRight = boxRight.intersect(boxReference);
}

template <int DIM>
inline float Sbvh<DIM>::computeSpatialSplit(const BoundingBox<DIM>& nodeBoundingBox,
											const std::vector<BoundingBox<DIM>>& referenceBoxes,
											int nodeStart, int nodeEnd, int splitDim, float& splitCoord,
											BoundingBox<DIM>& boxLeft, BoundingBox<DIM>& boxRight)
{
	// find the best split along splitDim
	float splitCost = maxFloat;
	splitCoord = 0.0f;

	Vector<DIM> extent = nodeBoundingBox.extent();
	float surfaceArea = nodeBoundingBox.surfaceArea();
	float volume = nodeBoundingBox.volume();

	// bin references
	float binWidth = extent[splitDim]/nBins;
	for (int b = 0; b < nBins; b++) {
		std::get<0>(bins[b]) = BoundingBox<DIM>();
		std::get<1>(bins[b]) = 0;
		std::get<2>(bins[b]) = 0;
	}

	for (int p = nodeStart; p < nodeEnd; p++) {
		// find the bins the reference is contained in
		const std::shared_ptr<Primitive<DIM>>& primitive = primitives[references[p]];
		int firstBinIndex = (int)((referenceBoxes[p].pMin[splitDim] - nodeBoundingBox.pMin[splitDim])/binWidth);
		int lastBinIndex = (int)((referenceBoxes[p].pMax[splitDim] - nodeBoundingBox.pMin[splitDim])/binWidth);
		firstBinIndex = clamp(firstBinIndex, 0, nBins - 1);
		lastBinIndex = clamp(lastBinIndex, 0, nBins - 1);
		BoundingBox<DIM> boxReference = referenceBoxes[p];

		// loop over those bins, splitting the reference and growing the bin boxes
		for (int b = firstBinIndex; b < lastBinIndex; b++) {
			BoundingBox<DIM> boxRefLeft, boxRefRight;
			float coord = nodeBoundingBox.pMin[splitDim] + (b + 1)*binWidth;
			splitPrimitive(primitive, splitDim, coord, boxReference, boxRefLeft, boxRefRight);
			std::get<0>(bins[b]).expandToInclude(boxRefLeft);
			boxReference = boxRefRight;
		}

		std::get<0>(bins[lastBinIndex]).expandToInclude(boxReference);
		std::get<1>(bins[firstBinIndex]) += 1; // increment number of entries
		std::get<2>(bins[lastBinIndex]) += 1; // increment number of exits
	}

	// sweep right to left to build right bin bounding boxes
	BoundingBox<DIM> boxRefRight;
	for (int b = nBins - 1; b > 0; b--) {
		boxRefRight.expandToInclude(std::get<0>(bins[b]));
		rightBinBoxes[b].first = boxRefRight;
		rightBinBoxes[b].second = std::get<2>(bins[b]);
		if (b != nBins - 1) rightBinBoxes[b].second += rightBinBoxes[b + 1].second;
	}

	// evaluate bin split costs
	BoundingBox<DIM> boxRefLeft;
	int nReferencesLeft = 0;
	for (int b = 1; b < nBins; b++) {
		boxRefLeft.expandToInclude(std::get<0>(bins[b - 1]));
		nReferencesLeft += std::get<1>(bins[b - 1]);

		if (nReferencesLeft > 0 && rightBinBoxes[b].second > 0) {
			float cost = computeSplitCost(boxRefLeft, rightBinBoxes[b].first, surfaceArea,
										  volume, nReferencesLeft, rightBinBoxes[b].second);

			if (cost < splitCost) {
				splitCost = cost;
				splitCoord = nodeBoundingBox.pMin[splitDim] + b*binWidth;
				boxLeft = boxRefLeft;
				boxRight = rightBinBoxes[b].first;
			}
		}
	}

	return splitCost;
}

template <int DIM>
inline int Sbvh<DIM>::performSpatialSplit(const BoundingBox<DIM>& boxLeft, const BoundingBox<DIM>& boxRight,
										  int splitDim, float splitCoord, int nodeStart, int& nodeEnd,
										  int& nReferencesAdded, int& nTotalReferences,
										  std::vector<BoundingBox<DIM>>& referenceBoxes,
										  std::vector<Vector<DIM>>& referenceCentroids)
{
	// categorize references into the following buckets:
	// [leftStart, leftEnd),
	// [leftEnd, rightStart) -> duplicates
	// [rightStart, reference.size())
	int leftStart = nodeStart;
	int leftEnd = nodeStart;
	int rightStart = nodeEnd;
	int rightEnd = nodeEnd;

	for (int i = leftStart; i < rightEnd; i++) {
		if (referenceBoxes[i].pMax[splitDim] <= splitCoord) {
			std::swap(references[i], references[leftEnd]);
			std::swap(referenceBoxes[i], referenceBoxes[leftEnd]);
			std::swap(referenceCentroids[i], referenceCentroids[leftEnd]);
			leftEnd++;
		}
	}

	for (int i = rightEnd - 1; i >= leftEnd; i--) {
		if (referenceBoxes[i].pMin[splitDim] >= splitCoord) {
			rightStart--;
			std::swap(references[i], references[rightStart]);
			std::swap(referenceBoxes[i], referenceBoxes[rightStart]);
			std::swap(referenceCentroids[i], referenceCentroids[rightStart]);
		}
	}

	// resize if memory is about to be exceeded
	int nPossibleNewReferences = rightStart - leftEnd;
	if (nTotalReferences + nPossibleNewReferences >= memoryBudget) {
		memoryBudget *= 2;
		references.resize(memoryBudget, -1);
		referenceBoxes.resize(memoryBudget);
		referenceCentroids.resize(memoryBudget);
	}

	if ((int)referencesToAdd.size() < nPossibleNewReferences) {
		referencesToAdd.resize(nPossibleNewReferences);
		referenceBoxesToAdd.resize(nPossibleNewReferences);
		referenceCentroidsToAdd.resize(nPossibleNewReferences);
	}

	// split or unsplit staddling references
	nReferencesAdded = 0;
	while (leftEnd < rightStart) {
		// split reference
		BoundingBox<DIM> boxRefLeft, boxRefRight;
		splitPrimitive(primitives[references[leftEnd]], splitDim, splitCoord,
					   referenceBoxes[leftEnd], boxRefLeft, boxRefRight);

		// compute unsplitting costs
		float costDuplicate, costUnsplitLeft, costUnsplitRight;
		computeUnsplittingCosts(boxLeft, boxRight, referenceBoxes[leftEnd], boxRefLeft,
								boxRefRight, leftEnd - leftStart, rightEnd - rightStart,
								costDuplicate, costUnsplitLeft, costUnsplitRight);

		if (costDuplicate < costUnsplitLeft && costDuplicate < costUnsplitRight) {
			// modify this reference box to contain the left split box
			referenceBoxes[leftEnd] = boxRefLeft;
			referenceCentroids[leftEnd] = boxRefLeft.centroid();

			// add right split box
			referencesToAdd[nReferencesAdded] = references[leftEnd];
			referenceBoxesToAdd[nReferencesAdded] = boxRefRight;
			referenceCentroidsToAdd[nReferencesAdded] = boxRefRight.centroid();

			nReferencesAdded++;
			leftEnd++;

		} else if (costUnsplitLeft < costDuplicate && costUnsplitLeft < costUnsplitRight) {
			// use reference box as is, but assign to the left of the split
			leftEnd++;

		} else {
			// use reference box as is, but assign to the right of the split
			rightStart--;
			std::swap(references[leftEnd], references[rightStart]);
			std::swap(referenceBoxes[leftEnd], referenceBoxes[rightStart]);
			std::swap(referenceCentroids[leftEnd], referenceCentroids[rightStart]);
		}
	}

	if (nReferencesAdded > 0) {
		// move entries between [nodeEnd, nTotalReferences) to
		// [nodeEnd + nReferencesAdded, nTotalReferences + nReferencesAdded)
		for (int i = nTotalReferences - 1; i >= nodeEnd; i--) {
			references[i + nReferencesAdded] = references[i];
			referenceBoxes[i + nReferencesAdded] = referenceBoxes[i];
			referenceCentroids[i + nReferencesAdded] = referenceCentroids[i];
		}

		// copy added references to range [nodeEnd, nodeEnd + nReferencesAdded)
		for (int i = 0; i < nReferencesAdded; i++) {
			references[nodeEnd + i] = referencesToAdd[i];
			referenceBoxes[nodeEnd + i] = referenceBoxesToAdd[i];
			referenceCentroids[nodeEnd + i] = referenceCentroidsToAdd[i];
		}
	}

	nodeEnd += nReferencesAdded;
	nTotalReferences += nReferencesAdded;
	return leftEnd;
}

template <int DIM>
inline int Sbvh<DIM>::buildRecursive(std::vector<BoundingBox<DIM>>& referenceBoxes,
									 std::vector<Vector<DIM>>& referenceCentroids,
									 std::vector<SbvhFlatNode<DIM>>& buildNodes,
									 int parent, int start, int end, int depth,
									 int& nTotalReferences)
{
	const int Untouched    = 0xffffffff;
	const int TouchedTwice = 0xfffffffd;
	maxDepth = std::max(depth, maxDepth);

	// add node to tree
	SbvhFlatNode<DIM> node;
	int currentNodeIndex = nNodes;
	int nReferences = end - start;

	nNodes++;
	node.parent = parent;
	node.start = start;
	node.nReferences = nReferences;
	node.rightOffset = Untouched;

	// calculate the bounding box for this node
	BoundingBox<DIM> bb, bc;
	for (int p = start; p < end; p++) {
		bb.expandToInclude(referenceBoxes[p]);
		bc.expandToInclude(referenceCentroids[p]);
	}

	node.splitDim = bc.maxDimension();
	node.box = bb;

	// if the number of references at this point is less than the leaf
	// size, then this will become a leaf (signified by rightOffset == 0)
	if (nReferences <= leafSize) {
		node.rightOffset = 0;
		nLeafs++;
	}

	buildNodes.emplace_back(node);

	// child touches parent...
	// special case: don't do this for the root
	if (parent != 0xfffffffc) {
		buildNodes[parent].rightOffset--;

		// when this is the second touch, this is the right child;
		// the right child sets up the offset for the flat tree
		if (buildNodes[parent].rightOffset == TouchedTwice) {
			buildNodes[parent].rightOffset = nNodes - 1 - parent;
		}
	}

	// if this is a leaf, no need to subdivide
	if (node.rightOffset == 0) return 0;

	// compute object split
	int splitDim;
	float splitCoord;
	bool isObjectSplitBetter = true;
	BoundingBox<DIM> boxLeft, boxRight, boxIntersected;
	float splitCost = computeObjectSplit(bb, bc, referenceBoxes, referenceCentroids,
										 start, end, splitDim, splitCoord, boxIntersected);

	// compute spatial split if intersected box is valid and not too small compared to the scene
	if (boxIntersected.isValid() &&
		((costHeuristic == CostHeuristic::SurfaceArea &&
		  boxIntersected.surfaceArea() > splitAlpha*rootSurfaceArea) ||
		 (costHeuristic == CostHeuristic::Volume &&
		  boxIntersected.volume() > splitAlpha*rootVolume))) {
		float spatialSplitCoord;
		float spatialSplitCost = computeSpatialSplit(bb, referenceBoxes, start, end,
													 splitDim, spatialSplitCoord,
													 boxLeft, boxRight);

		if (spatialSplitCost < splitCost) {
			isObjectSplitBetter = false;
			splitCoord = spatialSplitCoord;
			splitCost = spatialSplitCost;
		}
	}

	// partition the list of references on split
	int nReferencesAdded = 0;
	int mid = isObjectSplitBetter ? performObjectSplit(start, end, splitDim, splitCoord,
													   referenceBoxes, referenceCentroids) :
									performSpatialSplit(boxLeft, boxRight, splitDim, splitCoord,
														start, end, nReferencesAdded, nTotalReferences,
														referenceBoxes, referenceCentroids);

	// push left and right children
	int nReferencesAddedLeft = buildRecursive(referenceBoxes, referenceCentroids, buildNodes,
											  currentNodeIndex, start, mid, depth + 1,
											  nTotalReferences);
	int nReferencesAddedRight = buildRecursive(referenceBoxes, referenceCentroids, buildNodes,
											   currentNodeIndex, mid + nReferencesAddedLeft,
											   end + nReferencesAddedLeft, depth + 1,
											   nTotalReferences);
	int nTotalReferencesAdded = nReferencesAdded + nReferencesAddedLeft + nReferencesAddedRight;
	buildNodes[currentNodeIndex].nReferences += nTotalReferencesAdded;
	buildNodes[currentNodeIndex].splitDim = splitDim;

	return nTotalReferencesAdded;
}

template <int DIM>
inline void Sbvh<DIM>::build()
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	// precompute bounding boxes and centroids
	int nReferences = (int)primitives.size();
	std::vector<BoundingBox<DIM>> referenceBoxes;
	std::vector<Vector<DIM>> referenceCentroids;
	std::vector<SbvhFlatNode<DIM>> buildNodes;

	memoryBudget = nReferences*2;
	references.resize(memoryBudget, -1);
	referenceBoxes.resize(memoryBudget);
	referenceCentroids.resize(memoryBudget);
	buildNodes.reserve(nReferences*2);
	BoundingBox<DIM> boxRoot;

	for (int i = 0; i < nReferences; i++) {
		references[i] = i;
		referenceBoxes[i] = primitives[i]->boundingBox();
		referenceCentroids[i] = primitives[i]->centroid();
		boxRoot.expandToInclude(referenceBoxes[i]);
	}

	rootSurfaceArea = boxRoot.surfaceArea();
	rootVolume = boxRoot.volume();

	// build tree recursively
	int nTotalReferences = nReferences;
	int nReferencesAdded = buildRecursive(referenceBoxes, referenceCentroids, buildNodes,
										  0xfffffffc, 0, nReferences, 0, nTotalReferences);
	maxDepth = std::pow(2, std::ceil(std::log2(maxDepth)));

	// copy the temp node data to a flat array
	flatTree.reserve(nNodes);
	for (int n = 0; n < nNodes; n++) {
		flatTree.emplace_back(buildNodes[n]);
	}

	// resize references vector and clear working set
	references.resize(nReferences + nReferencesAdded);
	referencesToAdd.clear();
	referenceBoxesToAdd.clear();
	referenceCentroidsToAdd.clear();
	buckets.clear();
	rightBucketBoxes.clear();
	rightBinBoxes.clear();
	bins.clear();
}

template <int DIM>
inline BoundingBox<DIM> Sbvh<DIM>::boundingBox() const
{
	return flatTree.size() > 0 ? flatTree[0].box : BoundingBox<DIM>();
}

template <int DIM>
inline Vector<DIM> Sbvh<DIM>::centroid() const
{
	Vector<DIM> c = zeroVector<DIM>();
	int nPrimitives = (int)primitives.size();

	for (int p = 0; p < nPrimitives; p++) {
		c += primitives[p]->centroid();
	}

	return c/nPrimitives;
}

template <int DIM>
inline float Sbvh<DIM>::surfaceArea() const
{
	float area = 0.0f;
	for (int p = 0; p < (int)primitives.size(); p++) {
		area += primitives[p]->surfaceArea();
	}

	return area;
}

template <int DIM>
inline float Sbvh<DIM>::signedVolume() const
{
	float volume = 0.0f;
	for (int p = 0; p < (int)primitives.size(); p++) {
		volume += primitives[p]->signedVolume();
	}

	return volume;
}

template <int DIM>
inline bool Sbvh<DIM>::processSubtreeForIntersection(Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
													 bool checkOcclusion, bool countHits,
													 SbvhTraversal *subtree, float *boxHits,
													 int& hits, int& nodesVisited) const
{
	int stackPtr = 0;
	while (stackPtr >= 0) {
		// pop off the next node to work on
		int nodeIndex = subtree[stackPtr].node;
		float near = subtree[stackPtr].distance;
		stackPtr--;
		const SbvhFlatNode<DIM>& node(flatTree[nodeIndex]);

		// if this node is further than the closest found intersection, continue
		if (!countHits && near > r.tMax) continue;
		nodesVisited++;

		// is leaf -> intersect
		if (node.rightOffset == 0) {
			for (int p = 0; p < node.nReferences; p++) {
				const std::shared_ptr<Primitive<DIM>>& prim = primitives[references[node.start + p]];

				// check if primitive has already been seen
				bool seenPrim = false;
				int nInteractions = (int)is.size();
				for (int sp = 0; sp < nInteractions; sp++) {
					if (prim.get() == is[sp].primitive) {
						seenPrim = true;
						break;
					}
				}

				if (!seenPrim) {
					std::vector<Interaction<DIM>> cs;
					int hit = prim->intersect(r, cs, checkOcclusion, countHits);
					nodesVisited++;

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
			}

		} else { // not a leaf
			bool hit0 = flatTree[nodeIndex + 1].box.intersect(r, boxHits[0], boxHits[1]);
			bool hit1 = flatTree[nodeIndex + node.rightOffset].box.intersect(r, boxHits[2], boxHits[3]);

			// did we hit both nodes?
			if (hit0 && hit1) {
				// we assume that the left child is a closer hit...
				int closer = nodeIndex + 1;
				int other = nodeIndex + node.rightOffset;

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
				subtree[stackPtr].node = nodeIndex + node.rightOffset;
				subtree[stackPtr].distance = boxHits[2];
			}
		}
	}

	return false;
}

template <int DIM>
inline int Sbvh<DIM>::intersectFromNode(Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
										int nodeStartIndex, int& nodesVisited,
										bool checkOcclusion, bool countHits) const
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	LOG_IF(FATAL, nodeStartIndex < 0 || nodeStartIndex >= nNodes) << "Start node index: "
								 << nodeStartIndex << " out of range [0, " << nNodes << ")";
	int hits = 0;
	if (!countHits) is.resize(1);
	SbvhTraversal subtree[maxDepth];
	float boxHits[4];

	// push the start node onto the working set and process its subtree if it intersects ray
	if (flatTree[nodeStartIndex].box.intersect(r, boxHits[0], boxHits[1])) {
		subtree[0].node = nodeStartIndex;
		subtree[0].distance = boxHits[0];
		bool occluded = processSubtreeForIntersection(r, is, checkOcclusion, countHits,
													  subtree, boxHits, hits, nodesVisited);
		if (occluded) return 1;
	}

	int nodeParentIndex = flatTree[nodeStartIndex].parent;
	while (nodeParentIndex != 0xfffffffc) {
		// determine the sibling node's index
		int nodeSiblingIndex = nodeParentIndex + 1 == nodeStartIndex ?
							   nodeParentIndex + flatTree[nodeParentIndex].rightOffset :
							   nodeParentIndex + 1;

		// push the sibling node onto the working set and process its subtree if it intersects ray
		if (flatTree[nodeSiblingIndex].box.intersect(r, boxHits[2], boxHits[3])) {
			subtree[0].node = nodeSiblingIndex;
			subtree[0].distance = boxHits[2];
			bool occluded = processSubtreeForIntersection(r, is, checkOcclusion, countHits,
														  subtree, boxHits, hits, nodesVisited);
			if (occluded) return 1;
		}

		// update the start node index to its parent index
		nodeStartIndex = nodeParentIndex;
		nodeParentIndex = flatTree[nodeStartIndex].parent;
	}

	if (countHits) {
		std::sort(is.begin(), is.end(), compareInteractions<DIM>);
		is = removeDuplicates<DIM>(is);
		hits = (int)is.size();
	}

	return hits;
}

template <int DIM>
inline void Sbvh<DIM>::processSubtreeForClosestPoint(BoundingSphere<DIM>& s, Interaction<DIM>& i,
													 std::deque<SbvhTraversal>& subtree, float *boxHits,
													 bool& notFound, int& nodesVisited) const
{
	while (!subtree.empty()) {
		// pop off the next node to work on
		SbvhTraversal traversal = subtree.front();
		subtree.pop_front();

		int nodeIndex = traversal.node;
		float near = traversal.distance;
		const SbvhFlatNode<DIM>& node(flatTree[nodeIndex]);

		// if this node is further than the closest found primitive, continue
		if (near > s.r2) continue;
		nodesVisited++;

		// is leaf -> compute squared distance
		if (node.rightOffset == 0) {
			for (int p = 0; p < node.nReferences; p++) {
				const std::shared_ptr<Primitive<DIM>>& prim = primitives[references[node.start + p]];

				if (prim.get() != i.primitive) {
					Interaction<DIM> c;
					bool found = prim->findClosestPoint(s, c);
					nodesVisited++;

					// keep the closest point only
					if (found) {
						notFound = false;
						s.r2 = std::min(s.r2, c.d*c.d);
						i = c;
						i.nodeIndex = nodeIndex;
					}
				}
			}

		} else { // not a leaf
			bool hit0 = flatTree[nodeIndex + 1].box.overlap(s, boxHits[0], boxHits[1]);
			s.r2 = std::min(s.r2, boxHits[1]);

			bool hit1 = flatTree[nodeIndex + node.rightOffset].box.overlap(s, boxHits[2], boxHits[3]);
			s.r2 = std::min(s.r2, boxHits[3]);

			// is there overlap with both nodes?
			if (hit0 && hit1) {
				// we assume that the left child is a closer hit...
				int closer = nodeIndex + 1;
				int other = nodeIndex + node.rightOffset;

				// ... if the right child was actually closer, swap the relavent values
				if (boxHits[2] < boxHits[0]) {
					std::swap(boxHits[0], boxHits[2]);
					std::swap(boxHits[1], boxHits[3]);
					std::swap(closer, other);
				}

				// it's possible that the nearest object is still in the other side, but we'll
				// check the farther-away node later...

				// push the closer first, then the farther
				subtree.emplace_back(SbvhTraversal(closer, boxHits[0]));
				subtree.emplace_back(SbvhTraversal(other, boxHits[2]));

			} else if (hit0) {
				subtree.emplace_back(SbvhTraversal(nodeIndex + 1, boxHits[0]));

			} else if (hit1) {
				subtree.emplace_back(SbvhTraversal(nodeIndex + node.rightOffset, boxHits[2]));
			}
		}
	}
}

template <int DIM>
inline bool Sbvh<DIM>::findClosestPointFromNode(BoundingSphere<DIM>& s, Interaction<DIM>& i,
												int nodeStartIndex, int& nodesVisited) const
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	LOG_IF(FATAL, nodeStartIndex < 0 || nodeStartIndex >= nNodes) << "Start node index: "
								 << nodeStartIndex << " out of range [0, " << nNodes << ")";
	bool notFound = true;
	std::deque<SbvhTraversal> subtree;
	float boxHits[4];

	// push the start node onto the working set
	if (flatTree[nodeStartIndex].box.overlap(s, boxHits[0], boxHits[1])) {
		s.r2 = std::min(s.r2, boxHits[1]);
		subtree.emplace_back(SbvhTraversal(nodeStartIndex, boxHits[0]));
	}

	int nodeParentIndex = flatTree[nodeStartIndex].parent;
	while (nodeParentIndex != 0xfffffffc) {
		// determine the sibling node's index
		int nodeSiblingIndex = nodeParentIndex + 1 == nodeStartIndex ?
							   nodeParentIndex + flatTree[nodeParentIndex].rightOffset :
							   nodeParentIndex + 1;

		// push the sibling node onto the working set
		if (flatTree[nodeSiblingIndex].box.overlap(s, boxHits[2], boxHits[3])) {
			s.r2 = std::min(s.r2, boxHits[3]);
			subtree.emplace_back(SbvhTraversal(nodeSiblingIndex, boxHits[2]));
		}

		// update the start node index to its parent index
		nodeStartIndex = nodeParentIndex;
		nodeParentIndex = flatTree[nodeStartIndex].parent;
	}

	// process subtrees
	processSubtreeForClosestPoint(s, i, subtree, boxHits, notFound, nodesVisited);

	return !notFound;
}

} // namespace fcpw
