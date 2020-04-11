#include <stack>
#include <queue>

namespace fcpw {

struct SbvhTraversal {
	// constructor
	SbvhTraversal(int i_, float d_): i(i_), d(d_) {}

	// members
	int i; // node index
	float d; // minimum distance (parametric, squared, ...) to this node
};

template <int DIM>
inline Sbvh<DIM>::Sbvh(std::vector<std::shared_ptr<Primitive<DIM>>>& primitives_,
					   const CostHeuristic& costHeuristic_, float splitAlpha_,
					   int leafSize_, int nBuckets_, int nBins_):
costHeuristic(costHeuristic_),
splitAlpha(splitAlpha_),
rootSurfaceArea(0.0f),
rootVolume(0.0f),
nNodes(0),
nLeafs(0),
leafSize(leafSize_),
nBuckets(nBuckets_),
nBins(nBins_),
buckets(nBuckets, std::make_pair(BoundingBox<DIM>(true), 0)),
rightBucketBoxes(nBuckets, std::make_pair(BoundingBox<DIM>(true), 0)),
rightBinBoxes(nBuckets, std::make_pair(BoundingBox<DIM>(true), 0)),
bins(nBins, std::make_tuple(BoundingBox<DIM>(true), 0, 0)),
primitives(primitives_)
{
	using namespace std::chrono;
	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	build();

	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	duration<double> timeSpan = duration_cast<duration<double>>(t2 - t1);
	std::cout << "Built bvh with "
			  << nNodes << " nodes, "
			  << nLeafs << " leaves, "
			  << primitives.size() << " primitives, "
			  << references.size() << " references in "
			  << timeSpan.count() << " seconds" << std::endl;
}

template <int DIM>
inline float computeSplitCost(const CostHeuristic& costHeuristic,
							  const BoundingBox<DIM>& bboxLeft,
							  const BoundingBox<DIM>& bboxRight,
							  float parentSurfaceArea, float parentVolume,
							  int nReferencesLeft, int nReferencesRight)
{
	float cost = maxFloat;
	if (costHeuristic == CostHeuristic::SurfaceArea) {
		cost = (nReferencesLeft*bboxLeft.surfaceArea() +
				nReferencesRight*bboxRight.surfaceArea())/parentSurfaceArea;

	} else if (costHeuristic == CostHeuristic::OverlapSurfaceArea) {
		// set the cost to be negative if the left and right boxes don't overlap at all
		BoundingBox<DIM> bboxIntersected = bboxLeft.intersect(bboxRight);
		cost = (nReferencesLeft/bboxRight.surfaceArea() +
				nReferencesRight/bboxLeft.surfaceArea())*std::fabs(bboxIntersected.surfaceArea());
		if (!bboxIntersected.isValid()) cost *= -1;

	} else if (costHeuristic == CostHeuristic::Volume) {
		cost = (nReferencesLeft*bboxLeft.volume() +
				nReferencesRight*bboxRight.volume())/parentVolume;

	} else if (costHeuristic == CostHeuristic::OverlapVolume) {
		// set the cost to be negative if the left and right boxes don't overlap at all
		BoundingBox<DIM> bboxIntersected = bboxLeft.intersect(bboxRight);
		cost = (nReferencesLeft/bboxRight.volume() +
				nReferencesRight/bboxLeft.volume())*std::fabs(bboxIntersected.volume());
		if (!bboxIntersected.isValid()) cost *= -1;
	}

	return cost;
}

template <int DIM>
inline float Sbvh<DIM>::computeObjectSplit(const BoundingBox<DIM>& nodeBoundingBox,
										   const BoundingBox<DIM>& nodeCentroidBox,
										   const std::vector<BoundingBox<DIM>>& referenceBoxes,
										   const std::vector<Vector<DIM>>& referenceCentroids,
										   int nodeStart, int nodeEnd, int& splitDim, float& splitCoord,
										   BoundingBox<DIM>& bboxIntersected)
{
	float splitCost = maxFloat;
	splitDim = -1;
	splitCoord = 0.0f;
	bboxIntersected = BoundingBox<DIM>(false);

	if (costHeuristic != CostHeuristic::LongestAxisCenter) {
		Vector<DIM> extent = nodeBoundingBox.extent();
		float surfaceArea = nodeBoundingBox.surfaceArea();
		float volume = nodeBoundingBox.volume();

		// find the best split across all three dimensions
		for (int dim = 0; dim < DIM; dim++) {
			// ignore flat dimension
			if (extent(dim) < 1e-6) continue;

			// bin references into buckets
			float bucketWidth = extent(dim)/nBuckets;
			for (int b = 0; b < nBuckets; b++) {
				buckets[b].first = BoundingBox<DIM>(true);
				buckets[b].second = 0;
			}

			for (int p = nodeStart; p < nodeEnd; p++) {
				int bucketIndex = (int)((referenceCentroids[p](dim) - nodeBoundingBox.pMin(dim))/bucketWidth);
				bucketIndex = clamp(bucketIndex, 0, nBuckets - 1);
				buckets[bucketIndex].first.expandToInclude(referenceBoxes[p]);
				buckets[bucketIndex].second += 1;
			}

			// sweep right to left to build right bucket bounding boxes
			BoundingBox<DIM> bboxRight(true);
			for (int b = nBuckets - 1; b > 0; b--) {
				bboxRight.expandToInclude(buckets[b].first);
				rightBucketBoxes[b].first = bboxRight;
				rightBucketBoxes[b].second = buckets[b].second;
				if (b != nBuckets - 1) rightBucketBoxes[b].second += rightBucketBoxes[b + 1].second;
			}

			// evaluate bucket split costs
			BoundingBox<DIM> bboxLeft(true);
			int nReferencesLeft = 0;
			for (int b = 1; b < nBuckets; b++) {
				bboxLeft.expandToInclude(buckets[b - 1].first);
				nReferencesLeft += buckets[b - 1].second;
				float cost = computeSplitCost(costHeuristic, bboxLeft, rightBucketBoxes[b].first,
											  surfaceArea, volume, nReferencesLeft,
											  rightBucketBoxes[b].second);

				if (cost < splitCost) {
					splitCost = cost;
					splitDim = dim;
					splitCoord = nodeBoundingBox.pMin(dim) + b*bucketWidth;
					bboxIntersected = bboxLeft.intersect(bboxRight);
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
inline int Sbvh<DIM>::performObjectSplit(std::vector<BoundingBox<DIM>>& referenceBoxes,
										 std::vector<Vector<DIM>>& referenceCentroids,
										 int nodeStart, int nodeEnd, int splitDim, float splitCoord)
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
inline void Sbvh<DIM>::splitReference(int referenceIndex, int dim, float splitCoord,
									  const BoundingBox<DIM>& bboxReference,
									  BoundingBox<DIM>& bboxLeft, BoundingBox<DIM>& bboxRight) const
{
	// split primitive along the provided coordinate and axis
	primitives[referenceIndex]->split(dim, splitCoord, bboxLeft, bboxRight);

	// intersect with bounds
	bboxLeft = bboxLeft.intersect(bboxReference);
	bboxRight = bboxRight.intersect(bboxReference);
}

template <int DIM>
inline float Sbvh<DIM>::computeSpatialSplit(const BoundingBox<DIM>& nodeBoundingBox,
											const std::vector<BoundingBox<DIM>>& referenceBoxes,
											int nodeStart, int nodeEnd, int& splitDim, float& splitCoord)
{
	float splitCost = maxFloat;
	splitDim = -1;
	splitCoord = 0.0f;

	Vector<DIM> extent = nodeBoundingBox.extent();
	float surfaceArea = nodeBoundingBox.surfaceArea();
	float volume = nodeBoundingBox.volume();

	// find the best split across all three dimensions
	for (int dim = 0; dim < DIM; dim++) {
		// ignore flat dimension
		if (extent(dim) < 1e-6) continue;

		// bin references
		float binWidth = extent(dim)/nBins;
		for (int b = 0; b < nBins; b++) {
			std::get<0>(bins[b]) = BoundingBox<DIM>(true);
			std::get<1>(bins[b]) = 0;
			std::get<2>(bins[b]) = 0;
		}

		for (int p = nodeStart; p < nodeEnd; p++) {
			// find the bins the reference is contained in
			int firstBinIndex = (int)((referenceBoxes[p].pMin(dim) - nodeBoundingBox.pMin(dim))/binWidth);
			int lastBinIndex = (int)((referenceBoxes[p].pMax(dim) - nodeBoundingBox.pMin(dim))/binWidth);
			firstBinIndex = clamp(firstBinIndex, 0, nBins - 1);
			lastBinIndex = clamp(lastBinIndex, 0, nBins - 1);
			BoundingBox<DIM> bboxReference = referenceBoxes[p];

			// loop over those bins, splitting the reference and growing the bin boxes
			for (int b = firstBinIndex; b < lastBinIndex; b++) {
				BoundingBox<DIM> bboxLeft(true), bboxRight(true);
				float coord = nodeBoundingBox.pMin(dim) + (b + 1)*binWidth;
				splitReference(references[p], dim, coord, bboxReference, bboxLeft, bboxRight);
				std::get<0>(bins[b]).expandToInclude(bboxLeft);
				bboxReference = bboxRight;
			}

			std::get<0>(bins[lastBinIndex]).expandToInclude(bboxReference);
			std::get<1>(bins[firstBinIndex]) += 1; // increment number of entries
			std::get<2>(bins[lastBinIndex]) += 1; // increment number of exits
		}

		// sweep right to left to build right bin bounding boxes
		BoundingBox<DIM> bboxRight(true);
		for (int b = nBins - 1; b > 0; b--) {
			bboxRight.expandToInclude(std::get<0>(bins[b]));
			rightBinBoxes[b].first = bboxRight;
			rightBinBoxes[b].second = std::get<2>(bins[b]);
			if (b != nBins - 1) rightBinBoxes[b].second += rightBinBoxes[b + 1].second;
		}

		// evaluate bin split costs
		BoundingBox<DIM> bboxLeft(true);
		int nReferencesLeft = 0;
		for (int b = 1; b < nBins; b++) {
			bboxLeft.expandToInclude(std::get<0>(bins[b - 1]));
			nReferencesLeft += std::get<1>(bins[b - 1]);
			float cost = computeSplitCost(costHeuristic, bboxLeft, rightBinBoxes[b].first,
										  surfaceArea, volume, nReferencesLeft,
										  rightBinBoxes[b].second);

			if (cost < splitCost) {
				splitCost = cost;
				splitDim = dim;
				splitCoord = nodeBoundingBox.pMin(dim) + b*binWidth;
			}
		}
	}

	return splitCost;
}

template <int DIM>
inline int Sbvh<DIM>::performSpatialSplit(std::vector<BoundingBox<DIM>>& referenceBoxes,
										  std::vector<Vector<DIM>>& referenceCentroids,
										  int splitDim, float splitCoord, int nodeStart, int& nodeEnd,
										  int& nReferencesAdded)
{
	nReferencesAdded = 0;
	// TODO
	nodeEnd += nReferencesAdded;

	return 0;
}

template <int DIM>
inline int Sbvh<DIM>::buildRecursive(std::vector<BoundingBox<DIM>>& referenceBoxes,
									 std::vector<Vector<DIM>>& referenceCentroids,
									 std::vector<SbvhFlatNode<DIM>>& buildNodes,
									 int parent, int start, int end)
{
	const int Untouched    = 0xffffffff;
	const int TouchedTwice = 0xfffffffd;

	// add node to tree
	SbvhFlatNode<DIM> node;
	int currentNodeIndex = nNodes;
	int nReferences = end - start;

	nNodes++;
	node.start = start;
	node.nReferences = nReferences;
	node.rightOffset = Untouched;

	// calculate the bounding box for this node
	BoundingBox<DIM> bb(true), bc(true);
	for (int p = start; p < end; p++) {
		bb.expandToInclude(referenceBoxes[p]);
		bc.expandToInclude(referenceCentroids[p]);
	}

	node.bbox = bb;

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
	BoundingBox<DIM> bboxIntersected(false);
	bool isObjectSplitBetter = true;
	float splitCost = computeObjectSplit(bb, bc, referenceBoxes, referenceCentroids,
										 start, end, splitDim, splitCoord, bboxIntersected);

	// compute spatial split if intersected box is valid and not too small compared to the scene
	if ((costHeuristic == CostHeuristic::SurfaceArea &&
		 bboxIntersected.surfaceArea() > splitAlpha*rootSurfaceArea) ||
		(costHeuristic == CostHeuristic::Volume &&
		 bboxIntersected.volume() > splitAlpha*rootVolume)) {
		int spatialSplitDim;
		float spatialSplitCoord;
		float spatialSplitCost = computeSpatialSplit(bb, referenceBoxes, start, end,
													 spatialSplitDim, spatialSplitCoord);

		if (spatialSplitCost < splitCost) {
			// TODO: uncomment after implementing performSpatialSplit
			// isObjectSplitBetter = false;
			// splitDim = spatialSplitDim;
			// splitCoord = spatialSplitCoord;
			// splitCost = spatialSplitCost;
		}
	}

	// partition the list of references on split
	int nReferencesAdded = 0;
	int mid = isObjectSplitBetter ? performObjectSplit(referenceBoxes, referenceCentroids,
														start, end, splitDim, splitCoord) :
									performSpatialSplit(referenceBoxes, referenceCentroids,
														splitDim, splitCoord, start, end,
														nReferencesAdded);

	// push left and right children
	int nReferencesAddedLeft = buildRecursive(referenceBoxes, referenceCentroids, buildNodes,
											  currentNodeIndex, start, mid);
	int nReferencesAddedRight = buildRecursive(referenceBoxes, referenceCentroids, buildNodes,
											   currentNodeIndex, mid + nReferencesAddedLeft,
											   end + nReferencesAddedLeft);
	int nTotalReferencesAdded = nReferencesAdded + nReferencesAddedLeft + nReferencesAddedRight;
	buildNodes[currentNodeIndex].nReferences += nTotalReferencesAdded;

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
	BoundingBox<DIM> bboxRoot(true);

	for (int i = 0; i < nReferences; i++) {
		referenceBoxes.emplace_back(primitives[i]->boundingBox());
		referenceCentroids.emplace_back(primitives[i]->centroid());
		bboxRoot.expandToInclude(referenceBoxes[i]);
		references.emplace_back(i);
	}

	std::vector<SbvhFlatNode<DIM>> buildNodes;
	buildNodes.reserve(nReferences*2);

	rootSurfaceArea = bboxRoot.surfaceArea();
	rootVolume = bboxRoot.volume();

	// build tree recursively
	buildRecursive(referenceBoxes, referenceCentroids, buildNodes,
				   0xfffffffc, 0, nReferences);

	// copy the temp node data to a flat array
	flatTree.reserve(nNodes);
	for (int n = 0; n < nNodes; n++) {
		flatTree.emplace_back(buildNodes[n]);
	}
}

template <int DIM>
inline BoundingBox<DIM> Sbvh<DIM>::boundingBox() const
{
	return flatTree.size() > 0 ? flatTree[0].bbox : BoundingBox<DIM>(false);
}

template <int DIM>
inline Vector<DIM> Sbvh<DIM>::centroid() const
{
	return boundingBox().centroid();
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
inline int Sbvh<DIM>::intersect(Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
								bool checkOcclusion, bool countHits) const
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	int hits = 0;
	if (!countHits) is.resize(1);
	std::stack<SbvhTraversal> todo;
	float bbhits[4];
	int closer, other;

	// "push" on the root node to the working set
	todo.emplace(SbvhTraversal(0, minFloat));

	while (!todo.empty()) {
		// pop off the next node to work on
		SbvhTraversal traversal = todo.top();
		todo.pop();

		int ni = traversal.i;
		float near = traversal.d;
		const SbvhFlatNode<DIM>& node(flatTree[ni]);

		// if this node is further than the closest found intersection, continue
		if (!countHits && near > r.tMax) continue;

		// is leaf -> intersect
		if (node.rightOffset == 0) {
			for (int p = 0; p < node.nReferences; p++) {
				std::vector<Interaction<DIM>> cs;
				const std::shared_ptr<Primitive<DIM>>& prim = primitives[references[node.start + p]];
				int hit = prim->intersect(r, cs, checkOcclusion, countHits);

				// keep the closest intersection only
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

		} else { // not a leaf
			bool hit0 = flatTree[ni + 1].bbox.intersect(r, bbhits[0], bbhits[1]);
			bool hit1 = flatTree[ni + node.rightOffset].bbox.intersect(r, bbhits[2], bbhits[3]);

			// did we hit both nodes?
			if (hit0 && hit1) {
				// we assume that the left child is a closer hit...
				closer = ni + 1;
				other = ni + node.rightOffset;

				// ... if the right child was actually closer, swap the relavent values
				if (bbhits[2] < bbhits[0]) {
					std::swap(bbhits[0], bbhits[2]);
					std::swap(bbhits[1], bbhits[3]);
					std::swap(closer, other);
				}

				// it's possible that the nearest object is still in the other side, but we'll
				// check the farther-away node later...

				// push the farther first, then the closer
				todo.emplace(SbvhTraversal(other, bbhits[2]));
				todo.emplace(SbvhTraversal(closer, bbhits[0]));

			} else if (hit0) {
				todo.emplace(SbvhTraversal(ni + 1, bbhits[0]));

			} else if (hit1) {
				todo.emplace(SbvhTraversal(ni + node.rightOffset, bbhits[2]));
			}
		}
	}

	if (countHits) {
		// it is particularly important to remove duplicates, since primitive references
		// in a sbvh might be contained in multiple leaves
		std::sort(is.begin(), is.end(), compareInteractions<DIM>);
		is = removeDuplicates<DIM>(is);
		hits = (int)is.size();
	}

	return hits;
}

template <int DIM>
inline bool Sbvh<DIM>::findClosestPoint(BoundingSphere<DIM>& s, Interaction<DIM>& i) const
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	bool notFound = true;
	std::queue<SbvhTraversal> todo;
	float bbhits[4];
	int closer, other;

	// "push" on the root node to the working set
	todo.emplace(SbvhTraversal(0, minFloat));

	while (!todo.empty()) {
		// pop off the next node to work on
		SbvhTraversal traversal = todo.front();
		todo.pop();

		int ni = traversal.i;
		float near = traversal.d;
		const SbvhFlatNode<DIM>& node(flatTree[ni]);

		// if this node is further than the closest found primitive, continue
		if (near > s.r2) continue;

		// is leaf -> compute squared distance
		if (node.rightOffset == 0) {
			for (int p = 0; p < node.nReferences; p++) {
				Interaction<DIM> c;
				const std::shared_ptr<Primitive<DIM>>& prim = primitives[references[node.start + p]];
				bool found = prim->findClosestPoint(s, c);

				// keep the closest point only
				if (found) {
					notFound = false;
					s.r2 = std::min(s.r2, c.d*c.d);
					i = c;
				}
			}

		} else { // not a leaf
			bool hit0 = flatTree[ni + 1].bbox.overlaps(s, bbhits[0], bbhits[1]);
			s.r2 = std::min(s.r2, bbhits[1]);

			bool hit1 = flatTree[ni + node.rightOffset].bbox.overlaps(s, bbhits[2], bbhits[3]);
			s.r2 = std::min(s.r2, bbhits[3]);

			// is there overlap with both nodes?
			if (hit0 && hit1) {
				// we assume that the left child is a closer hit...
				closer = ni + 1;
				other = ni + node.rightOffset;

				// ... if the right child was actually closer, swap the relavent values
				if (bbhits[2] < bbhits[0]) {
					std::swap(bbhits[0], bbhits[2]);
					std::swap(bbhits[1], bbhits[3]);
					std::swap(closer, other);
				}

				// it's possible that the nearest object is still in the other side, but we'll
				// check the farther-away node later...

				// push the closer first, then the farther
				todo.emplace(SbvhTraversal(closer, bbhits[0]));
				todo.emplace(SbvhTraversal(other, bbhits[2]));

			} else if (hit0) {
				todo.emplace(SbvhTraversal(ni + 1, bbhits[0]));

			} else if (hit1) {
				todo.emplace(SbvhTraversal(ni + node.rightOffset, bbhits[2]));
			}
		}
	}

	return !notFound;
}

} // namespace fcpw
