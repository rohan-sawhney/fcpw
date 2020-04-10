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
					   const CostHeuristic& costHeuristic_, int leafSize_,
					   float splitAlpha_):
costHeuristic(costHeuristic_),
nNodes(0),
nLeafs(0),
leafSize(leafSize_),
splitAlpha(splitAlpha_),
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
		// cost can be negative, but that's a good thing, because the more negative the cost
		// the further apart the left and right boxes should be
		BoundingBox<DIM> bboxIntersected = bboxLeft.intersect(bboxRight);
		cost = (nReferencesLeft/bboxRight.surfaceArea() +
				nReferencesRight/bboxLeft.surfaceArea())*bboxIntersected.surfaceArea();

	} else if (costHeuristic == CostHeuristic::Volume) {
		cost = (nReferencesLeft*bboxLeft.volume() +
				nReferencesRight*bboxRight.volume())/parentVolume;

	} else if (costHeuristic == CostHeuristic::OverlapVolume) {
		// cost can be negative, but that's a good thing, because the more negative the cost
		// the further apart the left and right boxes should be
		BoundingBox<DIM> bboxIntersected = bboxLeft.intersect(bboxRight);
		cost = (nReferencesLeft/bboxRight.volume() +
				nReferencesRight/bboxLeft.volume())*bboxIntersected.volume();
	}

	return cost;
}

template <int DIM>
inline void evaluateBucketSplitCosts(const CostHeuristic& costHeuristic,
									 const BoundingBox<DIM>& nodeBoundingBox,
									 const std::vector<std::pair<BoundingBox<DIM>, int>>& buckets,
									 int nBuckets, float bucketWidth, float surfaceArea, float volume,
									 int dim, float& splitCost, int& splitDim, float& splitCoord)
{
	for (int b = 1; b < nBuckets; b++) {
		// compute left and right child boxes for this particular split
		BoundingBox<DIM> bboxLeft(true), bboxRight(true);
		int nReferencesLeft = 0, nReferencesRight = 0;

		for (int i = 0; i < b; i++) {
			bboxLeft.expandToInclude(buckets[i].first);
			nReferencesLeft += buckets[i].second;
		}

		for (int i = b; i < nBuckets; i++) {
			bboxRight.expandToInclude(buckets[i].first);
			nReferencesRight += buckets[i].second;
		}

		// compute split cost based on heuristic
		float cost = computeSplitCost(costHeuristic, bboxLeft, bboxRight, surfaceArea,
									  volume, nReferencesLeft, nReferencesRight);
		float coord = nodeBoundingBox.pMin(dim) + b*bucketWidth;

		if (cost < splitCost) {
			splitCost = cost;
			splitDim = dim;
			splitCoord = coord;
		}
	}
}

template <int DIM>
inline float computeObjectSplit(const CostHeuristic& costHeuristic,
								const BoundingBox<DIM>& nodeBoundingBox,
								const BoundingBox<DIM>& nodeCentroidBox,
								const std::vector<BoundingBox<DIM>>& referenceBoxes,
								const std::vector<Vector<DIM>>& referenceCentroids,
								int nodeStart, int nodeEnd, int& splitDim, float& splitCoord)
{
	float splitCost = maxFloat;
	splitDim = -1;
	splitCoord = 0.0f;

	if (costHeuristic != CostHeuristic::LongestAxisCenter) {
		// initialize buckets
		const int nBuckets = 8;
		Vector<DIM> extent = nodeBoundingBox.extent();
		float surfaceArea = nodeBoundingBox.surfaceArea();
		float volume = nodeBoundingBox.volume();
		std::vector<std::pair<BoundingBox<DIM>, int>> buckets(nBuckets,
							std::make_pair(BoundingBox<DIM>(true), 0));

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

			// evaluate bucket split costs
			evaluateBucketSplitCosts<DIM>(costHeuristic, nodeBoundingBox, buckets,
										  nBuckets, bucketWidth, surfaceArea, volume,
										  dim, splitCost, splitDim, splitCoord);
		}
	}

	// if no split dimension was chosen, fallback to LongestAxisCenter heuristic
	if (splitDim == -1) {
		splitDim = nodeCentroidBox.maxDimension();
		splitCoord = (nodeCentroidBox.pMin[splitDim] + nodeCentroidBox.pMax[splitDim])*0.5f;
	}

	return splitCost;
}

// computeSpatialSplit
// splitReference
// performSpatialSplit

template <int DIM>
inline void Sbvh<DIM>::buildRecursive(std::vector<BoundingBox<DIM>>& referenceBoxes,
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
	if (node.rightOffset == 0) return;

	// choose splitDim and splitCoord based on cost heuristic
	int splitDim;
	float splitCoord;
	float splitCost = computeObjectSplit<DIM>(costHeuristic, bb, bc,
											  referenceBoxes, referenceCentroids,
											  start, end, splitDim, splitCoord);

	// partition the list of references on this split
	int mid = start;
	for (int i = start; i < end; i++) {
		if (referenceCentroids[i][splitDim] < splitCoord) {
			std::swap(references[i], references[mid]);
			std::swap(referenceBoxes[i], referenceBoxes[mid]);
			std::swap(referenceCentroids[i], referenceCentroids[mid]);
			mid++;
		}
	}

	// if we get a bad split, just choose the center...
	if (mid == start || mid == end) {
		mid = start + (end - start)/2;
	}

	// push left and right children
	buildRecursive(referenceBoxes, referenceCentroids, buildNodes,
				   currentNodeIndex, start, mid);
	buildRecursive(referenceBoxes, referenceCentroids, buildNodes,
				   currentNodeIndex, mid, end);
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

	for (int i = 0; i < nReferences; i++) {
		referenceBoxes.emplace_back(primitives[i]->boundingBox());
		referenceCentroids.emplace_back(primitives[i]->centroid());
		references.emplace_back(i);
	}

	std::vector<SbvhFlatNode<DIM>> buildNodes;
	buildNodes.reserve(nReferences*2);

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
