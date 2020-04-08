#include <stack>
#include <queue>

namespace fcpw {

struct BvhBuildEntry {
	// constructor
	BvhBuildEntry(int parent_, int start_, int end_):
				  parent(parent_), start(start_), end(end_) {}

	// members
	int parent; // if non-zero then this is the index of the parent (used in offsets)
	int start, end; // the range of primitives in the primitive list covered by this node
};

struct BvhTraversal {
	// constructor
	BvhTraversal(int i_, float d_): i(i_), d(d_) {}

	// members
	int i; // node index
	float d; // minimum distance (parametric, squared, ...) to this node
};

template <int DIM>
inline Bvh<DIM>::Bvh(std::vector<std::shared_ptr<Primitive<DIM>>>& primitives_,
					 const CostHeuristic& costHeuristic_, int leafSize_):
nNodes(0),
nLeafs(0),
leafSize(leafSize_),
primitives(primitives_)
{
	using namespace std::chrono;
	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	build(costHeuristic_);

	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	duration<double> timeSpan = duration_cast<duration<double>>(t2 - t1);
	std::cout << "Built Bvh with "
			  << nNodes << " nodes, "
			  << nLeafs << " leaves, "
			  << primitives.size() << " primitives in "
			  << timeSpan.count() << " seconds" << std::endl;
}

template <int DIM>
inline float computeSplitCost(const CostHeuristic& costHeuristic,
							  const BoundingBox<DIM>& bboxLeft,
							  const BoundingBox<DIM>& bboxRight,
							  const BoundingBox<DIM>& bboxParent,
							  int nPrimitivesLeft, int nPrimitivesRight)
{
	float cost = maxFloat;
	if (costHeuristic == CostHeuristic::SurfaceArea) {
		cost = (nPrimitivesLeft*bboxLeft.surfaceArea() +
				nPrimitivesRight*bboxRight.surfaceArea())/bboxParent.surfaceArea();

	} else if (costHeuristic == CostHeuristic::OverlapSurfaceArea) {
		// cost can be negative, but that's a good thing, because the more negative the cost
		// the further apart the left and right boxes should be
		BoundingBox<DIM> bboxIntersected = bboxLeft.intersect(bboxRight);
		cost = (nPrimitivesLeft/bboxRight.surfaceArea() +
				nPrimitivesRight/bboxLeft.surfaceArea())*bboxIntersected.surfaceArea();

	} else if (costHeuristic == CostHeuristic::Volume) {
		cost = (nPrimitivesLeft*bboxLeft.volume() +
				nPrimitivesRight*bboxRight.volume())/bboxParent.volume();

	} else if (costHeuristic == CostHeuristic::OverlapVolume) {
		// cost can be negative, but that's a good thing, because the more negative the cost
		// the further apart the left and right boxes should be
		BoundingBox<DIM> bboxIntersected = bboxLeft.intersect(bboxRight);
		cost = (nPrimitivesLeft/bboxRight.volume() +
				nPrimitivesRight/bboxLeft.volume())*bboxIntersected.volume();
	}

	return cost;
}

template <int DIM>
inline float computeSplit(const CostHeuristic& costHeuristic,
						  const BvhFlatNode<DIM>& node,
						  const BoundingBox<DIM>& nodeCentroidBox,
						  const std::vector<BoundingBox<DIM>>& primitiveBoxes,
						  const std::vector<Vector<DIM>>& primitiveCentroids,
						  int& splitDim, float& splitCoord)
{
	float splitCost = maxFloat;
	splitDim = -1;
	splitCoord = 0.0f;

	if (costHeuristic != CostHeuristic::LongestAxisCenter) {
		// initialize buckets
		const int nBuckets = 8;
		Vector<DIM> extent = node.bbox.extent();
		std::vector<std::pair<BoundingBox<DIM>, int>> buckets(nBuckets,
							std::make_pair(BoundingBox<DIM>(false), 0));

		// find the best split across all three dimensions
		for (int dim = 0; dim < DIM; dim++) {
			// ignore flat dimension
			if (extent(dim) < 1e-6) continue;

			// bin primitives into buckets
			float bucketWidth = extent(dim)/nBuckets;
			for (int b = 0; b < nBuckets; b++) {
				buckets[b].first = BoundingBox<DIM>(false);
				buckets[b].second = 0;
			}

			for (int p = node.start; p < node.start + node.nPrimitives; p++) {
				int bucketIndex = (int)((primitiveCentroids[p](dim) - node.bbox.pMin(dim))/bucketWidth);
				bucketIndex = clamp(bucketIndex, 0, nBuckets - 1);
				buckets[bucketIndex].first.expandToInclude(primitiveBoxes[p]);
				buckets[bucketIndex].second += 1;
			}

			// evaluate split costs
			for (int b = 1; b < nBuckets; b++) {
				// compute left and right child boxes for this particular split
				BoundingBox<DIM> bboxLeft(false), bboxRight(false);
				int nPrimitivesLeft = 0, nPrimitivesRight = 0;

				for (int i = 0; i < b; i++) {
					bboxLeft.expandToInclude(buckets[i].first);
					nPrimitivesLeft += buckets[i].second;
				}

				for (int i = b; i < nBuckets; i++) {
					bboxRight.expandToInclude(buckets[i].first);
					nPrimitivesRight += buckets[i].second;
				}

				// compute split cost based on heuristic
				float cost = computeSplitCost(costHeuristic, bboxLeft, bboxRight, node.bbox,
											  nPrimitivesLeft, nPrimitivesRight);
				float coord = node.bbox.pMin(dim) + b*bucketWidth;

				if (cost < splitCost) {
					splitCost = cost;
					splitDim = dim;
					splitCoord = coord;
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
inline void Bvh<DIM>::build(const CostHeuristic& costHeuristic)
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	std::stack<BvhBuildEntry> todo;
	const int Untouched    = 0xffffffff;
	const int TouchedTwice = 0xfffffffd;

	// precompute bounding boxes and centroids
	int nPrimitives = (int)primitives.size();
	std::vector<BoundingBox<DIM>> primitiveBoxes;
	std::vector<Vector<DIM>> primitiveCentroids;

	for (int i = 0; i < nPrimitives; i++) {
		primitiveBoxes.emplace_back(primitives[i]->boundingBox());
		primitiveCentroids.emplace_back(primitives[i]->centroid());
	}

	// push the root
	todo.emplace(BvhBuildEntry(0xfffffffc, 0, nPrimitives));

	BvhFlatNode<DIM> node;
	std::vector<BvhFlatNode<DIM>> buildNodes;
	buildNodes.reserve(nPrimitives*2);

	while (!todo.empty()) {
		// pop the next item off the stack
		BvhBuildEntry buildEntry = todo.top();
		todo.pop();

		int start = buildEntry.start;
		int end = buildEntry.end;
		int nPrimitives = end - start;

		nNodes++;
		node.start = start;
		node.nPrimitives = nPrimitives;
		node.rightOffset = Untouched;

		// calculate the bounding box for this node
		BoundingBox<DIM> bb(true), bc(true);
		for (int p = start; p < end; p++) {
			bb.expandToInclude(primitiveBoxes[p]);
			bc.expandToInclude(primitiveCentroids[p]);
		}

		node.bbox = bb;

		// if the number of primitives at this point is less than the leaf
		// size, then this will become a leaf (signified by rightOffset == 0)
		if (nPrimitives <= leafSize) {
			node.rightOffset = 0;
			nLeafs++;
		}

		buildNodes.emplace_back(node);

		// child touches parent...
		// special case: don't do this for the root
		if (buildEntry.parent != 0xfffffffc) {
			buildNodes[buildEntry.parent].rightOffset--;

			// when this is the second touch, this is the right child;
			// the right child sets up the offset for the flat tree
			if (buildNodes[buildEntry.parent].rightOffset == TouchedTwice) {
				buildNodes[buildEntry.parent].rightOffset = nNodes - 1 - buildEntry.parent;
			}
		}

		// if this is a leaf, no need to subdivide
		if (node.rightOffset == 0) {
			continue;
		}

		// choose splitDim and splitCoord based on cost heuristic
		int splitDim;
		float splitCoord;
		float splitCost = computeSplit<DIM>(costHeuristic, node, bc, primitiveBoxes,
											primitiveCentroids, splitDim, splitCoord);

		// partition the list of primitives on this split
		int mid = start;
		for (int i = start; i < end; i++) {
			if (primitiveCentroids[i][splitDim] < splitCoord) {
				std::swap(primitives[i], primitives[mid]);
				std::swap(primitiveBoxes[i], primitiveBoxes[mid]);
				std::swap(primitiveCentroids[i], primitiveCentroids[mid]);
				mid++;
			}
		}

		// if we get a bad split, just choose the center...
		if (mid == start || mid == end) {
			mid = start + (end - start)/2;
		}

		// push right and left children
		todo.emplace(BvhBuildEntry(nNodes - 1, mid, end));
		todo.emplace(BvhBuildEntry(nNodes - 1, start, mid));
	}

	// copy the temp node data to a flat array
	flatTree.clear();
	flatTree.reserve(nNodes);
	for (int n = 0; n < nNodes; n++) {
		flatTree.emplace_back(buildNodes[n]);
	}
}

template <int DIM>
inline BoundingBox<DIM> Bvh<DIM>::boundingBox() const
{
	return flatTree.size() > 0 ? flatTree[0].bbox : BoundingBox<DIM>(false);
}

template <int DIM>
inline Vector<DIM> Bvh<DIM>::centroid() const
{
	return boundingBox().centroid();
}

template <int DIM>
inline float Bvh<DIM>::surfaceArea() const
{
	float area = 0.0f;
	for (int p = 0; p < (int)primitives.size(); p++) {
		area += primitives[p]->surfaceArea();
	}

	return area;
}

template <int DIM>
inline float Bvh<DIM>::signedVolume() const
{
	float volume = 0.0f;
	for (int p = 0; p < (int)primitives.size(); p++) {
		volume += primitives[p]->signedVolume();
	}

	return volume;
}

template <int DIM>
inline int Bvh<DIM>::intersect(Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
							   bool checkOcclusion, bool countHits) const
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	int hits = 0;
	if (!countHits) is.resize(1);
	std::stack<BvhTraversal> todo;
	float bbhits[4];
	int closer, other;

	// "push" on the root node to the working set
	todo.emplace(BvhTraversal(0, minFloat));

	while (!todo.empty()) {
		// pop off the next node to work on
		BvhTraversal traversal = todo.top();
		todo.pop();

		int ni = traversal.i;
		float near = traversal.d;
		const BvhFlatNode<DIM>& node(flatTree[ni]);

		// if this node is further than the closest found intersection, continue
		if (!countHits && near > r.tMax) continue;

		// is leaf -> intersect
		if (node.rightOffset == 0) {
			for (int p = 0; p < node.nPrimitives; p++) {
				std::vector<Interaction<DIM>> cs;
				const std::shared_ptr<Primitive<DIM>>& prim = primitives[node.start + p];
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
				todo.emplace(BvhTraversal(other, bbhits[2]));
				todo.emplace(BvhTraversal(closer, bbhits[0]));

			} else if (hit0) {
				todo.emplace(BvhTraversal(ni + 1, bbhits[0]));

			} else if (hit1) {
				todo.emplace(BvhTraversal(ni + node.rightOffset, bbhits[2]));
			}
		}
	}

	if (countHits) {
		std::sort(is.begin(), is.end(), compareInteractions<DIM>);
		is = removeDuplicates<DIM>(is);
		hits = (int)is.size();
	}

	return hits;
}

template <int DIM>
inline bool Bvh<DIM>::findClosestPoint(BoundingSphere<DIM>& s, Interaction<DIM>& i) const
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	bool notFound = true;
	std::queue<BvhTraversal> todo;
	float bbhits[4];
	int closer, other;

	// "push" on the root node to the working set
	todo.emplace(BvhTraversal(0, minFloat));

	while (!todo.empty()) {
		// pop off the next node to work on
		BvhTraversal traversal = todo.front();
		todo.pop();

		int ni = traversal.i;
		float near = traversal.d;
		const BvhFlatNode<DIM>& node(flatTree[ni]);

		// if this node is further than the closest found primitive, continue
		if (near > s.r2) {
			continue;
		}

		// is leaf -> compute squared distance
		if (node.rightOffset == 0) {
			for (int p = 0; p < node.nPrimitives; p++) {
				Interaction<DIM> c;
				const std::shared_ptr<Primitive<DIM>>& prim = primitives[node.start + p];
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
				todo.emplace(BvhTraversal(closer, bbhits[0]));
				todo.emplace(BvhTraversal(other, bbhits[2]));

			} else if (hit0) {
				todo.emplace(BvhTraversal(ni + 1, bbhits[0]));

			} else if (hit1) {
				todo.emplace(BvhTraversal(ni + node.rightOffset, bbhits[2]));
			}
		}
	}

	return !notFound;
}

} // namespace fcpw
