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

template <int DIM>
inline Bvh<DIM>::Bvh(std::vector<std::shared_ptr<Primitive<DIM>>>& primitives_, int leafSize_, int splittingMethod_, int binCount_):
nNodes(0),
nLeafs(0),
leafSize(leafSize_),
primitives(primitives_),
splittingMethod(splittingMethod_),
binCount(binCount_)
{
	using namespace std::chrono;
	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	build();

	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	duration<double> timeSpan = duration_cast<duration<double>>(t2 - t1);
	std::cout << "Built Bvh with "
			  << nNodes << " nodes, "
			  << nLeafs << " leaves, "
			  << primitives.size() << " primitives in "
			  << timeSpan.count() << " seconds" << std::endl;
}

template <int DIM>
inline void Bvh<DIM>::build()
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	std::stack<BvhBuildEntry> todo;
	const int Untouched    = 0xffffffff;
	const int TouchedTwice = 0xfffffffd;

	// precompute bounding boxes and centroids
	int nPrimitives = (int)primitives.size();
	std::vector<BoundingBox<DIM>> primitiveBoxes(nPrimitives);
	std::vector<Vector<DIM>> primitiveCentroids(nPrimitives);

	for (int i = 0; i < nPrimitives; i++) {
		primitiveBoxes[i] = primitives[i]->boundingBox();
		primitiveCentroids[i] = primitives[i]->centroid();
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
		BoundingBox<DIM> bb, bc;
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

		// set the split dimensions
		int splitDim = bc.maxDimension();

		float splitCoord = 0;

		// split on the center of the longest axis
		if(splittingMethod == 0){
		}
		else{
			costFunction<DIM> cost;
			switch(splittingMethod){
				case 0:
					splitCoord = (bc.pMin[splitDim] + bc.pMax[splitDim])*0.5;
					break;
				case 1:
					cost = &surfaceAreaCost;
					break;
				case 2:
					cost = &volumeCost;
					break;
				case 3:
					cost = &overlapSurfaceAreaCost;
					break;
				case 4:
					cost = &overlapVolumeCost;
					break;
				default:
					LOG(FATAL) << "Method number " << splittingMethod << " is an invalid splitting method";
			}
			if(splitCoord != 0){
				// TEMPORARY: it just auto generates a reference list; may have to change how a BVH is
				// built to reduce construction of these reference lists
				// or not try to overgeneralize probabilityHeuristic
				std::vector<ReferenceWrapper<DIM>> refs;
				for(int i = start; i < end; i++){
					refs.emplace_back(ReferenceWrapper<DIM>(primitives[i]->boundingBox(), i));
				}
				BvhSplit split = probabilityHeuristic(refs, bc, bb, binCount, cost);
				splitCoord = split.split;
				splitDim = split.axis;
			}
		}

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
	return flatTree.size() > 0 ? flatTree[0].bbox : BoundingBox<DIM>();
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
