#include <stack>
#include <queue>
#include <chrono>
#include <set>
#define DOTESTS false
#define NEAR_ZERO 1e-12

namespace fcpw{

    template <int DIM>
    struct SbvhBuildEntry{
        // constructor
        SbvhBuildEntry(int parent_, std::vector<ReferenceWrapper<DIM>> references_, int curDepth_):
                    parent(parent_), curDepth(curDepth_), references(references_){}
        int parent, curDepth;
        std::vector<ReferenceWrapper<DIM>> references;
    };

    template <int DIM>
    struct SplitBin{
        // constructor
        SplitBin(): bounds(BoundingBox<DIM>()), enter(0), exit(0){}
        BoundingBox<DIM> bounds;
        // sbvh entry and exit counters
        int enter, exit;
    };

    template <int DIM>
    inline Sbvh<DIM>::Sbvh(std::vector<std::shared_ptr<Primitive<DIM>>>& primitives_, int leafSize_, int splittingMethod_, int binCount_, bool doUnsplitting_, bool fillLeaves_):
        Bvh<DIM>(primitives_, leafSize_, splittingMethod_, binCount_, false), nReferences(0), fillLeaves(fillLeaves_), doUnsplitting(doUnsplitting_), nSpatialSplits(0), nObjectSplits(0){

        // setup timer code
        std::chrono::high_resolution_clock::time_point t_start, t_end;
        std::chrono::nanoseconds duration;
        double buildTime = 0;

        // construct and time sbvh construction
        t_start = std::chrono::high_resolution_clock::now();
        build();
        t_end = std::chrono::high_resolution_clock::now();
        duration = t_end - t_start;
        buildTime = (double)(duration.count()) / std::chrono::nanoseconds::period::den;

        // output some stats about finished SBVH
        std::string bvhSplitHeuristic;
        std::string sbvhSplitHeuristic;
        switch(Bvh<DIM>::splittingMethod){
            case 1:
                bvhSplitHeuristic = sbvhSplitHeuristic = "Volume";
                break;
            case 2:
                bvhSplitHeuristic = "Overlap Surface Area";
                sbvhSplitHeuristic = "Surface Area";
                break;
            case 3:
                bvhSplitHeuristic = "Overlap Volume";
                sbvhSplitHeuristic = "Volume";
                break;
            default:
                bvhSplitHeuristic = sbvhSplitHeuristic = "Surface Area";
                break;
        }
        std::cout << "Sbvh created with "
                    << Bvh<DIM>::nNodes << " nodes, "
                    << Bvh<DIM>::nLeaves << " leaves, "
                    << Bvh<DIM>::primitives.size() << " primitives, "
                    << Bvh<DIM>::depth << " depth, in "
                    << buildTime << " seconds, using the "
                    << bvhSplitHeuristic << " heuristic for the Bvh and the "
                    << sbvhSplitHeuristic << " heuristic for the Sbvh" << std::endl;
        // LOG(INFO) << "Number of spatial splits: " << nSpatialSplits << " Number of object splits: " << nObjectSplits << " Ratio of spatial splits to object splits: " << (nObjectSplits == 0 ? "NAN" : std::to_string((float)nSpatialSplits / nObjectSplits));
    }
    //The following is adapted from the RadeonRays SDK in order to better integrate it with our datatypes

    template <int DIM>
    inline bool Sbvh<DIM>::splitReference(const ReferenceWrapper<DIM>& reference, int dim, float split, ReferenceWrapper<DIM>& loRef, ReferenceWrapper<DIM>& hiRef){
        // set initial values for references
        loRef.index  = reference.index;
        hiRef.index = reference.index;
        loRef.bbox   = reference.bbox;
        hiRef.bbox  = reference.bbox;

        if(split > reference.bbox.pMin(dim) && split < reference.bbox.pMax(dim)){
            // if box straddles split plane, split according to primitive rules
            const std::shared_ptr<Primitive<DIM>>& primitive = Bvh<DIM>::primitives[reference.index];
            primitive->split(reference.bbox, loRef.bbox, hiRef.bbox, dim, split);
            return true;
        }
        return false;
    }

    template <int DIM>
    inline SbvhSplit<DIM> Sbvh<DIM>::splitProbabilityHeuristic(std::vector<ReferenceWrapper<DIM>>& references, BoundingBox<DIM> bc, BoundingBox<DIM> bb, int binCount, costFunction<DIM> cost){

        // setup container for multiple splits
        std::vector<SbvhSplit<DIM>> results;
        for(int i = 0; i < DIM; i++){
            results.emplace_back(SbvhSplit<DIM>());
            results.back().split = 0.0;
            results.back().cost = maxFloat;
            results.back().axis = i;
        }

        // setup useful containers
        std::vector<SplitBin<DIM>> bins = std::vector<SplitBin<DIM>>(binCount);
        std::vector<float> splits = std::vector<float>(binCount - 1);
        
        for(int dimension = 0; dimension < DIM; dimension++){
            // skip if bounding box is flat in this dimension
            if(bb.extent()[dimension] < NEAR_ZERO) continue;

            // select split result container corresponding to this dimension
            SbvhSplit<DIM>& res = results[dimension];

            // setup containers and variables for binning process
            float delta = bb.extent()[res.axis] / binCount;
            float loBound = bb.pMin(res.axis);
            for(int i = 0; i < binCount; i++){
                bins[i] = SplitBin<DIM>();
            }
            for(int i = 0; i < binCount - 1; i++){
                splits[i] = delta * (i + 1) + loBound;
            }

            // loop over primitives and bin them
            for(ReferenceWrapper<DIM> reference : references){
                // unpack reference
                BoundingBox<DIM> bbox = reference.bbox;
                int index = reference.index;

                // get hi and lo bound from bbox, compute index of bounds
                float refLoBound = bbox.pMin(res.axis);
                float refHiBound = bbox.pMax(res.axis);
                int loIndex = (refLoBound - loBound) / delta;
                int hiIndex = (refHiBound - loBound) / delta;

                // clamp bounds of reference
                loIndex = loIndex + (loIndex < 0 ? 1 : (loIndex >= splits.size() ? -1 : 0));
                hiIndex = hiIndex + (hiIndex < 0 ? 1 : (hiIndex >= splits.size() ? -1 : 0));
                if(loIndex < binCount - 1 && loIndex < hiIndex && std::fabs(splits[loIndex + 1] - refLoBound) < NEAR_ZERO){
                    // if lo bound is close to split plane, but rounding placed it at a lower bin, shift bin up by one
                    loIndex ++;
                }
                if(hiIndex > 0 && hiIndex > loIndex && (hiIndex == splits.size() || std::fabs(splits[hiIndex] - refHiBound) < NEAR_ZERO)){
                    // if hi bound is close to split plane, and rounding placed it at a higher bin, shift bin down by one
                    hiIndex --;
                }

                // split reference, bin splits
                bins[loIndex].enter++;
                bins[hiIndex].exit++;
                ReferenceWrapper<DIM> curRef = reference;
                for(int j = loIndex; j < hiIndex; j++){
                    ReferenceWrapper<DIM> loRef, hiRef;
                    if(splitReference(curRef, res.axis, splits[j], loRef, hiRef)){
                        bins[j].bounds.expandToInclude(loRef.bbox);
                        curRef = hiRef;
                    }
                }
                bins[hiIndex].bounds.expandToInclude(curRef.bbox);
            }

            // initialize variables to find optimal split
            BoundingBox<DIM> hiBins[binCount - 1];
            BoundingBox<DIM> loBox = BoundingBox<DIM>();
            BoundingBox<DIM> hiBox;
            int loCount = 0;
            int hiCount = references.size();
            res.cost = maxFloat;
            hiBins[binCount - 2] = bins.back().bounds;

            // fill array with boxes grown from the hi value along dim
            for(int i = binCount - 2; i > 0; i--){
                hiBins[i - 1] = hiBins[i];
                hiBins[i - 1].expandToInclude(bins[i].bounds);
            }

            // find optimal split scanning from lo bin to hi bin
            for(int i = 0; i < binCount - 1; i++){
                // grow lo box, choose shrunken hi box
                loBox.expandToInclude(bins[i].bounds);
                hiBox = hiBins[i];
                hiCount -= bins[i].exit;
                loCount += bins[i].enter;

                // get cost for current split and compare
                float tempCost = cost(loBox, hiBox, bb, loCount, hiCount, (bb.volume() < epsilon));
                if(tempCost < res.cost){
                    // update if current split is better than prior splits
                    res.loBox = loBox;
                    res.hiBox = hiBox;
                    res.cost = tempCost;
                    res.split = splits[i];
                    res.entry = hiCount;
                    res.exit = loCount;
                }
            }
        }

        // get and return best split
        SbvhSplit<DIM> bestSplit = results[0];
        for(int i = 1; i < DIM; i++){
            if(results[i].cost < bestSplit.cost){
                bestSplit = results[i];
            }
        }
        return bestSplit;
    }

    template <int DIM>
    inline void Sbvh<DIM>::build(){

        // setup stack of worker nodes
        std::stack<SbvhBuildEntry<DIM>> todo;
        const int Untouched    = 0xffffffff;
        const int TouchedTwice = 0xfffffffd;

        // setup initial references
        std::vector<ReferenceWrapper<DIM>> initReferences = std::vector<ReferenceWrapper<DIM>>();
        initReferences.reserve(Bvh<DIM>::primitives.size());
        for(int i = 0; i < Bvh<DIM>::primitives.size(); i++){
            initReferences.emplace_back(ReferenceWrapper<DIM>(Bvh<DIM>::primitives[i]->boundingBox(), i));
        }
        references.reserve(initReferences.size());
        // push the root
        todo.emplace(SbvhBuildEntry<DIM>(0xfffffffc, initReferences, 0));

        // setup buildnode 
        std::vector<BvhFlatNode<DIM>> buildNodes;
        buildNodes.reserve(Bvh<DIM>::primitives.size()*2);

        while (!todo.empty()) {
            // pop the next item off the stack
            SbvhBuildEntry<DIM> buildEntry = todo.top();
            todo.pop();

            // extract primitive reference and depth from buildentry
            std::vector<ReferenceWrapper<DIM>> curReferences = std::move(buildEntry.references);
            int parent = buildEntry.parent;
            int curDepth = buildEntry.curDepth;
            if(Bvh<DIM>::depth < curDepth){
                Bvh<DIM>::depth = curDepth;
            }

            // fill node data with reference info
            Bvh<DIM>::nNodes++;
            buildNodes.emplace_back(BvhFlatNode<DIM>());
            BvhFlatNode<DIM>& node = buildNodes.back();
            node.rightOffset = Untouched;
            node.parentIndex = parent;
            node.start = -1;
            node.nPrimitives = -1;

            // calculate the bounding box for this node
            BoundingBox<DIM> bb, bc;
            for(ReferenceWrapper<DIM> reference : curReferences){
               bb.expandToInclude(reference.bbox);
               bc.expandToInclude(reference.bbox.centroid()); 
            }
            node.bbox = bb;
            
            // leaf size test
            if(curReferences.size() <= Bvh<DIM>::leafSize){
                // is a leaf
                node.rightOffset = 0;
                Bvh<DIM>::nLeaves ++;
                node.start = references.size();
                node.nPrimitives = curReferences.size();
                nReferences += node.nPrimitives;
                references.insert(references.end(), curReferences.begin(), curReferences.end());

                std::vector<ReferenceWrapper<DIM>>().swap(curReferences);
            }

            // child touches parent...
            // special case: don't do this for the root
            if (parent != 0xfffffffc) {
                buildNodes[parent].rightOffset--;

                // when this is the second touch, this is the right child;
                // the right child sets up the offset for the flat tree
                if (buildNodes[parent].rightOffset == TouchedTwice) {
                    buildNodes[parent].rightOffset = Bvh<DIM>::nNodes - 1 - parent;
                }
            }

            // if leaf, no need to split
            if(node.rightOffset == 0){
                continue;
            }

            // get cost function
            costFunction<DIM> objectHeuristic;
            costFunction<DIM> spatialHeuristic;
            switch(Bvh<DIM>::splittingMethod){
                case 1:
                    objectHeuristic = spatialHeuristic = &volumeCost;
                    break;
                case 2:
                    objectHeuristic = &overlapSurfaceAreaCost;
                    spatialHeuristic = &surfaceAreaCost;
                    break;
                case 3:
                    objectHeuristic = &overlapVolumeCost;
                    spatialHeuristic = &volumeCost;
                    break;
                default:
                    objectHeuristic = spatialHeuristic = &surfaceAreaCost;
                    break;
            }

            // get results from sbvh and bvh methods for splitting plane location
            BvhSplit objectSplitRes = probabilityHeuristic(curReferences, bc, bb, Bvh<DIM>::binCount, objectHeuristic);
            SbvhSplit<DIM> spatialSplitRes = splitProbabilityHeuristic(curReferences, bc, bb, Bvh<DIM>::binCount, spatialHeuristic);

            // if leaf, no need to split
            if(node.rightOffset == 0){
                continue;
            }

            // interior node
            std::vector<ReferenceWrapper<DIM>> loRefs;
            std::vector<ReferenceWrapper<DIM>> hiRefs;
            loRefs.reserve(curReferences.size());
            hiRefs.reserve(curReferences.size());

            // determine whether to use object or spatial split
            bool isBvhNode = objectSplitRes.cost <= spatialSplitRes.cost;
            float split, cost;
            int splitDim;
            if(isBvhNode){
                cost = objectSplitRes.cost;
                split = objectSplitRes.split;
                splitDim = objectSplitRes.axis;
                nObjectSplits ++;
            }
            else{
                cost = spatialSplitRes.cost;
                split = spatialSplitRes.split;
                splitDim = spatialSplitRes.axis;
                nSpatialSplits ++;
            }



            // perform partition depending on previous determination
            BoundingBox<DIM> loBox = BoundingBox<DIM>();
            BoundingBox<DIM> hiBox = BoundingBox<DIM>();
            for(ReferenceWrapper<DIM> reference : curReferences){
                if(!isBvhNode
                && reference.bbox.pMin(splitDim) < split && reference.bbox.pMax(splitDim) > split
                && std::fabs(reference.bbox.pMin(splitDim) - split) > NEAR_ZERO
                && std::fabs(reference.bbox.pMax(splitDim) - split) > NEAR_ZERO){
                    // reference straddles split, and is sbvh, split reference and append
                    BoundingBox<DIM> bbox = reference.bbox;
                    int index = reference.index;
                    ReferenceWrapper<DIM> loRef = ReferenceWrapper<DIM>();
                    ReferenceWrapper<DIM> hiRef = ReferenceWrapper<DIM>();
                    splitReference(reference, splitDim, split, loRef, hiRef);

                    loRefs.emplace_back(loRef);
                    hiRefs.emplace_back(hiRef);
                    loBox.expandToInclude(loRef.bbox);
                    hiBox.expandToInclude(hiRef.bbox);
                }
                else{
                    // bvh behavior, fully in one bin or the other
                    if(reference.bbox.centroid()[splitDim] <= split){
                        loRefs.emplace_back(reference);
                        loBox.expandToInclude(reference.bbox);
                    }
                    else{
                        hiRefs.emplace_back(reference);
                        hiBox.expandToInclude(reference.bbox);
                    }
                }
            }

            // perform unsplitting
            if(!isBvhNode && doUnsplitting){
                // set up unsplitting variables
                int loCount = loRefs.size();
                int hiCount = hiRefs.size();
                int loIndex = 0;
                int hiIndex = 0;
                float loUnsplitCost, hiUnsplitCost;

                // perform unsplitting check on all refs
                while(loIndex < loRefs.size() && hiIndex < hiRefs.size()){
                    int loPrimIdx = loRefs[loIndex].index;
                    int hiPrimIdx = hiRefs[hiIndex].index;
                    if(loPrimIdx == hiPrimIdx){
                        // perform unsplitting on bounding boxes
                        BoundingBox<DIM> loUnsplitBox = loBox;
                        loUnsplitBox.expandToInclude(hiRefs[hiIndex].bbox);
                        BoundingBox<DIM> hiUnsplitBox = hiBox;
                        hiUnsplitBox.expandToInclude(loRefs[loIndex].bbox);
                        
                        // compute unsplitting costs
                        loUnsplitCost = spatialHeuristic(loUnsplitBox, hiBox, bb, loCount, hiCount - 1, false);
                        hiUnsplitCost = spatialHeuristic(loBox, hiUnsplitBox, bb, loCount - 1, hiCount, false);

                        // compare unsplitting costs and case on comparisons
                        if(loUnsplitCost < cost && loUnsplitCost < hiUnsplitCost){
                            // unsplitting is best on lo
                            loBox = loUnsplitBox;
                            hiCount --;
                            loRefs[loIndex].bbox.expandToInclude(hiRefs[hiIndex].bbox);
                            hiRefs.erase(hiRefs.begin() + hiIndex);
                            loIndex ++;
                        }
                        else if(hiUnsplitCost < cost){
                            // unsplitting is best on hi
                            hiBox = hiUnsplitBox;
                            loCount --;
                            hiRefs[hiIndex].bbox.expandToInclude(loRefs[loIndex].bbox);
                            loRefs.erase(loRefs.begin() + loIndex);
                            hiIndex ++;
                        }
                        else{
                            // unsplitting failed, move on
                            loIndex ++;
                            hiIndex ++;
                        }
                    }
                    else{
                        // two refs to diff prims
                        if(loPrimIdx < hiPrimIdx){
                            loIndex ++;
                        }
                        else{
                            hiIndex ++;
                        }
                    }
                }
            }

            // shrink container sizes
            loRefs.shrink_to_fit();
            hiRefs.shrink_to_fit();
 
            // child nodes go onto working set
            todo.emplace(SbvhBuildEntry<DIM>(Bvh<DIM>::nNodes - 1, hiRefs, curDepth + 1));
            todo.emplace(SbvhBuildEntry<DIM>(Bvh<DIM>::nNodes - 1, loRefs, curDepth + 1));
        }

        // shift temporary tree over to tree in SBVH
        Bvh<DIM>::flatTree.clear();
        Bvh<DIM>::flatTree.reserve(Bvh<DIM>::nNodes);
        for (int n = 0; n < Bvh<DIM>::nNodes; n++) {
            Bvh<DIM>::flatTree.emplace_back(buildNodes[n]);
        }
    }

    template <int DIM>
    inline int Sbvh<DIM>::intersect(Ray<DIM>& r, std::vector<Interaction<DIM>>& is, bool checkOcclusion, bool countHits) const{
        #ifdef PROFILE
            PROFILE_SCOPED();
        #endif

        // track visited primitives for double intersect check
        std::set<int> visitedPrims({});

        int hits = 0;
        if(!countHits) is.resize(1);
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
            const BvhFlatNode<DIM>& node(Bvh<DIM>::flatTree[ni]);

            // if this node is further than the closest found intersection, continue
            if (!countHits && near > r.tMax) continue;

            // is leaf -> intersect
            if (node.rightOffset == 0){
                for(int ridx = node.start; ridx < node.start + node.nPrimitives; ridx++){
                    const ReferenceWrapper<DIM>& reference = references[ridx];
                    // skip if primitive has been intersected before
                    if(countHits){
                        if(visitedPrims.find(reference.index) != visitedPrims.end()) continue;
                        else visitedPrims.emplace(reference.index);
                    }
                    std::vector<Interaction<DIM>> cs;
                    const std::shared_ptr<Primitive<DIM>>& prim = Bvh<DIM>::primitives[reference.index];
                    int hit = prim->intersect(r, cs, checkOcclusion, countHits);
                    
                    // keep the closest intersection only
                    if (hit > 0) {
                        hits += hit;
                        if (countHits){
                            is.insert(is.end(), cs.begin(), cs.end());
                        }
                        else{
                            r.tMax = std::min(r.tMax, cs[0].d);
                            is[0] = cs[0];
                        }
                        if(checkOcclusion) return 1;
                    }
                }
            } else { // not a leaf
                bool hit0 = Bvh<DIM>::flatTree[ni + 1].bbox.intersect(r, bbhits[0], bbhits[1]);
                bool hit1 = Bvh<DIM>::flatTree[ni + node.rightOffset].bbox.intersect(r, bbhits[2], bbhits[3]);

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

        if(countHits){
            std::sort(is.begin(), is.end(), compareInteractions<DIM>);
            is = removeDuplicates<DIM>(is);
            hits = (int)is.size();
        }

        return hits;
    }

    template <int DIM>
    inline bool Sbvh<DIM>::applyClosestPoint(BoundingSphere<DIM>& s, Interaction<DIM>& i, int pos) const{
        const ReferenceWrapper<DIM>& reference = references[pos];
        const std::shared_ptr<Primitive<DIM>>& primitive = Bvh<DIM>::primitives[reference.index];
        return primitive->findClosestPoint(s, i);
    }

    template <int DIM>
    inline void Sbvh<DIM>::convert(const int simdWidth, std::shared_ptr<Aggregate<DIM>>& mbvh){
        
    }

}