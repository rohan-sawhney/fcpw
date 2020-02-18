#include <stack>
#include <queue>
#include <chrono>
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
    struct SahBin{
        // constructor
        SahBin(): bounds(BoundingBox<DIM>()), enter(0), exit(0){}
        BoundingBox<DIM> bounds;
        // sbvh entry and exit counters
        int enter, exit;
    };

    template <int DIM>
    inline Sbvh<DIM>::Sbvh(std::vector<std::shared_ptr<Primitive<DIM>>>& primitives_, int leafSize_, int splittingMethod_, bool makeBvh_, int binCount_, bool fillLeaves_):
        nNodes(0), nLeaves(0), leafSize(leafSize_), splittingMethod(splittingMethod_), primitives(primitives_), makeBvh(makeBvh_), binCount(binCount_), depth(0), nRefs(0), fillLeaves(fillLeaves_){

        times = std::unique_ptr<std::vector<double>>(new std::vector<double>({0, 0, 0}));

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
        switch(splittingMethod){
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
        LOG(INFO) << "Sbvh created with "
                    << nNodes << " nodes, "
                    << nLeaves << " leaves, "
                    << primitives.size() << " primitives, "
                    << depth << " depth, in "
                    << buildTime << " seconds, "
                    << "using the " << bvhSplitHeuristic << " heuristic for the Bvh and the " << sbvhSplitHeuristic << " heuristic for the Sbvh";
    }

    template <int DIM>
    inline SbvhSplit<DIM> Sbvh<DIM>::probabilityHeuristic(
        std::vector<ReferenceWrapper<DIM>>& references,
        BoundingBox<DIM> bc, BoundingBox<DIM> bb,
        int binCount, costFunction<DIM> cost){

        // setup container for multiple splits
        std::vector<SbvhSplit<DIM>> results;
        for(int i = 0; i < DIM; i++){
            results.emplace_back(SbvhSplit<DIM>());
            results.back().split = 0.0;
            results.back().cost = maxDouble;
            results.back().axis = i;
        }

        // setup useful containers
        std::vector<SahBin<DIM>> bins = std::vector<SahBin<DIM>>(binCount);
        std::vector<double> splits = std::vector<double>(binCount - 1);

        for(int dimension = 0; dimension < DIM; dimension++){
            // skip if bounding box is flat in this dimension
            if(bc.extent()[dimension] < NEAR_ZERO) continue;

            // select a split container corresponding to this dimension
            SbvhSplit<DIM>& res = results[dimension];

            // setup containers and variables for binning process
            double delta = bc.extent()[res.axis] / binCount;
            double loBound = bc.pMin(res.axis);
            for(int i = 0; i < binCount; i++){
                bins[i] = SahBin<DIM>();
            }
            for(int i = 0; i < binCount - 1; i++){
                splits[i] = delta * (i + 1) + loBound;
            }

            // loop over references and bin reference
            for(ReferenceWrapper<DIM> reference : references){
                // parse reference
                BoundingBox<DIM> bbox = reference.bbox;
                int index = reference.index;
                double center = bbox.centroid()[res.axis];

                // get index of bbox via centroid
                int binIndex = (center - loBound) / delta;
                LOG_IF(FATAL, binIndex < -1 || binIndex > binCount) << "Out of bounds bin index even before adjustments, index is at " << binIndex << " with only " << binCount << "bins. Delta is " << delta;
                if(binIndex == -1){
                    // if centroid is on lo end of box but negative bin index due to rounding, increase index
                    binIndex ++;
                }
                if(binIndex == binCount){
                    // if centroid is on hi end of box but out of bounds index due to rounding, reduce index
                    binIndex --;
                }

                // bin reference
                bins[binIndex].bounds.expandToInclude(bbox);
                bins[binIndex].enter ++;
            }

            // initialize variables to find optimal split
            BoundingBox<DIM> hiBins[binCount - 1];
            BoundingBox<DIM> loBox = BoundingBox<DIM>();
            BoundingBox<DIM> hiBox;
            int loCount = 0;
            int hiCount = references.size();
            res.cost = maxDouble;
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
                hiCount -= bins[i].enter;
                loCount += bins[i].enter;

                // get cost for current split and compare
                double tempCost = cost(loBox, hiBox, loCount, hiCount, (bb.volume() < epsilon), bb.surfaceArea(), bb.volume());
                if(tempCost < res.cost){
                    // update if current split better than prior splits
                    res.loBox = loBox;
                    res.hiBox = hiBox;
                    res.cost = tempCost;
                    res.split = splits[i];
                    res.entry = loCount;
                    res.exit = hiCount;
                }
            }
        }

        // get and return best split
        SbvhSplit<DIM> bestSplit = results[0];
        for(int i = 0; i < DIM; i++){
            if(results[i].cost < bestSplit.cost){
                bestSplit = results[i];
            }
        }
        return bestSplit;
    }

    //The following is adapted from the RadeonRays SDK in order to better integrate it with our datatypes

    template <int DIM>
    inline bool Sbvh<DIM>::splitReference(const ReferenceWrapper<DIM>& reference, int dim, double split, ReferenceWrapper<DIM>& loRef, ReferenceWrapper<DIM>& hiRef){
        // set initial values for references
        loRef.index  = reference.index;
        hiRef.index = reference.index;
        loRef.bbox   = reference.bbox;
        hiRef.bbox  = reference.bbox;

        if(split > reference.bbox.pMin(dim) && split < reference.bbox.pMax(dim)){
            // if box straddles split plane, split according to primitive rules
            const std::shared_ptr<Primitive<DIM>>& primitive = primitives[reference.index];
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
            results.back().cost = maxDouble;
            results.back().axis = i;
        }

        // setup useful containers
        std::vector<SahBin<DIM>> bins = std::vector<SahBin<DIM>>(binCount);
        std::vector<double> splits = std::vector<double>(binCount - 1);
        
        for(int dimension = 0; dimension < DIM; dimension++){
            // skip if bounding box is flat in this dimension
            if(bb.extent()[dimension] < NEAR_ZERO) continue;

            // select split result container corresponding to this dimension
            SbvhSplit<DIM>& res = results[dimension];

            // setup containers and variables for binning process
            double delta = bb.extent()[res.axis] / binCount;
            double loBound = bb.pMin(res.axis);
            for(int i = 0; i < binCount; i++){
                bins[i] = SahBin<DIM>();
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
                double refLoBound = bbox.pMin(res.axis);
                double refHiBound = bbox.pMax(res.axis);
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
            res.cost = maxDouble;
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
                double tempCost = cost(loBox, hiBox, loCount, hiCount, (bb.volume() < epsilon), bb.surfaceArea(), bb.volume());
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
        initReferences.reserve(primitives.size());
        for(int i = 0; i < primitives.size(); i++){
            initReferences.emplace_back(ReferenceWrapper<DIM>(primitives[i]->boundingBox(), i));
        }
        references.reserve(initReferences.size());
        // push the root
        todo.emplace(SbvhBuildEntry<DIM>(0xfffffffc, initReferences, 0));

        // setup buildnode 
        std::vector<BvhFlatNode<DIM>> buildNodes;
        buildNodes.reserve(primitives.size()*2);

        while (!todo.empty()) {
            // pop the next item off the stack
            SbvhBuildEntry<DIM> buildEntry = todo.top();
            todo.pop();

            // extract primitive reference and depth from buildentry
            std::vector<ReferenceWrapper<DIM>> curReferences = std::move(buildEntry.references);
            int parent = buildEntry.parent;
            int curDepth = buildEntry.curDepth;
            if(depth < curDepth){
                depth = curDepth;
            }

            // fill node data with reference info
            nNodes++;
            buildNodes.emplace_back(BvhFlatNode<DIM>());
            BvhFlatNode<DIM>& node = buildNodes.back();
            node.rightOffset = Untouched;
            node.parentIndex = parent;
            node.start = -1;
            node.nPrims = -1;

            // calculate the bounding box for this node
            BoundingBox<DIM> bb, bc;
            for(ReferenceWrapper<DIM> reference : curReferences){
               bb.expandToInclude(reference.bbox);
               bc.expandToInclude(reference.bbox.centroid()); 
            }
            node.bbox = bb;
            
            // leaf size test
            if(curReferences.size() <= leafSize){
                // is a leaf
                node.rightOffset = 0;
                nLeaves ++;
                node.start = references.size();
                node.nPrims = curReferences.size();
                nRefs += node.nPrims;
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
                    buildNodes[parent].rightOffset = nNodes - 1 - parent;
                }
            }

            // if leaf, no need to split
            if(node.rightOffset == 0){
                continue;
            }

            // get cost function
            costFunction<DIM> objectHeuristic;
            costFunction<DIM> spatialHeuristic;
            switch(splittingMethod){
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
            SbvhSplit<DIM> objectSplitRes = probabilityHeuristic(curReferences, bc, bb, binCount, objectHeuristic);
            SbvhSplit<DIM> spatialSplitRes = makeBvh ? SbvhSplit<DIM>() : splitProbabilityHeuristic(curReferences, bc, bb, binCount, spatialHeuristic);

            // compare two costs received
            double cost = makeBvh || objectSplitRes.cost < spatialSplitRes.cost ? objectSplitRes.cost : spatialSplitRes.cost;

            // leaf cost test (check if better than bruteforce)
            if(false){//cost > costOperation * curReferences.size()){
                // is a leaf
                node.rightOffset = 0;
                nLeaves ++;
                node.start = references.size();
                node.nPrims = curReferences.size();
                nRefs += node.nPrims;
                for(ReferenceWrapper<DIM> reference : curReferences){
                    references.emplace_back(reference);
                }
                std::vector<ReferenceWrapper<DIM>>().swap(curReferences);
            }

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
            bool isBvhNode = makeBvh || objectSplitRes.cost <= spatialSplitRes.cost;
            cost         = isBvhNode ? objectSplitRes.cost  : spatialSplitRes.cost;
            double split = isBvhNode ? objectSplitRes.split : spatialSplitRes.split;
            int splitDim = isBvhNode ? objectSplitRes.axis  : spatialSplitRes.axis;


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
                double loUnsplitCost, hiUnsplitCost;

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
                        loUnsplitCost = spatialHeuristic(loUnsplitBox, hiBox, loCount, hiCount - 1, false, bb.surfaceArea(), bb.volume());
                        hiUnsplitCost = spatialHeuristic(loBox, hiUnsplitBox, loCount - 1, hiCount, false, bb.surfaceArea(), bb.volume());

                        // compare unsplitting costs
                        bool loUnsplit = loUnsplitCost < cost;
                        bool hiUnsplit = hiUnsplitCost < cost;

                        // case on comparisons
                        if(loUnsplit && loUnsplitCost < hiUnsplitCost){
                            // unsplitting is best on lo
                            loBox = loUnsplitBox;
                            hiCount --;
                            loRefs[loIndex].bbox.expandToInclude(hiRefs[hiIndex].bbox);
                            hiRefs.erase(hiRefs.begin() + hiIndex);
                            loIndex ++;
                        }
                        else if(hiUnsplit){
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
            todo.emplace(SbvhBuildEntry<DIM>(nNodes - 1, hiRefs, curDepth + 1));
            todo.emplace(SbvhBuildEntry<DIM>(nNodes - 1, loRefs, curDepth + 1));
        }

        // shift temporary tree over to tree in SBVH
        flatTree.clear();
        flatTree.reserve(nNodes);
        for (int n = 0; n < nNodes; n++) {
            flatTree.emplace_back(buildNodes[n]);
        }
    }

    template <int DIM>
    inline BoundingBox<DIM> Sbvh<DIM>::boundingBox() const{
        return flatTree.size() > 0 ? flatTree[0].bbox : BoundingBox<DIM>();
    }

    template <int DIM>
    inline Vector<DIM> Sbvh<DIM>::centroid() const{
        return boundingBox().centroid();
    }

    template <int DIM>
    inline double Sbvh<DIM>::surfaceArea() const{
        double area = 0.0;
        for(std::shared_ptr<Primitive<DIM>> p : primitives){
            area += p->surfaceArea();
        }
        return area;
    }

    template <int DIM>
    inline double Sbvh<DIM>::signedVolume() const{
        double volume = 0.0;
        for(std::shared_ptr<Primitive<DIM>> p : primitives){
            volume += p->signedVolume();
        }
        return volume;
    }

    template <int DIM>
    inline int Sbvh<DIM>::intersect(Ray<DIM>& r, Interaction<DIM>& i, bool countHits) const{
        #ifdef PROFILE
            PROFILE_SCOPED();
        #endif

        int hits = 0;
        std::stack<BvhTraversal> todo;
        double bbhits[4];
        int closer, other;
        int numVisited = 0;

        // "push" on the root node to the working set
        todo.emplace(BvhTraversal(0, minDouble));

        while (!todo.empty()) {
            numVisited ++;
            // pop off the next node to work on
            BvhTraversal traversal = todo.top();
            todo.pop();

            int ni = traversal.i;
            double near = traversal.d;
            const BvhFlatNode<DIM>& node(flatTree[ni]);

            // if this node is further than the closest found intersection, continue
            if (!countHits && near > r.tMax) {
                continue;
            }

            // is leaf -> intersect
            if (node.rightOffset == 0){// && node.references != nullptr) {
                for(int ridx = node.start; ridx < node.start + node.nPrims; ridx++){
                    Interaction<DIM> c;
                    const ReferenceWrapper<DIM>& reference = references[ridx];
                    const std::shared_ptr<Primitive<DIM>>& prim = primitives[reference.index];
                    int hit = prim->intersect(r, c);
                    
                    // keep the closest intersection only
                    if (hit > 0) {
                        if (c.d < r.tMax) {
                            if (!countHits) r.tMax = c.d;
                            i = c;
                        }

                        hits += hit;
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

        return hits;
    }

    template <int DIM>
    inline void Sbvh<DIM>::findClosestPoint(BoundingSphere<DIM>& s, Interaction<DIM>& i) const{
        #ifdef PROFILE
            PROFILE_SCOPED();
        #endif

        std::queue<BvhTraversal> todo;
        double bbhits[4];
        int closer, other;

        // "push" on the root node to the working set
        todo.emplace(BvhTraversal(0, minDouble));
        while (!todo.empty()) {

            // pop off the next node to work on
            BvhTraversal traversal = todo.front();
            todo.pop();

            int ni = traversal.i;
            double near = traversal.d;
            const BvhFlatNode<DIM>& node(flatTree[ni]);

            // if this node is further than the closest found primitive, continue
            if (near > s.r2) {
                continue;
            }

            // is leaf -> compute squared distance
            if (node.rightOffset == 0){
                for(int ridx = node.start; ridx < node.start + node.nPrims; ridx++){
                    Interaction<DIM> c;
                    const ReferenceWrapper<DIM>& reference = references[ridx];
                    double dMin;
                    double dMax;
                    bool temp = reference.bbox.overlaps(s, dMin, dMax);
                    if(temp){
                        const std::shared_ptr<Primitive<DIM>>& prim = primitives[reference.index];
                        prim->findClosestPoint(s, c);

                        // keep the closest point only
                        if (c.d < s.r2) {
                            s.r2 = c.d;
                            i = c;
                            i.d = std::sqrt(i.d);
                            LOG_IF(FATAL, i.primitive == nullptr) << "Primitive " << reference.index << " referenced is null in sbvh";
                        }
                    }
                }

            } else { // not a leaf
                bool hit0 = flatTree[ni + 1].bbox.overlaps(s, bbhits[0], bbhits[1]);
                if (s.r2 > bbhits[1]) s.r2 = bbhits[1];

                bool hit1 = flatTree[ni + node.rightOffset].bbox.overlaps(s, bbhits[2], bbhits[3]);
                if (s.r2 > bbhits[3]) s.r2 = bbhits[3];

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
    }

    template <int DIM>
    inline BoundingBox<DIM> Sbvh<DIM>::traverse(int& curIndex, int gotoIdx) const{
        // update provided curIndex with new index depending on what the goto number is, return bbox of node
        // gotoIdx order is preorder traversal, cases 0 to 2 incl
        LOG_IF(FATAL, curIndex < 0 || curIndex >= flatTree.size()) << "Index provided is out of bounds of tree";
        const BvhFlatNode<DIM>& node = flatTree[curIndex];
        switch(gotoIdx){
            case 0:
                // parent
                if(curIndex == 0){
                    // root node
                    return node.bbox;
                }
                curIndex = node.parentIndex;
                break;
            case 1:
                // leftChild
                if(node.rightOffset == 0){
                    // leaf node
                    return node.bbox;
                }
                curIndex ++;
                break;
            case 2:
                // rightChild
                if(node.rightOffset == 0){
                    // root node
                    return node.bbox;
                }
                curIndex += node.rightOffset;
                break;
            default:
                LOG(FATAL) << "Invalid traversal index";
                break;
        }
        return flatTree[curIndex].bbox;
    }

    template <int DIM>
    inline void Sbvh<DIM>::getBoxList(int curIndex, int topDepth, int bottomDepth,
    std::vector<Vector<DIM>>& boxVertices, std::vector<std::vector<int>>& boxEdges,
    std::vector<Vector<DIM>>& curBoxVertices, std::vector<std::vector<int>>& curBoxEdges) const{
        int selectedBoxIndex = curIndex;
        BoundingBox<DIM> bbox = flatTree[curIndex].bbox;
        bbox.vertices(curBoxVertices, curBoxEdges, false);
        boxVertices.clear();
        boxEdges.clear();

        while(topDepth > 0){
            if(curIndex > 0){
                curIndex = flatTree[curIndex].parentIndex;
                bottomDepth ++;
            }
            topDepth --;
        }

        std::queue<BvhTraversal> todo;
        todo.emplace(BvhTraversal(curIndex, minDouble, 0));
        int boxIndex = 0;
        int boxPow = 1;
        for(int i = 0; i < DIM; i++) boxPow *= 2;

        while(!todo.empty()){
            BvhTraversal traversal = todo.front();
            todo.pop();
            if(traversal.i != selectedBoxIndex){
                std::vector<Vector<DIM>> thisBoxVertices;
                bbox = flatTree[traversal.i].bbox;
                bbox.vertices(thisBoxVertices, curBoxEdges, false);
                boxVertices.insert(boxVertices.end(), thisBoxVertices.begin(), thisBoxVertices.end());
                for(std::vector<int> edge : curBoxEdges){
                    int boxOffset = boxIndex * boxPow;
                    boxEdges.emplace_back(std::vector<int>({edge[0] + boxOffset, edge[1] + boxOffset}));
                }
            }

            if(flatTree[traversal.i].rightOffset != 0 && traversal.depth < bottomDepth){
                todo.emplace(BvhTraversal(traversal.i + 1, 0, traversal.depth + 1));
                todo.emplace(BvhTraversal(traversal.i + flatTree[traversal.i].rightOffset, 0, traversal.depth + 1));
            }

            if(traversal.i != selectedBoxIndex) boxIndex ++;
        }
    }
}