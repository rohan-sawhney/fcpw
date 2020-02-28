#include <stack>
#include <queue>
#include <chrono>

namespace fcpw{

    struct BvhSimdBuildNode{
        BvhSimdBuildNode(int nodeIndex_, int parentIndex_, int depth_=0) :
            nodeIndex(nodeIndex_), parentIndex(parentIndex_), depth(depth_){}
        int nodeIndex, parentIndex, depth;
    };

    template <int DIM, int W>
    inline BvhSimd<DIM, W>::BvhSimd(
        const std::vector<BvhFlatNode<DIM>>& nodes_, 
        const std::vector<ReferenceWrapper<DIM>>& references_, 
        const std::vector<std::shared_ptr<Primitive<DIM>>>& primitives_, 
        const std::string& parentDescription_):
        nNodes(0), nLeaves(0), primitives(primitives_), bbox(nodes_[0].bbox),
        depth(0), nReferences(0), averageLeafSize(0), nPrimitives(primitives_.size()){

        std::chrono::high_resolution_clock::time_point t_start, t_end;
        std::chrono::nanoseconds duration;
        double buildTime = 0;

        t_start = std::chrono::high_resolution_clock::now();
        build(nodes_, references_);
        t_end = std::chrono::high_resolution_clock::now();
        duration = t_end - t_start;
        buildTime = (double)(duration.count()) / std::chrono::nanoseconds::period::den;

        averageLeafSize /= nLeaves;

        std::string simdMethod;
        switch(W){
            case 4:
                simdMethod = "SSE";
                break;
            case 8:
                simdMethod = "AVX";
                break;
            case 16:
                simdMethod = "AVX512";
                break;
            default:
                simdMethod = "INVALID";
                break;
        }
        LOG(INFO) << simdMethod << " Bvh created with "
                    << nNodes << " nodes, "
                    << nLeaves << " leaves with average size " << averageLeafSize << ", "
                    << nPrimitives << " primitives, "
                    << depth << " depth, in "
                    << buildTime << " seconds, "
                    << parentDescription_;
    }

    template <int DIM, int W>
    inline void BvhSimd<DIM, W>::build(const std::vector<BvhFlatNode<DIM>>& nodes, const std::vector<ReferenceWrapper<DIM>>& references){
        std::stack<BvhSimdBuildNode> todo;
        std::stack<BvhTraversal> nodeWorkingSet;
        std::vector<BvhSimdFlatNode<DIM, W>> buildNodes;
        std::vector<BvhSimdLeafNode<DIM, W>> buildLeaves;

        int maxDepth = W == 4 ? 2 : (W == 8 ? 3 : (W == 16 ? 4 : 0));
        LOG_IF(FATAL, maxDepth == 0) << "BvhSimd::build(): Provided width for SIMD is invalid";

        todo.emplace(BvhSimdBuildNode(0, -1, 0));

        // NEED TO INTEGRATE TRAVERSAL ORDER SOMEHOW INTO THIS

        // process todo
        while(!todo.empty()){
            // pop off node for processing
            BvhSimdBuildNode toBuild = todo.top();
            todo.pop();

            // process build node
            int nodeIndex = toBuild.nodeIndex;
            int parentIndex = toBuild.parentIndex;
            if(depth < toBuild.depth) depth = toBuild.depth;
            int toBuildDepth = toBuild.depth;
            const BvhFlatNode<DIM>& curNode = nodes[nodeIndex];

            // construct a new tree node
            if(parentIndex == -1 || curNode.rightOffset != 0){
                buildNodes.emplace_back(BvhSimdFlatNode<DIM, W>());
                BvhSimdFlatNode<DIM, W>& node = buildNodes.back();
                for(int i = 0; i < W; i++){
                    node.indices[i] = -1;
                    node.isLeaf[i] = false;
                    for(int j = 0; j < DIM; j++){
                        node.minBoxes[j][i] = 0;
                        node.maxBoxes[j][i] = 0;
                    }
                }
                nNodes ++;
            }
            int simdTreeIndex = buildNodes.size() - 1;

            // fill node with data (if not root node)
            if(parentIndex != -1){
                // setup useful variables
                BvhSimdFlatNode<DIM, W>& parentNode = buildNodes[parentIndex];
                int tempCounter = 0;
                while(parentNode.indices[tempCounter] != -1){
                    tempCounter ++;
                    LOG_IF(FATAL, tempCounter == W) << "Out of bounds number of child nodes!";
                }
                BoundingBox<DIM> bbox = curNode.bbox;

                // fill bbox data in node
                for(int i = 0; i < DIM; i++){
                    parentNode.minBoxes[i][tempCounter] = bbox.pMin(i);
                    parentNode.maxBoxes[i][tempCounter] = bbox.pMax(i);
                }

                // connect popped off node to associated build node
                if(curNode.rightOffset == 0){
                    // if node is leaf, construct the parallel leaf, link and move on

                    averageLeafSize += (float)curNode.nPrimitives;

                    float tripoints[3][DIM][W];
                    nLeaves ++;
                    buildLeaves.emplace_back(BvhSimdLeafNode<DIM, W>());
                    BvhSimdLeafNode<DIM, W>& leafNode = buildLeaves.back();
                    for(int i = 0; i < W; i++){
                        if(i >= curNode.nPrimitives){
                            leafNode.indices[i] = -1;
                            for(int j = 0; j < DIM; j++){
                                tripoints[0][j][i] = 0.;
                                tripoints[1][j][i] = 0.;
                                tripoints[2][j][i] = 0.;
                            }
                            continue;
                        }
                        leafNode.indices[i] = references[curNode.start + i].index;
                        const std::shared_ptr<Primitive<DIM>>& primitive = primitives[leafNode.indices[i]];
                        if(DIM == 3){
                            std::vector<Vector3f> vertices;
                            std::shared_ptr<Triangle> triangle = std::dynamic_pointer_cast<Triangle>(primitive);
                            triangle->getVertices(vertices);

                            const Vector3f& pa = vertices[0];
                            const Vector3f& pb = vertices[1];
                            const Vector3f& pc = vertices[2];

                            for(int j = 0; j < DIM; j++){
                                tripoints[0][j][i] = pa(j);
                                tripoints[1][j][i] = pb(j);
                                tripoints[2][j][i] = pc(j);
                                // leafNode.pa[j][i] = pa(j);
                                // leafNode.pb[j][i] = pb(j);
                                // leafNode.pc[j][i] = pc(j);
                                // leafNode.minBoxes[j][i] = references[curNode.start + i].bbox.pMin(j);
                                // leafNode.maxBoxes[j][i] = references[curNode.start + i].bbox.pMax(j);
                            }
                        }
                        else{
                            LOG(FATAL) << "Non triangular primitives not handled at the moment";
                        }
                    }

                    leafNode.initPoints(tripoints);

                    parentNode.indices[tempCounter] = buildLeaves.size() - 1;
                    parentNode.isLeaf[tempCounter] = true;
                    continue;
                }
                else{
                    // link node to build node
                    parentNode.indices[tempCounter] = simdTreeIndex;
                    parentNode.isLeaf[tempCounter] = false;
                }

            }

            // push grandchildren of node on top of processing stack (leftmost grandchild goes to top of stack)
            nodeWorkingSet.emplace(BvhTraversal(nodeIndex, 0, 0));
            while(!nodeWorkingSet.empty()){
                // pop off node for processing
                BvhTraversal traversal = nodeWorkingSet.top();
                nodeWorkingSet.pop();

                // parse node
                int ni = traversal.i;
                int d = traversal.depth;

                // process node
                if(d < maxDepth && nodes[ni].rightOffset != 0){
                    // node is not leaf and not grandchild, continue down tree
                    nodeWorkingSet.emplace(BvhTraversal(ni + 1, 0, d + 1));
                    nodeWorkingSet.emplace(BvhTraversal(ni + nodes[ni].rightOffset, 0, d + 1));
                }
                else{
                    // node is grandchild or leaf node (but not grandchild), push onto todo set
                    todo.emplace(BvhSimdBuildNode(ni, simdTreeIndex, toBuildDepth + 1));
                }
            }
        }

        flatTree.clear();
        flatTree.reserve(nNodes);
        for(int n = 0; n < buildNodes.size(); n++){
            flatTree.emplace_back(buildNodes[n]);
        }
        leaves.clear();
        leaves.reserve(nLeaves);
        for(int n = 0; n < buildLeaves.size(); n++){
            leaves.emplace_back(buildLeaves[n]);
        }
    }

    template <int DIM, int W>
    inline BoundingBox<DIM> BvhSimd<DIM, W>::boundingBox() const{
        return bbox;
    }

    template <int DIM, int W>
    inline Vector<DIM> BvhSimd<DIM, W>::centroid() const{
        return bbox.centroid();
    }

    template <int DIM, int W>
    inline float BvhSimd<DIM, W>::surfaceArea() const{
        float area = 0.0;
        for(std::shared_ptr<Primitive<DIM>> p : primitives){
            area += p->surfaceArea();
        }
        return area;
    }

    template <int DIM, int W>
    inline float BvhSimd<DIM, W>::signedVolume() const{
        float volume = 0.0;
        for(std::shared_ptr<Primitive<DIM>> p : primitives){
            volume += p->signedVolume();
        }
        return volume;
    }

    template <int DIM, int W>
    inline int BvhSimd<DIM, W>::intersect(Ray<DIM>& r, std::vector<Interaction<DIM>>& is, bool checkOcclusion, bool countHits) const{
        
        return 0;
    }

/* ---- CPQ ---- */

    template <int DIM, int W>
    inline bool BvhSimd<DIM, W>::findClosestPoint(BoundingSphere<DIM>& s, Interaction<DIM>& i) const{
        #ifdef PROFILE
            PROFILE_SCOPED();
        #endif

        std::queue<BvhTraversal> todo;
        SimdType resVec[2];
        SimdType leafVec[2];

        todo.emplace(BvhTraversal(0, minFloat));
        while(!todo.empty()){

            // pop off the next node to work on
            BvhTraversal traversal = todo.front();
            todo.pop();

            int ni = traversal.i;
            float near = traversal.d;
            const BvhSimdFlatNode<DIM, W>& node = flatTree[ni];

            if(near > s.r2){
                continue;
            }

            // do overlap test
            // NOTE: might be good to separate this into closest and furthest distance functions
            // so that furthest doesn't need to be done if query is out of range
            parallelOverlap(node.minBoxes, node.maxBoxes, s, resVec[0], resVec[1]);

            // int ordering[W];
            // float unpackedMin[W];
            // for(int j = 0; j < W; j++){
            //     ordering[j] = j;
            // }

            // // temp bubble sort ordering
            // bool isSorted = false;
            // while(!isSorted){
            //     isSorted = true;
            //     for(int j = 0; node.indices[j + 1] != -1 && j < W - 1; j++){
            //         if(unpackedMin[j] > unpackedMin[j + 1]){
            //             isSorted = false;
            //             std::swap(unpackedMin[j], unpackedMin[j + 1]);
            //             std::swap(ordering[j], ordering[j + 1]);
            //         }
            //     }
            // }
            
            // process overlapped nodes NOTE: ADD IN ORDERING ONCE THAT IS AVAILABLE
            for(int j = 0; node.indices[j] != -1 && j < W; j++){
                int index = j;//ordering[j];
                // only process if box is in bounds of query
                if((float)s.r2 > resVec[0][index]){
                    // aggressively shorten if box is fully contained in query
                    if((float)s.r2 > resVec[1][index]) s.r2 = (float)resVec[1][index];
                    if(!node.isLeaf[index]){
                        // if interior node, add to queue
                        if((float)s.r2 >= resVec[0][index])
                            todo.emplace(BvhTraversal(node.indices[index], resVec[0][index]));
                    }
                    else{
                        // if mbvh leaf, process triangles in parallel
                        const BvhSimdLeafNode<DIM, W>& leafNode = leaves[node.indices[index]];
                        ParallelInteraction<DIM, W> pi = ParallelInteraction<DIM, W>();
                        for(int k = 0; k < W; k++){
                            pi.indices[k] = leafNode.indices[k];
                        }
                        parallelTriangleOverlap2(leafNode.pa, leafNode.pb, leafNode.pc, s, pi);

                        float bestDistance;
                        float bestPoint[DIM];
                        int bestIndex; // = findClosestFromLeaf(s, leafNode);
                        pi.getBest(bestDistance, bestPoint, bestIndex);

                        // LOG(INFO) << "Node indices: " << leafNode.indices[0] << " " << leafNode.indices[1] << " " << leafNode.indices[2] << " " << leafNode.indices[3] << " Best index: " << bestIndex << " Best distance: " << bestDistance << " Current radius: " << s.r2;
                        bool updatedDistances = false;

                        if(bestIndex != -1 && bestDistance < (float)s.r2){
                            // updatedDistances = true;
                            s.r2 = (float)bestDistance;
                            // i = c;
                            i.p = Vector<DIM>();
                            for(int k = 0; k < DIM; k++){
                                i.p(k) = (float)bestPoint[k];
                            }
                            i.n = Vector<DIM>(); // TEMPORARY!!!!
                            i.d = std::sqrt(s.r2);
                            i.primitive = primitives[bestIndex].get();
                        }
                        // LOG_IF(FATAL, updatedDistances && i.primitive == nullptr) << "Primitive is null!";
                    }
                }
            }
        }
        LOG_IF(FATAL, i.primitive == nullptr) << "Primitive is null!";
        return s.r2 != maxFloat;
    }
}// namespace fcpw