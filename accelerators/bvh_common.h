#pragma once

#include "primitive.h"

#define COST_TRAVERSAL 0.125
#define COST_OPERATION 1.0

namespace fcpw{

    // default node type for BVH. Leaves are denoted with rightOffset of 0
    template <int DIM>
    struct BvhFlatNode{
        BoundingBox<DIM> bbox;
        int start, nPrimitives, rightOffset, parentIndex, siblingIndex;
        bool isOverlapping;
    };

    // contains information about optimal split location and dimension
    struct BvhSplit{
        int axis, hiCount, loCount;
        float cost, split;
    };

    // wrapper for a reference to a primitive
    template <int DIM>
    struct ReferenceWrapper{
        ReferenceWrapper(BoundingBox<DIM> bbox_, int index_) : bbox(bbox_), index(index_){}
        ReferenceWrapper() : bbox(BoundingBox<DIM>()), index(-1){}
        BoundingBox<DIM> bbox;
        int index;
    };

    // wrapper for class encoding traversal in a BVH
    struct BvhTraversalDepth {
        // constructor
        BvhTraversalDepth(int i_): i(i_), depth(0) {
        // #ifdef PROFILE
        //     PROFILE_SCOPED();
        // #endif
        }

        BvhTraversalDepth(int i_, int depth_): i(i_), depth(depth_) {
        // #ifdef PROFILE
        //     PROFILE_SCOPED();
        // #endif
        }

        // members
        int i, depth; // node index
    };

        // wrapper for class encoding traversal in a BVH
    struct BvhTraversal {
        // constructor
        BvhTraversal(int i_, float d_): i(i_), d(d_) {
        // #ifdef PROFILE
        //     PROFILE_SCOPED();
        // #endif
        }

        // members
        int i; // node index
        float d; // minimum distance (parametric, squared, ...) to this node
    };

    // cost function type (shared between BVH and SBVH)
    template <int DIM>
    using costFunction = float (*) (BoundingBox<DIM>& loBox, BoundingBox<DIM>& hiBox, const BoundingBox<DIM>& parent, int countsLo, int countsHi, bool isFlat);
    // typedef float (* costFunction)(BoundingBox<DIM>& consolidateLeft, BoundingBox<DIM>& consolidateRight, int countsLeft, int countsRight, bool isFlat, float bbSA, float bbVol, float costTraversal, float costOperation);

    // cost functions
    // surface area heuristic
    template <int DIM>
    inline float surfaceAreaCost(BoundingBox<DIM>& left, BoundingBox<DIM>& right, const BoundingBox<DIM>& parent, int countsLeft, int countsRight, bool isFlat){
        float leftSA = isFlat ? 2 * left.extent().sum() : left.surfaceArea();
        float rightSA = isFlat ? 2 * right.extent().sum() : right.surfaceArea();
        float parentSA = isFlat ? 2 * parent.extent().sum() : parent.surfaceArea();
        return (leftSA * countsLeft + rightSA * countsRight) / parentSA * COST_OPERATION + COST_TRAVERSAL;
    }
    // volume heuristic
    template <int DIM>
    inline float volumeCost(BoundingBox<DIM>& left, BoundingBox<DIM>& right, const BoundingBox<DIM>& parent, int countsLeft, int countsRight, bool isFlat){
        float leftVol = isFlat ? left.surfaceArea() / 2 : left.volume();
        float rightVol = isFlat ? right.surfaceArea() / 2 : right.volume();
        float parentVol = isFlat ? parent.surfaceArea() / 2 : parent.volume();
        return (leftVol * countsLeft + rightVol * countsRight) / parentVol * COST_OPERATION + COST_TRAVERSAL;
    }
    // overlap surface area heuristic
    template <int DIM>
    inline float overlapSurfaceAreaCost(BoundingBox<DIM>& left, BoundingBox<DIM>& right, const BoundingBox<DIM>& parent, int countsLeft, int countsRight, bool isFlat){
        BoundingBox<DIM> overlap = left.intersect(right);
        float overlapSA = isFlat ? 2 * overlap.extent().sum() : overlap.surfaceArea();
        float leftSA = isFlat ? 2 * left.extent().sum() : left.surfaceArea();
        float rightSA = isFlat ? 2 * right.extent().sum() : right.surfaceArea();
        return (countsLeft / rightSA + countsRight / leftSA) * overlapSA * COST_OPERATION + COST_TRAVERSAL;
    }
    // overlap volume heuristic
    template <int DIM>
    inline float overlapVolumeCost(BoundingBox<DIM>& left, BoundingBox<DIM>& right, const BoundingBox<DIM>& parent, int countsLeft, int countsRight, bool isFlat){
        BoundingBox<DIM> overlap = left.intersect(right);
        float overlapVol = isFlat ? overlap.surfaceArea() / 2 : overlap.volume();
        float leftVol = isFlat ? left.surfaceArea() / 2 : left.volume();
        float rightVol = isFlat ? right.surfaceArea() / 2 : right.volume();
        return (countsLeft / rightVol + countsRight / leftVol) * overlapVol * COST_OPERATION + COST_TRAVERSAL;
    }

    template <int DIM>
    struct HeuristicBin{
        HeuristicBin(): bbox(BoundingBox<DIM>()), count(0){}
        BoundingBox<DIM> bbox;
        int count;
    };

    // probability heuristic object splitting
    template <int DIM>
    inline BvhSplit probabilityHeuristic(const std::vector<ReferenceWrapper<DIM>>& references, const BoundingBox<DIM>& bc, const BoundingBox<DIM>& bb, const int binCount, costFunction<DIM> cost, int leafMod=0){
        
        // setup container for multiple splits
        std::vector<BvhSplit> results;
        for(int i = 0; i < DIM; i++){
            results.emplace_back(BvhSplit());
            results.back().split = 0.0;
            results.back().cost = maxFloat;
            results.back().axis = i;
        }

        // setup useful containers
        std::vector<HeuristicBin<DIM>> bins = std::vector<HeuristicBin<DIM>>(binCount);
        std::vector<float> splits = std::vector<float>(binCount - 1);

        for(int dimension = 0; dimension < DIM; dimension++){
            // skip if bounding box is flat in this dimension
            if(bc.extent()[dimension] < epsilon) continue;

            // select a split container corresponding to this dimension
            BvhSplit& res = results[dimension];

            // setup containers and variables for binning process
            float delta = bc.extent()[res.axis] / binCount;
            float loBound = bc.pMin(res.axis);
            for(int i = 0; i < binCount; i++){
                bins[i] = HeuristicBin<DIM>();
            }
            for(int i = 0; i < binCount - 1; i++){
                splits[i] = delta * (i + 1) + loBound;
            }

            // loop over bounding boxes and bin
            for(ReferenceWrapper<DIM> reference : references){
                // parse reference
                BoundingBox<DIM> bbox = reference.bbox;
                float center = bbox.centroid()[res.axis];

                // get index of bbox via centroid
                int binIndex = (center - loBound) / delta;
                LOG_IF(FATAL, binIndex < -1 || binIndex > binCount) << "Out of bounds bin index even before adjustments, index is at " << binIndex << " with only " << binCount << "bins. Delta is " << delta << " prim centroid at: " << center << " upper bound at " << bc.pMax(res.axis) << " lower bound at " << bc.pMin(res.axis);
                if(binIndex == -1){
                    // if centroid is on lo end of box but negative bin index due to rounding, increase index
                    binIndex ++;
                }
                if(binIndex == binCount){
                    // if centroid is on hi end of box but out of bounds index due to rounding, reduce index
                    binIndex --;
                }

                // bin reference
                bins[binIndex].bbox.expandToInclude(bbox);
                bins[binIndex].count ++;
            }

            // initialize variables to find optimal split
            BoundingBox<DIM> hiBins[binCount - 1];
            BoundingBox<DIM> loBox = BoundingBox<DIM>();
            BoundingBox<DIM> hiBox;
            int loCount = 0;
            int hiCount = references.size();
            res.cost = maxFloat;
            hiBins[binCount - 2] = bins.back().bbox;

            // fill array with boxes grown from the hi value along dim
            for(int i = binCount - 2; i > 0; i--){
                hiBins[i - 1] = hiBins[i];
                hiBins[i - 1].expandToInclude(bins[i].bbox);
            }

            // find optimal split scanning from lo bin to hi bin
            for(int i = 0; i < binCount - 1; i++){
                // grow lo box, choose shrunken hi box
                loBox.expandToInclude(bins[i].bbox);
                hiBox = hiBins[i];
                hiCount -= bins[i].count;
                loCount += bins[i].count;

                // get cost for current split and compare
                float tempCost;
                if(leafMod == 0){
                    tempCost = cost(loBox, hiBox, bb, loCount, hiCount, (bb.volume() < epsilon));
                }
                else{
                    tempCost = ((loCount % leafMod == 0) || (hiCount % leafMod == 0)) ? cost(loBox, hiBox, bb, loCount, hiCount, (bb.volume() < epsilon)) : maxFloat;
                }
                if(tempCost < res.cost){
                    // update if current split better than prior splits
                    res.cost = tempCost;
                    res.split = splits[i];
                    res.hiCount = hiCount;
                    res.loCount = loCount;
                }
            }
        }

        // get and return best split
        BvhSplit bestSplit = results[0];
        for(int i = 0; i < DIM; i++){
            if(results[i].cost < bestSplit.cost){
                bestSplit = results[i];
            }
        }
        return bestSplit;
    }
}