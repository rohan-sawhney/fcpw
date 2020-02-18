#pragma once

#include "primitive.h"

#define COST_TRAVERSAL 1.0
#define COST_OPERATION 1.0

namespace zombie{

    // default node type for BVH. Leaves are denoted with rightOffset of 0
    template <int DIM>
    struct BvhFlatNode{
        BoundingBox<DIM> bbox;
        int start, nPrims, rightOffset, parentIndex, siblingIndex;
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
    struct BvhTraversal {
        // constructor
        BvhTraversal(int i_, double d_): i(i_), d(d_) {
            depth = 0;
        }

        BvhTraversal(int i_, double d_, int depth_): i(i_), d(d_), depth(depth_) {}

        // members
        int i, depth; // node index
        double d; // minimum distance (parametric, squared, ...) to this node
    };

    // cost function type (shared between BVH and SBVH) (NEED TO ADDRESS DIFFERENCE IN NOT HAVING BASE COSTS IN BVH VERSION)
    template <int DIM>
    using costFunction = double (*) (BoundingBox<DIM>& loBox, BoundingBox<DIM>& hiBox, int countsLo, int countsHi, bool isFlat, double bbSA, double bbVol);
    // typedef double (* costFunction)(BoundingBox<DIM>& consolidateLeft, BoundingBox<DIM>& consolidateRight, int countsLeft, int countsRight, bool isFlat, double bbSA, double bbVol, double costTraversal, double costOperation);

    // cost functions
    // surface area heuristic
    template <int DIM>
    inline double surfaceAreaCost(BoundingBox<DIM>& left, BoundingBox<DIM>& right, int countsLeft, int countsRight, bool isFlat, double bbSA, double bbVol){
        double leftSA = isFlat ? 2 * left.extent().sum() : left.surfaceArea();
        double rightSA = isFlat ? 2 * right.extent().sum() : right.surfaceArea();
        return (leftSA * countsLeft + rightSA * countsRight) / bbSA * COST_OPERATION + COST_TRAVERSAL;
    }
    // volume heuristic
    template <int DIM>
    inline double volumeCost(BoundingBox<DIM>& left, BoundingBox<DIM>& right, int countsLeft, int countsRight, bool isFlat, double bbSA, double bbVol){
        double leftVol = isFlat ? left.surfaceArea() / 2 : left.volume();
        double rightVol = isFlat ? right.surfaceArea() / 2 : right.volume();
        return (leftVol * countsLeft + rightVol * countsRight) / bbVol * COST_OPERATION + COST_TRAVERSAL;
    }
    // overlap surface area heuristic
    template <int DIM>
    inline double overlapSurfaceAreaCost(BoundingBox<DIM>& left, BoundingBox<DIM>& right, int countsLeft, int countsRight, bool isFlat, double bbSA, double bbVol){
        BoundingBox<DIM> overlap = left.intersect(right);
        double overlapSA = isFlat ? 2 * overlap.extent().sum() : overlap.surfaceArea();
        double leftSA = isFlat ? 2 * left.extent().sum() : left.surfaceArea();
        double rightSA = isFlat ? 2 * right.extent().sum() : right.surfaceArea();
        return (countsLeft / rightSA + countsRight / leftSA) * overlapSA * COST_OPERATION + COST_TRAVERSAL;
    }
    // overlap volume heuristic
    template <int DIM>
    inline double overlapVolumeCost(BoundingBox<DIM>& left, BoundingBox<DIM>& right, int countsLeft, int countsRight, bool isFlat, double bbSA, double bbVol){
        BoundingBox<DIM> overlap = left.intersect(right);
        double overlapVol = isFlat ? overlap.surfaceArea() / 2 : overlap.volume();
        double leftVol = isFlat ? left.surfaceArea() / 2 : left.volume();
        double rightVol = isFlat ? right.surfaceArea() / 2 : right.volume();
        return (countsLeft / rightVol + countsRight / leftVol) * overlapVol * COST_OPERATION + COST_TRAVERSAL;
    }
}