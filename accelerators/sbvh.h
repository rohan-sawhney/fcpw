#pragma once

#include "bvh_common.h"
// #include "sbvh_simd.h"

namespace fcpw{
    
    template <int DIM>
    struct SbvhSplit{
        SbvhSplit(): axis(0), entry(0), exit(0), cost(maxDouble), split(maxDouble), loBox(BoundingBox<DIM>()), hiBox(BoundingBox<DIM>()){}
        int axis, entry, exit;
        double cost, split;
        BoundingBox<DIM> loBox, hiBox;
    };

    template <int DIM>
    class Sbvh: public Aggregate<DIM>{
        // friend class SbvhSimd;

        public:
        // constructor
	    Sbvh(std::vector<std::shared_ptr<Primitive<DIM>>>& primitives_, int leafSize_=4, int splittingMethod_=0, bool makeBvh_=false, int binCount_=16, bool fillLeaves_=false);

        // gets bounding box of SBVH
        BoundingBox<DIM> boundingBox() const;
        
        // gets centroid of bounding box of SBVH
        Vector<DIM> centroid() const;

        // gets surface area of bounding box of SBVH
        double surfaceArea() const;

        // gets signed volume of bounding box of SBVH
        double signedVolume() const;

        // gets ray intersection point
        int intersect(Ray<DIM>&r, Interaction<DIM>& i, bool countHits=false) const;

        // gets closest point
        void findClosestPoint(BoundingSphere<DIM>& s, Interaction<DIM>& i) const;
        
        // returns neighboring box
        BoundingBox<DIM> traverse(int& curIndex, int gotoIndex) const;

        // returns list of boxes around a currently selected box
    	void getBoxList(int curIndex, int topDepth, int bottomDepth, std::vector<Vector<DIM>>& boxVertices, std::vector<std::vector<int>>& boxEdges, std::vector<Vector<DIM>>& curBoxVertices, std::vector<std::vector<int>>& curBoxEdges) const;

        private:

        // constructs SBVH
        void build();

        // determines object box split (split by binning reference centroids) using some heuristic
        SbvhSplit<DIM> probabilityHeuristic(std::vector<ReferenceWrapper<DIM>>& references, BoundingBox<DIM> bc, BoundingBox<DIM> bb, int binCount, costFunction<DIM> cost);

        // splits reference along some axis        
        bool splitReference(const ReferenceWrapper<DIM>& reference, int dim, double split, ReferenceWrapper<DIM>& loRef, ReferenceWrapper<DIM>& hiRef);

        // determines spatial box split (split by binning references split along proposed split planes) using some heuristic
        SbvhSplit<DIM> splitProbabilityHeuristic(std::vector<ReferenceWrapper<DIM>>& references, BoundingBox<DIM> bc, BoundingBox<DIM> bb, int binCount, costFunction<DIM> cost);

        // member variables
        int splittingMethod, leafSize, nNodes, binCount, nLeaves, depth, nRefs;
        std::vector<std::shared_ptr<Primitive<DIM>>> primitives;
        const double epsilon = 1e-16;
        bool doUnsplitting = false;
        bool countIntersections = true;
        double buildTime;
        bool makeBvh;
        bool fillLeaves;

        public:
        // THE FOLLOWING ARE ONLY PUBLIC UNTIL I CAN FIGURE OUT CLASS FRIENDSHIP
        std::vector<ReferenceWrapper<DIM>> references;
        std::vector<BvhFlatNode<DIM>> flatTree;

        // debug
        std::unique_ptr<std::vector<double>> times;
    };
}

#include "sbvh.inl"