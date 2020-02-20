#pragma once

#include "bvh_common.h"
// #include "sbvh_simd.h"

namespace fcpw{
    
    template <int DIM>
    struct SbvhSplit{
        SbvhSplit(): axis(0), entry(0), exit(0), cost(maxFloat), split(maxFloat), loBox(BoundingBox<DIM>()), hiBox(BoundingBox<DIM>()){}
        int axis, entry, exit;
        float cost, split;
        BoundingBox<DIM> loBox, hiBox;
    };

    template <int DIM>
    class Sbvh: public Aggregate<DIM>{
        // friend class SbvhSimd;

        public:
        // constructor
	    Sbvh(std::vector<std::shared_ptr<Primitive<DIM>>>& primitives_, int leafSize_=4, int splittingMethod_=0, int binCount_=16, bool doUnsplitting_=false, bool fillLeaves_=false);

        // gets bounding box of SBVH
        BoundingBox<DIM> boundingBox() const;
        
        // gets centroid of bounding box of SBVH
        Vector<DIM> centroid() const;

        // gets surface area of bounding box of SBVH
        float surfaceArea() const;

        // gets signed volume of bounding box of SBVH
        float signedVolume() const;

        // gets ray intersection point
        int intersect(Ray<DIM>&r, std::vector<Interaction<DIM>>& is, bool checkOcclusion=false, bool countHits=false) const;

        // gets closest point
        bool findClosestPoint(BoundingSphere<DIM>& s, Interaction<DIM>& i) const;

        private:

        // constructs SBVH
        void build();

        // splits reference along some axis        
        bool splitReference(const ReferenceWrapper<DIM>& reference, int dim, float split, ReferenceWrapper<DIM>& loRef, ReferenceWrapper<DIM>& hiRef);

        // determines spatial box split (split by binning references split along proposed split planes) using some heuristic
        SbvhSplit<DIM> splitProbabilityHeuristic(std::vector<ReferenceWrapper<DIM>>& references, BoundingBox<DIM> bc, BoundingBox<DIM> bb, int binCount, costFunction<DIM> cost);

        // member variables
        std::vector<std::shared_ptr<Primitive<DIM>>> primitives;
        int splittingMethod, leafSize, nNodes, binCount, nLeaves, depth, nReferences, nPrimitives;
        bool fillLeaves, doUnsplitting;

        public:
        // THE FOLLOWING ARE ONLY PUBLIC UNTIL I CAN FIGURE OUT CLASS FRIENDSHIP
        std::vector<ReferenceWrapper<DIM>> references;
        std::vector<BvhFlatNode<DIM>> flatTree;
    };
}

#include "sbvh.inl"