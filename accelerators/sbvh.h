#pragma once

#include "bvh.h"

namespace fcpw{
    
    template <int DIM>
    struct SbvhSplit{
        SbvhSplit(): axis(0), entry(0), exit(0), cost(maxFloat), split(maxFloat), loBox(BoundingBox<DIM>()), hiBox(BoundingBox<DIM>()){}
        int axis, entry, exit;
        float cost, split;
        BoundingBox<DIM> loBox, hiBox;
    };

    template <int DIM>
    class Sbvh: public Bvh<DIM>{
        // friend class SbvhSimd;

        public:
        // constructor
	    Sbvh(std::vector<std::shared_ptr<Primitive<DIM>>>& primitives_, int leafSize_=4, int splittingMethod_=0, int binCount_=32, bool doUnsplitting_=false, bool fillLeaves_=false);

        // gets ray intersection point
        int intersect(Ray<DIM>&r, std::vector<Interaction<DIM>>& is, bool checkOcclusion=false, bool countHits=false) const;

        bool applyClosestPoint(BoundingSphere<DIM>& s, Interaction<DIM>& i, int pos) const;

        protected:

        // constructs SBVH
        void build();

        // splits reference along some axis        
        bool splitReference(const ReferenceWrapper<DIM>& reference, int dim, float split, ReferenceWrapper<DIM>& loRef, ReferenceWrapper<DIM>& hiRef);

        // determines spatial box split (split by binning references split along proposed split planes) using some heuristic
        SbvhSplit<DIM> splitProbabilityHeuristic(std::vector<ReferenceWrapper<DIM>>& references, BoundingBox<DIM> bc, BoundingBox<DIM> bb, int binCount, costFunction<DIM> cost);

        // member variables
        std::vector<ReferenceWrapper<DIM>> references;
        int nReferences, nPrimitives;
        bool fillLeaves, doUnsplitting;
        
        // debug
        int nSpatialSplits, nObjectSplits;
    };
}

#include "sbvh.inl"