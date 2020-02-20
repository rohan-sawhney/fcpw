#pragma once

#include "sbvh.h"
#include "bvh_simd_common.h"

namespace fcpw{

    template <int DIM, int W>
    class SbvhSimd: public Aggregate<DIM>{
        using SimdType = typename IntrinsicType<W>::type;

        public:
        // constructor
	    SbvhSimd(std::vector<std::shared_ptr<Primitive<DIM>>>& primitives_, int leafSize_=4, int splittingMethod_=0, int binCount_=16);

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

        // constructs SBVH4
        void build(std::vector<BvhFlatNode<DIM>> nodes, std::vector<ReferenceWrapper<DIM>> references);

        // member variables
        int splittingMethod, leafSize, nNodes, binCount, nLeaves, depth, nReferences, nPrimitives;
        std::vector<std::shared_ptr<Primitive<DIM>>> primitives;
        std::vector<BvhSimdFlatNode<DIM, W>> flatTree;
        std::vector<BvhSimdLeafNode<DIM, W>> leaves;
        BoundingBox<DIM> bbox;
    };
} // namespace fcpw

#include "sbvh_simd.inl"