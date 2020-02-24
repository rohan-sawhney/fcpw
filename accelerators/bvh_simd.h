#pragma once

#include "bvh_simd_common.h"

namespace fcpw{

    template <int DIM, int W>
    class BvhSimd: public Aggregate<DIM>{
        using SimdType = typename IntrinsicType<W>::type;

        // Notes:
        // We assume leaf size of original bvh is simd size or smaller
        // need to eventually handle case of greater than simd size leaves

        public:
        // constructor
	    BvhSimd(std::vector<BvhFlatNode<DIM>>& nodes, std::vector<ReferenceWrapper<DIM>>& references, std::vector<std::shared_ptr<Primitive<DIM>>>& primitives, std::string parentDescription = "");

        // gets bounding box of MBVH
        BoundingBox<DIM> boundingBox() const;
        
        // gets centroid of bounding box of MBVH
        Vector<DIM> centroid() const;

        // gets surface area of bounding box of MBVH
        float surfaceArea() const;

        // gets signed volume of bounding box of MBVH
        float signedVolume() const;

        // gets ray intersection point
        int intersect(Ray<DIM>&r, std::vector<Interaction<DIM>>& is, bool checkOcclusion=false, bool countHits=false) const;

        // gets closest point
        bool findClosestPoint(BoundingSphere<DIM>& s, Interaction<DIM>& i) const;

        private:

        // constructs mbvh
        void build(std::vector<BvhFlatNode<DIM>>& nodes, std::vector<ReferenceWrapper<DIM>>& references);

        // member variables
        int nNodes, nLeaves, depth, nReferences, nPrimitives;
        std::vector<std::shared_ptr<Primitive<DIM>>> primitives;
        std::vector<BvhSimdFlatNode<DIM, W>> flatTree;
        std::vector<BvhSimdLeafNode<DIM, W>> leaves;
        BoundingBox<DIM> bbox;
    };
} // namespace fcpw

#include "bvh_simd.inl"