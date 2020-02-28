#pragma once

#include "bvh_simd_common.h"
#include "bvh_common.h"

namespace fcpw{

    template <int DIM, int W>
    class BvhSimd: public Aggregate<DIM>{
        using SimdType = typename IntrinsicType<W>::type;

        // Notes:
        // We assume leaf size of original bvh is simd size or smaller
        // need to eventually handle case of greater than simd size leaves

        public:
        // constructor
	    BvhSimd(const std::vector<BvhFlatNode<DIM>>& nodes_, 
        const std::vector<ReferenceWrapper<DIM>>& references_, 
        const std::vector<std::shared_ptr<Primitive<DIM>>>& primitives_, 
        const std::string& parentDescription_ = "");

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

        // debug for simd triangle closest point
        int findClosestFromLeaf(BoundingSphere<DIM>& s, const BvhSimdLeafNode<DIM, W>& leaf) const{
            int res = -1;
            float bestDist = s.r2;
            for(int i = 0; i < W; i++){
                int primIndex = leaf.indices[i];
                if(primIndex == -1){
                    continue;
                }
                const std::shared_ptr<Primitive<DIM>>& primitive = primitives[primIndex];
                Interaction<DIM> c;
                bool temp = primitive->findClosestPoint(s, c);
                if(temp && c.d * c.d < bestDist){
                    bestDist = c.d * c.d;
                    res = primIndex;
                }
            }
            return res;
        }

        // constructs mbvh
        void build(const std::vector<BvhFlatNode<DIM>>& nodes, const std::vector<ReferenceWrapper<DIM>>& references);

        // member variables
        int nNodes, nLeaves, depth, nReferences, nPrimitives;
        std::vector<std::shared_ptr<Primitive<DIM>>> primitives;
        std::vector<BvhSimdFlatNode<DIM, W>> flatTree;
        std::vector<BvhSimdLeafNode<DIM, W>> leaves;
        BoundingBox<DIM> bbox;
        float averageLeafSize;
    };
} // namespace fcpw

#include "bvh_simd.inl"