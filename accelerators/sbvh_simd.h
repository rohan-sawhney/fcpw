#pragma once

#include "sbvh.h"
#include "simd.h"

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

        // constructs SBVH4
        void build(std::vector<BvhFlatNode<DIM>> nodes, std::vector<ReferenceWrapper<DIM>> references);

        // member variables
        int splittingMethod, leafSize, nNodes, binCount, nLeaves, depth, nRefs;
        std::vector<std::shared_ptr<Primitive<DIM>>> primitives;
        std::vector<BvhSimdFlatNode<DIM, W>> flatTree;
        std::vector<BvhSimdLeafNode<DIM, W>> leaves;
        const double epsilon = 1e-16;
        double buildTime;
        BoundingBox<DIM> bbox;
    };
} // namespace fcpw

#include "sbvh_simd.inl"