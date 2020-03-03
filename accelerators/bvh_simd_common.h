#pragma once

#include "immintrin.h"
#include <array>
#include "bounding_volumes.h"
#include "simd.h"

namespace fcpw{

    // TEMP NOTE: prep for simd optimized bvhs
    template <int DIM, int W>
    struct BvhSimdFlatNode{
        // pMin first then pMax
        // below is SSE example
        // x1 x2 x3 x4
        // y1 y2 y3 y4
        // ...
        using SimdType = typename IntrinsicType<W>::type;
        // using CounterType = typename CountType<W>::type;
        
        SimdType minBoxes[DIM];
        SimdType maxBoxes[DIM];
        int indices[W];
        char sortOrder[DIM];
        bool isLeaf[W];
    };

    template <int W>
    struct TestNode{
        simdBox_type<W> boxes;
        int indices[W];
        char sortOrder[3];
        bool isLeaf[W];
    };

    // triangles only
    template <int DIM, int W>
    struct BvhSimdLeafNode{
        using SimdType = typename IntrinsicType<W>::type;

        // SimdType minBoxes[DIM];
        // SimdType maxBoxes[DIM];
        SimdType pa[DIM];
        SimdType pb[DIM];
        SimdType pc[DIM];
        int indices[W]; // will later wrap in triangle data

        inline void initPoints(float pointCoords[3][DIM][W]){
            for(int i = 0; i < DIM; i++){
                initSimd(pointCoords[0][i], pa[i]);
                initSimd(pointCoords[1][i], pb[i]);
                initSimd(pointCoords[2][i], pc[i]);
            }
        }
    };

    template <int W>
    struct TestLeafNode{
        simdTriangle_type<W> triangles;
        int indices[W];
    };

    /* ---- Vectorized Functions ---- */

    template <int W>
    inline void simdBoxOverlap(simdFloat<W>& closestDistances, simdFloat<W>& furthestDistances, const simdPoint_type<W>& iPoint, const simdBox_type<W>& iBoxes){
        closestDistances = length2(max(max(iBoxes[0] - iPoint, iPoint - iBoxes[1]), zeroVector<W>()));
        furthestDistances = length2(max(iPoint - iBoxes[0], iBoxes[1] - iPoint));
    }

    template <int DIM, class T, int W>
    inline void parallelOverlap(const T boxMins[DIM], const T boxMaxs[DIM], BoundingSphere<DIM>& s, T& d2Min, T& d2Max){
        float sPos[DIM];
        for(int i = 0; i < DIM; i++){
            sPos[i] = (float)s.c(i);
        }
        simdBox_type<W> boxes;
        simdPoint_type<W> sc;
        simdFloat<W> closestDists = vecZero<W>();
        simdFloat<W> furthestDists = vecZero<W>();

        sc = embree::Vec3<simdFloat<W>>(vecf<W>(_mm_set1_ps(s.c(0))), vecf<W>(_mm_set1_ps(s.c(1))), vecf<W>(_mm_set1_ps(s.c(2))));
        boxes[0] = embree::Vec3<simdFloat<W>>(vecf<W>(boxMins[0]), vecf<W>(boxMins[1]), vecf<W>(boxMins[2]));
        boxes[1] = embree::Vec3<simdFloat<W>>(vecf<W>(boxMaxs[0]), vecf<W>(boxMaxs[1]), vecf<W>(boxMaxs[2]));

        simdBoxOverlap(closestDists, furthestDists, sc, boxes);

        d2Min = closestDists.vec;
        d2Max = furthestDists.vec;
    }

    template <int W>
    const simdFloat<W> simdTriPoint2(simdFloatVec<W>& oTriPoint, const simdTriangle_type<W>& iTri, const simdPoint_type<W>& iPoint){
		// Check if P in vertex region outside A
		const simdFloatVec<W> ab = iTri[1] - iTri[0];
		const simdFloatVec<W> ac = iTri[2] - iTri[0];
		const simdFloatVec<W> ap = iPoint - iTri[0];
		const simdFloat<W> d1 = dot(ab, ap);
		const simdFloat<W> d2 = dot(ac, ap);
		const simdBool<W> mask1 = (d1 <= simdFloat<W>(vecZero<4>().vec)) & (d2 <= simdFloat<W>(vecZero<4>().vec));
		oTriPoint = iTri[0];
		simdBool<W> exit(mask1);
		if (all(exit))
			return length2(oTriPoint - iPoint);  // barycentric coordinates (1,0,0)

		// Check if P in vertex region outside B
		const simdFloatVec<W> bp = iPoint - iTri[1];
		const simdFloat<W> d3 = dot(ab, bp);
		const simdFloat<W> d4 = dot(ac, bp);
		const simdBool<W> mask2 = (d3 >= simdFloat<W>(vecZero<4>().vec)) & (d4 <= d3);
		exit |= mask2;
		oTriPoint = select(mask2, iTri[1], oTriPoint);
		if (all(exit))
			return length2(oTriPoint - iPoint);  // barycentric coordinates (0,1,0)

		// Check if P in vertex region outside C
		const simdFloatVec<W> cp = iPoint - iTri[2];
		const simdFloat<W> d5 = dot(ab, cp);
		const simdFloat<W> d6 = dot(ac, cp);
		const simdBool<W> mask3 = (d6 >= simdFloat<W>(vecZero<4>().vec)) & (d5 <= d6);
		exit |= mask3;
		oTriPoint = select(mask3, iTri[2], oTriPoint);
		if (all(exit))
			return length2(oTriPoint - iPoint);  // barycentric coordinates (0,0,1)

		// Check if P in edge region of AB, if so return projection of P onto AB
		const simdFloat<W> vc = d1*d4 - d3*d2;
		const simdBool<W> mask4 = (vc <= simdFloat<W>(vecZero<4>().vec)) & (d1 >= simdFloat<W>(vecZero<4>().vec)) & (d3 <= simdFloat<W>(vecZero<4>().vec));
		exit |= mask4;
		const simdFloat<W> v1 = d1 / (d1 - d3);
		const simdFloatVec<W> answer1 = iTri[0] + v1 * ab;
		oTriPoint = select(mask4, answer1, oTriPoint);
		if (all(exit))
			return length2(oTriPoint - iPoint);  // barycentric coordinates (1-v,v,0)

		// Check if P in edge region of AC, if so return projection of P onto AC
		const simdFloat<W> vb = d5*d2 - d1*d6;
		const simdBool<W> mask5 = (vb <= simdFloat<W>(vecZero<4>().vec)) & (d2 >= simdFloat<W>(vecZero<4>().vec)) & (d6 <= simdFloat<W>(vecZero<4>().vec));
		exit |= mask5;
		const simdFloat<W> w1 = d2 / (d2 - d6);
		const simdFloatVec<W> answer2 = iTri[0] + w1 * ac;
		oTriPoint = select(mask5, answer2, oTriPoint);
		if (all(exit))
			return length2(oTriPoint - iPoint);  // barycentric coordinates (1-w,0,w)

		// Check if P in edge region of BC, if so return projection of P onto BC
		const simdFloat<W> va = d3*d6 - d5*d4;
		const simdBool<W> mask6 = (va <= simdFloat<W>(vecZero<4>().vec)) & ((d4 - d3) >= simdFloat<W>(vecZero<4>().vec)) & ((d5 - d6) >= simdFloat<W>(vecZero<4>().vec));
		exit |= mask6;
		simdFloat<W> w2 = (d4 - d3) / ((d4 - d3) + (d5 - d6));
		const simdFloatVec<W> answer3 = iTri[1] + w2 * (iTri[2] - iTri[1]);
		oTriPoint = select(mask6, answer3, oTriPoint);
		if (all(exit))
			return length2(oTriPoint - iPoint); // barycentric coordinates (0,1-w,w)

		// P inside face region. Compute Q through its barycentric coordinates (u,v,w)
		const simdFloat<W> denom = simdFloat<W>(_mm_set1_ps(1)) / (va + vb + vc);
		const simdFloat<W> v2 = vb * denom;
		const simdFloat<W> w3 = vc * denom;
		const simdFloatVec<W> answer4 = iTri[0] + ab * v2 + ac * w3;
		const simdBool<W> mask7 = andnot(exit, length2(answer4 - iPoint) < length2(oTriPoint - iPoint));
		oTriPoint = select(mask7, answer4, oTriPoint);
		return length2(oTriPoint - iPoint);  // = u*a + v*b + w*c, u = va * denom = 1.0f - v - w
	}

    template <int DIM, int W>
    struct ParallelInteraction{
        using SimdType = typename IntrinsicType<W>::type;
        SimdType distances;
        SimdType points[DIM];
        int indices[W];

        void getBest(float& distance, float point[DIM], int& index){
            distance = distances[0];
            index = indices[0];
            for(int i = 0; i < DIM; i++){
                point[i] = points[i][0];
            }
            for(int i = 1; indices[i] != -1 && i < W; i++){
                if(distance > distances[i]){
                    distance = distances[i];
                    index = indices[i];
                    for(int j = 0; j < DIM; j++){
                        point[j] = points[j][i];
                    }
                }
            }
        }
    };

    template <int W, class T>
    inline void parallelTriangleOverlap(const T pa[3], const T pb[3], const T pc[3], BoundingSphere<3>& s, ParallelInteraction<3, W>& i){
        float sPos[3];
        for(int j = 0; j < 3; j++){
            sPos[j] = (float)s.c(j);
        }
        simdFloatVec<W> resPts;
        simdTriangle_type<W> tri;
        simdPoint_type<W> sc;


        sc = embree::Vec3<simdFloat<W>>(vecf<W>(_mm_set1_ps(s.c(0))), vecf<W>(_mm_set1_ps(s.c(1))), vecf<W>(_mm_set1_ps(s.c(2))));
        tri[0] = embree::Vec3<simdFloat<W>>(vecf<W>(pa[0]), vecf<W>(pa[1]), vecf<W>(pa[2]));
        tri[1] = embree::Vec3<simdFloat<W>>(vecf<W>(pb[0]), vecf<W>(pb[1]), vecf<W>(pb[2]));
        tri[2] = embree::Vec3<simdFloat<W>>(vecf<W>(pc[0]), vecf<W>(pc[1]), vecf<W>(pc[2]));
        simdFloat<W> res = simdTriPoint2<W>(resPts, tri, sc);

        i.distances = res.vec;
        i.points[0] = resPts.x.vec;
        i.points[1] = resPts.y.vec;
        i.points[2] = resPts.z.vec;
    }

} //namespace fcpw