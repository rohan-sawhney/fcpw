#pragma once

#include "immintrin.h"
#include <array>
#include "bounding_volumes.h"
#include "common/simd/sse.h"
#include "common/math/vec2.h"
#include "common/math/vec3.h"
#include "common/math/vec4.h"

// #include "common/simd/avx.h"

namespace fcpw{

    template<int W>
    struct UndefinedSIMDType{};

    template <int W>
    struct IntrinsicType{
        using type = UndefinedSIMDType<W>;
        static_assert((W == 4) || (W == 8) || (W == 16), "Provided SIMD width is not a valid SIMD width");
    };

    template <>
    struct IntrinsicType<4>{
        using type = __m128;
    };

    template <>
    struct IntrinsicType<8>{
        using type = __m256;
    };

    template <>
    struct IntrinsicType<16>{
        using type = __m512;
    };

    template <int W>
    struct UndefinedCounterType{};
    
    // template <int W>
    // struct CountType{
    //     using type = UndefinedCounterType<W>;
    //     static_assert((W == 4) || (W == 8) || (W == 16), "Provided SIMD width is not a valid SIMD width");
    // };

    // template <>
    // struct CountType<4>{
    //     using type = char;
    // };

    // template <>
    // struct CountType<8>{
    //     using type = char;
    // };

    // template <>
    // struct CountType<16>{
    //     using type = short;
    // };

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
        bool isLeaf[W];
    };

    template <int DIM, int W>
    struct BvhSimdLeafNode{
        using SimdType = typename IntrinsicType<W>::type;

        // SimdType minBoxes[DIM];
        // SimdType maxBoxes[DIM];
        SimdType pa[DIM];
        SimdType pb[DIM];
        SimdType pc[DIM];
        int indices[W]; // will later wrap in triangle data
    };

    /* ---- Vectorized Functions ---- */

    // parallel bbox overlap test
    template <class T>
    inline void parallelComputeSquaredDistance(const int dim, const T boxMins[], const T boxMaxs[], const float sPos[], const float& sRad, T& d2Min, T& d2Max){
        LOG(FATAL) << "Provided type is not a SIMD float vector type";
    }

    template <>
    inline void parallelComputeSquaredDistance<__m512>(const int dim, const __m512 boxMins[], const __m512 boxMaxs[], const float sPos[], const float& sRad, __m512& d2Min, __m512& d2Max){
        d2Min = _mm512_setzero_ps();
        d2Max = _mm512_setzero_ps();
        __m512 temp;
        __m512 pos;
        for(int i = 0; i < dim; i++){
            pos = _mm512_set1_ps((float)sPos[i]);
            temp = _mm512_max_ps(_mm512_sub_ps(boxMins[i], pos), _mm512_max_ps(_mm512_setzero_ps(), _mm512_sub_ps(pos, boxMaxs[i])));
            d2Min = _mm512_add_ps(d2Min, _mm512_mul_ps(temp, temp));

            temp = _mm512_max_ps(_mm512_sub_ps(boxMins[i], pos), _mm512_sub_ps(pos, boxMaxs[i]));
            d2Max = _mm512_add_ps(d2Max, _mm512_mul_ps(temp, temp));
        }
    }
    
    template <>
    inline void parallelComputeSquaredDistance<__m256>(const int dim, const __m256 boxMins[], const __m256 boxMaxs[], const float sPos[], const float& sRad, __m256& d2Min, __m256& d2Max){
        d2Min = _mm256_setzero_ps();
        d2Max = _mm256_setzero_ps();
        __m256 temp;
        __m256 pos;
        for(int i = 0; i < dim; i++){
            pos = _mm256_set1_ps((float)sPos[i]);
            temp = _mm256_max_ps(_mm256_sub_ps(boxMins[i], pos), _mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(pos, boxMaxs[i])));
            d2Min = _mm256_add_ps(d2Min, _mm256_mul_ps(temp, temp));

            temp = _mm256_max_ps(_mm256_sub_ps(boxMins[i], pos), _mm256_sub_ps(pos, boxMaxs[i]));
            d2Max = _mm256_add_ps(d2Max, _mm256_mul_ps(temp, temp));
        }
    }
    
    template <>
    inline void parallelComputeSquaredDistance<__m128>(const int dim, const __m128 boxMins[], const __m128 boxMaxs[], const float sPos[], const float& sRad, __m128& d2Min, __m128& d2Max){
        d2Min = _mm_setzero_ps();
        d2Max = _mm_setzero_ps();
        __m128 temp, pos, maxDif, minDif;
        for(int i = 0; i < 3; i++){
            pos = _mm_set1_ps((float)sPos[i]);
            minDif = _mm_sub_ps(boxMins[i], pos);
            maxDif = _mm_sub_ps(pos, boxMaxs[i]);
            temp = _mm_max_ps(minDif, _mm_max_ps(_mm_setzero_ps(), maxDif));
            d2Min = _mm_add_ps(d2Min, _mm_mul_ps(temp, temp));

            minDif = _mm_xor_ps(minDif, _mm_set1_ps(-0.0));
            maxDif = _mm_xor_ps(maxDif, _mm_set1_ps(-0.0));
            temp = _mm_max_ps(minDif, maxDif);
            d2Max = _mm_add_ps(d2Max, _mm_mul_ps(temp, temp));
        }
    }

    template <int DIM, class T>
    inline void parallelOverlap(const T boxMins[DIM], const T boxMaxs[DIM], BoundingSphere<DIM>& s, T& d2Min, T& d2Max){
        float sPos[DIM];
        for(int i = 0; i < DIM; i++){
            sPos[i] = (float)s.c(i);
        }
        parallelComputeSquaredDistance(DIM, boxMins, boxMaxs, sPos, s.r2, d2Min, d2Max);
    }

    // parallel triangle computation function
    template <class T>
    inline void parallelComputeTriangleDistance(const int dim, const T pa[], const T pb[], const T pc[], const float x[], T& d, T pt[], int idx[]){
        LOG(FATAL) << "Provided type is not a SIMD float vector type";
    }

    template <>
    inline void parallelComputeTriangleDistance<__m128>(const int dim, const __m128 pa[], const __m128 pb[], const __m128 pc[], const float x[], __m128& d, __m128 pt[], int idx[]){
        __m128 px[dim];
        __m128 temp;
        for(int i = 0; i < dim; i++){
            px[i] = _mm_set1_ps(x[i]);
        }
        __m128 exit = _mm_set_ps(idx[3] != -1 ? 0x00000000 : 0x11111111, idx[2] != -1 ? 0x00000000 : 0x11111111, idx[1] != -1 ? 0x00000000 : 0x11111111, idx[0] != -1 ? 0x00000000 : 0x11111111);

        // check barycentric coord (1, 0, 0)
        __m128 d1 = _mm_setzero_ps();
        __m128 d2 = _mm_setzero_ps();
        __m128 ab[dim], ac[dim];
        __m128 vx;
        for(int i = 0; i < dim; i++){
            ab[i] = _mm_sub_ps(pb[i], pa[i]);
            ac[i] = _mm_sub_ps(pc[i], pa[i]);
            vx = _mm_sub_ps(px[i], pa[i]);
            d1 = _mm_add_ps(_mm_mul_ps(ab[i], vx), d1);
            d2 = _mm_add_ps(_mm_mul_ps(ac[i], vx), d2);
        }
        __m128 mask = _mm_and_ps(_mm_cmple_ps(d1, _mm_setzero_ps()),_mm_cmple_ps(d2, _mm_setzero_ps()));
        if(_mm_movemask_ps(mask) != 0xf){
            exit = _mm_or_ps(exit, mask);
            for(int i = 0; i < dim; i++){
                pt[i] = _mm_blendv_ps(pt[i], pa[i], mask);
            }
            if(_mm_movemask_ps(exit) == 0xf){
                d = _mm_setzero_ps();
                for(int i = 0; i < dim; i++){
                    temp = _mm_sub_ps(pt[i], px[i]);
                    d = _mm_add_ps(d, _mm_mul_ps(temp, temp));
                }
                return;
            }
        }

        // exit = mask;
        // for(int i = 0; i < dim; i++){
        //     pt[i] = pa[i];
        // }
        // if(_mm_movemask_ps(exit) == 0xf) continue;

        // check barycentric coord (0, 1, 0)
        __m128 d3 = _mm_setzero_ps();
        __m128 d4 = _mm_setzero_ps();
        for(int i = 0; i < dim; i++){
            vx = _mm_sub_ps(px[i], pb[i]);
            d3 = _mm_add_ps(_mm_mul_ps(ab[i], vx), d3);
            d4 = _mm_add_ps(_mm_mul_ps(ac[i], vx), d4);
        }
        mask = _mm_and_ps(_mm_cmpge_ps(d3, _mm_setzero_ps()),_mm_cmple_ps(d4, d3));
        if(_mm_movemask_ps(mask) != 0xf){
            exit = _mm_or_ps(exit, mask);
            for(int i = 0; i < dim; i++){
                pt[i] = _mm_blendv_ps(pt[i], pb[i], mask);
            }
            if(_mm_movemask_ps(exit) == 0xf){
                d = _mm_setzero_ps();
                for(int i = 0; i < dim; i++){
                    temp = _mm_sub_ps(pt[i], px[i]);
                    d = _mm_add_ps(d, _mm_mul_ps(temp, temp));
                }
                return;
            }
        }

        // check barycentric coord (0, 0, 1)
        __m128 d5 = _mm_setzero_ps();
        __m128 d6 = _mm_setzero_ps();
        for(int i = 0; i < dim; i++){
            vx = _mm_sub_ps(px[i], pc[i]);
            d5 = _mm_add_ps(_mm_mul_ps(ab[i], vx), d5);
            d6 = _mm_add_ps(_mm_mul_ps(ac[i], vx), d6);
        }
        mask = _mm_and_ps(_mm_cmpge_ps(d6, _mm_setzero_ps()),_mm_cmple_ps(d5, d6));
        if(_mm_movemask_ps(mask) != 0xf){
            exit = _mm_or_ps(exit, mask);
            for(int i = 0; i < dim; i++){
                pt[i] = _mm_blendv_ps(pt[i], pc[i], mask);
            }
            if(_mm_movemask_ps(exit) == 0xf){
                d = _mm_setzero_ps();
                for(int i = 0; i < dim; i++){
                    temp = _mm_sub_ps(pt[i], px[i]);
                    d = _mm_add_ps(d, _mm_mul_ps(temp, temp));
                }
                return;
            }
        }

        // check barycentric coord (1 - v, v, 0)
        __m128 v;
        __m128 vc = _mm_sub_ps(_mm_mul_ps(d1, d4), _mm_mul_ps(d3, d2));
        mask = _mm_and_ps(_mm_cmple_ps(vc, _mm_setzero_ps()),_mm_and_ps(_mm_cmpge_ps(d1, _mm_setzero_ps()), _mm_cmple_ps(d3, _mm_setzero_ps())));
        if(_mm_movemask_ps(mask) != 0xf){
            exit = _mm_or_ps(exit, mask);
            v = _mm_div_ps(d1, _mm_sub_ps(d1, d3));
            for(int i = 0; i < dim; i++){
                pt[i] = _mm_blendv_ps(pt[i], _mm_add_ps(pa[i], _mm_mul_ps(v, ab[i])), mask);
            }
            if(_mm_movemask_ps(exit) == 0xf){
                d = _mm_setzero_ps();
                for(int i = 0; i < dim; i++){
                    temp = _mm_sub_ps(pt[i], px[i]);
                    d = _mm_add_ps(d, _mm_mul_ps(temp, temp));
                }
                return;
            }
        }

        // check barycentric coord (1 - w, 0, w)
        __m128 w;
        __m128 vb = _mm_sub_ps(_mm_mul_ps(d5, d2), _mm_mul_ps(d1, d6));
        mask = _mm_and_ps(_mm_cmple_ps(vb, _mm_setzero_ps()),_mm_and_ps(_mm_cmpge_ps(d2, _mm_setzero_ps()), _mm_cmple_ps(d6, _mm_setzero_ps())));
        if(_mm_movemask_ps(mask) != 0xf){
            exit = _mm_or_ps(exit, mask);
            w = _mm_div_ps(d2, _mm_sub_ps(d2, d6));
            for(int i = 0; i < dim; i++){
                pt[i] = _mm_blendv_ps(pt[i], _mm_add_ps(pa[i], _mm_mul_ps(w, ac[i])), mask);
            }
            if(_mm_movemask_ps(exit) == 0xf){
                d = _mm_setzero_ps();
                for(int i = 0; i < dim; i++){
                    temp = _mm_sub_ps(pt[i], px[i]);
                    d = _mm_add_ps(d, _mm_mul_ps(temp, temp));
                }
                return;
            }
        }

        // check barycentric coord (0, 1 - w, w)
        __m128 va = _mm_sub_ps(_mm_mul_ps(d3, d6), _mm_mul_ps(d5, d4));
        mask = _mm_and_ps(_mm_cmple_ps(va, _mm_setzero_ps()),_mm_and_ps(_mm_cmpge_ps(_mm_sub_ps(d4, d3), _mm_setzero_ps()), _mm_cmpge_ps(_mm_sub_ps(d5, d6), _mm_setzero_ps())));
        if(_mm_movemask_ps(mask) != 0xf){
            exit = _mm_or_ps(exit, mask);
            w = _mm_div_ps(_mm_sub_ps(d4, d3), _mm_add_ps(_mm_sub_ps(d4, d3), _mm_sub_ps(d5, d6)));
            for(int i = 0; i < dim; i++){
                pt[i] = _mm_blendv_ps(pt[i], _mm_add_ps(pb[i], _mm_mul_ps(w, _mm_sub_ps(pc[i], pb[i]))), mask);
            }
            if(_mm_movemask_ps(exit) == 0xf){
                d = _mm_setzero_ps();
                for(int i = 0; i < dim; i++){
                    temp = _mm_sub_ps(pt[i], px[i]);
                    d = _mm_add_ps(d, _mm_mul_ps(temp, temp));
                }
                return;
            }
        }

        // check barycentric coord (u, v, w)
        __m128 denom = _mm_div_ps(_mm_set1_ps(1.0), _mm_add_ps(va, _mm_add_ps(vb, vc)));
        v = _mm_mul_ps(vb, denom);
        w = _mm_mul_ps(vc, denom);
        mask = _mm_xor_ps(exit, _mm_set1_ps(0xffffffff));
        for(int i = 0; i < dim; i++){
            pt[i] = _mm_blendv_ps(pt[i], _mm_add_ps(pa[i], _mm_add_ps(_mm_mul_ps(ab[i], v), _mm_mul_ps(ac[i], w))), mask);
        }
        d = _mm_setzero_ps();
        for(int i = 0; i < dim; i++){
            temp = _mm_sub_ps(pt[i], px[i]);
            d = _mm_add_ps(d, _mm_mul_ps(temp, temp));
        }
    }

    // The following are taken and adapted from the following citation:
    /*
    Evan Shellshear and Robin Ytterlid, Fast Distance Queries for Triangles, Lines, and Points using SSE Instructions, Journal of Computer Graphics Techniques (JCGT), vol. 3, no. 4, 86-110, 2014

    http://jcgt.org/published/0003/04/05/   */

    struct sseb
    {
        sseb(__m128 m128_) : m128(m128_){}
        __m128 m128;
        void operator |=(const sseb& a){
            m128 = _mm_or_ps(a.m128, m128);
        }
    };
    const sseb operator &(const sseb& a, const sseb& b){
        return sseb(_mm_and_ps(a.m128, b.m128));
    }
    const sseb operator |(const sseb& a, const sseb& b){
        return sseb(_mm_or_ps(a.m128, b.m128));
    }
    const sseb operator ^(const sseb& a, const sseb& b){
        return sseb(_mm_xor_ps(a.m128, b.m128));
    }
    bool all(const sseb& b){
        return _mm_movemask_ps(b.m128) == 0xf;
    }
    bool any(const sseb& b){
        return _mm_movemask_ps(b.m128) != 0x0;
    }
    bool none(const sseb& b){
        return _mm_movemask_ps(b.m128) == 0x0;
    }

    struct ssef{
        ssef(__m128 m128_) : m128(m128_){}
        __m128 m128;
    };
    const ssef operator +(const ssef& a, const ssef&b){
        return ssef(_mm_add_ps(a.m128, b.m128));
    }
    const ssef operator -(const ssef& a, const ssef&b){
        return ssef(_mm_sub_ps(a.m128, b.m128));
    }
    const ssef operator *(const ssef& a, const ssef&b){
        return ssef(_mm_mul_ps(a.m128, b.m128));
    }
    const ssef operator /(const ssef& a, const ssef&b){
        return ssef(_mm_div_ps(a.m128, b.m128));
    }
    const sseb operator >=(const ssef& a, const ssef&b){
        return sseb(_mm_cmpge_ps(a.m128, b.m128));
    }
    const sseb operator >(const ssef& a, const ssef&b){
        return sseb(_mm_cmpgt_ps(a.m128, b.m128));
    }
    const sseb operator <=(const ssef& a, const ssef&b){
        return sseb(_mm_cmple_ps(a.m128, b.m128));
    }
    const sseb operator <(const ssef& a, const ssef&b){
        return sseb(_mm_cmplt_ps(a.m128, b.m128));
    }
    const ssef min(const ssef& a, const ssef&b){
        return ssef(_mm_min_ps(a.m128, b.m128));
    }
    const ssef max(const ssef& a, const ssef&b){
        return ssef(_mm_max_ps(a.m128, b.m128));
    }
    const ssef sqr(const ssef& a){
        return ssef(_mm_mul_ps(a.m128, a.m128));
    }
    const ssef sqrt(const ssef& a){
        return ssef(_mm_sqrt_ps(a.m128));
    }
    const ssef select(const sseb& mask, const ssef& t, const ssef& f){
        return ssef(_mm_blendv_ps(f.m128, t.m128, mask.m128));
    }
    static const __m128 vecZero = _mm_setzero_ps();
    static const __m128 vecOne = _mm_set1_ps(0xffff);

	typedef sseb simdBool;
	typedef ssef simdFloat;

	//typedef avxf simdFloat;
	//typedef avxb simdBool;
	//typedef avxi simdInt;

	typedef embree::Vec3<simdBool>	simdBoolVec;
	typedef embree::Vec3<simdFloat> simdFloatVec;

	typedef std::array<simdFloatVec, 3>		simdTriangle_type;
	typedef std::array<simdFloatVec, 2>		simdLine_type;
	typedef simdFloatVec					simdPoint_type;

    inline simdFloat length2(const embree::Vec3<simdFloat>& a){
        return ssef(embree::dot(a, a));
    }

    __forceinline const simdFloatVec select(const simdBool& s, const simdFloatVec& t, const simdFloatVec& f) {
		return simdFloatVec(select(s, t.x, f.x), select(s, t.y, f.y), select(s, t.z, f.z));
	}

	__forceinline const simdLine_type select(const simdBool& s, const simdLine_type& t, const simdLine_type& f) {
		const simdFloatVec start(select(s, t[0].x, f[0].x), select(s, t[0].y, f[0].y), select(s, t[0].z, f[0].z));
		const simdFloatVec end(select(s, t[1].x, f[1].x), select(s, t[1].y, f[1].y), select(s, t[1].z, f[1].z));
		const simdLine_type result = { start, end };
		return result;
	}

    const simdFloat simdTriPoint2(simdFloatVec& oTriPoint, const simdTriangle_type& iTri, const simdPoint_type& iPoint){
		// Check if P in vertex region outside A
		const simdFloatVec ab = iTri[1] - iTri[0];
		const simdFloatVec ac = iTri[2] - iTri[0];
		const simdFloatVec ap = iPoint - iTri[0];
		const simdFloat d1 = dot(ab, ap);
		const simdFloat d2 = dot(ac, ap);
		const simdBool mask1 = (d1 <= simdFloat(vecZero)) & (d2 <= simdFloat(vecZero));
		oTriPoint = iTri[0];
		simdBool exit(mask1);
		if (all(exit))
			return length2(oTriPoint - iPoint);  // barycentric coordinates (1,0,0)

		// Check if P in vertex region outside B
		const simdFloatVec bp = iPoint - iTri[1];
		const simdFloat d3 = dot(ab, bp);
		const simdFloat d4 = dot(ac, bp);
		const simdBool mask2 = (d3 >= simdFloat(vecZero)) & (d4 <= d3);
		exit |= mask2;
		oTriPoint = select(mask2, iTri[1], oTriPoint);
		if (all(exit))
			return length2(oTriPoint - iPoint);  // barycentric coordinates (0,1,0)

		// Check if P in vertex region outside C
		const simdFloatVec cp = iPoint - iTri[2];
		const simdFloat d5 = dot(ab, cp);
		const simdFloat d6 = dot(ac, cp);
		const simdBool mask3 = (d6 >= simdFloat(vecZero)) & (d5 <= d6);
		exit |= mask3;
		oTriPoint = select(mask3, iTri[2], oTriPoint);
		if (all(exit))
			return length2(oTriPoint - iPoint);  // barycentric coordinates (0,0,1)

		// Check if P in edge region of AB, if so return projection of P onto AB
		const simdFloat vc = d1*d4 - d3*d2;
		const simdBool mask4 = (vc <= simdFloat(vecZero)) & (d1 >= simdFloat(vecZero)) & (d3 <= simdFloat(vecZero));
		exit |= mask4;
		const simdFloat v1 = d1 / (d1 - d3);
		const simdFloatVec answer1 = iTri[0] + v1 * ab;
		oTriPoint = select(mask4, answer1, oTriPoint);
		if (all(exit))
			return length2(oTriPoint - iPoint);  // barycentric coordinates (1-v,v,0)

		// Check if P in edge region of AC, if so return projection of P onto AC
		const simdFloat vb = d5*d2 - d1*d6;
		const simdBool mask5 = (vb <= simdFloat(vecZero)) & (d2 >= simdFloat(vecZero)) & (d6 <= simdFloat(vecZero));
		exit |= mask5;
		const simdFloat w1 = d2 / (d2 - d6);
		const simdFloatVec answer2 = iTri[0] + w1 * ac;
		oTriPoint = select(mask5, answer2, oTriPoint);
		if (all(exit))
			return length2(oTriPoint - iPoint);  // barycentric coordinates (1-w,0,w)

		// Check if P in edge region of BC, if so return projection of P onto BC
		const simdFloat va = d3*d6 - d5*d4;
		const simdBool mask6 = (va <= simdFloat(vecZero)) & ((d4 - d3) >= simdFloat(vecZero)) & ((d5 - d6) >= simdFloat(vecZero));
		exit |= mask6;
		simdFloat w2 = (d4 - d3) / ((d4 - d3) + (d5 - d6));
		const simdFloatVec answer3 = iTri[1] + w2 * (iTri[2] - iTri[1]);
		oTriPoint = select(mask6, answer3, oTriPoint);
		if (all(exit))
			return length2(oTriPoint - iPoint); // barycentric coordinates (0,1-w,w)

		// P inside face region. Compute Q through its barycentric coordinates (u,v,w)
		const simdFloat denom = simdFloat(_mm_set1_ps(1)) / (va + vb + vc);
		const simdFloat v2 = vb * denom;
		const simdFloat w3 = vc * denom;
		const simdFloatVec answer4 = iTri[0] + ab * v2 + ac * w3;
		const simdBool mask7 = length2(answer4 - iPoint) < length2(oTriPoint - iPoint);
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

    template <int DIM, int W, class T>
    inline void parallelTriangleOverlap(const T pa[DIM], const T pb[DIM], const T pc[DIM], BoundingSphere<DIM>& s, ParallelInteraction<DIM, W>& i){
        float sPos[DIM];
        for(int j = 0; j < DIM; j++){
            sPos[j] = (float)s.c(j);
        }
        parallelComputeTriangleDistance(DIM, pa, pb, pc, sPos, i.distances, i.points, i.indices);
    }

    template <int W, class T>
    inline void parallelTriangleOverlap2(const T pa[3], const T pb[3], const T pc[3], BoundingSphere<3>& s, ParallelInteraction<3, W>& i){
        float sPos[3];
        for(int j = 0; j < 3; j++){
            sPos[j] = (float)s.c(j);
        }
        simdFloatVec resPts;
        simdTriangle_type tri;
        simdPoint_type sc;
        sc = embree::Vec3<simdFloat>(ssef(_mm_set1_ps(s.c(0))), ssef(_mm_set1_ps(s.c(1))), ssef(_mm_set1_ps(s.c(2))));
        tri[0] = embree::Vec3<simdFloat>(ssef(pa[0]), ssef(pa[1]), ssef(pa[2]));
        tri[1] = embree::Vec3<simdFloat>(ssef(pb[0]), ssef(pb[1]), ssef(pb[2]));
        tri[2] = embree::Vec3<simdFloat>(ssef(pc[0]), ssef(pc[1]), ssef(pc[2]));
        simdFloat res = simdTriPoint2(resPts, tri, sc);

        i.distances = res.m128;
        i.points[0] = resPts.x.m128;
        i.points[1] = resPts.y.m128;
        i.points[2] = resPts.z.m128;
    }

} //namespace fcpw