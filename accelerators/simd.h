#pragma once

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
        inline void init(float inits[], type toInit){
            LOG(INFO) << "Provided SIMD width is not a valid SIMD width";
        }
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

    inline void initSimd(const float inits[4], __m128& vec){
        vec = _mm_set_ps(inits[3], inits[2], inits[1], inits[0]);
    }
    inline void initSimd(const float inits[8], __m256& vec){
        vec = _mm256_set_ps(inits[7], inits[6], inits[5], inits[4],
                            inits[3], inits[2], inits[1], inits[0]);
    }
    inline void initSimd(const float inits[16], __m512& vec){
        vec = _mm512_set_ps(inits[15], inits[14], inits[13], inits[12],
                            inits[11], inits[10], inits[9], inits[8],
                            inits[7], inits[6], inits[5], inits[4],
                            inits[3], inits[2], inits[1], inits[0]);
    }

    // The following are taken and adapted from the following citation:
    /*
    Evan Shellshear and Robin Ytterlid, Fast Distance Queries for Triangles, Lines, and Points using SSE Instructions, Journal of Computer Graphics Techniques (JCGT), vol. 3, no. 4, 86-110, 2014

    http://jcgt.org/published/0003/04/05/   */

    // structs

    template <int W>
    struct vecb
    {
    };

    template <>
    struct vecb<4>
    {
        vecb(__m128 vec_) : vec(vec_){}
        __m128 vec;
        void operator |=(const vecb<4>& a){
            vec = _mm_or_ps(a.vec, vec);
        }
    };

    template <>
    struct vecb<8>
    {
        vecb(__m256 m256_) : m256(m256_){}
        __m256 m256;
        void operator |=(const vecb<8>& a){
            m256 = _mm256_or_ps(a.m256, m256);
        }
    };

    template <int W>
    struct vecf{
    };

    template <>
    struct vecf<4>{
        vecf(__m128 vec_) : vec(vec_){}
        __m128 vec;
    };

    template <>
    struct vecf<8>{
        vecf(__m256 m256_) : m256(m256_){}
        __m256 m256;
    };

    // SSE bool operators

    inline const vecb<4> operator &(const vecb<4>& a, const vecb<4>& b){
        return vecb<4>(_mm_and_ps(a.vec, b.vec));
    }
    inline const vecb<4> operator |(const vecb<4>& a, const vecb<4>& b){
        return vecb<4>(_mm_or_ps(a.vec, b.vec));
    }
    inline const vecb<4> operator ^(const vecb<4>& a, const vecb<4>& b){
        return vecb<4>(_mm_xor_ps(a.vec, b.vec));
    }
    inline bool all(const vecb<4>& b){
        return _mm_movemask_ps(b.vec) == 0xf;
    }
    inline bool any(const vecb<4>& b){
        return _mm_movemask_ps(b.vec) != 0x0;
    }
    inline bool none(const vecb<4>& b){
        return _mm_movemask_ps(b.vec) == 0x0;
    }
    inline vecb<4> andnot(const vecb<4>& a, const vecb<4>& b){
        return vecb<4>(_mm_andnot_ps(a.vec, b.vec));
    }   

    // SSE float operators

    inline const vecf<4> operator +(const vecf<4>& a, const vecf<4>&b){
        return vecf<4>(_mm_add_ps(a.vec, b.vec));
    }
    inline const vecf<4> operator -(const vecf<4>& a, const vecf<4>&b){
        return vecf<4>(_mm_sub_ps(a.vec, b.vec));
    }
    inline const vecf<4> operator *(const vecf<4>& a, const vecf<4>&b){
        return vecf<4>(_mm_mul_ps(a.vec, b.vec));
    }
    inline const vecf<4> operator /(const vecf<4>& a, const vecf<4>&b){
        return vecf<4>(_mm_div_ps(a.vec, b.vec));
    }
    inline const vecb<4> operator >=(const vecf<4>& a, const vecf<4>&b){
        return vecb<4>(_mm_cmpge_ps(a.vec, b.vec));
    }
    inline const vecb<4> operator >(const vecf<4>& a, const vecf<4>&b){
        return vecb<4>(_mm_cmpgt_ps(a.vec, b.vec));
    }
    inline const vecb<4> operator <=(const vecf<4>& a, const vecf<4>&b){
        return vecb<4>(_mm_cmple_ps(a.vec, b.vec));
    }
    inline const vecb<4> operator <(const vecf<4>& a, const vecf<4>&b){
        return vecb<4>(_mm_cmplt_ps(a.vec, b.vec));
    }
    inline const vecf<4> min(const vecf<4>& a, const vecf<4>&b){
        return vecf<4>(_mm_min_ps(a.vec, b.vec));
    }
    inline const vecf<4> max(const vecf<4>& a, const vecf<4>&b){
        return vecf<4>(_mm_max_ps(a.vec, b.vec));
    }
    inline const vecf<4> sqr(const vecf<4>& a){
        return vecf<4>(_mm_mul_ps(a.vec, a.vec));
    }
    inline const vecf<4> sqrt(const vecf<4>& a){
        return vecf<4>(_mm_sqrt_ps(a.vec));
    }
    inline const vecf<4> select(const vecb<4>& mask, const vecf<4>& t, const vecf<4>& f){
        return vecf<4>(_mm_blendv_ps(f.vec, t.vec, mask.vec));
    }
    
    static const __m128 sseZero = _mm_setzero_ps();
    static const __m128 sseOne = _mm_set1_ps(0xffff);

    inline const vecf<4> vecZero(){
        return vecf<4>(sseZero);
    }
    inline const vecf<4> vecOne(){
        return vecf<4>(sseOne);
    }


    // template <int W>
	using simdBool  = vecb<4>;
    using simdFloat = vecf<4>;

	//typedef avxf simdFloat;
	//typedef avxb simdBool;
	//typedef avxi simdInt;

	typedef embree::Vec3<simdBool>	simdBoolVec;
	typedef embree::Vec3<simdFloat> simdFloatVec;

	typedef std::array<simdFloatVec, 3>		simdTriangle_type;
	typedef std::array<simdFloatVec, 2>		simdLine_type;
	typedef simdFloatVec					simdPoint_type;

    inline simdFloat length2(const embree::Vec3<simdFloat>& a){
        return vecf<4>(embree::dot(a, a));
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

}// namespace fcpw