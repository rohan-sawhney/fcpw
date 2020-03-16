#pragma once

#include "common/math/vec2.h"
#include "common/math/vec3.h"
#include "common/math/vec4.h"

#include "common/simd/sse.h"
#include "common/simd/avx.h"
#include "common/simd/avx512.h"

namespace fcpw{

    // translation from simd width to simd type via partial template specialization
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

    // inits an sse vector
    inline void initSimd(const float inits[4], __m128& vec){
        vec = _mm_set_ps(inits[3], inits[2], inits[1], inits[0]);
    }
    // inits an avx vector
    inline void initSimd(const float inits[8], __m256& vec){
        vec = _mm256_set_ps(inits[7], inits[6], inits[5], inits[4],
                            inits[3], inits[2], inits[1], inits[0]);
    }
    // inits an avx512 vector
    inline void initSimd(const float inits[16], __m512& vec){
        vec = _mm512_set_ps(inits[15], inits[14], inits[13], inits[12],
                            inits[11], inits[10], inits[9], inits[8],
                            inits[7], inits[6], inits[5], inits[4],
                            inits[3], inits[2], inits[1], inits[0]);
    }

    // inits an sse vector with a constant value
    inline void initSimd(const float init, __m128& vec){
        vec = _mm_set1_ps(init);
    }
    // inits an avx vector with a constant value
    inline void initSimd(const float init, __m256& vec){
        vec = _mm256_set1_ps(init);
    }
    // inits an avx512 vector with a constant value
    inline void initSimd(const float init, __m512& vec){
        vec = _mm512_set1_ps(init);
    }

    // The following are taken and adapted from the following citation:
    /*
    Evan Shellshear and Robin Ytterlid, Fast Distance Queries for Triangles, Lines, and Points using SSE Instructions, Journal of Computer Graphics Techniques (JCGT), vol. 3, no. 4, 86-110, 2014

    http://jcgt.org/published/0003/04/05/   */

    static const __m128 sseZero = _mm_setzero_ps();
    static const __m128 sseOne = _mm_set1_ps(0xffff);

    static const __m256 avxZero = _mm256_setzero_ps();
    static const __m256 avxOne = _mm256_set1_ps(0xffff);

    static const __m512 avx512Zero = _mm512_setzero_ps();
    static const __m512 avx512One = _mm512_set1_ps(0xffff);

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
        vecb(__m256 vec_) : vec(vec_){}
        __m256 vec;
        void operator |=(const vecb<8>& a){
            vec = _mm256_or_ps(a.vec, vec);
        }
    };

    template <>
    struct vecb<16>
    {
        vecb(__mmask16 vec_) : vec(vec_){}
        __mmask16 vec;
        void operator |=(const vecb<16>& a){
            vec = _kor_mask16(a.vec, vec);
        }
    };

    template <int W>
    struct vecf{
    };

    template <>
    struct vecf<4>{
        vecf(__m128 vec_) {
            V.vec = vec_;
        }
        vecf() {
            V.vec = sseZero;
        }
        union{
            __m128 vec;
            float v[4];
        } V;
    };

    template <>
    struct vecf<8>{
        vecf(__m256 vec_) {
            V.vec = vec_;
        }
        vecf() {
            V.vec = avxZero;
        }
        union{
            __m256 vec;
            float v[8];
        } V;
    };

    template <>
    struct vecf<16>{
        vecf(__m512 vec_) {
            V.vec = vec_;
        }
        vecf() {
            V.vec = avx512Zero;
        }
        union{
            __m512 vec;
            float v[16];
        } V;
    };

    template <int W>
    inline const vecf<W> vecZero(){
        return vecf<W>();
    }

    template <int W>
    inline const vecf<W> vecOne(){
        return vecf<W>();
    }

    template <int W>
    inline const vecf<W> vecInit(const float f){
        LOG(FATAL) << "Invalid simd width";
        return vecf<W>();
    }

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
        return vecf<4>(_mm_add_ps(a.V.vec, b.V.vec));
    }
    inline const vecf<4> operator -(const vecf<4>& a, const vecf<4>&b){
        return vecf<4>(_mm_sub_ps(a.V.vec, b.V.vec));
    }
    inline const vecf<4> operator *(const vecf<4>& a, const vecf<4>&b){
        return vecf<4>(_mm_mul_ps(a.V.vec, b.V.vec));
    }
    inline const vecf<4> operator /(const vecf<4>& a, const vecf<4>&b){
        return vecf<4>(_mm_div_ps(a.V.vec, b.V.vec));
    }
    inline const vecb<4> operator >=(const vecf<4>& a, const vecf<4>&b){
        return vecb<4>(_mm_cmpge_ps(a.V.vec, b.V.vec));
    }
    inline const vecb<4> operator >(const vecf<4>& a, const vecf<4>&b){
        return vecb<4>(_mm_cmpgt_ps(a.V.vec, b.V.vec));
    }
    inline const vecb<4> operator <=(const vecf<4>& a, const vecf<4>&b){
        return vecb<4>(_mm_cmple_ps(a.V.vec, b.V.vec));
    }
    inline const vecb<4> operator <(const vecf<4>& a, const vecf<4>&b){
        return vecb<4>(_mm_cmplt_ps(a.V.vec, b.V.vec));
    }
    inline const vecf<4> min(const vecf<4>& a, const vecf<4>&b){
        return vecf<4>(_mm_min_ps(a.V.vec, b.V.vec));
    }
    inline const vecf<4> max(const vecf<4>& a, const vecf<4>&b){
        return vecf<4>(_mm_max_ps(a.V.vec, b.V.vec));
    }
    inline const vecf<4> sqr(const vecf<4>& a){
        return vecf<4>(_mm_mul_ps(a.V.vec, a.V.vec));
    }
    inline const vecf<4> sqrt(const vecf<4>& a){
        return vecf<4>(_mm_sqrt_ps(a.V.vec));
    }
    inline const vecf<4> select(const vecb<4>& mask, const vecf<4>& t, const vecf<4>& f){
        return vecf<4>(_mm_blendv_ps(f.V.vec, t.V.vec, mask.vec));
    }

    template <>
    inline const vecf<4> vecZero(){
        return vecf<4>(sseZero);
    }
    template <>
    inline const vecf<4> vecOne(){
        return vecf<4>(sseOne);
    }

    template <>
    inline const vecf<4> vecInit(const float f){
        return vecf<4>(_mm_set1_ps(f));
    }

    // AVX bool operators

    inline const vecb<8> operator &(const vecb<8>& a, const vecb<8>& b){
        return vecb<8>(_mm256_and_ps(a.vec, b.vec));
    }
    inline const vecb<8> operator |(const vecb<8>& a, const vecb<8>& b){
        return vecb<8>(_mm256_or_ps(a.vec, b.vec));
    }
    inline const vecb<8> operator ^(const vecb<8>& a, const vecb<8>& b){
        return vecb<8>(_mm256_xor_ps(a.vec, b.vec));
    }
    inline bool all(const vecb<8>& b){
        return _mm256_movemask_ps(b.vec) == 0xff;
    }
    inline bool any(const vecb<8>& b){
        return _mm256_movemask_ps(b.vec) != 0x00;
    }
    inline bool none(const vecb<8>& b){
        return _mm256_movemask_ps(b.vec) == 0x00;
    }
    inline vecb<8> andnot(const vecb<8>& a, const vecb<8>& b){
        return vecb<8>(_mm256_andnot_ps(a.vec, b.vec));
    }

    // AVX float operators

    inline const vecf<8> operator +(const vecf<8>& a, const vecf<8>&b){
        return vecf<8>(_mm256_add_ps(a.V.vec, b.V.vec));
    }
    inline const vecf<8> operator -(const vecf<8>& a, const vecf<8>&b){
        return vecf<8>(_mm256_sub_ps(a.V.vec, b.V.vec));
    }
    inline const vecf<8> operator *(const vecf<8>& a, const vecf<8>&b){
        return vecf<8>(_mm256_mul_ps(a.V.vec, b.V.vec));
    }
    inline const vecf<8> operator /(const vecf<8>& a, const vecf<8>&b){
        return vecf<8>(_mm256_div_ps(a.V.vec, b.V.vec));
    }
    inline const vecb<8> operator >=(const vecf<8>& a, const vecf<8>&b){
        return vecb<8>(_mm256_cmp_ps(a.V.vec, b.V.vec, _CMP_GE_OS));
    }
    inline const vecb<8> operator >(const vecf<8>& a, const vecf<8>&b){
        return vecb<8>(_mm256_cmp_ps(a.V.vec, b.V.vec, _CMP_GT_OS));
    }
    inline const vecb<8> operator <=(const vecf<8>& a, const vecf<8>&b){
        return vecb<8>(_mm256_cmp_ps(a.V.vec, b.V.vec, _CMP_LE_OS));
    }
    inline const vecb<8> operator <(const vecf<8>& a, const vecf<8>&b){
        return vecb<8>(_mm256_cmp_ps(a.V.vec, b.V.vec, _CMP_LT_OS));
    }
    inline const vecf<8> min(const vecf<8>& a, const vecf<8>&b){
        return vecf<8>(_mm256_min_ps(a.V.vec, b.V.vec));
    }
    inline const vecf<8> max(const vecf<8>& a, const vecf<8>&b){
        return vecf<8>(_mm256_max_ps(a.V.vec, b.V.vec));
    }
    inline const vecf<8> sqr(const vecf<8>& a){
        return vecf<8>(_mm256_mul_ps(a.V.vec, a.V.vec));
    }
    inline const vecf<8> sqrt(const vecf<8>& a){
        return vecf<8>(_mm256_sqrt_ps(a.V.vec));
    }
    inline const vecf<8> select(const vecb<8>& mask, const vecf<8>& t, const vecf<8>& f){
        return vecf<8>(_mm256_blendv_ps(f.V.vec, t.V.vec, mask.vec));
    }

    template <>
    inline const vecf<8> vecZero(){
        return vecf<8>(avxZero);
    }
    template <>
    inline const vecf<8> vecOne(){
        return vecf<8>(avxOne);
    }

    template <>
    inline const vecf<8> vecInit(const float f){
        return vecf<8>(_mm256_set1_ps(f));
    }

    // AVX512 bool operators

    inline const vecb<16> operator &(const vecb<16>& a, const vecb<16>& b){
        return vecb<16>(_kand_mask16(a.vec, b.vec));
    }
    inline const vecb<16> operator |(const vecb<16>& a, const vecb<16>& b){
        return vecb<16>(_kor_mask16(a.vec, b.vec));
    }
    inline const vecb<16> operator ^(const vecb<16>& a, const vecb<16>& b){
        return vecb<16>(_kxor_mask16(a.vec, b.vec));
    }
    inline bool all(const vecb<16>& b){
        return b.vec == 0xffff;
    }
    inline bool any(const vecb<16>& b){
        return b.vec != 0x0000;
    }
    inline bool none(const vecb<16>& b){
        return b.vec == 0x0000;
    }
    inline vecb<16> andnot(const vecb<16>& a, const vecb<16>& b){
        return vecb<16>(_kand_mask16(_knot_mask16(a.vec), b.vec));
    }

    // AVX512 float operators

    inline const vecf<16> operator +(const vecf<16>& a, const vecf<16>&b){
        return vecf<16>(_mm512_add_ps(a.V.vec, b.V.vec));
    }
    inline const vecf<16> operator -(const vecf<16>& a, const vecf<16>&b){
        return vecf<16>(_mm512_sub_ps(a.V.vec, b.V.vec));
    }
    inline const vecf<16> operator *(const vecf<16>& a, const vecf<16>&b){
        return vecf<16>(_mm512_mul_ps(a.V.vec, b.V.vec));
    }
    inline const vecf<16> operator /(const vecf<16>& a, const vecf<16>&b){
        return vecf<16>(_mm512_div_ps(a.V.vec, b.V.vec));
    }
    inline const vecb<16> operator >=(const vecf<16>& a, const vecf<16>&b){
        return vecb<16>(_mm512_cmp_ps_mask(a.V.vec, b.V.vec, _CMP_GE_OS));
    }
    inline const vecb<16> operator >(const vecf<16>& a, const vecf<16>&b){
        return vecb<16>(_mm512_cmp_ps_mask(a.V.vec, b.V.vec, _CMP_GT_OS));
    }
    inline const vecb<16> operator <=(const vecf<16>& a, const vecf<16>&b){
        return vecb<16>(_mm512_cmp_ps_mask(a.V.vec, b.V.vec, _CMP_LE_OS));
    }
    inline const vecb<16> operator <(const vecf<16>& a, const vecf<16>&b){
        return vecb<16>(_mm512_cmp_ps_mask(a.V.vec, b.V.vec, _CMP_LT_OS));
    }
    inline const vecf<16> min(const vecf<16>& a, const vecf<16>&b){
        return vecf<16>(_mm512_min_ps(a.V.vec, b.V.vec));
    }
    inline const vecf<16> max(const vecf<16>& a, const vecf<16>&b){
        return vecf<16>(_mm512_max_ps(a.V.vec, b.V.vec));
    }
    inline const vecf<16> sqr(const vecf<16>& a){
        return vecf<16>(_mm512_mul_ps(a.V.vec, a.V.vec));
    }
    inline const vecf<16> sqrt(const vecf<16>& a){
        return vecf<16>(_mm512_sqrt_ps(a.V.vec));
    }
    inline const vecf<16> select(const vecb<16>& mask, const vecf<16>& t, const vecf<16>& f){
        return vecf<16>(_mm512_mask_blend_ps(mask.vec, f.V.vec, t.V.vec));
    }

    template <>
    inline const vecf<16> vecZero(){
        return vecf<16>(avx512Zero);
    }
    template <>
    inline const vecf<16> vecOne(){
        return vecf<16>(avx512One);
    }

    template <>
    inline const vecf<16> vecInit(const float f){
        return vecf<16>(_mm512_set1_ps(f));
    }

    // typedefs to contain embree vectors

    template <int W>
	using simdBool  = vecb<W>;
    template <int W>
    using simdFloat = vecf<W>;

    template <int W>
    using simdBoolVec   = embree::Vec3<vecb<W>>;
    template <int W>
    using simdFloatVec  = embree::Vec3<vecf<W>>;

    // typedefs encoding geometric quantities (3d)

    template <int W>
    using simdTriangle_type = std::array<simdFloatVec<W>, 3>;
    template <int W>
    using simdLine_type     = std::array<simdFloatVec<W>, 2>;
    template <int W>
    using simdPoint_type    = simdFloatVec<W>;
    template <int W>
    using simdBox_type      = std::array<simdFloatVec<W>, 2>;

    // returns squared length of vector
    template <int W>
    inline simdFloat<W> length2(const embree::Vec3<simdFloat<W>>& a){
        return vecf<W>(embree::dot(a, a));
    }

    // returns zero vector
    template <int W>
    inline const simdFloatVec<W> zeroVector(){
        return embree::Vec3<simdFloat<W>>(vecZero<W>(), vecZero<W>(), vecZero<W>());
    }

    // given some simd boolean vector and two simd float 3d vectors, merges float vectors into one depending on bool vector
    template <int W>
    __forceinline const simdFloatVec<W> select(const simdBool<W>& s, const simdFloatVec<W>& t, const simdFloatVec<W>& f) {
		return simdFloatVec<W>(select(s, t.x, f.x), select(s, t.y, f.y), select(s, t.z, f.z));
	}

    template <int W>
	__forceinline const simdLine_type<W> select(const simdBool<W>& s, const simdLine_type<W>& t, const simdLine_type<W>& f) {
		const simdFloatVec<W> start(select(s, t[0].x, f[0].x), select(s, t[0].y, f[0].y), select(s, t[0].z, f[0].z));
		const simdFloatVec<W> end(select(s, t[1].x, f[1].x), select(s, t[1].y, f[1].y), select(s, t[1].z, f[1].z));
		const simdLine_type<W> result = { start, end };
		return result;
	}

}// namespace fcpw