// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#pragma once

#include <stdint.h>

#include <cuda_runtime.h>

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx { namespace i4f16 {

#if defined(__CUDA_ARCH__) && !defined(__INTELLISENSE__)
#if defined(__CUDACC_RTC__) || (defined(__clang__) && defined(__CUDA__))
#define PRAGMA_UNROLL _Pragma("unroll")
#define PRAGMA_NO_UNROLL _Pragma("unroll 1")
#else
#define PRAGMA_UNROLL #pragma unroll
#define PRAGMA_NO_UNROLL #pragma unroll 1
#endif
#else
#define PRAGMA_UNROLL
#define PRAGMA_NO_UNROLL
#endif

__inline__ __device__ uint32_t smem_cast(void const* ptr) {
    uint32_t smem_int_ptr;

    asm("{.reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, "
        "smem_ptr; }\n"
        : "=r"(smem_int_ptr)
        : "l"(ptr));

    return smem_int_ptr;
}

__inline__ __device__ void load_cache(uint& d0, uint& d1, uint& d2, uint& d3, uint32_t smem_int_ptr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                 : "r"(smem_int_ptr));
}

__inline__ __device__ void load_cache(uint& d0, uint& d1, uint32_t smem_int_ptr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(d0), "=r"(d1) : "r"(smem_int_ptr));
}

template <typename T, int N>
struct Array {
    T a[N];

    __device__ __host__ constexpr T& operator[](int i) noexcept {
        return a[i];
    }
    __device__ __host__ constexpr const T& operator[](int i) const noexcept {
        return a[i];
    }
};

template <typename T>
__inline__ __device__ void cp_async_a(uint32_t smem_int_ptr, const T* __restrict__ src, bool mask) {
    constexpr int cp_size = sizeof(T);
    static_assert(cp_size == 16, "cp.async.cg requreis cp_size == 16");
    asm volatile("{\n"
                 "  .reg .pred p;\n"
                 "  setp.ne.b32 p, %0, 0;\n"
                 "  @p cp.async.ca.shared.global [%1], [%2], %3;\n"
                 "}\n" ::"r"((int)mask),
                 "r"(smem_int_ptr), "l"(src), "n"(cp_size));
}

template <typename T>
__inline__ __device__ void cp_async_b(uint32_t smem_int_ptr, const T* __restrict__ src, bool mask) {
    constexpr int cp_size = sizeof(T);
    static_assert(cp_size == 16, "cp.async.cg requreis cp_size == 16");
    asm volatile("{\n"
                 "  .reg .pred p;\n"
                 "  setp.ne.b32 p, %0, 0;\n"
                 //  "  @p cp.async.cg.shared.global.L2::128B [%1], [%2], %3;\n"  // origin
                 "  @p cp.async.cg.shared.global.L2::256B [%1], [%2], %3;\n"
                 //  "  @p cp.async.ca.shared.global [%1], [%2], %3;\n"
                 "}\n" ::"r"((int)mask),
                 "r"(smem_int_ptr), "l"(src), "n"(cp_size));
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

template <>
__device__ __forceinline__ void cp_async_wait<0>() {
    asm volatile("cp.async.wait_all;\n" ::);
}

__device__ __forceinline__ void cp_async_fence() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void dequantilize(uint32_t value, uint32_t (&result)[4]) {
    uint32_t* h = reinterpret_cast<uint32_t*>(result);
    uint32_t const i4s = reinterpret_cast<uint32_t const&>(value);

    // First, we extract the i4s and construct an intermediate fp16 number.
    static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint32_t BOTTOM_MASK = 0x000f000f;
    static constexpr uint32_t TOP_MASK = 0x00f000f0;
    static constexpr uint32_t I4s_TO_F16s_MAGIC_NUM = 0x64006400;

    // Note that the entire sequence only requires 1 shift instruction. This is
    // thanks to the register packing format and the fact that we force our
    // integers to be unsigned, and account for this in the fp16 subtractions. In
    // addition, I exploit the fact that sub and fma have the same throughput in
    // order to convert elt_23 and elt_67 to fp16 without having to shift them to
    // the bottom bits before hand.

    // Shift right by 8 to now consider elt_45 and elt_67. Issue first to hide RAW
    // dependency if we issue immediately before required.
    const uint32_t top_i4s = i4s >> 8;
    // Extract elt_01 - (i4s & 0x000f000f) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[0])
                 : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    // Extract elt_23 (i4s & 0x00f000f0) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[1])
                 : "r"(i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    // Extract elt_45 (top_i4s & 0x000f000f) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[2])
                 : "r"(top_i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    // Extract elt_67 (top_i4s & 0x00f000f0) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[3])
                 : "r"(top_i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));

    // I use inline PTX below because I am not sure if the compiler will emit
    // float2half instructions if I use the half2 ctor. In this case, I chose
    // performance reliability over code readability.

    // This is the half2 {1032, 1032} represented as an integer.
    static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64086408;
    // This is the half2 {1 / 16, 1 / 16} represented as an integer.
    static constexpr uint32_t ONE_SIXTEENTH = 0x2c002c00;
    // This is the half2 {-72, -72} represented as an integer.
    static constexpr uint32_t NEG_72 = 0xd480d480;

    // Finally, we construct the output numbers.
    // Convert elt_01
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));
    // Convert elt_23
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[1]) : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_72));
    // Convert elt_45
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[2]) : "r"(h[2]), "r"(FP16_TOP_MAGIC_NUM));
    // Convert elt_67
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[3]) : "r"(h[3]), "r"(ONE_SIXTEENTH), "r"(NEG_72));
}

__device__ __forceinline__ uint32_t scale(uint32_t value, uint32_t scale) {
    uint32_t result = 0;
    asm volatile("mul.f16x2 %0, %1, %2;\n" : "=r"(result) : "r"(value), "r"(scale));
    return result;
}

__device__ __forceinline__ void mma_s16816_half(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                                uint32_t b1, float& c0, float& c1, float& c2, float& c3) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                 "{%0, %1, %2, %3},"
                 "{%4, %5, %6, %7},"
                 "{%8, %9},"
                 "{%0, %1, %2, %3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void load(uint32_t& reg0, uint32_t& reg1, uint32_t& reg2, uint32_t& reg3,
                                     const uint32_t& addr) {
    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(reg0), "=r"(reg1), "=r"(reg2), "=r"(reg3)
                 : "r"(addr));
}

__device__ __forceinline__
void loadg(uint32_t &reg0, uint32_t &reg1,
            uint32_t &reg2, uint32_t &reg3,
            const void *addr, bool pred_guard) {
    asm volatile (
        "{\n"
        "   .reg .pred p;\n"
        "   setp.ne.b32 p, %5, 0;\n"
        "   @p ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n"
        "   @!p mov.u32 %0, %6;\n"
        "   @!p mov.u32 %1, %6;\n"
        "   @!p mov.u32 %2, %6;\n"
        "   @!p mov.u32 %3, %6;\n"
        "}\n"
        : "=r"(reg0), "=r"(reg1), "=r"(reg2), "=r"(reg3)
        : "l"(addr), "r"((int)pred_guard), "n"(0)
    );
}


template <typename AccessType>
struct global_store {
    __inline__ __device__ global_store(AccessType const& D, void* ptr, bool pred_guard) {
        uint4 const& data = reinterpret_cast<uint4 const&>(D);
        asm volatile("{\n"
                     "  .reg .pred p;\n"
                     "  setp.ne.b32 p, %5, 0;\n"
                     "  @p st.global.weak.v4.u32 [%0], {%1, %2, %3, %4};\n"
                     "}\n"
                     :
                     : "l"(ptr), "r"(data.x), "r"(data.y), "r"(data.z), "r"(data.w), "r"((int)pred_guard));
    }
};

__device__ __forceinline__ Array<uint32_t, 4> scale(const uint32_t (&value)[4], Array<half, 4> scale) {
    Array<uint32_t, 4> result;
    const uint32_t* h = reinterpret_cast<const uint32_t*>(value);
    uint32_t* r = reinterpret_cast<uint32_t*>(&result);
    uint16_t* scale_u16 = reinterpret_cast<uint16_t*>(&scale);
    uint32_t scale_batch[4];
    asm volatile("mov.b32 %0, {%4, %4};\n"
                 "mov.b32 %1, {%5, %5};\n"
                 "mov.b32 %2, {%6, %6};\n"
                 "mov.b32 %3, {%7, %7};\n"
                 : "=r"(scale_batch[0]), "=r"(scale_batch[1]), "=r"(scale_batch[2]), "=r"(scale_batch[3])
                 : "h"(scale_u16[0]), "h"(scale_u16[1]), "h"(scale_u16[2]), "h"(scale_u16[3]));
    // Convert elt_01
    asm volatile("mul.f16x2 %0, %1, %2;\n" : "=r"(r[0]) : "r"(h[0]), "r"(scale_batch[0]));
    // Convert elt_23
    asm volatile("mul.f16x2 %0, %1, %2;\n" : "=r"(r[1]) : "r"(h[1]), "r"(scale_batch[1]));
    // Convert elt_45
    asm volatile("mul.f16x2 %0, %1, %2;\n" : "=r"(r[2]) : "r"(h[2]), "r"(scale_batch[2]));
    // Convert elt_67
    asm volatile("mul.f16x2 %0, %1, %2;\n" : "=r"(r[3]) : "r"(h[3]), "r"(scale_batch[3]));

    return result;
}

__device__ constexpr int const_max(int a, int b) {
    return (b > a ? b : a);
}

__device__ constexpr int const_min(int a, int b) {
    return (b < a ? b : a);
}

template <int Contiguous, int Strided>
struct LinearShape {
    static int constexpr kContiguous = Contiguous;
    static int constexpr kStrided = Strided;
    static int constexpr kCount = Contiguous * Strided;
};

}}}}}} // namespace ppl::kernel::llm::cuda::pmx::i4f16
