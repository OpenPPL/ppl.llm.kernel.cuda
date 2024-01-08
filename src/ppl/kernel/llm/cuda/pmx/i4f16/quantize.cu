// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except input compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to input writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "ppl/kernel/llm/cuda/pmx/i4f16/quantize.h"

#include "ppl/common/log.h"
#include "cudakernel/common/common.cuh"

#include "../quant_utils.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx { namespace i4f16 {

// thread_block must be 32
template<int32_t GROUP_SIZE>
__global__
void minmax_quantize_fp16_kernel(
    const fp16_t* input, // [N, K], fp16
    const int64_t N,
    const int64_t K,
    uint16_t *output, // [N/4, K], int4x4
    fp16_t* scale // [K/group_size, N], fp16
) {
    constexpr int32_t WARP_SIZE = 32;
    constexpr int32_t PACK_SIZE = 4;
    constexpr int32_t TILE_K = WARP_SIZE;

    constexpr int32_t KPT = GROUP_SIZE / TILE_K;
    constexpr int32_t MAX_VPT16 = 8;
    static_assert(KPT <= MAX_VPT16);

    fp16_t local_x[PACK_SIZE * KPT];
    fp16_t local_s[PACK_SIZE];
    uint16_t local_q[KPT];

    const int64_t n4_idx = blockIdx.x;
    const int64_t k_idx = (blockIdx.y * TILE_K + threadIdx.x) * KPT;

    #pragma unroll
    for (int32_t i = 0; i < PACK_SIZE; i++) {
        copy<sizeof(local_x) / PACK_SIZE>(
            &input[(n4_idx * PACK_SIZE + i) * K + k_idx],
            &local_x[i * KPT]);
    }

    #pragma unroll
    for (int32_t j = 0; j < KPT; j++) {
        local_q[j] = 0;
    }

    #pragma unroll
    for (int32_t i = 0; i < PACK_SIZE; i++) {
        fp32_t local_max = 0.0f;

        #pragma unroll
        for (int32_t j = 0; j < KPT; j++) {
            auto fp32_absx = abs(__half2float(local_x[i * KPT + j]));
            local_max = max(fp32_absx, local_max);
        }

        // reduction inside a thread warp
        #pragma unroll
        for (int32_t mask = TILE_K / 2; mask >= 1; mask /= 2) {
            local_max = max(__shfl_xor_sync(uint32_t(-1), local_max, mask), local_max);
        }
        // broadcast to threads
        local_max = __shfl_sync(uint32_t(-1), local_max, 0);

        fp32_t scale_val = min_max_range_to_scale(local_max, QTHRESHOLD, INT4_QLEVEL);
        local_s[i] = __float2half(scale_val);

        // quant and pack int4 to int4x4
        #pragma unroll
        for (int32_t j = 0; j < KPT; j++) {
            int32_t temp = quant_scalar<fp32_t, fp32_t, false>(
                __half2float(local_x[i * KPT + j]), scale_val, INT4_QMIN, INT4_QMAX);
            local_q[j] += uint16_t(temp - INT4_QMIN) << (i * 4);
        }
    }

    copy<sizeof(local_q)>(local_q, &output[n4_idx * K + k_idx]);
    if (threadIdx.x == 0) {
        copy<sizeof(local_s)>(
            local_s, &scale[(k_idx / GROUP_SIZE) * N + n4_idx * PACK_SIZE]);
    }
}

ppl::common::RetCode minmax_quantize_fp16(
    cudaStream_t stream,
    const void* input, // [N, K], fp16
    const int64_t N,
    const int64_t K, // must aligned 128 now
    const int64_t group_size,
    void* quantized, // [N/4, K], int4x4
    void* scale // [K/group_size, N], fp16
)
{
    constexpr int32_t WARP_SIZE = 32;
    constexpr int32_t PACK_SIZE = 4;
    constexpr int32_t TILE_K = WARP_SIZE;

    switch(group_size){
        case 128: {
            const int32_t KPT = 128 / TILE_K;
            const dim3 launch_block = {
                (unsigned int)((N + PACK_SIZE - 1) / PACK_SIZE),
                (unsigned int)((K / KPT + TILE_K - 1) / TILE_K),
                1};
            minmax_quantize_fp16_kernel<128><<<launch_block, TILE_K, 0, stream>>>(
                (fp16_t*)(input), N, K, (uint16_t*)(quantized), (fp16_t*)(scale)
            );
            break;
        }
        default:
            LOG(ERROR) << "only support group_size == 128";
            return ppl::common::RC_UNSUPPORTED;
    }
    return ppl::common::RC_SUCCESS;
}

__device__ inline
void unpack_int4x4_to_fp16x4(const uint16_t in, fp16_t* out) {
    uint32_t *_out = reinterpret_cast<uint32_t*>(out);
    uint32_t a = in & 0x00007777;
    uint32_t b = (in & 0x00008888) >> 3;

    uint32_t c, d;
    static constexpr uint32_t LUT_0 = 0x07060504;
    static constexpr uint32_t LUT_1 = 0x03020100;
    static constexpr uint32_t LUT_2 = 0x00000800;
    static constexpr uint32_t start_byte_for_fp16 = 0x64006400;
    asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(c) : "n"(LUT_1), "n"(LUT_0), "r"(a));
    asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(d) : "n"(LUT_2), "n"(LUT_1), "r"(b));
    c += d;

    static constexpr uint32_t mask_for_elt_01 = 0x5150;
    static constexpr uint32_t mask_for_elt_23 = 0x5352;

    asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(_out[0]) : "r"(c), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
    asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(_out[1]) : "r"(c), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));

    static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64086408;
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(_out[0]) : "r"(_out[0]), "r"(I8s_TO_F16s_MAGIC_NUM));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(_out[1]) : "r"(_out[1]), "r"(I8s_TO_F16s_MAGIC_NUM));
}

template<int32_t TILE_N, int32_t TILE_K, int32_t VPT, int32_t GROUP_SIZE>
__global__
void minmax_dequantize_fp16_kernel(
    const uint16_t* input, // [N/4, K], int4x4
    const fp16_t* scale, // [K/group_size, N], fp16
    const int64_t N,
    const int64_t K,
    fp16_t *output // [N, K], fp16
) {
    constexpr int32_t PACK_SIZE = 4;
    uint16_t local_x[VPT]; fp16_t local_y[PACK_SIZE * VPT]; // [N4, KV]
    fp16_t local_s[PACK_SIZE]; fp16_t decoded[PACK_SIZE];

    const int64_t n4_idx = blockIdx.x * TILE_N + threadIdx.x;
    const int64_t k_idx = (blockIdx.y * TILE_K + threadIdx.y) * VPT;

    copy<sizeof(local_x)>(&input[n4_idx * K + k_idx], local_x);
    copy<sizeof(local_s)>(&scale[(k_idx / GROUP_SIZE) * N + n4_idx * PACK_SIZE], local_s);

    #pragma unroll
    for (auto i = 0; i < VPT; ++i) {
        unpack_int4x4_to_fp16x4(local_x[i], decoded);
        # pragma unroll
        for(auto j = 0; j < PACK_SIZE; j++){
            local_y[j * VPT + i] = decoded[j] * local_s[j];
        }
    }

    #pragma unroll
    for (auto i = 0; i < PACK_SIZE; i++) {
        copy<sizeof(local_y) / PACK_SIZE>(
            &local_y[i * VPT],
            &output[(n4_idx * PACK_SIZE + i) * K + k_idx]);
    }
}

ppl::common::RetCode minmax_dequantize_fp16(
    cudaStream_t stream,
    const void* input, // [N/4, K], int4x4
    const void* scale, // [K/group_size, N], fp16
    const int64_t N,
    const int64_t K, // must aligned 128 now
    const int64_t group_size,
    void* output // [N, K], fp16
) {
    constexpr int32_t TILE_N = 4;
    constexpr int32_t TILE_K = 32;
    constexpr int32_t VPT = 4;
    constexpr int32_t PACK_SIZE = 4;

    const dim3 launch_block = {
        (unsigned int)((N / PACK_SIZE + TILE_N - 1) / TILE_N),
        (unsigned int)((K / VPT + TILE_K - 1) / TILE_K),
        1};
    const dim3 thread_block = {TILE_N, TILE_K, 1};

    switch (group_size) {
        case 128:
            minmax_dequantize_fp16_kernel<TILE_N, TILE_K, VPT, 128>
            <<<launch_block, thread_block, 0, stream>>>(
                (uint16_t*)input, (fp16_t*)scale, N, K, (fp16_t*)output
            );
            break;
        default:
            LOG(ERROR) << "only support group_size == 128";
            return ppl::common::RC_UNSUPPORTED;
    }

    return ppl::common::RC_SUCCESS;
}

}}}}}}

