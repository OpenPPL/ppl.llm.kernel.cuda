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

#include "ppl/kernel/llm/cuda/pmx/multi_head_cache_attention.h"
#include "ppl/common/log.h"

#include "ppl/kernel/llm/cuda/xformer/fmha.h"
#include "cudakernel/common/common.cuh"

#include <cuda_fp16.h>
#include <float.h> // need for FLT_MAX

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

struct dynamic_batching_decoding_cache_attention_kernel_param {
    half* query;
    half* output;
    int8_t* cache;
    half* scale;
    int64_t* cachestarts;
    int64_t* kvstarts;
    float attn_scale;
    int64_t layer_idx;
    int64_t kv_head_shift;       // !!! Use this if (num_heads/num_kv_heads) is power of 2  or zero, otherwise set SHIFT_KV to false.
    int64_t num_kv_repeats;       // And then we will use this one to compute kv_head_idx, but the performance will lost 10%
    int64_t query_stride_s;
    int64_t output_stride_s;
    int64_t cache_stride_s;
    int64_t cache_stride_l;
    int64_t cache_stride_h;
    int64_t cache_stride_kv;

    struct {
        int32_t* block_counter;
        float* partial_max;
        float* partial_sum;
        half* partial_out;
    } multi_block;
};

struct dynamic_batching_kv_cache_quantize_kernel_param {
    half* current_key; // (S, KVH..., D)
    half* current_value; // (S, KVH..., D)
    int64_t* seqstarts; // (B + 1)
    int64_t* cachestarts;// (B)
    int64_t* start_pos; // (B)
    int64_t num_layer;
    int64_t layer_idx;
    int64_t num_kv_heads;
    int64_t current_key_stride_s;
    int64_t current_value_stride_s;
    int64_t cache_stride_s;
    int64_t cache_stride_l;
    int64_t cache_stride_h;
    int64_t cache_stride_kv;
    int8_t* cache;
    half* scale;
};

template<int32_t HEAD_SIZE, int32_t VPT, int32_t TPB> // 8 fp16 occupy 128 bytes, which can be loaded by a single thread at once.
__global__
void dynamic_batching_kv_cache_quantize_kernel(dynamic_batching_kv_cache_quantize_kernel_param p)
{
    if (blockIdx.y < p.seqstarts[blockIdx.x + 1] - p.seqstarts[blockIdx.x]) {
        constexpr int64_t thd_per_head_size = HEAD_SIZE / VPT;
        const int64_t batch_id = blockIdx.x;
        const int64_t seq_idx = blockIdx.y;
        const int64_t tid = blockIdx.z * TPB + threadIdx.x;

        if (tid < HEAD_SIZE * p.num_kv_heads / VPT) {
            const int64_t input_token_idx = p.seqstarts[batch_id] + seq_idx;
            const int64_t cache_token_idx = p.cachestarts[batch_id] + seq_idx + p.start_pos[batch_id];
            const int64_t key_out_offset = cache_token_idx * p.cache_stride_s + p.layer_idx * p.cache_stride_l;
            auto key_in_ptr = p.current_key + input_token_idx * p.current_key_stride_s;
            auto value_in_ptr = p.current_value + input_token_idx * p.current_value_stride_s;

            const int64_t kv_head_idx = tid / thd_per_head_size;
            const int64_t dim_idx = (tid % thd_per_head_size) * VPT;
            const int64_t scale_dim_idx = dim_idx / VPT;
            const int64_t input_idx = kv_head_idx * HEAD_SIZE + dim_idx;

            half key_in[VPT]; int8_t key_out[VPT];
            half value_in[VPT]; int8_t value_out[VPT];

            copy<sizeof(half) * VPT>(&key_in_ptr[input_idx], key_in);
            copy<sizeof(half) * VPT>(&value_in_ptr[input_idx], value_in);

            const int64_t key_out_idx
                = key_out_offset
                + kv_head_idx * p.cache_stride_h
                + dim_idx;
            const int64_t value_out_idx
                = key_out_idx
                + p.cache_stride_kv;

            const int64_t key_scale_out_idx = (key_out_idx - dim_idx) / VPT + scale_dim_idx;
            const int64_t value_scale_out_idx = key_scale_out_idx + p.cache_stride_kv / VPT;

            // calculate kv scale
            const half eps = 1e-5f;
            half key_max = 0.0f;
            half value_max = 0.0f;

            #pragma unroll
            for (int32_t i = 0; i < VPT; i ++){
                key_max = key_max > __habs(key_in[i]) ? key_max : __habs(key_in[i]);
                value_max = value_max > __habs(value_in[i]) ? value_max : __habs(value_in[i]);
            }

            half key_scale = __float2half(__half2float(key_max) / 127.0f);
            half value_scale = __float2half(__half2float(value_max) / 127.0f);
            key_scale = key_scale > eps ? key_scale : eps;
            value_scale = value_scale > eps ? value_scale : eps;

            #pragma unroll
            for (int32_t i = 0; i < VPT; i ++){
                key_out[i] = (int8_t)__half2short_rn(key_in[i] / key_scale);
                value_out[i] = (int8_t)__half2short_rn(value_in[i] / value_scale);
            }

            copy<sizeof(int8_t) * VPT>(key_out, &p.cache[key_out_idx]);
            copy<sizeof(int8_t) * VPT>(value_out, &p.cache[value_out_idx]);

            p.scale[key_scale_out_idx] = key_scale;
            p.scale[value_scale_out_idx] = value_scale;
        }
    }
}

template<int32_t THREAD_GROUP_SIZE, int32_t ELEMENT_NUM>
__device__ inline
float attn_thread_group_dot(half* local_q, half* local_k)
{
    // Helper function for QK Dot.
    // [TODO] It should be optimized by type fp32x4.

    float qk = 0.0f;
# pragma unroll
    for(int32_t i = 0; i < ELEMENT_NUM; i++) {
        qk += __half2float(local_q[i]) * __half2float(local_k[i]);
    }
#pragma unroll
    for (int32_t mask = THREAD_GROUP_SIZE / 2; mask >= 1; mask /= 2) {
        qk += __shfl_xor_sync(uint32_t(-1), qk, mask);
    }
    return qk;
}

template<int32_t THREAD_GROUP_SIZE>
__device__ inline
float attn_thread_group_reduce_sum(float qk)
{
#pragma unroll
    for (int32_t mask = THREAD_GROUP_SIZE / 2; mask >= 1; mask /= 2) {
        qk += __shfl_xor_sync(uint32_t(-1), qk, mask);
    }
    return qk;
}

template<int32_t WPT>
__device__ inline
float attn_block_reduce_max(float reducing, float* shared_mem)
{
    // Helper function for reduce softmax qkmax.
    constexpr int32_t WARP_SIZE = 32;
    const int32_t lane_id = threadIdx.x % WARP_SIZE;
    const int32_t warp_id = threadIdx.x / WARP_SIZE;

# pragma unroll
    for (int32_t mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
        reducing = fmaxf(reducing, __shfl_xor_sync(uint32_t(-1), reducing, mask));
    }

    if (lane_id == 0) {
        shared_mem[warp_id] = reducing;
    }
    __syncthreads();

    if (lane_id < WPT) reducing = shared_mem[lane_id];
    else reducing = -FLT_MAX;

# pragma unroll
    for (int32_t mask = WPT / 2; mask >= 1; mask /= 2) {
        reducing = fmaxf(reducing, __shfl_xor_sync(uint32_t(-1), reducing, mask));
    }

    reducing = __shfl_sync(uint32_t(-1), reducing, 0);
    return reducing;
}

template<int32_t WPT>
__device__ inline
float attn_block_reduce_sum(float reducing, float *shared_mem)
{
    // Helper function for reduce softmax exp sum.
    constexpr int32_t WARP_SIZE = 32;
    const int32_t lane_id = threadIdx.x % WARP_SIZE;
    const int32_t warp_id = threadIdx.x / WARP_SIZE;

# pragma unroll
    for (int32_t mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
        reducing += __shfl_xor_sync(uint32_t(-1), reducing, mask);
    }

    if (lane_id == 0) shared_mem[warp_id] = reducing;
    __syncthreads();

    if (lane_id < WPT) reducing = shared_mem[lane_id];

# pragma unroll
    for (int32_t mask = WPT / 2; mask >= 1; mask /= 2) {
        reducing += __shfl_xor_sync(uint32_t(-1), reducing, mask);
    }
    reducing = __shfl_sync(uint32_t(-1), reducing, 0);
    return reducing;
}


template<
    int32_t HEAD_SIZE,
    int32_t THREAD_GROUP_SIZE,        // how many threads inside a group
    int32_t TPB,
    int32_t QUANT_GROUP,
    int32_t MULTI_BLOCK,    // do flash decoding if more than 1
    bool SHIFT_KV>
__global__
void dynamic_batching_decoding_cache_attention_fp16_kernel(dynamic_batching_decoding_cache_attention_kernel_param p)
{
    /***
    * You have to remember that this Kernel was created by a brother on the night of July 20, 2023. On that day,
    * Beijing experienced the strongest rainstorm since the beginning of summer.

    DecodingAttention is a special operator designed specifically for large language models(LLM) decoding.

    It requires that the length of each input Query is always 1,
        while the Key and Value can have different lengths.

    This operator supports padding removal optimization, meaning that Q, K, and V all need to have their tokens
        concentrated in one sentence for input, with shapes like Q: [seq_lens, num_heads, head_size],
        and K: [context_lens, num_kv_heads, head_size].

    Since the Query sentence length is always 1, this operator is literally a fused matrix-vector multiplications operation.
        It does not utilize tensor cores for computation.

    The calculation logic is divided into three steps: gemv(QK) + softmax(Attention) + gemv(KV).
        In the provided code, it has already been split into these three parts.
    ***/

    /* --- Decoding Attention Kernel Implementation --- */
    static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480;
    static constexpr uint32_t mask_for_elt_01     = 0x5150;
    static constexpr uint32_t mask_for_elt_23     = 0x5352;
    static constexpr uint32_t start_byte_for_fp16 = 0x64646464;

    constexpr int64_t WARP_SIZE = 32;                              // warp size
    constexpr int64_t WPT       = TPB / WARP_SIZE;                 // warp per thread block
    constexpr int64_t GPW       = WARP_SIZE / THREAD_GROUP_SIZE;       // thread group per warp
    constexpr int64_t GPT       = WARP_SIZE / THREAD_GROUP_SIZE * WPT; // thread group per thread block

    // const int64_t num_heads     = gridDim.x;
    const int64_t batch_size    = gridDim.y;
    const int64_t head_idx      = blockIdx.x;
    const int64_t batch_idx     = blockIdx.y;
    const int64_t block_idx     = blockIdx.z;
    constexpr int64_t VEC_SIZE  = 16 / sizeof(half);  // 128 bits

    // ------------------------------------------------ //
    // Step 1. Load Q into Thread Reg.
    constexpr int64_t VEC_LEN = (HEAD_SIZE / VEC_SIZE) / THREAD_GROUP_SIZE;

    static_assert((HEAD_SIZE / THREAD_GROUP_SIZE) % VEC_SIZE == 0);
    static_assert(HEAD_SIZE % THREAD_GROUP_SIZE == 0);
    static_assert(QUANT_GROUP == 8);

    constexpr int64_t QUANT_GROUP_SHIFT = 3;

    // The elements in Q, K, and V will be evenly distributed across each thread group.
    half local_q[VEC_SIZE * VEC_LEN];

    const int64_t warp_id       = threadIdx.x / WARP_SIZE;
    const int64_t warp_lane_id  = threadIdx.x % WARP_SIZE;
    const int64_t group_id      = warp_lane_id / THREAD_GROUP_SIZE;
    const int64_t group_lane_id = warp_lane_id % THREAD_GROUP_SIZE;

    const int64_t cache_offset_s  = p.cachestarts[batch_idx];
    const int64_t kv_head_idx     = SHIFT_KV
                                    ? (head_idx >> p.kv_head_shift)
                                    : (head_idx / p.num_kv_repeats);

    #pragma unroll
    for (int64_t i = 0; i < VEC_LEN; i++) {
        // copy 128(16 * 8) bits from Q to Local Q

        copy<sizeof(half) * VEC_SIZE>(
            &p.query[
                batch_idx * p.query_stride_s +
                head_idx * HEAD_SIZE +
                (group_lane_id + i * THREAD_GROUP_SIZE) * VEC_SIZE
            ],
            &local_q[i * VEC_SIZE]);
    }
    // ------------------------------------------------ //
    // Step 2. Solve QK Dot

    // In the process of handling the QK matrix multiplication, we will divide a complete Thread Warp into several Thread groups.
    // Each thread group reads the entire Query and saves it in registers.
    // Then, each thread group iterates through the vectors in the Key and performs dot products with the Query.
    // During this process, a WARP performs multiple vector dot product operations at once.
    // At the same time, we also record the maximum current_value of the dot product results for later use in the softmax operation.
    const int64_t context_len           = p.kvstarts[batch_idx + 1] - p.kvstarts[batch_idx];
    const int64_t context_len_per_block = (context_len + MULTI_BLOCK - 1) / MULTI_BLOCK;
    const int64_t block_base_id         = block_idx * context_len_per_block;
    const int64_t block_context_len     = context_len >= context_len_per_block * (block_idx + 1) ? context_len_per_block : context_len - block_base_id;

    extern __shared__ float logits[];
    float partial_qk_max = -FLT_MAX;

    for (int64_t base_id = warp_id * GPW; base_id < block_context_len; base_id += GPT) {
        int8_t local_k_quant[VEC_SIZE * VEC_LEN];
        half local_k_scale[VEC_LEN];
        const int64_t block_context_id = base_id + group_id;

        float qk_dot = 0.0f;

        // all thread groups within a warp must be launched together.
        if (block_context_id < block_context_len) {
            const int64_t key_offset
                            = (cache_offset_s + block_base_id + block_context_id) * p.cache_stride_s
                            + p.layer_idx * p.cache_stride_l
                            + kv_head_idx * p.cache_stride_h
                            + group_lane_id * VEC_SIZE;
            #pragma unroll
            for (int64_t i = 0; i < VEC_LEN; i++) {
                // copy 128(16 * 8) bits from K to Local K
                const int64_t key_idx = key_offset + i * THREAD_GROUP_SIZE * VEC_SIZE;
                copy<sizeof(int8_t) * VEC_SIZE>(&p.cache[key_idx],  &local_k_quant[i * VEC_SIZE]);
                const int64_t key_scale_idx = key_idx >> QUANT_GROUP_SHIFT;
                local_k_scale[i] = p.scale[key_scale_idx];

                #pragma unroll
                for(int64_t k = 0; k < VEC_SIZE; k++) {
                    local_k_quant[i * VEC_SIZE + k] += 128;
                }
                half result[8];
                uint32_t*      h   = reinterpret_cast<uint32_t*>(result);
                uint32_t const i8s = reinterpret_cast<uint32_t const&>(*(local_k_quant + i * VEC_SIZE));
                asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h[0]) : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
                asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h[1]) : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));
                asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(I8s_TO_F16s_MAGIC_NUM));
                asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[1]) : "r"(h[1]), "r"(I8s_TO_F16s_MAGIC_NUM));
                uint32_t*      h_2   = reinterpret_cast<uint32_t*>(result+4);
                uint32_t const i8s_2 = reinterpret_cast<uint32_t const&>(*(local_k_quant + i * VEC_SIZE + 4));
                asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h_2[0]) : "r"(i8s_2), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
                asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h_2[1]) : "r"(i8s_2), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));
                asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h_2[0]) : "r"(h_2[0]), "r"(I8s_TO_F16s_MAGIC_NUM));
                asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h_2[1]) : "r"(h_2[1]), "r"(I8s_TO_F16s_MAGIC_NUM));
                #pragma unroll
                for (int64_t j = 0; j < VEC_SIZE; j++) {
                    qk_dot += __half2float(local_q[i * VEC_SIZE + j]) * __half2float(local_k_scale[i] * result[j]);
                }
            }
        }

        // Ready for QK Dot
        qk_dot = p.attn_scale * attn_thread_group_reduce_sum<THREAD_GROUP_SIZE>(qk_dot);

        if (group_lane_id == 0 && block_context_id < block_context_len) {
            logits[block_context_id] = qk_dot;
            partial_qk_max = fmaxf(qk_dot, partial_qk_max);
       }
    }

    // ------------------------------------------------ //
    // Step 3. Softmax

    // The process of solving softmax is divided into two stages.
    // First, we need to reduce partial_qk_max in two dimensions: WARP and ThreadBlock.
    // Afterward, we use reduced partial_qk_max to perform softmax calculations,
    //    the results will all be stored in shared memory.
    __shared__ float red_smem[WPT];

    // reduce partial_qk_max in thread block and boardcast
    partial_qk_max = attn_block_reduce_max<WPT>(partial_qk_max, red_smem);

    // Softmax Kernel Logic Start here
    float partial_exp_sum = 0.0f;
    for (int64_t block_context_id = threadIdx.x; block_context_id < block_context_len; block_context_id += TPB){
        logits[block_context_id] -= partial_qk_max;
        logits[block_context_id] = exp(logits[block_context_id]);
        partial_exp_sum += logits[block_context_id];
    }

    // block reduce sum on partial_exp_sum
    // Warp per thread block must be power-of-2 for reducation, check attn_block_reduce_sum kernel.
    static_assert(WPT == 2 || WPT == 4 || WPT == 8 || WPT == 16 || WPT == 32 || WPT == 64);
    partial_exp_sum = attn_block_reduce_sum<WPT>(partial_exp_sum, red_smem);

    if (MULTI_BLOCK == 1) {
        const float inv_sum = __fdividef(1.f, partial_exp_sum + 1e-6f);
        for (int64_t block_context_id = threadIdx.x; block_context_id < block_context_len; block_context_id += TPB) {
            logits[block_context_id] *= inv_sum;
        }
        __syncthreads(); // Must have this.
    }
    else if (threadIdx.x == 0) {
        p.multi_block.partial_max[
            head_idx * batch_size * MULTI_BLOCK +
            batch_idx * MULTI_BLOCK +
            block_idx]
            = partial_qk_max;
        p.multi_block.partial_sum[
            head_idx * batch_size * MULTI_BLOCK +
            batch_idx * MULTI_BLOCK +
            block_idx]
            = partial_exp_sum;
    }

    // ------------------------------------------------ //
    // Step 4. Solve logits * V

    int8_t local_v_quant[VEC_SIZE * VEC_LEN];
    float local_v[VEC_SIZE * VEC_LEN];
    half local_v_scale[VEC_LEN];

    #pragma unroll
    for(int32_t i = 0; i < VEC_SIZE * VEC_LEN; i++) {
        local_v[i] = 0;
    }

    for (int64_t base_id = warp_id * GPW; base_id < block_context_len; base_id += GPT) {
        const int64_t block_context_id = base_id + group_id;
        // all thread groups within a warp must be launched together.
        if (block_context_id < block_context_len){
            const int64_t value_offset
                            = (cache_offset_s + block_base_id + block_context_id) * p.cache_stride_s
                            + p.layer_idx * p.cache_stride_l
                            + kv_head_idx * p.cache_stride_h
                            + group_lane_id * VEC_SIZE
                            + p.cache_stride_kv;
            #pragma unroll
            for (int64_t i = 0; i < VEC_LEN; i++) {
                // copy 128(16 * 8) bits from V to Local V
                const int64_t value_idx = value_offset + i * THREAD_GROUP_SIZE * VEC_SIZE;
                copy<sizeof(int8_t) * VEC_SIZE>(&p.cache[value_idx],  &local_v_quant[i * VEC_SIZE]);
                const int64_t value_scale_idx = value_idx >> QUANT_GROUP_SHIFT;
                local_v_scale[i] = p.scale[value_scale_idx];

                #pragma unroll
                for(int64_t k = 0; k < VEC_SIZE; k++) {
                    local_v_quant[i * VEC_SIZE + k] += 128;
                }
                half result[8];
                uint32_t*      h   = reinterpret_cast<uint32_t*>(result);
                uint32_t const i8s = reinterpret_cast<uint32_t const&>(*(local_v_quant + i * VEC_SIZE));
                asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h[0]) : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
                asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h[1]) : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));
                asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(I8s_TO_F16s_MAGIC_NUM));
                asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[1]) : "r"(h[1]), "r"(I8s_TO_F16s_MAGIC_NUM));
                uint32_t*      h_2   = reinterpret_cast<uint32_t*>(result+4);
                uint32_t const i8s_2 = reinterpret_cast<uint32_t const&>(*(local_v_quant + i * VEC_SIZE + 4));
                asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h_2[0]) : "r"(i8s_2), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
                asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h_2[1]) : "r"(i8s_2), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));
                asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h_2[0]) : "r"(h_2[0]), "r"(I8s_TO_F16s_MAGIC_NUM));
                asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h_2[1]) : "r"(h_2[1]), "r"(I8s_TO_F16s_MAGIC_NUM));
                #pragma unroll
                for (int64_t j = 0; j < VEC_SIZE; j++) {
                    local_v[i * VEC_SIZE + j] += __half2float(local_v_scale[i] * result[j]) * logits[block_context_id];
                }
            }
        }
    }

    #pragma unroll
    for (int32_t i = 0; i < VEC_SIZE * VEC_LEN; i++) {
        #pragma unroll
        for (int32_t mask = THREAD_GROUP_SIZE; mask <= WARP_SIZE >> 1; mask = mask << 1) {
            local_v[i] += __shfl_xor_sync(uint32_t(-1), local_v[i], mask);
        }
    }
    //for now, every warp's each thread group got the partial result inside a warp
    //we need to add up each warp's first thread group by reusing the logits smem

    // wait for logits to be reused
    __syncthreads();

    constexpr int64_t WORK_THREAD = WPT * THREAD_GROUP_SIZE * VEC_LEN;
    constexpr int64_t VPT = 16;
    constexpr int64_t V32PT = 16 / sizeof(float);

    const int32_t v_warp_id  = threadIdx.x % WPT;
    const int32_t v_group_id = (threadIdx.x / WPT) % THREAD_GROUP_SIZE;
    const int32_t v_vec_id   = threadIdx.x / (WPT * THREAD_GROUP_SIZE);

    half local_out[VEC_SIZE];

    // save local_v to shared memory
    if (warp_lane_id < THREAD_GROUP_SIZE) {
        #pragma unroll
        for (int32_t i = 0; i < VEC_LEN * VEC_SIZE; i += V32PT) {
            copy<VPT>(
                &local_v[i],
                &logits[
                    i * WPT * THREAD_GROUP_SIZE +
                    warp_lane_id * WPT * V32PT +
                    ((warp_id + warp_lane_id) % WPT) * V32PT]);
        }
    }

    __syncthreads();

    // WPT reduce
    if (threadIdx.x < WORK_THREAD) {
        #pragma unroll
        for (int32_t i = 0; i < VEC_SIZE; i+= V32PT) {
            copy<VPT>(
                &logits[
                    v_vec_id * VEC_SIZE * WPT * THREAD_GROUP_SIZE +
                    i * WPT * THREAD_GROUP_SIZE +
                    v_group_id * WPT * V32PT +
                    ((v_warp_id + v_group_id) % WPT) * V32PT],
                &local_v[i]);
        }
        #pragma unroll
        for (int32_t i = 0; i < VEC_SIZE; i++) {
            #pragma unroll
            for (int32_t mask = WPT / 2; mask >= 1; mask /= 2) {
                local_v[i] += __shfl_xor_sync(uint32_t(-1), local_v[i], mask);
            }
            local_out[i] = __float2half(local_v[i]);
        }
        if (v_warp_id == 0) {
            half* partial_out = MULTI_BLOCK == 1 ?
            &p.output[
                batch_idx * p.output_stride_s +
                head_idx * HEAD_SIZE +
                v_vec_id * THREAD_GROUP_SIZE * VEC_SIZE +
                v_group_id * VEC_SIZE] :
            &p.multi_block.partial_out[
                head_idx * batch_size * HEAD_SIZE * MULTI_BLOCK +
                batch_idx * HEAD_SIZE * MULTI_BLOCK +
                v_vec_id * THREAD_GROUP_SIZE * MULTI_BLOCK * VEC_SIZE +
                v_group_id * MULTI_BLOCK * VEC_SIZE +
                block_idx * VEC_SIZE];
            copy<VPT>(local_out, partial_out);
        }
    }

    // Flash decoding
    if (MULTI_BLOCK > 1) {
        __syncthreads();

        bool last_block = false;
        // Make sure every block finishs the partial computation.
        if (threadIdx.x == 0)
        {
            if (atomicAdd(&p.multi_block.block_counter[head_idx * batch_size + batch_idx], 1) == MULTI_BLOCK - 1)
            {
                last_block = true;
            }
        }

        // The last block do the final computation.
        if (__syncthreads_or(last_block)) {
            const int64_t multi_block_idx = threadIdx.x % MULTI_BLOCK;
            const int64_t hbb = head_idx * batch_size * MULTI_BLOCK + batch_idx * MULTI_BLOCK + multi_block_idx;

            float final_qk_max = warp_lane_id < MULTI_BLOCK ? p.multi_block.partial_max[hbb] : -FLT_MAX;
            # pragma unroll
            for (int32_t mask = MULTI_BLOCK / 2; mask >= 1; mask /= 2) {
                final_qk_max = fmaxf(final_qk_max, __shfl_xor_sync(uint32_t(-1), final_qk_max, mask));
            }
            final_qk_max = __shfl_sync(uint32_t(-1), final_qk_max, 0);

            float final_exp_sum = warp_lane_id < MULTI_BLOCK ? p.multi_block.partial_sum[hbb] * exp(p.multi_block.partial_max[hbb] - final_qk_max) : 0.0f;
            # pragma unroll
            for (int32_t mask = MULTI_BLOCK / 2; mask >= 1; mask /= 2) {
                final_exp_sum += __shfl_xor_sync(uint32_t(-1), final_exp_sum, mask);
            }
            final_exp_sum = __shfl_sync(uint32_t(-1), final_exp_sum, 0);

            const float final_inv_sum = __fdividef(1.f, final_exp_sum + 1e-6f);

            const int64_t head_dim_idx_base  = (int64_t)(threadIdx.x / MULTI_BLOCK) * VEC_SIZE;
            const int64_t head_dim_idx_stride = TPB / MULTI_BLOCK * VEC_SIZE;

            for (int64_t head_dim_idx = head_dim_idx_base; head_dim_idx < HEAD_SIZE; head_dim_idx += head_dim_idx_stride) {
                half final_out[VEC_SIZE];
                float tmp_max = p.multi_block.partial_max[
                    head_idx * batch_size * MULTI_BLOCK +
                    batch_idx * MULTI_BLOCK +
                    multi_block_idx];
                copy<VEC_SIZE*sizeof(half)>(
                    &p.multi_block.partial_out[
                        head_idx * batch_size * HEAD_SIZE * MULTI_BLOCK +
                        batch_idx * HEAD_SIZE * MULTI_BLOCK +
                        head_dim_idx * MULTI_BLOCK +
                        multi_block_idx * VEC_SIZE],
                    final_out);

                #pragma unroll
                for (int32_t i = 0; i < VEC_SIZE; i++) {
                    float float_out = __half2float(final_out[i]) * exp(tmp_max - final_qk_max) * final_inv_sum;
                    # pragma unroll
                    for (int32_t mask = MULTI_BLOCK / 2; mask >= 1; mask /= 2) {
                        float_out += __shfl_xor_sync(uint32_t(-1), float_out, mask);
                    }
                    final_out[i] = __float2half(float_out);
                }

                if (multi_block_idx == 0) {
                    copy<VPT>(
                        final_out,
                        &p.output[
                            batch_idx * p.output_stride_s +
                            head_idx * HEAD_SIZE +
                            head_dim_idx]);
                }
            }
        }
    }
}



ppl::common::RetCode dynamic_batch_multi_head_cache_attention_prepare(
    const cudaStream_t stream,
    const cudaDeviceProp& device_prop,
    const ppl::common::TensorShape* query_shape,
    const void* query, // (S, ..., D)
    const ppl::common::TensorShape* current_key_shape,
    const void* current_key, // (S, ..., D)
    const ppl::common::TensorShape* current_value_shape,
    const void* current_value, // (S, ..., D)
    const ppl::common::TensorShape* attn_mask_shape,
    const void* attn_mask, // (seqstarts[-1], kvstarts[-1]), (num_heads, seqstarts[-1], kvstarts[-1])
    const void* seqstarts, // (B + 1)
    const void* kvstarts, // (B + 1)
    const void* cachestarts, // (B)
    const void* start_pos, // (B)
    const bool is_causal,
    const int64_t batch,
    const int64_t decoding_batches,
    const int64_t max_seqlen,
    const int64_t max_kvlen,
    const int64_t layer_idx,
    const int64_t num_layer,
    const int64_t num_heads,
    const int64_t num_kv_heads,
    const int64_t head_dim,
    const int32_t cache_mode,
    const int64_t cache_stride_s,
    const int64_t cache_stride_l,
    const int64_t cache_stride_h,
    const int64_t cache_stride_kv,
    void* cache, // int8 (S, L, 2, KVH, D), (L, KVH, S, 2, D)
    void* scale, // float16 (S, L, 2, KVH, D/8), (L, KVH, S, 2, D/8)
    const ppl::common::TensorShape* output_shape,
    void* output, // (S, .., D)
    int64_t* multi_block_buffer_size,
    dynamic_batch_multi_head_cache_attention_config* config)
{
    if (query_shape->GetDim(1) != num_heads) {
        LOG(ERROR) << "currnetly do not support leading dim on query's num_heads";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (output_shape->GetDim(1) != num_heads) {
        LOG(ERROR) << "currnetly do not support leading dim on output's num_heads";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (current_key_shape->GetDim(1) != num_kv_heads) {
        LOG(ERROR)
            << "currnetly do not support leading dim on current_key's num_kv_heads";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (current_value_shape->GetDim(1) != num_kv_heads) {
        LOG(ERROR)
            << "currnetly do not support leading dim on current_value's num_kv_heads";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (attn_mask != nullptr) {
        LOG(ERROR) << "currnetly do not support attn_mask";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (cache_mode != 0) {
        LOG(ERROR) << "currently only support cache_mode == 0";
        return ppl::common::RC_UNSUPPORTED;
    }

    int64_t kv_head_shift = 0;
    int64_t kv_repeats = num_heads / num_kv_heads;
    while (kv_repeats >>= 1)
        ++kv_head_shift;
    if ((num_kv_heads << kv_head_shift) != num_heads) {
        LOG(ERROR) << "currently only support (num_heads/num_kv_heads) is power of 2 or zero.";
        return ppl::common::RC_UNSUPPORTED;
    }

    const int64_t prefill_batches = batch - decoding_batches;
    const float attn_scale = float(1.0 / std::sqrt(float(head_dim)));

    const int64_t q_stride_s = query_shape->GetDim(1) * head_dim;
    const int64_t k_stride_s = current_key_shape->GetDim(1) * head_dim;
    const int64_t v_stride_s = current_value_shape->GetDim(1) * head_dim;
    const int64_t o_stride_s = output_shape->GetDim(1) * head_dim;

    constexpr int64_t TPB = 256;
    constexpr int64_t VPT = 8;

    if (num_heads != num_kv_heads) {
        LOG(ERROR) << "currently do not support GQA, whose num_heads != num_kv_heads";
        return ppl::common::RC_UNSUPPORTED;
    }

    {
        if (head_dim != 128) {
            LOG(ERROR) << "currently kv_cache_quantize_kernel only support head_dim == 128.";
            return ppl::common::RC_UNSUPPORTED;
        }
        dim3 grid(batch, max_seqlen, (num_kv_heads * head_dim / VPT + TPB - 1) / TPB);
        dynamic_batching_kv_cache_quantize_kernel<128, VPT, TPB><<<grid, TPB, 0, stream>>>(
            {(half*)current_key,
            (half*)current_value,
            (int64_t*)seqstarts,
            (int64_t*)cachestarts,
            (int64_t*)start_pos,
            num_layer,
            layer_idx,
            num_kv_heads,
            k_stride_s,
            v_stride_s,
            cache_stride_s,
            cache_stride_l,
            cache_stride_h,
            cache_stride_kv,
            (int8_t*)cache,
            (half*)scale});
    }

    const int64_t kernel_total_blocks = num_heads * decoding_batches;

    // get sm num
    int32_t device_id;
    int32_t multi_processor_count;
    cudaGetDevice(&device_id);
    cudaDeviceGetAttribute(&multi_processor_count, cudaDevAttrMultiProcessorCount, device_id);

    // get multi block size
    int64_t multi_block_size = 1;
    if (decoding_batches > 0 && kernel_total_blocks < multi_processor_count && max_kvlen >= 1024) {
        int64_t tmp_size = TPB / (head_dim / VPT);
        while (multi_block_size < tmp_size) {
            multi_block_size <<= 1;
        }
    }

    // get block size
    int64_t threads_per_block;
    if (multi_block_size > 1 || decoding_batches == 0) {
        threads_per_block = TPB;
    }
    else {
        // threads_per_block = 512;
        int32_t num_blocks_per_sm = -1;
        auto kernel_fn = dynamic_batching_decoding_cache_attention_fp16_kernel<64, 4, TPB, 8, 1, true>;
        switch (head_dim) {
            case 64:
                kernel_fn = dynamic_batching_decoding_cache_attention_fp16_kernel<64, 4, TPB, 8, 1, true>;
                break;
            case 96:
                kernel_fn = dynamic_batching_decoding_cache_attention_fp16_kernel<96, 4, TPB, 8, 1, true>;
                break;
            case 128:
                kernel_fn = dynamic_batching_decoding_cache_attention_fp16_kernel<128, 8, TPB, 8, 1, true>;
                break;
            case 256:
                kernel_fn = dynamic_batching_decoding_cache_attention_fp16_kernel<256, 16, TPB, 8, 1, true>;
                break;
            default:
                LOG(ERROR) << "cache flash decoding attention do not support head dim " << head_dim;
                return ppl::common::RC_UNSUPPORTED;
        }
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel_fn, TPB, 0);
        int64_t block_size_factor = (multi_processor_count * num_blocks_per_sm + kernel_total_blocks - 1) / kernel_total_blocks;
        block_size_factor = min(block_size_factor, (int64_t)num_blocks_per_sm);
        threads_per_block = min(TPB * block_size_factor, 1024L);
        if (threads_per_block <= 256) {
            threads_per_block = 256;
        }
        else if (threads_per_block <= 512) {
            threads_per_block = 512;
        }
        else {
            threads_per_block = 1024;
        }
    }

    // get decoding shared memory size
    int64_t decoding_shm_size = 0;
    if(decoding_batches > 0) {
        constexpr int64_t WARP_SIZE = 32;
        const int64_t WPT = threads_per_block / WARP_SIZE;
        const int64_t reduce_shm_size = threads_per_block / WARP_SIZE * sizeof(float);
        const int64_t max_multi_block_kvlen = (max_kvlen * sizeof(float) + multi_block_size - 1) / multi_block_size;
        const int64_t logits_size = max(max_multi_block_kvlen, WPT * head_dim * sizeof(float));
        decoding_shm_size = reduce_shm_size + logits_size;
    }

    config->query           = const_cast<void*>(query);
    config->output          = output;
    config->cache           = cache;
    config->scale           = scale;
    config->cachestarts     = const_cast<void*>(cachestarts);
    config->kvstarts        = const_cast<void*>(kvstarts);
    config->attn_scale      = attn_scale;
    config->layer_idx       = layer_idx;
    config->kv_head_shift   = kv_head_shift;
    config->num_kv_repeats  = kv_repeats;
    config->q_stride_s      = q_stride_s;
    config->k_stride_s      = k_stride_s;
    config->v_stride_s      = v_stride_s;
    config->o_stride_s      = o_stride_s;
    config->cache_stride_s  = cache_stride_s;
    config->cache_stride_l  = cache_stride_l;
    config->cache_stride_h  = cache_stride_h;
    config->cache_stride_kv = cache_stride_kv;

    config->query_shape         = const_cast<ppl::common::TensorShape*>(query_shape);
    config->current_key         = const_cast<void*>(current_key);
    config->current_value       = const_cast<void*>(current_value);
    config->seqstarts           = const_cast<void*>(seqstarts);
    config->prefill_batches     = prefill_batches;
    config->decoding_batches    = decoding_batches;
    config->max_seqlen          = max_seqlen;
    config->max_kvlen           = max_kvlen;
    config->num_heads           = num_heads;
    config->head_dim            = head_dim;
    config->num_kv_heads        = num_kv_heads;
    config->is_causal           = is_causal;

    config->threads_per_block           = threads_per_block;
    config->decoding_shm_size           = decoding_shm_size;
    config->multi_block_size            = multi_block_size;
    config->multi_block_output_size     = decoding_batches * num_heads * head_dim * multi_block_size * sizeof(half);
    config->multi_block_sum_size        = decoding_batches * num_heads * multi_block_size * sizeof(float);
    config->multi_block_counter_size    = decoding_batches * num_heads * sizeof(int32_t);
    config->multi_block_tmpbuffer       = nullptr;

    *multi_block_buffer_size = config->multi_block_output_size + config->multi_block_sum_size*2 + config->multi_block_counter_size;

    return ppl::common::RC_SUCCESS;
}



template<int32_t TPB, bool DO_MULTI_BLOCK>
ppl::common::RetCode dynamic_batching_dynamic_threads_decoding_cache_attention(
    const cudaStream_t stream,
    const int64_t num_heads,
    const int64_t head_dim,
    const int64_t decoding_batches,
    const int64_t kernel_shm_size,
    const int64_t multi_block_size,
    dynamic_batching_decoding_cache_attention_kernel_param p
)
{
    const int64_t RAW_SHM_SIZE = 48 * 1024;

    auto kernel_fn = dynamic_batching_decoding_cache_attention_fp16_kernel<64, 4, TPB, 8, 1, true>;
    if (DO_MULTI_BLOCK) {
        kernel_fn = dynamic_batching_decoding_cache_attention_fp16_kernel<64, 4, TPB, 8, 32, true>;
        switch (head_dim) {
            case 64:
                kernel_fn = dynamic_batching_decoding_cache_attention_fp16_kernel<64, 4, TPB, 8, 32, true>;
                break;
            case 96:
                kernel_fn = dynamic_batching_decoding_cache_attention_fp16_kernel<96, 4, TPB, 8, 32, true>;
                break;
            case 128:
                kernel_fn = dynamic_batching_decoding_cache_attention_fp16_kernel<128, 8, TPB, 8, 16, true>;
                break;
            case 256:
                kernel_fn = dynamic_batching_decoding_cache_attention_fp16_kernel<256, 16, TPB, 8, 8, true>;
                break;
            default:
                LOG(ERROR) << "cache flash decoding attention do not support head dim " << head_dim;
                return ppl::common::RC_UNSUPPORTED;
        }
    } else {
        switch (head_dim) {
            case 64:
                kernel_fn = dynamic_batching_decoding_cache_attention_fp16_kernel<64, 4, TPB, 8, 1, true>;
                break;
            case 96:
                kernel_fn = dynamic_batching_decoding_cache_attention_fp16_kernel<96, 4, TPB, 8, 1, true>;
                break;
            case 128:
                kernel_fn = dynamic_batching_decoding_cache_attention_fp16_kernel<128, 8, TPB, 8, 1, true>;
                break;
            case 256:
                kernel_fn = dynamic_batching_decoding_cache_attention_fp16_kernel<256, 16, TPB, 8, 1, true>;
                break;
            default:
                LOG(ERROR) << "cache flash decoding attention do not support head dim " << head_dim;
                return ppl::common::RC_UNSUPPORTED;
        }
    }
    if (kernel_shm_size > RAW_SHM_SIZE) {
        auto cuda_err = cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, kernel_shm_size);
        if (cuda_err == cudaErrorInvalidValue) {
            LOG(ERROR) << "this gpu does not have enough shared-memory cache flash decoding attention requires";
            return ppl::common::RC_UNSUPPORTED;
        }
    }
    const dim3 grid_size = {(unsigned int)num_heads, (unsigned int)decoding_batches, (unsigned int)multi_block_size};
    kernel_fn<<<grid_size, TPB, kernel_shm_size, stream>>>(p);

    return ppl::common::RC_SUCCESS;
}



ppl::common::RetCode dynamic_batch_multi_head_cache_attention(
    const cudaStream_t stream,
    const cudaDeviceProp& device_prop,
    dynamic_batch_multi_head_cache_attention_config config) // (S, .., D)
{
    void* query         = config.query;
    void* output        = config.output;
    void* cache         = config.cache;
    void* scale         = config.scale;
    void* cachestarts   = config.cachestarts;
    void* kvstarts      = config.kvstarts;
    const float attn_scale          = config.attn_scale;
    const int64_t layer_idx         = config.layer_idx;
    const int64_t kv_head_shift     = config.kv_head_shift;
    const int64_t num_kv_repeats    = config.num_kv_repeats;
    const int64_t q_stride_s        = config.q_stride_s;
    const int64_t k_stride_s        = config.k_stride_s;
    const int64_t v_stride_s        = config.v_stride_s;
    const int64_t o_stride_s        = config.o_stride_s;
    const int64_t cache_stride_s    = config.cache_stride_s;
    const int64_t cache_stride_l    = config.cache_stride_l;
    const int64_t cache_stride_h    = config.cache_stride_h;
    const int64_t cache_stride_kv   = config.cache_stride_kv;

    ppl::common::TensorShape* query_shape = config.query_shape;
    void* current_key   = config.current_key;
    void* current_value = config.current_value;
    void* seqstarts     = config.seqstarts;
    const int64_t prefill_batches   = config.prefill_batches;
    const int64_t decoding_batches  = config.decoding_batches;
    const int64_t max_seqlen        = config.max_seqlen;
    const int64_t max_kvlen         = config.max_kvlen;
    const int64_t num_heads         = config.num_heads;
    const int64_t num_kv_heads      = config.num_kv_heads;
    const int64_t head_dim          = config.head_dim;
    const bool is_causal            = config.is_causal;

    struct dynamic_batching_decoding_cache_attention_kernel_param p;
    p.query = (half*)query;
    p.output = (half*)output;
    p.cache = (int8_t*)cache;
    p.scale = (half*)scale;
    p.cachestarts = (int64_t*)cachestarts;
    p.kvstarts = (int64_t*)kvstarts;
    p.attn_scale = attn_scale;
    p.layer_idx = layer_idx;
    p.kv_head_shift = kv_head_shift;
    p.num_kv_repeats = num_kv_repeats;
    p.query_stride_s = q_stride_s;
    p.output_stride_s = o_stride_s;
    p.cache_stride_s = cache_stride_s;
    p.cache_stride_l = cache_stride_l;
    p.cache_stride_h = cache_stride_h;
    p.cache_stride_kv = cache_stride_kv;

    if(decoding_batches > 0) {
        const int64_t MAX_SHM_SIZE = device_prop.sharedMemPerBlockOptin;

        const int64_t kernel_shm_size           = config.decoding_shm_size;
        const int64_t threads_per_block         = config.threads_per_block;
        const int64_t multi_block_size          = config.multi_block_size;
        const int64_t multi_block_sum_size      = config.multi_block_sum_size;
        const int64_t multi_block_counter_size  = config.multi_block_counter_size;
        const int64_t multi_block_output_size   = config.multi_block_output_size;
        void* multi_block_tmpbuffer       = config.multi_block_tmpbuffer;

        if (kernel_shm_size <= MAX_SHM_SIZE) {
            ppl::common::RetCode status;
            if (multi_block_size > 1) {
                p.multi_block.partial_out   = (half*)multi_block_tmpbuffer;
                p.multi_block.partial_max   = reinterpret_cast<float*>((char*)multi_block_tmpbuffer + multi_block_output_size);
                p.multi_block.partial_sum   = reinterpret_cast<float*>((char*)multi_block_tmpbuffer + multi_block_output_size + multi_block_sum_size);
                p.multi_block.block_counter = reinterpret_cast<int32_t*>((char*)multi_block_tmpbuffer + multi_block_output_size + multi_block_sum_size * 2);
                cudaMemset(p.multi_block.block_counter, 0, multi_block_counter_size);

                status = dynamic_batching_dynamic_threads_decoding_cache_attention<256, true>(
                    stream,
                    num_heads,
                    head_dim,
                    decoding_batches,
                    kernel_shm_size,
                    multi_block_size,
                    p);
            } else if (threads_per_block == 256) {
                status = dynamic_batching_dynamic_threads_decoding_cache_attention<256, false>(
                    stream,
                    num_heads,
                    head_dim,
                    decoding_batches,
                    kernel_shm_size,
                    multi_block_size,
                    p);
            } else if (threads_per_block == 512) {
                status = dynamic_batching_dynamic_threads_decoding_cache_attention<512, false>(
                    stream,
                    num_heads,
                    head_dim,
                    decoding_batches,
                    kernel_shm_size,
                    multi_block_size,
                    p);
            } else {
                status = dynamic_batching_dynamic_threads_decoding_cache_attention<1024, false>(
                    stream,
                    num_heads,
                    head_dim,
                    decoding_batches,
                    kernel_shm_size,
                    multi_block_size,
                    p);
            }
            if (status != ppl::common::RC_SUCCESS) {
                return ppl::common::RC_UNSUPPORTED;
            }
        } else {
            LOG(ERROR) << "shm not enough, cache flash decoding attention is unsupported.";
            return ppl::common::RC_UNSUPPORTED;
        }
    }

    if (prefill_batches > 0) {
        const int64_t custom_mask_type = is_causal ? 1 : 0;
        const void* prefill_seqstart_q = ((int64_t*)seqstarts) + decoding_batches;

        return llm::cuda::xformer::fmha(
            stream,
            device_prop,
            query_shape->GetDataType(),
            query,
            current_key,
            current_value,
            nullptr,
            prefill_seqstart_q,
            prefill_seqstart_q,
            prefill_batches,
            0, q_stride_s, head_dim,
            0, k_stride_s, head_dim,
            0, v_stride_s, head_dim,
            0, 0, 0,
            o_stride_s,
            max_seqlen,
            max_kvlen,
            num_heads,
            num_kv_heads,
            head_dim,
            custom_mask_type,
            attn_scale,
            output
        );
    } else {
        return ppl::common::RC_SUCCESS;
    }
}

}}}}}
