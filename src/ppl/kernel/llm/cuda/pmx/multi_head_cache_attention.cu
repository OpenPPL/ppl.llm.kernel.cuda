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
    half* attn_mask;
    half* output;
    int8_t* cache;
    half* scale;
    int64_t* cachestarts;
    int64_t* kvstarts;
    float attn_scale;
    int64_t layer_idx;
    int64_t num_kv_repeats;
    int64_t query_stride_s;
    int64_t output_stride_s;
    int64_t mask_stride_s;
    int64_t mask_stride_h;
    int64_t cache_stride_s;
    int64_t cache_stride_l;
    int64_t cache_stride_h;
    int64_t cache_stride_kv;

    struct {
        int32_t* block_counter;
        float* log_sum_exp;
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
    int64_t head_dim;
    int64_t current_key_stride_s;
    int64_t current_value_stride_s;
    int64_t cache_stride_s;
    int64_t cache_stride_l;
    int64_t cache_stride_h;
    int64_t cache_stride_kv;
    int8_t* cache;
    half* scale;
};

template<int32_t VPT, int32_t TPB> // 8 fp16 occupy 128 bytes, which can be loaded by a single thread at once.
__global__
void dynamic_batching_kv_cache_quantize_kernel(dynamic_batching_kv_cache_quantize_kernel_param p)
{
    if (blockIdx.x < p.seqstarts[blockIdx.y + 1] - p.seqstarts[blockIdx.y]) {
        const int64_t thr_per_head = p.head_dim / VPT;
        const int64_t batch_id = blockIdx.y;
        const int64_t seq_idx = blockIdx.x;
        const int64_t tid = blockIdx.z * TPB + threadIdx.x;

        if (tid < p.num_kv_heads * p.head_dim / VPT) {
            const int64_t input_token_idx = p.seqstarts[batch_id] + seq_idx;
            const int64_t cache_token_idx = p.cachestarts[batch_id] + seq_idx + p.start_pos[batch_id];
            const int64_t key_out_offset = cache_token_idx * p.cache_stride_s + p.layer_idx * p.cache_stride_l;
            auto key_in_ptr = p.current_key + input_token_idx * p.current_key_stride_s;
            auto value_in_ptr = p.current_value + input_token_idx * p.current_value_stride_s;

            const int64_t kv_head_idx = tid / (thr_per_head);
            const int64_t dim_idx = (tid % thr_per_head) * VPT;
            const int64_t scale_dim_idx = dim_idx / VPT;
            const int64_t input_idx = kv_head_idx * p.head_dim + dim_idx;

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
    bool ATTN_MASK>
__global__
void dynamic_batching_decoding_cache_sharemem_attention_fp16_kernel(dynamic_batching_decoding_cache_attention_kernel_param p)
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
    const int32_t head_idx      = blockIdx.x;
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
    const int32_t kv_head_idx     = head_idx / int32_t(p.num_kv_repeats);

    half *attn_mask = nullptr;
    if (ATTN_MASK) {
        attn_mask = p.attn_mask
                + p.mask_stride_h * head_idx
                + batch_idx * p.mask_stride_s
                + p.kvstarts[batch_idx];
    }

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
    const int64_t block_context_beg     = block_idx * context_len_per_block;
    const int64_t block_context_len     = context_len >= context_len_per_block * (block_idx + 1) ? context_len_per_block : context_len - block_context_beg;

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
                            = (cache_offset_s + block_context_beg + block_context_id) * p.cache_stride_s
                            + p.layer_idx * p.cache_stride_l
                            + p.cache_stride_h * kv_head_idx
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
            if (ATTN_MASK)
                qk_dot += __half2float(attn_mask[block_context_id]);
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

    if (MULTI_BLOCK > 1 && threadIdx.x == 0) {
        p.multi_block.log_sum_exp[
            batch_size * MULTI_BLOCK * head_idx +
            batch_idx * MULTI_BLOCK +
            block_idx]
            = partial_qk_max + log(partial_exp_sum);
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
        if (block_context_id < block_context_len) {
            const int64_t value_offset
                            = (cache_offset_s + block_context_beg + block_context_id) * p.cache_stride_s
                            + p.layer_idx * p.cache_stride_l
                            + p.cache_stride_h * kv_head_idx
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

    const float inv_sum = __fdividef(1.f, partial_exp_sum + 1e-6f);
    #pragma unroll
    for (int32_t i = 0; i < VEC_SIZE * VEC_LEN; i++) {
        local_v[i] *= inv_sum;
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
    constexpr int64_t WORK_WARP = (WORK_THREAD + WARP_SIZE - 1) / WARP_SIZE;
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
    if (warp_id < WORK_WARP) {
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
        } else {
            for (int32_t i = 0; i < VEC_SIZE * VEC_LEN; i+= 1) {
                local_v[i] = 0.f;
            }
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
                batch_size * HEAD_SIZE * MULTI_BLOCK * head_idx +
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
        if (threadIdx.x == 0) {
            if (atomicAdd(&p.multi_block.block_counter[batch_size * head_idx + batch_idx], 1) == MULTI_BLOCK - 1) {
                last_block = true;
            }
        }

        // The last block do the final computation.
        if (__syncthreads_or(last_block)) {
            const int64_t multi_block_idx = threadIdx.x % MULTI_BLOCK;
            const int64_t hbb = batch_size * MULTI_BLOCK * head_idx + batch_idx * MULTI_BLOCK + multi_block_idx;

            float local_log_sum_exp = warp_lane_id < MULTI_BLOCK ? p.multi_block.log_sum_exp[hbb] : -FLT_MAX;
            float max_log_sum_exp = local_log_sum_exp;
            # pragma unroll
            for (int32_t mask = MULTI_BLOCK / 2; mask >= 1; mask /= 2) {
                max_log_sum_exp = fmaxf(max_log_sum_exp, __shfl_xor_sync(uint32_t(-1), max_log_sum_exp, mask));
            }
            max_log_sum_exp = __shfl_sync(uint32_t(-1), max_log_sum_exp, 0);

            float local_scale = warp_lane_id < MULTI_BLOCK ? exp(local_log_sum_exp - max_log_sum_exp) : 0.f;
            float scale_sum = local_scale;
            # pragma unroll
            for (int32_t mask = MULTI_BLOCK / 2; mask >= 1; mask /= 2) {
                scale_sum += __shfl_xor_sync(uint32_t(-1), scale_sum, mask);
            }
            scale_sum = __shfl_sync(uint32_t(-1), scale_sum, 0);

            float *scale_smem = logits;
            int scale_id = warp_id * MULTI_BLOCK + warp_lane_id;
            if (warp_lane_id < MULTI_BLOCK && scale_id < WARP_SIZE) {
                scale_smem[scale_id] = local_scale / scale_sum;
            }
            __syncthreads();

            const int64_t head_dim_idx_base  = (int64_t)(threadIdx.x / MULTI_BLOCK) * VEC_SIZE;
            const int64_t head_dim_idx_stride = TPB / MULTI_BLOCK * VEC_SIZE;

            for (int64_t head_dim_idx = head_dim_idx_base; head_dim_idx < HEAD_SIZE; head_dim_idx += head_dim_idx_stride) {
                half final_out[VEC_SIZE];
                local_scale = scale_smem[warp_lane_id];
                copy<VEC_SIZE*sizeof(half)>(
                    &p.multi_block.partial_out[
                        batch_size * HEAD_SIZE * MULTI_BLOCK * head_idx +
                        batch_idx * HEAD_SIZE * MULTI_BLOCK +
                        head_dim_idx * MULTI_BLOCK +
                        multi_block_idx * VEC_SIZE],
                    final_out);

                #pragma unroll
                for (int32_t i = 0; i < VEC_SIZE; i++) {
                    float float_out = __half2float(final_out[i]) * local_scale;
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



template<
    int32_t HEAD_SIZE,
    int32_t THREAD_GROUP_SIZE,        // how many threads inside a group
    int32_t TPB,
    int32_t QUANT_GROUP,
    int32_t MULTI_BLOCK,    // do flash decoding if more than 1
    bool ATTN_MASK>
__global__
void dynamic_batching_decoding_cache_infinity_attention_fp16_kernel(dynamic_batching_decoding_cache_attention_kernel_param p)
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
    const int32_t head_idx      = blockIdx.x;
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
    const int32_t kv_head_idx     = head_idx / p.num_kv_repeats;

    half *attn_mask = nullptr;
    if (ATTN_MASK) {
        attn_mask = p.attn_mask
                + p.mask_stride_h * head_idx
                + batch_idx * p.mask_stride_s
                + p.kvstarts[batch_idx];
    }

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
    const int64_t block_context_beg     = block_idx * context_len_per_block;
    const int64_t block_context_len     = context_len >= context_len_per_block * (block_idx + 1) ? context_len_per_block : context_len - block_context_beg;

    extern __shared__ float logits[];
    float thread_qk_max = -FLT_MAX;
    float partial_exp_sum = 0.0f;

    float local_v[VEC_SIZE * VEC_LEN];
    #pragma unroll
    for(int32_t i = 0; i < VEC_SIZE * VEC_LEN; i++) {
        local_v[i] = 0;
    }

    for (int64_t base_id = warp_id * GPW; base_id < block_context_len; base_id += GPT) {
        float local_v_new[VEC_SIZE * VEC_LEN];
        int8_t local_k_quant[VEC_SIZE * VEC_LEN], local_v_quant[VEC_SIZE * VEC_LEN];
        half local_k_scale[VEC_LEN], local_v_scale[VEC_LEN];
        const int64_t block_context_id = base_id + group_id;

        float qk_dot = 0.0f;

        // all thread groups within a warp must be launched together.
        if (block_context_id < block_context_len) {
            const int64_t key_offset
                            = (cache_offset_s + block_context_beg + block_context_id) * p.cache_stride_s
                            + p.layer_idx * p.cache_stride_l
                            + p.cache_stride_h * kv_head_idx
                            + group_lane_id * VEC_SIZE;
            const int64_t value_offset = key_offset + p.cache_stride_kv;
            #pragma unroll
            for (int64_t i = 0; i < VEC_LEN; i++) {
                // copy 128(16 * 8) bits from K to Local K
                const int64_t key_idx = key_offset + i * THREAD_GROUP_SIZE * VEC_SIZE;
                copy<sizeof(int8_t) * VEC_SIZE>(&p.cache[key_idx],  &local_k_quant[i * VEC_SIZE]);
                const int64_t key_scale_idx = key_idx >> QUANT_GROUP_SHIFT;
                local_k_scale[i] = p.scale[key_scale_idx];

                // copy 128(16 * 8) bits from V to Local V
                const int64_t value_idx = value_offset + i * THREAD_GROUP_SIZE * VEC_SIZE;
                copy<sizeof(int8_t) * VEC_SIZE>(&p.cache[value_idx],  &local_v_quant[i * VEC_SIZE]);
                const int64_t value_scale_idx = value_idx >> QUANT_GROUP_SHIFT;
                local_v_scale[i] = p.scale[value_scale_idx];

                #pragma unroll
                for(int64_t k = 0; k < VEC_SIZE; k++) {
                    local_k_quant[i * VEC_SIZE + k] += 128;
                    local_v_quant[i * VEC_SIZE + k] += 128;
                }

                half result_k[8];
                uint32_t*      h_k   = reinterpret_cast<uint32_t*>(result_k);
                uint32_t const i8s_k = reinterpret_cast<uint32_t const&>(*(local_k_quant + i * VEC_SIZE));
                asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h_k[0]) : "r"(i8s_k), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
                asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h_k[1]) : "r"(i8s_k), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));
                asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h_k[0]) : "r"(h_k[0]), "r"(I8s_TO_F16s_MAGIC_NUM));
                asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h_k[1]) : "r"(h_k[1]), "r"(I8s_TO_F16s_MAGIC_NUM));
                uint32_t*      h_k_2   = reinterpret_cast<uint32_t*>(result_k+4);
                uint32_t const i8s_k_2 = reinterpret_cast<uint32_t const&>(*(local_k_quant + i * VEC_SIZE + 4));
                asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h_k_2[0]) : "r"(i8s_k_2), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
                asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h_k_2[1]) : "r"(i8s_k_2), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));
                asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h_k_2[0]) : "r"(h_k_2[0]), "r"(I8s_TO_F16s_MAGIC_NUM));
                asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h_k_2[1]) : "r"(h_k_2[1]), "r"(I8s_TO_F16s_MAGIC_NUM));
                #pragma unroll
                for (int64_t j = 0; j < VEC_SIZE; j++) {
                    qk_dot += __half2float(local_q[i * VEC_SIZE + j]) * __half2float(local_k_scale[i] * result_k[j]);
                }

                half result_v[8];
                uint32_t*      h_v   = reinterpret_cast<uint32_t*>(result_v);
                uint32_t const i8s_v = reinterpret_cast<uint32_t const&>(*(local_v_quant + i * VEC_SIZE));
                asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h_v[0]) : "r"(i8s_v), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
                asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h_v[1]) : "r"(i8s_v), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));
                asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h_v[0]) : "r"(h_v[0]), "r"(I8s_TO_F16s_MAGIC_NUM));
                asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h_v[1]) : "r"(h_v[1]), "r"(I8s_TO_F16s_MAGIC_NUM));
                uint32_t*      h_v_2   = reinterpret_cast<uint32_t*>(result_v+4);
                uint32_t const i8s_v_2 = reinterpret_cast<uint32_t const&>(*(local_v_quant + i * VEC_SIZE + 4));
                asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h_v_2[0]) : "r"(i8s_v_2), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
                asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h_v_2[1]) : "r"(i8s_v_2), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));
                asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h_v_2[0]) : "r"(h_v_2[0]), "r"(I8s_TO_F16s_MAGIC_NUM));
                asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h_v_2[1]) : "r"(h_v_2[1]), "r"(I8s_TO_F16s_MAGIC_NUM));
                #pragma unroll
                for (int64_t j = 0; j < VEC_SIZE; j++) {
                    local_v_new[i * VEC_SIZE + j] = __half2float(local_v_scale[i] * result_v[j]);
                }
            }
        }

        qk_dot = p.attn_scale * attn_thread_group_reduce_sum<THREAD_GROUP_SIZE>(qk_dot);

        if (block_context_id < block_context_len) {
            if (ATTN_MASK) {
                qk_dot += __half2float(attn_mask[block_context_id]);
            }
            if (qk_dot > thread_qk_max) {
                float logic_scale = exp(thread_qk_max - qk_dot);
                thread_qk_max = qk_dot;
                partial_exp_sum = partial_exp_sum * logic_scale + 1.f;
                #pragma unroll
                for(int32_t i = 0; i < VEC_SIZE * VEC_LEN; i++) {
                    local_v[i] = local_v[i] * logic_scale + local_v_new[i];
                }
            } else {
                float exp_logic = exp(qk_dot - thread_qk_max);
                partial_exp_sum += exp_logic;
                #pragma unroll
                for(int32_t i = 0; i < VEC_SIZE * VEC_LEN; i++) {
                    local_v[i] = local_v[i] + local_v_new[i] * exp_logic;
                }
            }
        }
    }

    __shared__ float red_smem[WPT];

    // reduce partial_qk_max in thread block and boardcast
    float partial_qk_max = attn_block_reduce_max<WPT>(thread_qk_max, red_smem);

    if (partial_qk_max > thread_qk_max) {
        float logic_scale = exp(thread_qk_max - partial_qk_max);
        partial_exp_sum *= logic_scale;
        #pragma unroll
        for(int32_t i = 0; i < VEC_SIZE * VEC_LEN; i++) {
            local_v[i] *= logic_scale;
        }
    }

    // block reduce sum on partial_exp_sum
    // Warp per thread block must be power-of-2 for reducation, check attn_block_reduce_sum kernel.
    static_assert(WPT == 2 || WPT == 4 || WPT == 8 || WPT == 16 || WPT == 32 || WPT == 64);
    partial_exp_sum = attn_block_reduce_sum<WPT>(partial_exp_sum, red_smem) / THREAD_GROUP_SIZE;

    if (MULTI_BLOCK > 1 && threadIdx.x == 0) {
        p.multi_block.log_sum_exp[
            batch_size * MULTI_BLOCK * head_idx +
            batch_idx * MULTI_BLOCK +
            block_idx]
            = partial_qk_max + log(partial_exp_sum);
    }

    const float inv_sum = __fdividef(1.f, partial_exp_sum + 1e-6f);
    #pragma unroll
    for (int32_t i = 0; i < VEC_SIZE * VEC_LEN; i++) {
        local_v[i] *= inv_sum;
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
    constexpr int64_t WORK_WARP = (WORK_THREAD + WARP_SIZE - 1) / WARP_SIZE;
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
    if (warp_id < WORK_WARP) {
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
        } else {
            for (int32_t i = 0; i < VEC_SIZE * VEC_LEN; i+= 1) {
                local_v[i] = 0.f;
            }
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
                batch_size * HEAD_SIZE * MULTI_BLOCK * head_idx +
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
        if (threadIdx.x == 0) {
            if (atomicAdd(&p.multi_block.block_counter[batch_size * head_idx + batch_idx], 1) == MULTI_BLOCK - 1) {
                last_block = true;
            }
        }

        // The last block do the final computation.
        if (__syncthreads_or(last_block)) {
            const int64_t multi_block_idx = threadIdx.x % MULTI_BLOCK;
            const int64_t hbb = batch_size * MULTI_BLOCK * head_idx + batch_idx * MULTI_BLOCK + multi_block_idx;

            float local_log_sum_exp = warp_lane_id < MULTI_BLOCK ? p.multi_block.log_sum_exp[hbb] : -FLT_MAX;
            float max_log_sum_exp = local_log_sum_exp;
            # pragma unroll
            for (int32_t mask = MULTI_BLOCK / 2; mask >= 1; mask /= 2) {
                max_log_sum_exp = fmaxf(max_log_sum_exp, __shfl_xor_sync(uint32_t(-1), max_log_sum_exp, mask));
            }
            max_log_sum_exp = __shfl_sync(uint32_t(-1), max_log_sum_exp, 0);

            float local_scale = warp_lane_id < MULTI_BLOCK ? exp(local_log_sum_exp - max_log_sum_exp) : 0.f;
            float scale_sum = local_scale;
            # pragma unroll
            for (int32_t mask = MULTI_BLOCK / 2; mask >= 1; mask /= 2) {
                scale_sum += __shfl_xor_sync(uint32_t(-1), scale_sum, mask);
            }
            scale_sum = __shfl_sync(uint32_t(-1), scale_sum, 0);

            float *scale_smem = logits;
            int scale_id = warp_id * MULTI_BLOCK + warp_lane_id;
            if (warp_lane_id < MULTI_BLOCK && scale_id < WARP_SIZE) {
                scale_smem[scale_id] = local_scale / scale_sum;
            }
            __syncthreads();

            const int64_t head_dim_idx_base  = (int64_t)(threadIdx.x / MULTI_BLOCK) * VEC_SIZE;
            const int64_t head_dim_idx_stride = TPB / MULTI_BLOCK * VEC_SIZE;

            for (int64_t head_dim_idx = head_dim_idx_base; head_dim_idx < HEAD_SIZE; head_dim_idx += head_dim_idx_stride) {
                half final_out[VEC_SIZE];
                local_scale = scale_smem[warp_lane_id];
                copy<VEC_SIZE*sizeof(half)>(
                    &p.multi_block.partial_out[
                        batch_size * HEAD_SIZE * MULTI_BLOCK * head_idx +
                        batch_idx * HEAD_SIZE * MULTI_BLOCK +
                        head_dim_idx * MULTI_BLOCK +
                        multi_block_idx * VEC_SIZE],
                    final_out);

                #pragma unroll
                for (int32_t i = 0; i < VEC_SIZE; i++) {
                    float float_out = __half2float(final_out[i]) * local_scale;
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

std::pair<ppl::common::RetCode, dynamic_batching_multi_head_cache_attention_config>
dynamic_batching_multi_head_cache_attention_prepare(
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
    void* output) // (S, .., D)
{
    dynamic_batching_multi_head_cache_attention_config config{0};

    if (query_shape->GetDim(1) != num_heads) {
        LOG(ERROR) << "currnetly do not support leading dim on query's num_heads";
        return {ppl::common::RC_UNSUPPORTED, config};
    }

    if (output_shape->GetDim(1) != num_heads) {
        LOG(ERROR) << "currnetly do not support leading dim on output's num_heads";
        return {ppl::common::RC_UNSUPPORTED, config};
    }

    if (current_key_shape->GetDim(1) != num_kv_heads) {
        LOG(ERROR)
            << "currnetly do not support leading dim on current_key's num_kv_heads";
        return {ppl::common::RC_UNSUPPORTED, config};
    }

    if (current_value_shape->GetDim(1) != num_kv_heads) {
        LOG(ERROR)
            << "currnetly do not support leading dim on current_value's num_kv_heads";
        return {ppl::common::RC_UNSUPPORTED, config};
    }

    if (cache_mode != 0) {
        LOG(ERROR) << "currently only support cache_mode == 0";
        return {ppl::common::RC_UNSUPPORTED, config};
    }

    constexpr int64_t TPB = 256;
    constexpr int64_t VPT = 8;

    int64_t kv_repeats = num_heads / num_kv_heads;
    if ((num_kv_heads * kv_repeats) != num_heads) {
        LOG(ERROR) << "only support num_heads % num_kv_heads == 0.";
        return {ppl::common::RC_UNSUPPORTED, config};
    }

    const int64_t prefill_batches = batch - decoding_batches;
    const float attn_scale = float(1.0 / std::sqrt(float(head_dim)));

    const int64_t q_stride_s = query_shape->GetDim(1) * head_dim;
    const int64_t k_stride_s = current_key_shape->GetDim(1) * head_dim;
    const int64_t v_stride_s = current_value_shape->GetDim(1) * head_dim;
    const int64_t o_stride_s = output_shape->GetDim(1) * head_dim;

    int64_t mask_stride_s = 0;
    int64_t mask_stride_h = 0;
    if (attn_mask && attn_mask_shape->CalcElementsExcludingPadding() > 0) {
        if (attn_mask_shape->GetDimCount() == 3) {
            mask_stride_h = attn_mask_shape->GetDim(1) * attn_mask_shape->GetDim(2);
            mask_stride_s = attn_mask_shape->GetDim(2);
        } else if (attn_mask_shape->GetDimCount() == 2) {
            mask_stride_s = attn_mask_shape->GetDim(1);
        } else {
            LOG(ERROR) << "attn_mask must be 2d or 3d";
            return {ppl::common::RC_UNSUPPORTED, config};
        }
        if (mask_stride_s % VPT != 0) {
            LOG(ERROR) << "last dimension of attn_mask must be aligned with " << VPT;
            return {ppl::common::RC_UNSUPPORTED, config};
        }
    }


    const int64_t decoding_attention_total_blocks = num_heads * decoding_batches;
    const int32_t multi_processor_count = device_prop.multiProcessorCount;

    // get multi block size
    int64_t decoding_multi_block_size = 1;
    if (decoding_batches > 0 && decoding_attention_total_blocks < multi_processor_count && max_kvlen >= 1024) {
        while (decoding_multi_block_size < TPB / (head_dim / VPT)) {
            decoding_multi_block_size <<= 1;
        }
    }

    // get block size
    int64_t decoding_threads_per_block = TPB;
    if (decoding_attention_total_blocks < multi_processor_count * 0.9f &&
        decoding_multi_block_size == 1 &&
        decoding_batches > 0) {
        int32_t num_blocks_per_sm = -1;
        auto kernel_fn = dynamic_batching_decoding_cache_sharemem_attention_fp16_kernel<64, 4, TPB, 8, 1, false>;
        switch (head_dim) {
            case 64:
                kernel_fn = dynamic_batching_decoding_cache_sharemem_attention_fp16_kernel<64, 4, TPB, 8, 1, false>;
                break;
            case 96:
                kernel_fn = dynamic_batching_decoding_cache_sharemem_attention_fp16_kernel<96, 4, TPB, 8, 1, false>;
                break;
            case 128:
                kernel_fn = dynamic_batching_decoding_cache_sharemem_attention_fp16_kernel<128, 8, TPB, 8, 1, false>;
                break;
            case 256:
                kernel_fn = dynamic_batching_decoding_cache_sharemem_attention_fp16_kernel<256, 16, TPB, 8, 1, false>;
                break;
            default:
                LOG(ERROR) << "cache flash decoding attention do not support head dim " << head_dim;
                return {ppl::common::RC_UNSUPPORTED, config};
        }
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel_fn, TPB, 0);
        int64_t block_size_factor = (multi_processor_count * num_blocks_per_sm + decoding_attention_total_blocks - 1) / decoding_attention_total_blocks;
        block_size_factor = std::min<int64_t>(block_size_factor, num_blocks_per_sm);
        decoding_threads_per_block = std::min<int64_t>(TPB * block_size_factor, 1024);
        if (decoding_threads_per_block >= 1024 && max_kvlen > 512) {
            decoding_threads_per_block = 1024;
        } else if (decoding_threads_per_block >= 512 && max_kvlen > 256) {
            decoding_threads_per_block = 512;
        } else {
            decoding_threads_per_block = 256;
        }
    }

    // get decoding shared memory size
    bool use_infinity_mhca = false;
    int64_t decoding_shm_size = 0;
    if (decoding_batches > 0) {
        constexpr int64_t WARP_SIZE = 32;
        const int64_t WPT = decoding_threads_per_block / WARP_SIZE;
        const int64_t reduce_shm_size = decoding_threads_per_block / WARP_SIZE * sizeof(float);
        const int64_t max_multi_block_kvlen = (max_kvlen * sizeof(float) + decoding_multi_block_size - 1) / decoding_multi_block_size;
        const int64_t logits_size = max(max_multi_block_kvlen, WPT * head_dim * sizeof(float));
        decoding_shm_size = reduce_shm_size + logits_size;
        use_infinity_mhca = decoding_shm_size > (int64_t)device_prop.sharedMemPerBlockOptin;
    }

    if (use_infinity_mhca) {
        constexpr int64_t WARP_SIZE = 32;
        const int64_t WPT = decoding_threads_per_block / WARP_SIZE;
        const int64_t reduce_shm_size = decoding_threads_per_block / WARP_SIZE * sizeof(float);
        const int64_t logits_size = WPT * head_dim * sizeof(float);
        decoding_shm_size = reduce_shm_size + logits_size;
    }

    config.device_prop = const_cast<cudaDeviceProp*>(&device_prop);
    config.query_shape = const_cast<ppl::common::TensorShape*>(query_shape);
    config.query = const_cast<void*>(query);
    config.current_key_shape = const_cast<ppl::common::TensorShape*>(current_key_shape);
    config.current_key = const_cast<void*>(current_key);
    config.current_value_shape = const_cast<ppl::common::TensorShape*>(current_value_shape);
    config.current_value = const_cast<void*>(current_value);
    config.attn_mask_shape = const_cast<ppl::common::TensorShape*>(attn_mask_shape);
    config.attn_mask = const_cast<void*>(attn_mask);

    config.seqstarts = const_cast<void*>(seqstarts);
    config.kvstarts = const_cast<void*>(kvstarts);
    config.cachestarts = const_cast<void*>(cachestarts);
    config.start_pos = const_cast<void*>(start_pos);

    config.cache = cache;
    config.scale = scale;

    config.output_shape = const_cast<ppl::common::TensorShape*>(output_shape);
    config.output = output;

    config.is_causal = is_causal;
    config.batch = batch;
    config.decoding_batches = decoding_batches;
    config.max_seqlen = max_seqlen;
    config.max_kvlen = max_kvlen;
    config.layer_idx =layer_idx;
    config.num_layer = num_layer;
    config.num_heads = num_heads;
    config.num_kv_heads = num_kv_heads;
    config.head_dim = head_dim;
    config.cache_mode = cache_mode;
    config.cache_stride_s = cache_stride_s;
    config.cache_stride_l = cache_stride_l;
    config.cache_stride_h = cache_stride_h;
    config.cache_stride_kv = cache_stride_kv;

    config.prefill_batches = prefill_batches;
    config.q_stride_s = q_stride_s;
    config.k_stride_s = k_stride_s;
    config.v_stride_s = v_stride_s;
    config.o_stride_s = o_stride_s;

    config.mask_stride_s = mask_stride_s;
    config.mask_stride_h = mask_stride_h;

    config.attn_scale = attn_scale;
    config.num_kv_repeats = kv_repeats;
    config.use_infinity_mhca = use_infinity_mhca;

    config.decoding_threads_per_block = decoding_threads_per_block;
    config.decoding_shm_size = decoding_shm_size;
    config.decoding_multi_block_size = decoding_multi_block_size;
    if (config.decoding_multi_block_size > 1) {
        config.decoding_multi_block_output_size = decoding_batches * num_heads * head_dim * decoding_multi_block_size * sizeof(half);;
        config.decoding_multi_block_sum_size = decoding_batches * num_heads * decoding_multi_block_size * sizeof(float);
        config.decoding_multi_block_counter_size = decoding_batches * num_heads * sizeof(int32_t);
    }

    config.temp_buffer_size
        = config.decoding_multi_block_output_size
        + config.decoding_multi_block_sum_size
        + config.decoding_multi_block_counter_size;
    config.temp_buffer = nullptr;

    return {ppl::common::RC_SUCCESS, config};
}


template<int32_t TPB, bool ATTN_MASK, bool DO_MULTI_BLOCK>
ppl::common::RetCode dynamic_batching_decoding_cache_attention(
    const cudaStream_t stream,
    const dynamic_batching_multi_head_cache_attention_config &cfg,
    const dynamic_batching_decoding_cache_attention_kernel_param &p
)
{
    const int64_t RAW_SHM_SIZE = 48 * 1024;

    auto kernel_fn = dynamic_batching_decoding_cache_infinity_attention_fp16_kernel<64, 4, TPB, 8, 1, ATTN_MASK>;
    if (cfg.use_infinity_mhca) {
        if (DO_MULTI_BLOCK) {
            switch (cfg.head_dim) {
                case 64:
                    kernel_fn = dynamic_batching_decoding_cache_infinity_attention_fp16_kernel<64, 4, TPB, 8, 32, ATTN_MASK>;
                    break;
                case 96:
                    kernel_fn = dynamic_batching_decoding_cache_infinity_attention_fp16_kernel<96, 4, TPB, 8, 32, ATTN_MASK>;
                    break;
                case 128:
                    kernel_fn = dynamic_batching_decoding_cache_infinity_attention_fp16_kernel<128, 8, TPB, 8, 16, ATTN_MASK>;
                    break;
                case 256:
                    kernel_fn = dynamic_batching_decoding_cache_infinity_attention_fp16_kernel<256, 16, TPB, 8, 8, ATTN_MASK>;
                    break;
                default:
                    LOG(ERROR) << "cache flash decoding attention do not support head dim " << cfg.head_dim;
                    return ppl::common::RC_UNSUPPORTED;
            }
        } else {
            switch (cfg.head_dim) {
                case 64:
                    kernel_fn = dynamic_batching_decoding_cache_infinity_attention_fp16_kernel<64, 4, TPB, 8, 1, ATTN_MASK>;
                    break;
                case 96:
                    kernel_fn = dynamic_batching_decoding_cache_infinity_attention_fp16_kernel<96, 4, TPB, 8, 1, ATTN_MASK>;
                    break;
                case 128:
                    kernel_fn = dynamic_batching_decoding_cache_infinity_attention_fp16_kernel<128, 8, TPB, 8, 1, ATTN_MASK>;
                    break;
                case 256:
                    kernel_fn = dynamic_batching_decoding_cache_infinity_attention_fp16_kernel<256, 16, TPB, 8, 1, ATTN_MASK>;
                    break;
                default:
                    LOG(ERROR) << "cache flash decoding attention do not support head dim " << cfg.head_dim;
                    return ppl::common::RC_UNSUPPORTED;
            }
        }
    } else {
        if (DO_MULTI_BLOCK) {
            switch (cfg.head_dim) {
                case 64:
                    kernel_fn = dynamic_batching_decoding_cache_sharemem_attention_fp16_kernel<64, 4, TPB, 8, 32, ATTN_MASK>;
                    break;
                case 96:
                    kernel_fn = dynamic_batching_decoding_cache_sharemem_attention_fp16_kernel<96, 4, TPB, 8, 32, ATTN_MASK>;
                    break;
                case 128:
                    kernel_fn = dynamic_batching_decoding_cache_sharemem_attention_fp16_kernel<128, 8, TPB, 8, 16, ATTN_MASK>;
                    break;
                case 256:
                    kernel_fn = dynamic_batching_decoding_cache_sharemem_attention_fp16_kernel<256, 16, TPB, 8, 8, ATTN_MASK>;
                    break;
                default:
                    LOG(ERROR) << "cache flash decoding attention do not support head dim " << cfg.head_dim;
                    return ppl::common::RC_UNSUPPORTED;
            }
        } else {
            switch (cfg.head_dim) {
                case 64:
                    kernel_fn = dynamic_batching_decoding_cache_sharemem_attention_fp16_kernel<64, 4, TPB, 8, 1, ATTN_MASK>;
                    break;
                case 96:
                    kernel_fn = dynamic_batching_decoding_cache_sharemem_attention_fp16_kernel<96, 4, TPB, 8, 1, ATTN_MASK>;
                    break;
                case 128:
                    kernel_fn = dynamic_batching_decoding_cache_sharemem_attention_fp16_kernel<128, 8, TPB, 8, 1, ATTN_MASK>;
                    break;
                case 256:
                    kernel_fn = dynamic_batching_decoding_cache_sharemem_attention_fp16_kernel<256, 16, TPB, 8, 1, ATTN_MASK>;
                    break;
                default:
                    LOG(ERROR) << "cache flash decoding attention do not support head dim " << cfg.head_dim;
                    return ppl::common::RC_UNSUPPORTED;
            }
        }
    }

    if (cfg.decoding_shm_size > RAW_SHM_SIZE) {
        auto cuda_err = cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, cfg.decoding_shm_size);
        if (cuda_err == cudaErrorInvalidValue) {
            LOG(ERROR) << "this gpu does not have enough shared-memory cache flash decoding attention requires";
            return ppl::common::RC_UNSUPPORTED;
        }
    }
    const dim3 grid_size = {
        (unsigned int)cfg.num_heads,
        (unsigned int)cfg.decoding_batches,
        (unsigned int)cfg.decoding_multi_block_size};
    kernel_fn<<<grid_size, TPB, cfg.decoding_shm_size, stream>>>(p);

    return ppl::common::RC_SUCCESS;
}

template<int32_t TPB, bool DO_MULTI_BLOCK>
ppl::common::RetCode dynamic_batching_decoding_cache_attention(
    const cudaStream_t stream,
    const dynamic_batching_multi_head_cache_attention_config &cfg,
    const dynamic_batching_decoding_cache_attention_kernel_param &p
)
{
    if (p.attn_mask) {
        return dynamic_batching_decoding_cache_attention<TPB, true, DO_MULTI_BLOCK>(stream, cfg, p);
    } else {
        return dynamic_batching_decoding_cache_attention<TPB, false, DO_MULTI_BLOCK>(stream, cfg, p);
    }
}


ppl::common::RetCode dynamic_batching_multi_head_cache_attention(
    const cudaStream_t stream,
    const dynamic_batching_multi_head_cache_attention_config &cfg) // (S, .., D)
{
    {
        constexpr int64_t TPB = 256;
        constexpr int64_t VPT = 8;

        if (cfg.head_dim % VPT != 0) {
            LOG(ERROR) << "head_dim must be aligned with " << VPT << ", currently get " << cfg.head_dim;
            return ppl::common::RC_UNSUPPORTED;
        }

        dim3 grid(cfg.max_seqlen, cfg.batch, (cfg.num_kv_heads * cfg.head_dim / VPT + TPB - 1) / TPB);
        dynamic_batching_kv_cache_quantize_kernel<VPT, TPB><<<grid, TPB, 0, stream>>>(
            {(half*)cfg.current_key,
            (half*)cfg.current_value,
            (int64_t*)cfg.seqstarts,
            (int64_t*)cfg.cachestarts,
            (int64_t*)cfg.start_pos,
            cfg.num_layer,
            cfg.layer_idx,
            cfg.num_kv_heads,
            cfg.head_dim,
            cfg.k_stride_s,
            cfg.v_stride_s,
            cfg.cache_stride_s,
            cfg.cache_stride_l,
            cfg.cache_stride_h,
            cfg.cache_stride_kv,
            (int8_t*)cfg.cache,
            (half*)cfg.scale});
    }

    struct dynamic_batching_decoding_cache_attention_kernel_param p{0};
    p.query = (half*)cfg.query;
    p.attn_mask = (half*)cfg.attn_mask;
    p.output = (half*)cfg.output;
    p.cache = (int8_t*)cfg.cache;
    p.scale = (half*)cfg.scale;
    p.cachestarts = (int64_t*)cfg.cachestarts;
    p.kvstarts = (int64_t*)cfg.kvstarts;
    p.attn_scale = cfg.attn_scale;
    p.layer_idx = cfg.layer_idx;
    p.num_kv_repeats = cfg.num_kv_repeats;
    p.query_stride_s = cfg.q_stride_s;
    p.output_stride_s = cfg.o_stride_s;
    p.mask_stride_s = cfg.mask_stride_s;
    p.mask_stride_h = cfg.mask_stride_h;
    p.cache_stride_s = cfg.cache_stride_s;
    p.cache_stride_l = cfg.cache_stride_l;
    p.cache_stride_h = cfg.cache_stride_h;
    p.cache_stride_kv = cfg.cache_stride_kv;

    if(cfg.decoding_batches > 0) {
        const int64_t MAX_SHM_SIZE = cfg.device_prop->sharedMemPerBlockOptin;

        if (cfg.decoding_shm_size <= MAX_SHM_SIZE) {
            ppl::common::RetCode status = ppl::common::RC_UNSUPPORTED;
            if (cfg.decoding_multi_block_size > 1) {
                p.multi_block.partial_out   = (half*)cfg.temp_buffer;
                p.multi_block.log_sum_exp   = reinterpret_cast<float*>((char*)cfg.temp_buffer
                    + cfg.decoding_multi_block_output_size);
                p.multi_block.block_counter = reinterpret_cast<int32_t*>((char*)cfg.temp_buffer
                    + cfg.decoding_multi_block_output_size
                    + cfg.decoding_multi_block_sum_size);
                cudaMemsetAsync(p.multi_block.block_counter, 0, cfg.decoding_multi_block_counter_size, stream);
                status = dynamic_batching_decoding_cache_attention<256, true>(stream, cfg, p);
            } else if (cfg.decoding_threads_per_block == 256) {
                status = dynamic_batching_decoding_cache_attention<256, false>(stream, cfg, p);
            } else if (cfg.decoding_threads_per_block == 512) {
                status = dynamic_batching_decoding_cache_attention<512, false>(stream, cfg, p);
            } else if (cfg.decoding_threads_per_block == 1024) {
                status = dynamic_batching_decoding_cache_attention<1024, false>(stream, cfg, p);
            }
            if (status != ppl::common::RC_SUCCESS) {
                LOG(ERROR) << "unsupported decoding_multi_block_size and decoding_threads_per_block";
                return ppl::common::RC_UNSUPPORTED;
            }
        } else {
            LOG(ERROR) << "shm not enough, cache flash decoding attention is unsupported.";
            return ppl::common::RC_UNSUPPORTED;
        }
    }

    if (cfg.prefill_batches > 0) {
        const int64_t custom_mask_type = cfg.is_causal ? 1 : 0;
        const void* prefill_seqstart_q = ((int64_t*)cfg.seqstarts) + cfg.decoding_batches;

        return llm::cuda::xformer::fmha(
            stream,
            *cfg.device_prop,
            cfg.query_shape->GetDataType(),
            cfg.query,
            cfg.current_key,
            cfg.current_value,
            cfg.attn_mask,
            prefill_seqstart_q,
            prefill_seqstart_q,
            cfg.prefill_batches,
            0, cfg.q_stride_s, cfg.head_dim,
            0, cfg.k_stride_s, cfg.head_dim,
            0, cfg.v_stride_s, cfg.head_dim,
            0, cfg.mask_stride_s, cfg.mask_stride_h,
            cfg.o_stride_s,
            cfg.max_seqlen,
            cfg.max_kvlen,
            cfg.num_heads,
            cfg.num_kv_heads,
            cfg.head_dim,
            custom_mask_type,
            cfg.attn_scale,
            cfg.output
        );
    } else {
        return ppl::common::RC_SUCCESS;
    }
}

}}}}}
