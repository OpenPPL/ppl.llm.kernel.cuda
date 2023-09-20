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

#include "ppl/kernel/llm/cuda/pmx/column_parallel_linear.h"
#include "ppl/common/log.h"

#include "../../../../../llm/xformer_fmha/xformer_fmha.h"
#include "cudakernel/common/common.cuh"

#include <cuda_fp16.h>
#include <float.h> // need for FLT_MAX

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

struct decoding_cache_attention_param {
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
};


template<int32_t HEAD_SIZE, int32_t VPT> // 8 fp16 occupy 128 bytes, which can be loaded by a single thread at once.
__global__
void dynamic_batching_kv_cache_quantize_kernel(
    const half* current_key, // (S, KVH..., D)
    const half* current_value, // (S, KVH..., D)
    const int64_t* seqstarts, // (B + 1)
    const int64_t* cachestarts,// (B)
    const int64_t* start_pos, // (B)
    const int64_t num_layer,
    const int64_t layer_idx,
    const int64_t num_kv_heads,
    const int64_t current_key_stride_s,
    const int64_t current_value_stride_s,
    const int64_t cache_stride_s,
    const int64_t cache_stride_l,
    const int64_t cache_stride_h,
    const int64_t cache_stride_kv,
    int8_t* cache,
    half* scale)
{
    if (blockIdx.y < seqstarts[blockIdx.x + 1] - seqstarts[blockIdx.x]) {
        constexpr int64_t thd_per_head_size = HEAD_SIZE / VPT;
        const int64_t batch_id = blockIdx.x;
        const int64_t seq_idx = blockIdx.y;
        const int64_t input_token_idx = seqstarts[batch_id] + seq_idx;
        const int64_t cache_token_idx = cachestarts[batch_id] + seq_idx + start_pos[batch_id];
        const int64_t key_out_offset = cache_token_idx * cache_stride_s + layer_idx * cache_stride_l;
        auto key_in_ptr = current_key + input_token_idx * current_key_stride_s;
        auto value_in_ptr = current_value + input_token_idx * current_value_stride_s;

        for (int32_t tid = threadIdx.x; tid < HEAD_SIZE * num_kv_heads / VPT; tid += blockDim.x) {
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
                + kv_head_idx * cache_stride_h
                + dim_idx;
            const int64_t value_out_idx
                = key_out_idx
                + cache_stride_kv;

            const int64_t key_scale_out_idx = (key_out_idx - dim_idx) / VPT + scale_dim_idx;
            const int64_t value_scale_out_idx = key_scale_out_idx + cache_stride_kv / VPT;

            // calculate kv scale
            const half eps = 1e-5f;
            const half fact = 127.f;
            half key_scale = 0.0f;
            half value_scale = 0.0f;

            #pragma unroll
            for (int32_t i = 0; i < VPT; i ++){
                key_scale = key_scale > __habs(key_in[i]) ? key_scale : __habs(key_in[i]);
                value_scale = value_scale > __habs(value_in[i]) ? value_scale : __habs(value_in[i]);
            }

            key_scale = key_scale / fact; 
            value_scale = value_scale / fact;
            key_scale = key_scale > eps ? key_scale : eps;
            value_scale = value_scale > eps ? value_scale : eps;

            #pragma unroll
            for (int32_t i = 0; i < VPT; i ++){
                key_out[i] = (int8_t)__half2short_rn(key_in[i] / key_scale);
                value_out[i] = (int8_t)__half2short_rn(value_in[i] / value_scale);
            }

            copy<sizeof(int8_t) * VPT>(key_out, &cache[key_out_idx]);
            copy<sizeof(int8_t) * VPT>(value_out, &cache[value_out_idx]);

            scale[key_scale_out_idx] = key_scale;
            scale[value_scale_out_idx] = value_scale;
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
    bool SHIFT_KV>
__global__
void dynamic_batching_decoding_cache_attention_fp16_kernel(struct decoding_cache_attention_param p)
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
    const int64_t head_idx      = blockIdx.x;
    const int64_t batch_idx     = blockIdx.y;
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
    const int64_t context_len = p.kvstarts[batch_idx + 1] - p.kvstarts[batch_idx];
    extern __shared__ float logits[];
    float qk_max = -FLT_MAX;

    for (int64_t base_id = warp_id * GPW; base_id < context_len; base_id += GPT) {
        int8_t local_k_quant[VEC_SIZE * VEC_LEN];
        half local_k[VEC_SIZE * VEC_LEN];
        half local_k_scale[VEC_LEN];
        const int64_t context_id = base_id + group_id;

        // all thread groups within a warp must be launched together.
        if (context_id >= context_len){
            memset(local_k, 0, sizeof(local_k));
        } else {
            const int64_t key_offset
                            = (cache_offset_s + context_id) * p.cache_stride_s
                            + p.layer_idx * p.cache_stride_l
                            + kv_head_idx * p.cache_stride_h
                            + group_lane_id * VEC_SIZE;
            #pragma unroll
            for (int64_t i = 0; i < VEC_LEN; i++) {
                // copy 128(16 * 8) bits from K to Local K
                const int64_t key_idx = key_offset + i * THREAD_GROUP_SIZE * VEC_SIZE;
                copy<sizeof(int8_t) * VEC_SIZE>(&p.cache[key_idx],  &local_k_quant[i * VEC_SIZE]);
                #pragma unroll
                for(int k = i * VEC_SIZE; k < (i + 1) * VEC_SIZE; k++) {
                    local_k_quant[k] += 128;
                }

                const int64_t key_scale_idx = key_idx >> QUANT_GROUP_SHIFT;
                local_k_scale[i] = p.scale[key_scale_idx];
            }


            #pragma unroll
            for (int64_t i = 0; i < VEC_LEN; i++) {
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
                    local_k[i * VEC_SIZE + j] = local_k_scale[i] * result[j];
                }
            }
        }

        // Ready for QK Dot
        const float qk_dot
            = p.attn_scale
            * attn_thread_group_dot<THREAD_GROUP_SIZE, VEC_LEN * VEC_SIZE>(local_q, local_k);

        if (group_lane_id == 0 && context_id < context_len) {
            logits[context_id] = qk_dot;
            qk_max = fmaxf(qk_dot, qk_max);
        }
    }

    // ------------------------------------------------ //
    // Step 3. Softmax

    // The process of solving softmax is divided into two stages. 
    // First, we need to reduce qk_max in two dimensions: WARP and ThreadBlock. 
    // Afterward, we use reduced qk_max to perform softmax calculations,
    //    the results will all be stored in shared memory.
    __shared__ float red_smem[WPT];

    // reduce qk_max in thread block and boardcast
    qk_max = attn_block_reduce_max<WPT>(qk_max, red_smem);

    // Softmax Kernel Logic Start here
    float exp_sum = 0.0f;
    for (int64_t context_id = threadIdx.x; context_id < context_len; context_id += TPB){
        logits[context_id] -= qk_max;
        logits[context_id] = exp(logits[context_id]);
        exp_sum += logits[context_id];
    }

    // block reduce sum on exp_sum
    // Warp per thread block must be power-of-2 for reducation, check attn_block_reduce_sum kernel.
    static_assert(WPT == 2 || WPT == 4 || WPT == 8 || WPT == 16 || WPT == 32 || WPT == 64);
    exp_sum = attn_block_reduce_sum<WPT>(exp_sum, red_smem);

    const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
    for (int64_t context_id = threadIdx.x; context_id < context_len; context_id += TPB) {
        logits[context_id] *= inv_sum;
    }
    __syncthreads(); // Must have this.

    // ------------------------------------------------ //
    // Step 4. Solve logits * V

    int8_t local_v_quant[VEC_SIZE * VEC_LEN];
    float local_v[VEC_SIZE * VEC_LEN];
    half local_v_scale[VEC_LEN];

    #pragma unroll
    for(int32_t i = 0; i < VEC_SIZE * VEC_LEN; i++) {
        local_v[i] = 0;
    }

    for (int64_t base_id = warp_id * GPW; base_id < context_len; base_id += GPT) {
        const int64_t context_id = base_id + group_id;
        // all thread groups within a warp must be launched together.
        if (context_id < context_len){
            const int64_t value_offset
                            = (cache_offset_s + context_id) * p.cache_stride_s
                            + p.layer_idx * p.cache_stride_l
                            + kv_head_idx * p.cache_stride_h
                            + group_lane_id * VEC_SIZE
                            + p.cache_stride_kv;
            #pragma unroll
            for (int64_t i = 0; i < VEC_LEN; i++) {
                // copy 128(16 * 8) bits from V to Local V
                const int64_t value_idx = value_offset + i * THREAD_GROUP_SIZE * VEC_SIZE;
                copy<sizeof(int8_t) * VEC_SIZE>(&p.cache[value_idx],  &local_v_quant[i * VEC_SIZE]);
                #pragma unroll
                for(int k = i * VEC_SIZE; k < (i + 1) * VEC_SIZE; k++) {
                    local_v_quant[k] += 128;
                }

                const int64_t value_scale_idx = value_idx >> QUANT_GROUP_SHIFT;
                local_v_scale[i] = p.scale[value_scale_idx];
            }

            #pragma unroll
            for (int64_t i = 0; i < VEC_LEN; i++) {
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
                    local_v[i * VEC_SIZE + j] += __half2float(local_v_scale[i] * result[j]) * logits[context_id];
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
    __syncthreads();

    // do some reuse
    for (int64_t i = threadIdx.x; i < HEAD_SIZE; i += TPB){
        logits[i] = 0;
    }

    __syncthreads();

    if (warp_lane_id < THREAD_GROUP_SIZE) {
        #pragma unroll
        for (int32_t i = 0; i < VEC_LEN; i++) {
            #pragma unroll
            for (int32_t j = 0; j < VEC_SIZE; j++) {
                atomicAdd(
                    logits + i * THREAD_GROUP_SIZE * VEC_SIZE + warp_lane_id * VEC_SIZE + j,
                    local_v[i * VEC_SIZE + j]
                );
            }
        }
    }

    __syncthreads();

    for (int64_t i = threadIdx.x; i < HEAD_SIZE; i += TPB){
        p.output[batch_idx * p.output_stride_s + head_idx * HEAD_SIZE + i] = logits[i];
    }
}



ppl::common::RetCode dynamic_batch_multi_head_cache_attention(
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

    const int64_t q_num = query_shape->GetDim(0);
    const int64_t kv_num = current_key_shape->GetDim(0);

    const int64_t decodeing_q_num = decoding_batches;
    const int64_t prefill_qkv_num = q_num - decodeing_q_num;
    const float attn_scale = float(1.0 / std::sqrt(float(head_dim)));

    const int64_t q_stride_s = query_shape->GetDim(1) * head_dim;
    const int64_t k_stride_s = current_key_shape->GetDim(1) * head_dim;
    const int64_t v_stride_s = current_value_shape->GetDim(1) * head_dim;
    const int64_t o_stride_s = output_shape->GetDim(1) * head_dim;

    struct decoding_cache_attention_param p;
    p.query = (half*)query;
    p.output = (half*)output;
    p.cache = (int8_t*)cache;
    p.scale = (half*)scale;
    p.cachestarts = (int64_t*)cachestarts;
    p.kvstarts = (int64_t*)kvstarts;
    p.attn_scale = attn_scale;
    p.layer_idx = layer_idx;
    p.kv_head_shift = kv_head_shift;
    p.num_kv_repeats = kv_repeats;
    p.query_stride_s = q_stride_s;
    p.output_stride_s = o_stride_s;
    p.cache_stride_s = cache_stride_s;
    p.cache_stride_l = cache_stride_l;
    p.cache_stride_h = cache_stride_h;
    p.cache_stride_kv = cache_stride_kv;

    if (num_heads != num_kv_heads) {
        LOG(ERROR) << "currently do not support GQA, whose num_heads != num_kv_heads";
        return ppl::common::RC_UNSUPPORTED;
    }

    {
        if (head_dim != 128) {
            LOG(ERROR) << "currently kv_cache_quantize_kernel only support head_dim == 128.";
            return ppl::common::RC_UNSUPPORTED;
        }
        dim3 grid(batch, max_seqlen);
        const int32_t block_size = GetBlockSize(num_kv_heads * head_dim / 8);
        dynamic_batching_kv_cache_quantize_kernel<128, 8><<<grid, block_size, 0, stream>>>(
            (half*)current_key,
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
            (half*)scale);
    }

    if(decodeing_q_num > 0) {
        constexpr int64_t WARP_SIZE = 32;
        constexpr int64_t TPB = 256;
        constexpr int64_t MAX_SHM_SIZE = 48 * 1024;

        constexpr int64_t reduce_shm_size = TPB / WARP_SIZE * sizeof(float);
        const int64_t logits_size = max(max_kvlen * sizeof(float), head_dim * sizeof(float));

        if (reduce_shm_size + logits_size <= MAX_SHM_SIZE) {
            const dim3 grid_size = {(unsigned int)num_heads, (unsigned int)decodeing_q_num, 1};
            switch (head_dim){
                case 64:
                    dynamic_batching_decoding_cache_attention_fp16_kernel<64, 4, 256, 8, true>
                    <<<grid_size, 256, logits_size, stream>>>(p);
                    break;
                case 96:
                    dynamic_batching_decoding_cache_attention_fp16_kernel<96, 4, 256, 8, true>
                    <<<grid_size, 256, logits_size, stream>>>(p);
                    break;
                case 128:
                    dynamic_batching_decoding_cache_attention_fp16_kernel<128, 8, 256, 8, true>
                    <<<grid_size, 256, logits_size, stream>>>(p);
                    break;
                case 256:
                    dynamic_batching_decoding_cache_attention_fp16_kernel<256, 16, 256, 8, true>
                    <<<grid_size, 256, logits_size, stream>>>(p);
                    break;
                default:
                    LOG(ERROR) << "cache flash decoding attention do not support head dim " << head_dim;
                    return ppl::common::RC_UNSUPPORTED;
            }
        } else {
            LOG(ERROR) << "shm not enough, cache flash decoding attention is unsupported.";
            return ppl::common::RC_UNSUPPORTED;
        }
    }

    if(prefill_qkv_num > 0) {
        ppl::common::TensorShape prefill_query_shape(*query_shape);
        ppl::common::TensorShape prefill_key_shape(*current_key_shape);
        ppl::common::TensorShape prefill_seqstart_q_shape;

        prefill_query_shape.Reshape({1, q_num, num_heads, head_dim});
        prefill_key_shape.Reshape({1, kv_num, num_kv_heads, head_dim});
        prefill_seqstart_q_shape.Reshape({batch + 1 - decodeing_q_num});

        const int64_t custom_mask_type = is_causal ? 1 : 0;
        const void* prefill_seqstart_q = ((int64_t*)seqstarts) + decodeing_q_num;

        PPLCUDAFMHAForwardImp(device_prop, stream,
            &prefill_query_shape, query,
            &prefill_key_shape, current_key,
            &prefill_key_shape, current_value,
            nullptr, nullptr,
            &prefill_seqstart_q_shape, prefill_seqstart_q,
            &prefill_seqstart_q_shape, prefill_seqstart_q,
            nullptr, nullptr,
            max_seqlen, custom_mask_type, attn_scale,
            &prefill_query_shape, output);

        return ppl::common::RC_SUCCESS;
    } else {
        return ppl::common::RC_SUCCESS;
    }
}

}}}}}
