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

#ifndef __PPL_KERNEL_LLM_CUDA_PMX_MULTI_HEAD_CACHE_ATTENTION_H__
#define __PPL_KERNEL_LLM_CUDA_PMX_MULTI_HEAD_CACHE_ATTENTION_H__

#include "ppl/kernel/llm/cuda/common/general_include.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

class dynamic_batching_multi_head_cache_attention {
public:
    typedef int32_t decoding_algo_t;

    struct decoding_algo {
        static const decoding_algo_t UNKNOWN = 0;
        static const decoding_algo_t SHAREMEM_MHCA = 1;
        static const decoding_algo_t INFINITY_MHCA = 2;
        static const decoding_algo_t INFINITY_GQCA = 3;
    };

    struct config {
        cudaDeviceProp* device_prop;

        ppl::common::datatype_t datatype;

        void* query;
        void* current_key;
        void* current_value;
        void* attn_mask;

        void* seqstarts;
        void* kvstarts;
        void* cachestarts;
        void* start_pos;
        void* alibi_slopes;

        void* cache;
        void* scale;

        void* output;

        bool is_causal;
        int64_t batch;
        int64_t decoding_batches;
        int64_t max_seqlen;
        int64_t max_kvlen;
        int64_t layer_idx;
        int64_t num_layer;
        int64_t num_heads;
        int64_t num_kv_heads;
        int64_t head_dim;
        int32_t cache_mode;
        int32_t page_size;
        int32_t page_shift;
        int32_t page_mask;
        int64_t cache_stride_s;
        int64_t cache_stride_l;
        int64_t cache_stride_h;
        int64_t cache_stride_kv;
        int64_t cachestarts_stride_b;

        // produce by prepare function
        int64_t prefill_batches;
        int64_t q_stride_s;
        int64_t k_stride_s;
        int64_t v_stride_s;
        int64_t o_stride_s;

        int64_t mask_stride_s;
        int64_t mask_stride_h;

        float attn_scale;
        int64_t num_kv_repeats;
        decoding_algo_t decoding_algo;

        int64_t decoding_threads_per_block;
        int64_t decoding_multi_block_size;
        int64_t decoding_multi_block_partial_out_size;
        int64_t decoding_multi_block_partial_log_sum_exp_size;
        int64_t decoding_multi_block_counter_size;

        int64_t workspace_size;
        void *workspace;

        bool enable_cache_prefill;
    } cfg {0};


    // must call it before any forward
    //
    // remember to set workspace by workspace size after call it/before call forward
    ppl::common::RetCode heuristic_prepare(
        const cudaDeviceProp& device_prop,
        const ppl::common::TensorShape* query_shape,
        const void* query, // (Sq, ..., D)
        const ppl::common::TensorShape* current_key_shape,
        const void* current_key, // (Skv, ..., D)
        const ppl::common::TensorShape* current_value_shape,
        const void* current_value, // (Skv, ..., D)
        const ppl::common::TensorShape* attn_mask_shape,
        const void* attn_mask, // (seqstarts[-1], aligned(kvstarts[-1], 8)), (num_heads, seqstarts[-1], aligned(kvstarts[-1], 8))
        const void* seqstarts, // (B + 1)
        const void* kvstarts, // (B + 1)
        const void* cachestarts, // (B)
        const void* start_pos, // (B)
        const void* alibi_slopes, // (num_head)
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
        const int64_t page_size,
        const int64_t cache_stride_s,
        const int64_t cache_stride_l,
        const int64_t cache_stride_h,
        const int64_t cache_stride_kv,
        const int64_t cachestarts_stride_b,
        const bool enable_cache_prefill,
        const bool enable_sharemem_mhca,
        const bool enable_infinity_mhca,
        const bool enable_infinity_gqca,
        const int32_t specify_decoding_multi_block, // 0 for off, 1 for heuristic, 2 for always on
        const int32_t specify_decoding_tpb, // 0 for not specify, only accept 256,512
        void* cache, // int8 (S, L, 2, KVH, D), (L, KVH, S, 2, D)
        void* scale, // float16 (S, L, 2, KVH, D/8), (L, KVH, S, 2, D/8)
        const ppl::common::TensorShape* output_shape,
        void* output); // (S, .., D)

    // per stage forward
    // kvstore must be the first stage
    ppl::common::RetCode forward_kvstore(const cudaStream_t stream);
    ppl::common::RetCode forward_decode(const cudaStream_t stream);
    ppl::common::RetCode forward_prefill(const cudaStream_t stream);


    // all in one forward
    ppl::common::RetCode forward(const cudaStream_t stream);
};

}}}}}

#endif
