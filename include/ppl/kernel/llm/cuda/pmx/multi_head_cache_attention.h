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

uint64_t CalcMultiBlockSize(const int64_t grid_size, const int64_t max_kvlen, const int64_t head_dim);

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
    void* multi_block_counter,
    void* multi_block_max,
    void* multi_block_sum,
    void* multi_block_out,
    void* cache, // int8 (S, L, 2, KVH, D), (L, KVH, S, 2, D)
    void* scale, // float16 (S, L, 2, KVH, D/8), (L, KVH, S, 2, D/8)
    const ppl::common::TensorShape* output_shape,
    void* output); // (S, .., D)

}}}}}

#endif
