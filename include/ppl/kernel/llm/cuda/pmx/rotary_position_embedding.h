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

#ifndef __PPL_KERNEL_LLM_CUDA_PMX_ROTARY_POSITION_EMBEDDING_H__
#define __PPL_KERNEL_LLM_CUDA_PMX_ROTARY_POSITION_EMBEDDING_H__

#include "ppl/kernel/llm/cuda/common/general_include.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

template <typename T>
ppl::common::RetCode rotary_position_embedding(
    cudaStream_t stream,
    const ppl::common::TensorShape* query_shape,
    const void* query, // (batch, seqlen, ..., head_dim), dim[2] is the leading dim of heads
    const ppl::common::TensorShape* key_shape,
    const void* key,  // (batch, seqlen, ..., head_dim), dim[2] is the leading dim of heads
    const void* cu_start_pos,
    const int64_t start_pos,
    const float theta,
    const int32_t bypass_key,
    const int64_t rotary_dim,
    const int64_t num_heads,
    const int64_t num_key_heads,
    const ppl::common::TensorShape* rotated_query_shape,
    void* rotated_query, // (batch, seqlen, ..., head_dim), dim[2] is the leading dim of heads
    const ppl::common::TensorShape* rotated_key_shape,
    void* rotated_key); // (batch, seqlen, ..., head_dim), dim[2] is the leading dim of heads

template <typename T>
ppl::common::RetCode dynamic_batching_rotary_position_embedding(
    cudaStream_t stream,
    const ppl::common::TensorShape* query_shape,
    const void* query, // (seqstarts[-1], ..., head_dim), dim[1] is the leading dim of heads
    const ppl::common::TensorShape* key_shape,
    const void* key, // (seqstarts[-1], ..., head_dim), dim[1] is the leading dim of heads
    const void* start_pos,
    const void* seqstarts,
    const float theta,
    const int32_t bypass_key,
    const int64_t rotary_dim,
    const int64_t batch,
    const int64_t num_heads,
    const int64_t num_key_heads,
    const int64_t max_seqlen,
    const ppl::common::TensorShape* rotated_query_shape,
    void* rotated_query, // (seqstarts[-1], ..., head_dim), dim[1] is the leading dim of heads
    const ppl::common::TensorShape* rotated_key_shape,
    void* rotated_key); // (seqstarts[-1], ..., head_dim), dim[1] of shape may not be num_heads

}}}}}

#endif
