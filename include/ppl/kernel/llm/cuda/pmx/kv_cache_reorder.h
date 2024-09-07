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

#ifndef __PPL_KERNEL_LLM_CUDA_PMX_REORDER_H__
#define __PPL_KERNEL_LLM_CUDA_PMX_REORDER_H__

#include "ppl/kernel/llm/cuda/common/general_include.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

struct Stride {
    int p_stride_cache;
    int l_stride_cache;
    int kv_stride_cache;
    int h_stride_cache;
    int t_stride_cache;
    int d_stride_cache;

    int p_stride_scale;
    int l_stride_scale;
    int kv_stride_scale;
    int h_stride_scale;
    int t_stride_scale;
    int d_stride_scale;
};

ppl::common::RetCode kv_cache_reorder(
    cudaStream_t stream,
    const void* kv_cache,
    const void* kv_scale,
    const void* page_ids,
    const Stride& src_stride,
    const Stride& dst_stride,
    int64_t page_nums,
    int64_t layers,
    int64_t kv_heads,
    int64_t tokens, 
    int64_t head_dim,
    int64_t group_size,
    int64_t tp_rank, 
    int64_t tp_size,
    void* outer_mem
);

ppl::common::RetCode kv_cache_invert_reorder(
    cudaStream_t stream,
    const void* outer_mem,
    const void* page_ids,
    const Stride& src_stride,
    const Stride& dst_stride,
    int64_t page_nums,
    int64_t layers,
    int64_t kv_heads,
    int64_t tokens, 
    int64_t head_dim,
    int64_t group_size,
    int64_t tp_rank, 
    int64_t tp_size,
    void* kv_cache,
    void* kv_scale
);

}}}}}

#endif