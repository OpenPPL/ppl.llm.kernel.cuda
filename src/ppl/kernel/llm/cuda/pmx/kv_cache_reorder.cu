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

#include "ppl/kernel/llm/cuda/pmx/kv_cache_reorder.h"
#include "cudakernel/common/common.cuh"
#include "ppl/common/log.h"
#include <cuda_fp16.h>
#include <iostream>

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

__global__ void kv_cache_reorder_kernel(
    const char* kv_cache,
    const char* kv_scale,
    const Stride src_stride,
    const Stride dst_stride,
    const int64_t* page_ids,
    int64_t page_nums,
    int64_t head_dim,
    int64_t group_size,
    int64_t tp_rank,
    int64_t tp_size,
    char* outer_mem
) {
    int idx_l = blockIdx.y;
    int idx_kv = threadIdx.y;
    int idx_h = blockIdx.z / page_nums;
    int idx_t = blockIdx.x;
    int idx_d = threadIdx.x;
    int page_id_offset = blockIdx.z % page_nums;
    int page_id = page_ids[page_id_offset];

    int page_cache_bytes = src_stride.p_stride_cache;
    int page_scale_bytes = src_stride.p_stride_scale;
    int page_size = blockDim.x;

    int src_cache_idx = page_id * page_size * head_dim + idx_l * src_stride.l_stride_cache + idx_kv * src_stride.kv_stride_cache + idx_h * src_stride.h_stride_cache + idx_t * src_stride.t_stride_cache + idx_d * src_stride.d_stride_cache;
    int dst_cache_idx = idx_d * dst_stride.d_stride_cache + idx_t * dst_stride.t_stride_cache + idx_kv * dst_stride.kv_stride_cache + idx_l * dst_stride.l_stride_cache + idx_h * dst_stride.h_stride_cache;

    char* outer_cache = outer_mem + (page_cache_bytes + page_scale_bytes) * tp_size * page_id_offset + tp_rank * page_cache_bytes;
    outer_cache[dst_cache_idx] = kv_cache[src_cache_idx];

    if (idx_d >= head_dim / group_size * 2) {
        return;
    }
    int src_scale_idx = page_id * page_size * (head_dim / group_size * 2) + idx_l * src_stride.l_stride_scale + idx_kv * src_stride.kv_stride_scale + idx_h * src_stride.h_stride_scale + idx_t * src_stride.t_stride_scale + idx_d * src_stride.d_stride_scale;
    int dst_scale_idx = idx_d * dst_stride.d_stride_scale + idx_t * dst_stride.t_stride_scale + idx_kv * dst_stride.kv_stride_scale + idx_l * dst_stride.l_stride_scale + idx_h * dst_stride.h_stride_scale;
    char* outer_scale = outer_cache + (tp_size - tp_rank) * page_cache_bytes + tp_rank * page_scale_bytes;
    outer_scale[dst_scale_idx] = kv_scale[src_scale_idx];
}

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
) {
    const dim3 block(head_dim, 2);
    const dim3 grid(tokens, layers, kv_heads * page_nums);

    kv_cache_reorder_kernel<<<grid, block, 0, stream>>>(
        (char*)kv_cache,
        (char*)kv_scale,
        src_stride,
        dst_stride,
        (int64_t*)page_ids,
        page_nums,
        head_dim,
        group_size,
        tp_rank,
        tp_size,
        (char*)outer_mem
    );

    return ppl::common::RC_SUCCESS;
}


__global__ void kv_cache_invert_reorder_kernel(
    const char* outer_mem,
    const Stride src_stride,
    const Stride dst_stride,
    const int64_t* page_ids,
    int64_t page_nums,
    int64_t head_dim,
    int64_t group_size,
    int64_t tp_rank,
    int64_t tp_size,
    char* kv_cache,
    char* kv_scale
) {
    int idx_l = blockIdx.y;
    int idx_kv = threadIdx.y;
    int idx_h = blockIdx.z / page_nums;
    int idx_t = blockIdx.x;
    int idx_d = threadIdx.x;

    int page_id_offset = blockIdx.z % page_nums;
    int page_id = page_ids[page_id_offset];
    int page_size = blockDim.x;

    int page_cache_bytes = src_stride.p_stride_cache;
    int page_scale_bytes = src_stride.p_stride_scale;

    int src_cache_idx = idx_l * src_stride.l_stride_cache + idx_kv * src_stride.kv_stride_cache + idx_h * src_stride.h_stride_cache + idx_t * src_stride.t_stride_cache + idx_d * src_stride.d_stride_cache;
    int dst_cache_idx = page_id * page_size * head_dim + idx_d * dst_stride.d_stride_cache + idx_t * dst_stride.t_stride_cache + idx_kv * dst_stride.kv_stride_cache + idx_l * dst_stride.l_stride_cache + idx_h * dst_stride.h_stride_cache;

    const char* input_cache = outer_mem + (page_cache_bytes + page_scale_bytes) * tp_size *  page_id_offset + tp_rank * page_cache_bytes;
    kv_cache[dst_cache_idx] = input_cache[src_cache_idx];

    if (idx_d >= head_dim / group_size * 2) {
        return;
    }

    int src_scale_idx = idx_l * src_stride.l_stride_scale + idx_kv * src_stride.kv_stride_scale + idx_h * src_stride.h_stride_scale + idx_t * src_stride.t_stride_scale + idx_d * src_stride.d_stride_scale;
    int dst_scale_idx = page_id * page_size * (head_dim / group_size * 2) + idx_d * dst_stride.d_stride_scale + idx_t * dst_stride.t_stride_scale + idx_kv * dst_stride.kv_stride_scale + idx_l * dst_stride.l_stride_scale + idx_h * dst_stride.h_stride_scale;
    const char* input_scale = input_cache + (tp_size - tp_rank) * page_cache_bytes + tp_rank * page_scale_bytes;
    kv_scale[dst_scale_idx] = input_scale[src_scale_idx];
}


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
) {
    const dim3 block(head_dim, 2);
    const dim3 grid(tokens, layers, kv_heads * page_nums);

    kv_cache_invert_reorder_kernel<<<grid, block, 0, stream>>>(
        (char*)outer_mem,
        src_stride,
        dst_stride,
        (int64_t*)page_ids,
        page_nums,
        head_dim,
        group_size,
        tp_rank,
        tp_size,
        (char*)kv_cache,
        (char*)kv_scale
    );
    return ppl::common::RC_SUCCESS;
}

}}}}}
