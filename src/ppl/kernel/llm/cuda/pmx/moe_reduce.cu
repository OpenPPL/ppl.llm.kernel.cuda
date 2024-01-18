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

#include "ppl/kernel/llm/cuda/pmx/moe_reduce.h"
#include "cudakernel/common/common.cuh"
#include "ppl/common/log.h"

#include <cub/cub.cuh>
#include <cuda_fp16.h>

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

__global__ 
void moe_reduce_sum_kernel(
    const half* y_expand_permute,  
    const half* expert_weights, 
    const int64_t* invert_permutation, 
    const int64_t cols, 
    const int64_t num_experts_per_token, 
    half* y_reduced) 
{
    for(int col_id=threadIdx.x; col_id < cols; col_id += blockDim.x) {

        const int64_t origin_row = blockIdx.x;
        half* reduced_row_ptr = y_reduced + origin_row * cols;

        half thread_output = __float2half(0.0f);

        for(int k_idx = 0; k_idx < num_experts_per_token; ++k_idx) {
            const int64_t expand_origin_row = blockIdx.x * num_experts_per_token + k_idx;
            const int64_t expand_permuted_row = invert_permutation[expand_origin_row];
            
            const half* expanded_permuted_row_ptr = y_expand_permute + expand_permuted_row * cols;

            const half row_scale = expert_weights[expand_origin_row];
            
            thread_output = thread_output + row_scale * expanded_permuted_row_ptr[col_id];
        }

        reduced_row_ptr[col_id] = thread_output;
    }
}

ppl::common::RetCode moe_reduce(
    const cudaStream_t stream,
    const ppl::common::TensorShape* y_expand_permute_shape, 
    const void* y_expand_permute,   // [tokens, num_experts_per_token, dim]
    const void* expert_weights,     // [tokens, num_experts_per_token]
    const void* invert_permutation, // [tokens, num_experts_per_token]
    const int64_t num_experts_per_token,
    void* y_reduced) 
{
    if (y_expand_permute_shape->GetDim(y_expand_permute_shape->GetDimCount() - 2) != num_experts_per_token) {
        LOG(ERROR) << "Y_expand_permute.shape[-2] != num_experts_per_token";
        return ppl::common::RC_OTHER_ERROR;
    }

    const int64_t expand_tokens = y_expand_permute_shape->CalcElementsToDimensionExcludingPadding(y_expand_permute_shape->GetDimCount() - 1);
    const int64_t tokens = expand_tokens / num_experts_per_token;

    const int64_t dim = y_expand_permute_shape->GetDim(y_expand_permute_shape->GetDimCount() - 1);

    const int TPB = min(1024, (int32_t)dim);
    const int BPG = tokens;
    
    moe_reduce_sum_kernel<<<BPG, TPB, 0, stream>>>(
        (half*)y_expand_permute, 
        (half*)expert_weights,
        (int64_t*)invert_permutation,
        dim,
        num_experts_per_token,
        (half*)y_reduced);

    return ppl::common::RC_SUCCESS;
}

}}}}}