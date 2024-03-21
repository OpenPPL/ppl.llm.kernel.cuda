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

#ifndef __PPL_KERNEL_LLM_CUDA_PMX_MOE_SELECT_H__
#define __PPL_KERNEL_LLM_CUDA_PMX_MOE_SELECT_H__

#include "ppl/kernel/llm/cuda/common/general_include.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

struct moe_select_config {
    int64_t sorted_expert_ids_size;
    int64_t expert_ids_size;
    int64_t source_row_size;
    int64_t permute_token_idx_size;
    int64_t sort_buffer_size;
    int64_t temp_buffer_size;
};

moe_select_config moe_select_prepare(const ppl::common::TensorShape* invert_permutation_shape, const int64_t num_experts);

ppl::common::RetCode moe_select(
    const cudaStream_t stream,
    const ppl::common::TensorShape* x_shape,
    const void* x,
    const ppl::common::TensorShape* scores_shape,
    const void* scores,
    const int64_t num_experts,
    const int64_t num_experts_per_token,
    const moe_select_config& config,
    void* temp_buffer,
    void* x_expand_permute,
    void* expert_weights,
    void* invert_permutation,
    void* expert_offset);

}}}}}

#endif