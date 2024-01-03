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

#ifndef __PPL_KERNEL_LLM_CUDA_PMX_I4F16_COLUMN_PARALLEL_LINEAR_H__
#define __PPL_KERNEL_LLM_CUDA_PMX_I4F16_COLUMN_PARALLEL_LINEAR_H__

#include "ppl/kernel/llm/cuda/common/general_include.h"

#include "ppl/common/cuda/nccl_utils.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx { namespace i4f16 {

ppl::common::RetCode column_parallel_linear(
    const cudaStream_t stream,
    const void* handle,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* weight_shape,
    const void* weight,
    const void* weight_scale,
    const ppl::common::TensorShape* bias_shape,
    const void* bias,
    const int64_t in_features,
    const int64_t out_features,
    const ppl::common::NcclParam* nccl_param,
    const bool gather_output,
    void* gather_buffer,
    const int64_t gemm_workspace_size,
    void* gemm_workspace,
    const ppl::common::TensorShape* output_shape,
    void* output);

}}}}}}

#endif
