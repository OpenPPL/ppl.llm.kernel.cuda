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

#include "ppl/kernel/llm/cuda/pmx/i4f16/row_parallel_linear.h"
#include "ppl/kernel/llm/cuda/pmx/i4f16/gemm.h"
#include "ppl/common/log.h"

#include <cuda_fp16.h>

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx { namespace i4f16 {

ppl::common::RetCode row_parallel_linear(
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
    const bool input_is_parallel,
    void* split_buffer,
    const int64_t gemm_workspace_size,
    void* gemm_workspace,
    const ppl::common::TensorShape* output_shape,
    void* output)
{
    if (!input_is_parallel) {
        LOG(ERROR) << "currnetly only support parallel input";
        return ppl::common::RC_UNSUPPORTED;
    }

    // input (M, K/w)
    // weight (N, K/w)
    // output (M, N)

    const int64_t M = input_shape->CalcElementsToDimensionExcludingPadding(input_shape->GetDimCount() - 1);
    const int64_t N = out_features;
    const int64_t Kw = in_features / nccl_param->size;

    ppl::common::RetCode status;

    status = i4f16::gemm(
        stream,
        handle,
        input,
        weight,
        weight_scale,
        bias,
        M,
        N,
        Kw,
        gemm_workspace_size,
        gemm_workspace,
        output);

    if (ppl::common::RC_SUCCESS != status)
        return status;

    if (nccl_param->size > 1) {
        return ppl::common::NcclAllReduceSum<half>(
            (half*)output,
            (half*)output,
            M * N,
            nccl_param,
            stream);
    }

    return ppl::common::RC_SUCCESS;
}

}}}}}}
