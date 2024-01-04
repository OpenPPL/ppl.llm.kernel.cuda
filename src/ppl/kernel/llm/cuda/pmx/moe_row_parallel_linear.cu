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

#include "ppl/kernel/llm/cuda/pmx/moe_row_parallel_linear.h"
#include "ppl/common/log.h"

#include <cuda_fp16.h>

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

ppl::common::RetCode moe_row_parallel_linear(
    const cudaStream_t stream,
    const cublasLtHandle_t& cublaslt_handle,
    const cublasLtMatmulAlgo_t* algo,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* weight_shape,
    const void* weight,
    const ppl::common::TensorShape* bias_shape,
    const void* bias,
    const ppl::common::TensorShape* offset_shape,
    const void* expert_offset,
    const int64_t in_features,
    const int64_t out_features,
    const ppl::common::NcclParam* nccl_param,
    const bool input_is_parallel,
    void* split_buffer,
    const int64_t cublas_workspace_size,
    void* cublas_workspace,
    const ppl::common::TensorShape* output_shape,
    void* output)
{
    if (!input_is_parallel) {
        LOG(ERROR) << "currnetly only support parallel input";
        return ppl::common::RC_UNSUPPORTED;
    }

    // input [seqlen * num_experts_per_token, hidden_dim/w]
    // weight [num_experts_per_token, hidden_dim_out, hidden_dim/w]
    // output [seqlen * num_experts_per_token, hidden_dim_out]

    const int64_t M = input_shape->CalcElementsToDimensionExcludingPadding(input_shape->GetDimCount() - 1);
    const int64_t N = out_features;
    const int64_t Kw = in_features / nccl_param->size;
    const int64_t* offset64_ptr = (const int64_t*)expert_offset;
    const int64_t num_experts = weight_shape->GetDim(0);
    const void *bias_ = nullptr;
    ppl::common::RetCode status;
    for (int i = 0; i < num_experts; ++i) {
        const int64_t start = offset64_ptr[i];
        const int64_t end = offset64_ptr[i + 1];
        if (end - start <= 0) {
            continue;
        }
        if (bias != nullptr) {
            bias_ = (char*)bias + start * N * ppl::common::GetSizeOfDataType(bias_shape->GetDataType());
        }
        const void *input_ = (char*)input + start * Kw * ppl::common::GetSizeOfDataType(input_shape->GetDataType());
        const void *weight_ = (char*)weight + start * N * Kw * ppl::common::GetSizeOfDataType(weight_shape->GetDataType());
        void *gemm_output_ = output + start * N * ppl::common::GetSizeOfDataType(output_shape->GetDataType());
        status = ppl::kernel::llm::cuda::cublas::gemm(
            stream,
            cublaslt_handle,
            algo,
            false,
            Kw,
            input_shape->GetDataType(),
            input_,
            true,
            Kw,
            weight_shape->GetDataType(),
            weight_,
            bias_,
            end - start,
            N,
            Kw,
            1.0f,
            0.0f,
            cublas_workspace_size,
            cublas_workspace,
            N,
            output_shape->GetDataType(),
            gemm_output_);
    
    }


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

}}}}}
