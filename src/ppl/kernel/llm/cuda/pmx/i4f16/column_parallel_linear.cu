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

#include "ppl/kernel/llm/cuda/pmx/i4f16/column_parallel_linear.h"
#include "ppl/kernel/llm/cuda/pmx/i4f16/gemm.h"
#include "ppl/common/log.h"

#include "cudakernel/memory/transpose.h"

#include <cuda_fp16.h>

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
    void* output)
{
    // input (M, K)
    // weight (N/w, K)
    // gemm_output (M, Nw)
    // output (M, N)

    const int64_t M = input_shape->CalcElementsToDimensionExcludingPadding(input_shape->GetDimCount() - 1);
    const int64_t Nw = out_features / nccl_param->size;
    const int64_t K = in_features;

    void *gemm_output = output;
    if (gather_output && nccl_param->size > 1) {
        gemm_output = (char*)gather_buffer
            + nccl_param->rank * M * Nw * ppl::common::GetSizeOfDataType(output_shape->GetDataType());
    }

    ppl::common::RetCode status;

    status = i4f16::gemm(
        stream,
        handle,
        input,
        weight,
        weight_scale,
        bias,
        M,
        Nw,
        K,
        gemm_workspace_size,
        gemm_workspace,
        gemm_output);

    if (ppl::common::RC_SUCCESS != status)
        return status;

    if (gather_output && nccl_param->size > 1) {
        status = ppl::common::NcclAllGather<half>(
            (half*)gemm_output,
            (half*)gather_buffer,
            M * Nw,
            nccl_param,
            stream);
        if (ppl::common::RC_SUCCESS != status)
            return status;

        // gather_buffer(w, M, N/w)
        status = PPLCUDATranspose01ForwardImp(
            stream, gather_buffer,
            output_shape->GetDataType(),
            nccl_param->size,
            M,
            Nw,
            output);
        if (ppl::common::RC_SUCCESS != status)
            return status;
    }

    return ppl::common::RC_SUCCESS;
}

}}}}}}
