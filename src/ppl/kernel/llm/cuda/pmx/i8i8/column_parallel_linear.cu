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

#include "ppl/kernel/llm/cuda/pmx/i8i8/column_parallel_linear.h"
#include "ppl/common/log.h"

#include "cudakernel/memory/transpose.h"
#include "ppl/kernel/llm/cuda/pmx/i8i8/quantize.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx { namespace i8i8 {

// input should be m*k, CUBLASLT_ORDER_COL32
// weight should be n*k, CUBLASLT_ORDER_COL32_2R_4R4 or CUBLASLT_ORDER_COL4_4R2_8C
// output is m*n, CUBLASLT_ORDER_COL32
ppl::common::RetCode column_parallel_linear(
    const cudaStream_t stream,
    const cublasLtHandle_t& cublaslt_handle,
    const cublasLtMatmulAlgo_t* algo,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* weight_shape,
    const void* weight,
    const ppl::common::TensorShape* bias_shape,
    const void* bias,
    const void* scale_M,
    const void* scale_N,
    const float down_scale_M,
    const float down_scale_N,
    const int64_t in_features,
    const int64_t out_features,
    const matrix_layout_t weight_layout,
    ppl::common::NcclParam* nccl_param,
    const bool gather_output,
    void* gather_buffer,
    void* quant_buffer,
    const int64_t cublas_workspace_size,
    void* cublas_workspace,
    ppl::kernel::llm::cuda::cublas::AlgoCache* cublas_algo_cache,
    const ppl::common::TensorShape* output_shape,
    void* output)
{
    // input (M, K)
    // weight (N/w, K)
    // gemm_output (M, Nw)
    // output (M, N)

    if (bias && bias_shape->GetDataType() != ppl::common::DATATYPE_FLOAT16) {
        LOG(ERROR) << "only support fp16 bias";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (weight_layout != MATRIX_LAYOUT_ROW_MAJOR &&
        weight_layout != MATRIX_LAYOUT_COL4_4R2_8C &&
        weight_layout != MATRIX_LAYOUT_COL32_2R_4R4)
    {
        LOG(ERROR) << "unsupported weight layout:" << (int32_t)weight_layout;
        return ppl::common::RC_UNSUPPORTED;
    }

    const bool use_col32_gemm = weight_layout != MATRIX_LAYOUT_ROW_MAJOR;
    const bool use_4r4_weight = weight_layout == MATRIX_LAYOUT_COL32_2R_4R4;

    const int64_t M = input_shape->CalcElementsToDimensionExcludingPadding(input_shape->GetDimCount() - 1);
    const int64_t Nw = out_features / nccl_param->size;
    const int64_t K = in_features;

    void *gemm_output = output;
    void *dequant_output = output;
    if (gather_output) {
        gemm_output = quant_buffer;
    }
    if (gather_output && nccl_param->size > 1) {
        dequant_output = (char*)gather_buffer
            + nccl_param->rank * M * Nw * ppl::common::GetSizeOfDataType(output_shape->GetDataType());
    }

    ppl::common::RetCode status = ppl::common::RC_SUCCESS;

    // LOG(ERROR) << "M" << M << ", N" << Nw << ", K" << K;

    if (!use_col32_gemm) {
        status = ppl::kernel::llm::cuda::cublas::gemm_i8i8i32(
            stream,
            cublaslt_handle,
            algo,
            false,
            K,
            input_shape->GetDataType(),
            input,
            true,
            K,
            weight_shape->GetDataType(),
            weight,
            nullptr,
            M,
            Nw,
            K,
            1,
            0,
            cublas_workspace_size,
            cublas_workspace,
            cublas_algo_cache,
            Nw,
            ppl::common::DATATYPE_INT32,
            gemm_output);
    } else {
        status = ppl::kernel::llm::cuda::cublas::gemm_i8i8i32_col32(
            stream,
            cublaslt_handle,
            input,
            weight,
            M,
            Nw,
            K,
            use_4r4_weight,
            gemm_output);
    }

    if (ppl::common::RC_SUCCESS != status)
        return status;

    if (gather_output) {
        status = ppl::kernel::llm::cuda::pmx::i8i8::minmax_dequantize_fp16(
            stream,
            gemm_output,
            bias,
            scale_M,
            scale_N,
            M,
            Nw,
            down_scale_M,
            down_scale_N,
            (
                use_col32_gemm ?
                MATRIX_LAYOUT_COL32 :
                MATRIX_LAYOUT_ROW_MAJOR
            ),
            dequant_output
        );

        if (ppl::common::RC_SUCCESS != status)
            return status;
    }

    if (gather_output && nccl_param->size > 1) {
        status = ppl::common::NcclAllGather<half>(
            (half*)dequant_output,
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
