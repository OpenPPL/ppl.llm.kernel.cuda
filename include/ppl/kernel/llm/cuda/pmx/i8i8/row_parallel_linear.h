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

#ifndef __PPL_KERNEL_LLM_CUDA_PMX_I8I8_ROW_PARALLEL_LINEAR_H__
#define __PPL_KERNEL_LLM_CUDA_PMX_I8I8_ROW_PARALLEL_LINEAR_H__

#include "ppl/kernel/llm/cuda/common/general_include.h"

#include "ppl/kernel/llm/cuda/cublas/gemm.h"
#include "ppl/common/cuda/nccl_utils.h"

#include "quantize.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx { namespace i8i8 {

// input should be m*k, CUBLASLT_ORDER_COL32
// weight should be n*k, CUBLASLT_ORDER_COL32_2R_4R4 or CUBLASLT_ORDER_COL4_4R2_8C
// output is m*n, CUBLASLT_ORDER_COL32
ppl::common::RetCode row_parallel_linear(
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
    const bool input_is_parallel,
    void* split_buffer,
    void* quant_buffer,
    const int64_t cublas_workspace_size,
    void* cublas_workspace,
    ppl::kernel::llm::cuda::cublas::AlgoCache* cublas_algo_cache,
    const ppl::common::TensorShape* output_shape,
    void* output);

}}}}}}

#endif
