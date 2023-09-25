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

#ifndef __PPL_KERNEL_LLM_CUDA_PMX_CUBLAS_GEMM_H__
#define __PPL_KERNEL_LLM_CUDA_PMX_CUBLAS_GEMM_H__

#include "ppl/kernel/llm/cuda/common/general_include.h"

#include "gemm_algo.h"

#include "ppl/common/types.h"

#include <cublasLt.h>

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace cublas {

// translate cublas col major gemm to row major gemm
ppl::common::RetCode gemm(
    const cudaStream_t stream,
    const cublasLtHandle_t& cublaslt_handle,
    const cublasLtMatmulAlgo_t* algo,
    const bool transa,
    const int64_t lda,
    const ppl::common::datatype_t typea,
    const void* A,
    const bool transb,
    const int64_t ldb,
    const ppl::common::datatype_t typeb,
    const void* B,
    const void* bias,
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const float alpha,
    const float beta,
    const int64_t workspace_size,
    void* workspace,
    const int64_t ldc,
    const ppl::common::datatype_t typec,
    void* C);

ppl::common::RetCode gemm_i8i8i32(
    const cudaStream_t stream,
    const cublasLtHandle_t& cublaslt_handle,
    const cublasLtMatmulAlgo_t* algo,
    const bool transa, // must be false
    const int64_t lda, // transa ? M : K;
    const ppl::common::datatype_t typea, // int8
    const void* A, // int8
    const bool transb, // must be true
    const int64_t ldb, // transb ? K : N;
    const ppl::common::datatype_t typeb, // int8
    const void* B, // int8
    const void* bias, // int32
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const int32_t alpha, // int32-C need
    const int32_t beta, // int32-C need
    const int64_t workspace_size,
    void* workspace,
    cublaslt_algo_cache_t* algo_cache,
    const int64_t ldc, // N
    const ppl::common::datatype_t typec, // int32
    void* C); // int32

ppl::common::RetCode gemm_i8i8i32_col32(
    const cudaStream_t stream,
    const cublasLtHandle_t& cublaslt_handle,
    const void* input_col32, // int8
    const void* kernel, // int8
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const bool use_4r4_kernel,
    void* output_col32); // int32

}}}}}

#endif
