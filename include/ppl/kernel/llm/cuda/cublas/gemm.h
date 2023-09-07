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

#ifndef __PPL_KERNEL_LLM_CUDA_PMX_CUBLAS_MATMUL_H__
#define __PPL_KERNEL_LLM_CUDA_PMX_CUBLAS_MATMUL_H__

#include "ppl/kernel/llm/cuda/common/general_include.h"

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

}}}}}

#endif
