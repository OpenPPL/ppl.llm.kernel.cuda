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

#ifndef __PPL_KERNEL_LLM_CUDA_PMX_CUBLAS_GEMM_ALGO_H__
#define __PPL_KERNEL_LLM_CUDA_PMX_CUBLAS_GEMM_ALGO_H__

#include "ppl/kernel/llm/cuda/common/general_include.h"

#include "ppl/common/types.h"

#include <cublasLt.h>
#include <array>
#include <map>

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace cublas {

using cublaslt_matrix_layout_t = std::tuple<cudaDataType_t, cublasLtOrder_t, uint64_t, uint64_t>;
using cublaslt_algo_cache_idx_t = std::tuple<cublasLtMatmulDesc_t, std::array<cublaslt_matrix_layout_t, 4>>;
using cublaslt_algo_cache_t = std::map<cublaslt_algo_cache_idx_t, cublasLtMatmulAlgo_t>;

static inline cublaslt_matrix_layout_t create_cublas_matrix_layout(cublasLtMatrixLayout_t Mdesc) {
    size_t       return_size;
    cublaslt_matrix_layout_t m_layout;

    cublasLtMatrixLayoutGetAttribute(
        Mdesc, CUBLASLT_MATRIX_LAYOUT_TYPE, &std::get<0>(m_layout), sizeof(std::get<0>(m_layout)), &return_size);
    cublasLtMatrixLayoutGetAttribute(
        Mdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &std::get<1>(m_layout), sizeof(std::get<1>(m_layout)), &return_size);
    cublasLtMatrixLayoutGetAttribute(
        Mdesc, CUBLASLT_MATRIX_LAYOUT_ROWS, &std::get<2>(m_layout), sizeof(std::get<2>(m_layout)), &return_size);
    cublasLtMatrixLayoutGetAttribute(
        Mdesc, CUBLASLT_MATRIX_LAYOUT_COLS, &std::get<3>(m_layout), sizeof(std::get<3>(m_layout)), &return_size);

    return m_layout;
}

std::pair<ppl::common::RetCode, cublasLtMatmulAlgo_t> cublaslt_find_best_algo(
    const cudaStream_t     stream,
    const cublasLtHandle_t&lightHandle,
    cublasLtMatmulDesc_t   computeDesc,
    const void*            alpha,
    const void*            A,
    cublasLtMatrixLayout_t Adesc,
    const void*            B,
    cublasLtMatrixLayout_t Bdesc,
    const void*            beta,
    const void*            C,
    cublasLtMatrixLayout_t Cdesc,
    void*                  D,
    cublasLtMatrixLayout_t Ddesc,
    const int64_t          workspace_size,
    void*                  workspace);

}}}}}

#endif
