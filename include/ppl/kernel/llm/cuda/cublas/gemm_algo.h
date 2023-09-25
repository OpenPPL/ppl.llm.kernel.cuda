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
#include <unordered_map>

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace cublas {

using cublaslt_matmul_desc_t = std::array<uint64_t, 12>;
using cublaslt_matrix_layout_t = std::tuple<
    cudaDataType_t, cublasLtOrder_t, uint64_t, uint64_t>;
struct cublaslt_algo_cache_idx_t {
    cublaslt_matmul_desc_t matmul_desc;
    std::array<cublaslt_matrix_layout_t, 4> matrix_descs;
};

struct cublaslt_algo_cache_idx_hash {
    std::size_t operator()(const cublaslt_algo_cache_idx_t& k) const {
        return    std::hash<uint64_t>()(k.matmul_desc[0])
               ^ (std::hash<uint64_t>()(k.matmul_desc[1]) << 1)
               ^ (std::hash<uint64_t>()(k.matmul_desc[2]) << 2)
               ^ (std::hash<uint64_t>()(k.matmul_desc[3]) << 3)
               ^ (std::hash<uint64_t>()(k.matmul_desc[4]) << 4)
               ^ (std::hash<uint64_t>()(k.matmul_desc[5]) << 5)
               ^ (std::hash<uint64_t>()(k.matmul_desc[6]) << 6)
               ^ (std::hash<uint64_t>()(k.matmul_desc[7]) << 7)
               ^ (std::hash<uint64_t>()(k.matmul_desc[8]) << 8)
               ^ (std::hash<uint64_t>()(k.matmul_desc[9]) << 9)
               ^ (std::hash<uint64_t>()(k.matmul_desc[10]) << 10)
               ^ (std::hash<uint64_t>()(k.matmul_desc[11]) << 11)
               ^ (std::hash<int>()(std::get<0>(k.matrix_descs[0])) << 12)
               ^ (std::hash<int>()(std::get<1>(k.matrix_descs[0])) << 13)
               ^ (std::hash<uint64_t>()(std::get<2>(k.matrix_descs[0])) << 14)
               ^ (std::hash<uint64_t>()(std::get<3>(k.matrix_descs[0])) << 15)
               ^ (std::hash<int>()(std::get<0>(k.matrix_descs[1])) << 16)
               ^ (std::hash<int>()(std::get<1>(k.matrix_descs[1])) << 17)
               ^ (std::hash<uint64_t>()(std::get<2>(k.matrix_descs[1])) << 18)
               ^ (std::hash<uint64_t>()(std::get<3>(k.matrix_descs[1])) << 19)
               ^ (std::hash<int>()(std::get<0>(k.matrix_descs[2])) << 20)
               ^ (std::hash<int>()(std::get<1>(k.matrix_descs[2])) << 21)
               ^ (std::hash<uint64_t>()(std::get<2>(k.matrix_descs[2])) << 22)
               ^ (std::hash<uint64_t>()(std::get<3>(k.matrix_descs[2])) << 23)
               ^ (std::hash<int>()(std::get<0>(k.matrix_descs[3])) << 24)
               ^ (std::hash<int>()(std::get<1>(k.matrix_descs[3])) << 25)
               ^ (std::hash<uint64_t>()(std::get<2>(k.matrix_descs[3])) << 26)
               ^ (std::hash<uint64_t>()(std::get<3>(k.matrix_descs[3])) << 27);
    }
};

struct cublaslt_algo_cache_idx_equal {
    bool operator()(const cublaslt_algo_cache_idx_t& a, const cublaslt_algo_cache_idx_t& b) const {
        return a.matmul_desc == b.matmul_desc && a.matrix_descs == b.matrix_descs;
    }
};

using cublaslt_algo_cache_t = std::unordered_map<
    cublaslt_algo_cache_idx_t,
    cublasLtMatmulAlgo_t,
    cublaslt_algo_cache_idx_hash,
    cublaslt_algo_cache_idx_equal>;

static inline cublaslt_matmul_desc_t create_cublas_matmul_desc(cublasLtMatmulDesc_t Mdesc) {
    cublaslt_matmul_desc_t m_desc = {
        Mdesc->data[0], Mdesc->data[1], Mdesc->data[2], Mdesc->data[3], 
        Mdesc->data[4], Mdesc->data[5], Mdesc->data[6], Mdesc->data[7], 
        Mdesc->data[8], Mdesc->data[9], Mdesc->data[10], Mdesc->data[11], 
    };

    return m_desc;
}

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
