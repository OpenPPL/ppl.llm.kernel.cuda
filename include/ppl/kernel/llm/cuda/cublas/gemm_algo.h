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

struct MatmulDesc {
    cublasOperation_t transa;
    cublasOperation_t transb;

    bool operator==(const MatmulDesc &other) const {
        return transa == other.transa
            && transb == other.transb;
    }
};

struct MatrixLayout {
    cudaDataType_t datatype;
    cublasLtOrder_t layout;
    uint64_t rows;
    uint64_t cols;

    bool operator==(const MatrixLayout &other) const {
        return datatype == other.datatype
            && layout == other.layout
            && rows == other.rows
            && cols == other.cols;
    }
};

struct AlgoCacheIndex {
    MatmulDesc matmul_desc;
    std::array<MatrixLayout, 4> matrix_descs;

    bool operator==(const AlgoCacheIndex &other) const {
        return matmul_desc == other.matmul_desc
            && matrix_descs == other.matrix_descs;
    }
};

struct AlgoCacheIndexHash {
    std::size_t operator()(const AlgoCacheIndex& k) const {
        return    std::hash<int>()(k.matmul_desc.transa)
               ^ (std::hash<int>()(k.matmul_desc.transb) << 1)
               ^ (std::hash<int>()(k.matrix_descs[0].datatype) << 2)
               ^ (std::hash<int>()(k.matrix_descs[0].layout) << 3)
               ^ (std::hash<uint64_t>()(k.matrix_descs[0].rows) << 4)
               ^ (std::hash<uint64_t>()(k.matrix_descs[0].cols) << 5)
               ^ (std::hash<int>()(k.matrix_descs[1].datatype) << 6)
               ^ (std::hash<int>()(k.matrix_descs[1].layout) << 7)
               ^ (std::hash<uint64_t>()(k.matrix_descs[1].rows) << 8)
               ^ (std::hash<uint64_t>()(k.matrix_descs[1].cols) << 9)
               ^ (std::hash<int>()(k.matrix_descs[2].datatype) << 10)
               ^ (std::hash<int>()(k.matrix_descs[2].layout) << 11)
               ^ (std::hash<uint64_t>()(k.matrix_descs[2].rows) << 12)
               ^ (std::hash<uint64_t>()(k.matrix_descs[2].cols) << 13)
               ^ (std::hash<int>()(k.matrix_descs[3].datatype) << 14)
               ^ (std::hash<int>()(k.matrix_descs[3].layout) << 15)
               ^ (std::hash<uint64_t>()(k.matrix_descs[3].rows) << 16)
               ^ (std::hash<uint64_t>()(k.matrix_descs[3].cols) << 17);
    }
};

using AlgoCache = std::unordered_map<
    AlgoCacheIndex,
    cublasLtMatmulAlgo_t,
    AlgoCacheIndexHash>;

static inline MatmulDesc convert_matmul_desc(cublasLtMatmulDesc_t Mdesc) {
    size_t       return_size;
    MatmulDesc m_desc;

    cublasLtMatmulDescGetAttribute(
        Mdesc, CUBLASLT_MATMUL_DESC_TRANSA, &m_desc.transa, sizeof(m_desc.transa), &return_size);
    cublasLtMatmulDescGetAttribute(
        Mdesc, CUBLASLT_MATMUL_DESC_TRANSB, &m_desc.transb, sizeof(m_desc.transb), &return_size);

    return m_desc;
}

static inline MatrixLayout convert_matrix_layout(cublasLtMatrixLayout_t Mdesc) {
    size_t       return_size;
    MatrixLayout m_layout;

    cublasLtMatrixLayoutGetAttribute(
        Mdesc, CUBLASLT_MATRIX_LAYOUT_TYPE, &m_layout.datatype, sizeof(m_layout.datatype), &return_size);
    cublasLtMatrixLayoutGetAttribute(
        Mdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &m_layout.layout, sizeof(m_layout.layout), &return_size);
    cublasLtMatrixLayoutGetAttribute(
        Mdesc, CUBLASLT_MATRIX_LAYOUT_ROWS, &m_layout.rows, sizeof(m_layout.rows), &return_size);
    cublasLtMatrixLayoutGetAttribute(
        Mdesc, CUBLASLT_MATRIX_LAYOUT_COLS, &m_layout.cols, sizeof(m_layout.cols), &return_size);

    return m_layout;
}

std::pair<ppl::common::RetCode, cublasLtMatmulAlgo_t> find_best_algo(
    const cudaStream_t     stream,
    const cublasLtHandle_t&lightHandle,
    const std::vector<int>&banned_algo_ids, // some algo does invalid read in my unittest, what are you doing nvidia?
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
