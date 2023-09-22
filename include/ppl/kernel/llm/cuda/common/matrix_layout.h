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

#ifndef __PPL_KERNEL_LLM_CUDA_COMMON_MATRIX_LAYOUT_H__
#define __PPL_KERNEL_LLM_CUDA_COMMON_MATRIX_LAYOUT_H__

#include "ppl/kernel/llm/cuda/common/general_include.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda {

enum matrix_layout_t {
    MATRIX_LAYOUT_ROW_MAJOR = 0,
    MATRIX_LAYOUT_COL_MAJOR = 1,
    MATRIX_LAYOUT_COL32 = 2,
    MATRIX_LAYOUT_COL32_2R_4R4 = 3,
    MATRIX_LAYOUT_COL4_4R2_8C = 4,
};

template<matrix_layout_t layout>
struct MatrixLayoutHelper {};

template<>
struct MatrixLayoutHelper<MATRIX_LAYOUT_ROW_MAJOR> {
    int64_t rows;
    int64_t cols;

    inline void __host__ __device__ Init(const int64_t _rows, const int64_t _cols) {
        rows = _rows;
        cols = _cols;
    }

    inline int64_t __host__ __device__ GetOffset(const int64_t row_id, const int64_t col_id) {
        return row_id * cols + col_id;
    }
};

template<>
struct MatrixLayoutHelper<MATRIX_LAYOUT_COL_MAJOR> {
    int64_t rows;
    int64_t cols;

    inline void __host__ __device__ Init(const int64_t _rows, const int64_t _cols) {
        rows = _rows;
        cols = _cols;
    }

    inline int64_t __host__ __device__ GetOffset(const int64_t row_id, const int64_t col_id) {
        return col_id * rows + row_id;
    }
};

template<>
struct MatrixLayoutHelper<MATRIX_LAYOUT_COL32> {
    int64_t rows;
    int64_t cols;

    inline void __host__ __device__ Init(const int64_t _rows, const int64_t _cols) {
        rows = _rows;
        cols = _cols;
    }

    // COL32 (ceil(cols/32), rows, 32)
    inline int64_t __host__ __device__ GetOffset(const int64_t row_id, const int64_t col_id) {
        return (col_id & (~(31LL))) * rows + (row_id << 5) + (col_id & 31);
    }
};

template<>
struct MatrixLayoutHelper<MATRIX_LAYOUT_COL4_4R2_8C> {
    int64_t rows;
    int64_t cols;

    int64_t rows_32;

    inline void __host__ __device__ Init(const int64_t _rows, const int64_t _cols) {
        rows = _rows;
        cols = _cols;
        rows_32 = rows << 5;
    }

    // IDK WTF is this
    // refer: https://docs.nvidia.com/cuda/cublas/index.html#cublasltorder-t
    // src: src/fastertransformer/kernels/quantize_weight.cu
    // CUBLASLT_ORDER_COL4_4R2_8C
    inline int64_t __host__ __device__ GetOffset(const int64_t row_id, const int64_t col_id) {
        int64_t new_col = col_id >> 5;
        int64_t new_row =  // CUBLASLT_ORDER_COL4_4R2_8C
                    ////row_id/8 is the number of tile of (8 rows 32 columns) -- column-major
                    ////row_id%2 is even row, otherwise odd row
                    ////col_id%COL32_/8 is the number tile of (8 rows 8 columns)
            (((((row_id >> 3) << 3) + ((row_id & 1) << 2) + ((col_id & 31) >> 3)) << 5) +
            ////col_id%8 >= 4 is the right half of (8 rows 8 columns) tile
            ////(row_id%8/2) is (the row id of alternating 4 rows) - 1
            (((((col_id & 7) >= 4) ? 4 : 0) + ((row_id & 7) >> 1)) << 2) +
            ////col_id%4 is the id of 4 cols
            (col_id & 3));
        return new_col * rows_32 + new_row;
    }
};

template<>
struct MatrixLayoutHelper<MATRIX_LAYOUT_COL32_2R_4R4> {
    int64_t rows;
    int64_t cols;

    int64_t rows_32;

    inline void __host__ __device__ Init(const int64_t _rows, const int64_t _cols) {
        rows = _rows;
        cols = _cols;
        rows_32 = rows << 5;
    }

    // IDK WTF is this
    // refer: https://docs.nvidia.com/cuda/cublas/index.html#cublasltorder-t
    // src: src/fastertransformer/kernels/quantize_weight.cu
    // CUBLASLT_ORDER_COL32_2R_4R4
    inline int64_t __host__ __device__ GetOffset(const int64_t row_id, const int64_t col_id) {
        int64_t new_col     = col_id >> 5;
        int64_t row_in_tile = row_id & 31;
        int64_t col_in_tile = col_id & 31;
        int64_t new_row     =  // CUBLASLT_ORDER_COL32_2R_4R4
            (((row_id >> 5) << 10) +
            //(((row%8)/2*4+row/8)*2+row%2)*32+col
            (((((((row_in_tile & 7) >> 1) << 2) + (row_in_tile >> 3)) << 1) + (row_in_tile & 1)) << 5) + col_in_tile);
        return new_col * rows_32 + new_row;
    }
};

}}}}

#endif
