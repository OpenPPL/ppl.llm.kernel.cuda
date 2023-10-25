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

#include "ppl/kernel/llm/cuda/cublas/gemm.h"
#include "ppl/common/log.h"

#include <cuda_fp16.h>

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace cublas {

#if (CUDART_VERSION < 11000)
template <typename T>
__global__ void cublas_gemm_add_bias_kernel(int64_t num_elems, int64_t bias_elems, T* output, const T* bias) {
    const int64_t output_idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (output_idx >= num_elems)
        return;

    output[output_idx] += bias[output_idx % bias_elems];
}
#endif

#define CUBLAS_CHECK_RC(X) do { \
        cublasStatus_t __status = (X); \
        if (__status != CUBLAS_STATUS_SUCCESS) { \
            LOG(ERROR) << "cublasLt failed: " << cublasLtGetStatusString(__status); \
            return ppl::common::RC_DEVICE_RUNTIME_ERROR; \
        } \
    } while (0)

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
    void* C)
{
    if (typea != ppl::common::DATATYPE_FLOAT16) {
        LOG(ERROR) << "only support fp16 A matrix";
        return ppl::common::RC_UNSUPPORTED;
    }
    if (typeb != ppl::common::DATATYPE_FLOAT16) {
        LOG(ERROR) << "only support fp16 B matrix";
        return ppl::common::RC_UNSUPPORTED;
    }
    if (typec != ppl::common::DATATYPE_FLOAT16) {
        LOG(ERROR) << "only support fp16 C matrix";
        return ppl::common::RC_UNSUPPORTED;
    }

    cublasLtMatmulDesc_t operationDesc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;

    cublasOperation_t cublas_transa = transa == true ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t cublas_transb = transb == true ? CUBLAS_OP_T : CUBLAS_OP_N;
    cudaDataType_t scaleType = CUDA_R_16F;

#if (CUDART_VERSION >= 11000)
    cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
    CUBLAS_CHECK_RC(cublasLtMatmulDescCreate(&operationDesc, computeType, scaleType));
#else
    cudaDataType_t computeType = scaleType;
    CUBLAS_CHECK_RC(cublasLtMatmulDescCreate(&operationDesc, computeType));
#endif

    // exchange A & B to col-major
    CUBLAS_CHECK_RC(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &cublas_transb, sizeof(cublas_transb)));
    CUBLAS_CHECK_RC(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &cublas_transa, sizeof(cublas_transa)));

#if (CUDART_VERSION >= 11000)
    if (bias != nullptr) {
        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        CUBLAS_CHECK_RC(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
        CUBLAS_CHECK_RC(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(void*)));
    }
#endif
    // create matrix descriptors, we are good with the details here so no need to set any extra attributes
    CUBLAS_CHECK_RC(cublasLtMatrixLayoutCreate(&Adesc, scaleType, cublas_transa == CUBLAS_OP_N ? K : M, cublas_transa == CUBLAS_OP_N ? M : K, lda));
    CUBLAS_CHECK_RC(cublasLtMatrixLayoutCreate(&Bdesc, scaleType, cublas_transb == CUBLAS_OP_N ? N : K, cublas_transb == CUBLAS_OP_N ? K : N, ldb));
    CUBLAS_CHECK_RC(cublasLtMatrixLayoutCreate(&Cdesc, scaleType, N, M, ldc));

    CUBLAS_CHECK_RC(cublasLtMatmul(
        cublaslt_handle,
        operationDesc,
        (const void*)(&alpha),
        B,
        Bdesc,
        A,
        Adesc,
        (const void*)(&beta),
        C,
        Cdesc,
        C,
        Cdesc,
        algo,
        workspace,
        workspace_size,
        stream));

#if (CUDART_VERSION < 11000)
    if (bias != nullptr) {
        const int64_t num_elems = M * N;
        const int64_t block_size = 128;
        const int64_t blocks = (num_elems + block_size - 1) / block_size;
        dim3 grid_size(blocks, 1, 1);
        cublas_gemm_add_bias_kernel<half><<<grid_size, block_size, 0, stream>>>(M * N, N, (half*)C, (const half*)bias);
    }
#endif

    CUBLAS_CHECK_RC(cublasLtMatmulDescDestroy(operationDesc));
    CUBLAS_CHECK_RC(cublasLtMatrixLayoutDestroy(Adesc));
    CUBLAS_CHECK_RC(cublasLtMatrixLayoutDestroy(Bdesc));
    CUBLAS_CHECK_RC(cublasLtMatrixLayoutDestroy(Cdesc));
    
    return ppl::common::RC_SUCCESS;
}

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
    AlgoCache* algo_cache,
    const int64_t ldc, // N
    const ppl::common::datatype_t typec, // int32
    void* C) // int32
{
    if (typea != ppl::common::DATATYPE_INT8) {
        LOG(ERROR) << "only support int8 A matrix";
        return ppl::common::RC_UNSUPPORTED;
    }
    if (typeb != ppl::common::DATATYPE_INT8) {
        LOG(ERROR) << "only support int8 B matrix";
        return ppl::common::RC_UNSUPPORTED;
    }
    if (typec != ppl::common::DATATYPE_INT32) {
        LOG(ERROR) << "only support int32 C matrix";
        return ppl::common::RC_UNSUPPORTED;
    }
    if (transa != false) {
        LOG(ERROR) << "only support non-transposed A matrix";
        return ppl::common::RC_UNSUPPORTED;
    }
    if (transb != true) {
        LOG(ERROR) << "only support transposed B matrix";
        return ppl::common::RC_UNSUPPORTED;
    }

    cublasLtMatmulDesc_t operationDesc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;

    cublasOperation_t cublas_transa = transa == true ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t cublas_transb = transb == true ? CUBLAS_OP_T : CUBLAS_OP_N;
    cudaDataType_t scaleType = CUDA_R_32I;
    cudaDataType_t abType = CUDA_R_8I;
    cudaDataType_t cType = CUDA_R_32I;

#if (CUDART_VERSION >= 11000)
    cublasComputeType_t computeType = CUBLAS_COMPUTE_32I;
    CUBLAS_CHECK_RC(cublasLtMatmulDescCreate(&operationDesc, computeType, scaleType));
#else
    cudaDataType_t computeType = scaleType;
    CUBLAS_CHECK_RC(cublasLtMatmulDescCreate(&operationDesc, computeType));
#endif

    // exchange A & B to col-major
    CUBLAS_CHECK_RC(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &cublas_transb, sizeof(cublas_transb)));
    CUBLAS_CHECK_RC(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &cublas_transa, sizeof(cublas_transa)));

#if (CUDART_VERSION >= 11000)
    if (bias != nullptr) {
        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        CUBLAS_CHECK_RC(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
        CUBLAS_CHECK_RC(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(void*)));
    }
#endif

    // create matrix descriptors, we are good with the details here so no need to set any extra attributes
    CUBLAS_CHECK_RC(cublasLtMatrixLayoutCreate(&Adesc, abType, cublas_transa == CUBLAS_OP_N ? K : M, cublas_transa == CUBLAS_OP_N ? M : K, lda));
    CUBLAS_CHECK_RC(cublasLtMatrixLayoutCreate(&Bdesc, abType, cublas_transb == CUBLAS_OP_N ? N : K, cublas_transb == CUBLAS_OP_N ? K : N, ldb));
    CUBLAS_CHECK_RC(cublasLtMatrixLayoutCreate(&Cdesc, cType, N, M, ldc));

    // find algo online
    cublasLtMatmulAlgo_t algo_value;
    bool                 found_algo = false;
    if (algo == nullptr) {
        // create matrix descriptors for algo select, align MNK to power of 2
        int64_t M_shift = 0;
        int64_t M_aligned = M;
        while (M_aligned >>= 2)
            ++M_shift;
        M_aligned = 1 << (M_shift * 2);

        int64_t K_shift = 0;
        int64_t K_aligned = K;
        while (K_aligned >>= 2)
            ++K_shift;
        K_aligned = 1 << (K_shift * 2);
        K_aligned = std::max<int64_t>(K / 256 * 256, K_aligned);

        int64_t N_shift = 0;
        int64_t N_aligned = N;
        while (N_aligned >>= 2)
            ++N_shift;
        N_aligned = 1 << (N_shift * 2);
        N_aligned = std::max<int64_t>(N / 256 * 256, N_aligned);

        cublasLtMatrixLayout_t AAdesc = nullptr, BBdesc = nullptr, CCdesc = nullptr;

        CUBLAS_CHECK_RC(cublasLtMatrixLayoutCreate(&AAdesc, abType, cublas_transa == CUBLAS_OP_N ? K_aligned : M_aligned, cublas_transa == CUBLAS_OP_N ? M_aligned : K_aligned, lda));
        CUBLAS_CHECK_RC(cublasLtMatrixLayoutCreate(&BBdesc, abType, cublas_transb == CUBLAS_OP_N ? N_aligned : K_aligned, cublas_transb == CUBLAS_OP_N ? K_aligned : N_aligned, ldb));
        CUBLAS_CHECK_RC(cublasLtMatrixLayoutCreate(&CCdesc, cType, N_aligned, M_aligned, ldc));

        AlgoCacheIndex cache_idx{
            convert_matmul_desc(operationDesc),
            {convert_matrix_layout(BBdesc),
            convert_matrix_layout(AAdesc),
            convert_matrix_layout(CCdesc),
            convert_matrix_layout(CCdesc)}};

        auto algo_cache_it = algo_cache->find(cache_idx);
        if (algo_cache_it == algo_cache->end()) {
            LOG(DEBUG) << "cublas finding algo, (M,N,K) = (" << M << "," << N << "," << K <<"), aligned to ("
                << M_aligned << "," << N_aligned << "," << K_aligned <<")";
            auto result =
                find_best_algo(
                    stream,
                    cublaslt_handle,
                    {20}, // ban this algo because it will give "Invalid __global__ read"
                    operationDesc,
                    (const void*)(&alpha),
                    B,
                    BBdesc,
                    A,
                    AAdesc,
                    (const void*)(&beta),
                    C,
                    CCdesc,
                    C,
                    CCdesc,
                    workspace_size,
                    workspace);
            if (result.first == ppl::common::RC_SUCCESS) {
                algo_cache->emplace(cache_idx, result.second);
                algo_value = result.second;
                found_algo = true;
            } else {
                LOG(ERROR) << "cublas find algo failed, (M,N,K) = (" << M << "," << N << "," << K <<")";
                return result.first;
            }
        } else {
            algo_value = algo_cache_it->second;
            found_algo = true;
        }

        CUBLAS_CHECK_RC(cublasLtMatrixLayoutDestroy(AAdesc));
        CUBLAS_CHECK_RC(cublasLtMatrixLayoutDestroy(BBdesc));
        CUBLAS_CHECK_RC(cublasLtMatrixLayoutDestroy(CCdesc));
    }

    CUBLAS_CHECK_RC(cublasLtMatmul(
        cublaslt_handle,
        operationDesc,
        (const void*)(&alpha),
        B,
        Bdesc,
        A,
        Adesc,
        (const void*)(&beta),
        C,
        Cdesc,
        C,
        Cdesc,
        found_algo ? &algo_value : algo,
        workspace,
        workspace_size,
        stream));

#if (CUDART_VERSION < 11000)
    if (bias != nullptr) {
        const int64_t num_elems = M * N;
        const int64_t block_size = 128;
        const int64_t blocks = (num_elems + block_size - 1) / block_size;
        dim3 grid_size(blocks, 1, 1);
        cublas_gemm_add_bias_kernel<int32_t><<<grid_size, block_size, 0, stream>>>(M * N, N, (int32_t*)C, (const int32_t*)bias);
    }
#endif

    CUBLAS_CHECK_RC(cublasLtMatmulDescDestroy(operationDesc));
    CUBLAS_CHECK_RC(cublasLtMatrixLayoutDestroy(Adesc));
    CUBLAS_CHECK_RC(cublasLtMatrixLayoutDestroy(Bdesc));
    CUBLAS_CHECK_RC(cublasLtMatrixLayoutDestroy(Cdesc));

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode gemm_i8i8i32_col32(
    const cudaStream_t stream,
    const cublasLtHandle_t& cublaslt_handle,
    const void* input_col32,
    const void* kernel,
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const bool use_4r4_kernel,
    void* output_col32)
{
    cublasOperation_t opTranspose = CUBLAS_OP_T;
#if (CUDART_VERSION >= 11000)
    cublasComputeType_t computeType = CUBLAS_COMPUTE_32I;
#else
    cudaDataType_t computeType = CUDA_R_32I;
#endif

    cublasLtMatmulDesc_t   matmulDesc = nullptr;
    cublasLtMatrixLayout_t AtransformDesc = nullptr;
    cublasLtMatrixLayout_t BtransformDesc = nullptr;
    cublasLtMatrixLayout_t CtransformDesc = nullptr;
    cublasLtOrder_t        order_COL32    = CUBLASLT_ORDER_COL32;

    cublasLtOrder_t order_matrixB;
#if (CUDART_VERSION >= 11000)
    if (use_4r4_kernel) {
        order_matrixB = CUBLASLT_ORDER_COL32_2R_4R4;
    } else {
        order_matrixB = CUBLASLT_ORDER_COL4_4R2_8C;
    }
#else
    order_matrixB = CUBLASLT_ORDER_COL4_4R2_8C;
#endif

    int ldaTransform = 32 * M;
    int ldbTransform;
    if (use_4r4_kernel) {
        ldbTransform = 32 * ((N + 32 - 1) / 32) * 32;
    }
    else {
        ldbTransform = 32 * ((N + 8 - 1) / 8) * 8;
    }
    int ldcTransform = 32 * M;

    // create matmulDesc
#if (CUDART_VERSION >= 11000)
    CUBLAS_CHECK_RC(cublasLtMatmulDescCreate(&matmulDesc, computeType, CUDA_R_32I));
#else
    CUBLAS_CHECK_RC(cublasLtMatmulDescCreate(&matmulDesc, computeType));
#endif
    CUBLAS_CHECK_RC(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opTranspose, sizeof(cublasOperation_t)));
    CUBLAS_CHECK_RC(cublasLtMatrixLayoutCreate(&AtransformDesc, CUDA_R_8I, M, K, ldaTransform));
    CUBLAS_CHECK_RC(cublasLtMatrixLayoutSetAttribute(AtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32)));
    CUBLAS_CHECK_RC(cublasLtMatrixLayoutCreate(&BtransformDesc, CUDA_R_8I, N, K, ldbTransform));
    CUBLAS_CHECK_RC(cublasLtMatrixLayoutSetAttribute(
        BtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_matrixB, sizeof(order_matrixB)));
    CUBLAS_CHECK_RC(cublasLtMatrixLayoutCreate(&CtransformDesc, CUDA_R_32I, M, N, ldcTransform));
    CUBLAS_CHECK_RC(cublasLtMatrixLayoutSetAttribute(CtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32)));

    int alphaI = 1;
    int betaI  = 0;

    // get algo
    cublasLtMatmulAlgo_t algo;
    {
        int algoId;
        if (use_4r4_kernel) {
            algoId = 7;
        }
        else {
            algoId = 6;
        }
        int swizzle         = 0;
        int customOption    = 0;
        int tile            = CUBLASLT_MATMUL_TILE_128x128;
        int splitK_val      = 0;
        int reductionScheme = 0;
        CUBLAS_CHECK_RC(cublasLtMatmulAlgoInit(
            cublaslt_handle, computeType, CUDA_R_32I, CUDA_R_8I, CUDA_R_8I, CUDA_R_32I, CUDA_R_32I, algoId, &algo));
        CUBLAS_CHECK_RC(cublasLtMatmulAlgoConfigSetAttribute(
            &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &(customOption), sizeof(customOption)));
        CUBLAS_CHECK_RC(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &(tile), sizeof(tile)));
        CUBLAS_CHECK_RC(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &(splitK_val), sizeof(splitK_val)));
        CUBLAS_CHECK_RC(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(swizzle), sizeof(swizzle)));
        CUBLAS_CHECK_RC(cublasLtMatmulAlgoConfigSetAttribute(
            &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &(reductionScheme), sizeof(int)));
#if (CUDART_VERSION >= 11000)
        int stages;
        if (use_4r4_kernel) {
            stages = CUBLASLT_MATMUL_STAGES_64x3;
        }
        else {
            stages = CUBLASLT_MATMUL_STAGES_64x1;
        }
        CUBLAS_CHECK_RC(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &(stages), sizeof(stages)));
#endif
    }

    CUBLAS_CHECK_RC(cublasLtMatmul(
        cublaslt_handle,
        matmulDesc,
        &alphaI,
        input_col32,
        AtransformDesc,
        kernel,
        BtransformDesc,
        &betaI,
        output_col32,
        CtransformDesc,
        output_col32,
        CtransformDesc,
        &algo,
        nullptr,
        0,
        stream));

    CUBLAS_CHECK_RC(cublasLtMatmulDescDestroy(matmulDesc));
    CUBLAS_CHECK_RC(cublasLtMatrixLayoutDestroy(AtransformDesc));
    CUBLAS_CHECK_RC(cublasLtMatrixLayoutDestroy(BtransformDesc));
    CUBLAS_CHECK_RC(cublasLtMatrixLayoutDestroy(CtransformDesc));

    return ppl::common::RC_SUCCESS;
}

}}}}}
