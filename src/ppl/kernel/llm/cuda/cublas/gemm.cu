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

    // descriptors are no longer needed as all GPU work was already enqueued
    if (Cdesc) CUBLAS_CHECK_RC(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) CUBLAS_CHECK_RC(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) CUBLAS_CHECK_RC(cublasLtMatrixLayoutDestroy(Adesc));
    if (operationDesc) CUBLAS_CHECK_RC(cublasLtMatmulDescDestroy(operationDesc));
    
    return ppl::common::RC_SUCCESS;
}

}}}}}
