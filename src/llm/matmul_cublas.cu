#include "cudakernel/llm/matmul_cublas.h"
#include <stdio.h>
#if __CUDACC_VER_MAJOR__ >= 9 && !defined(_WIN64)
#include <mutex>
static bool g_is_less_volta_deivce_ = false;
static std::once_flag is_less_volta_deivce_onceflag;
#endif

static void isLessVolta() {
#if __CUDACC_VER_MAJOR__ >= 9 && !defined(_WIN64)
    int dev_id_;
    CUresult result_  = cuCtxGetDevice(&dev_id_);
    if (result_ != CUDA_SUCCESS ) { g_is_less_volta_deivce_ = false; }
    cudaDeviceProp deviceProp;
    auto err = cudaGetDeviceProperties(&deviceProp, dev_id_);
    if (err != cudaSuccess) { g_is_less_volta_deivce_ = false; }
    if (deviceProp.major < 7) {
        g_is_less_volta_deivce_  = true;
    } else {
        g_is_less_volta_deivce_ = false;
    }
#endif
}

inline void checkCublasStatus(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS API failed with status %d\n", status);
        throw std::logic_error("cuBLAS API failed");
    }
}

inline void checkCudaStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        printf("cuda API failed with status %d: %s\n", status, cudaGetErrorString(status));
        throw std::logic_error("cuda API failed");
    }
}

uint64_t PPLCUDAMatMulCublasGetRuntimeBufSize(
        const ppl::common::TensorShape* input_shape,
        const ppl::common::TensorShape* weight_shape) {
    auto dim_count0 = input_shape->GetDimCount();
    auto dim_count1 = weight_shape->GetDimCount();
    int m_id = dim_count0 - 2;
    uint64_t batch = 1;
    if (dim_count1 == 2){
        batch = 1;
    } else {
        for (int i = 0; i < m_id; i++) 
            batch *= input_shape->GetDim(i);
    }
    return 1024 * 1024 * 4 * batch;
}

double PPLCUDAMatMulCublasSelectKernel(
            const cudaStream_t stream,
            const cublasLtHandle_t& ltHandle,
            const GemmKernelParam &param, 
            const ppl::common::TensorShape* input_shape,
            const void* input,
            const ppl::common::TensorShape* weight_shape,
            const void* weight,
            const ppl::common::TensorShape* output_shape,
            void* output,
            const ppl::common::TensorShape* bias_shape,
            const void* bias,
            const int64_t batch,
            void* workspace, 
            size_t workspaceSize,
            cublasLtMatmulAlgo_t& algo) {
    auto dim_count0 = input_shape->GetDimCount();
    auto dim_count1 = weight_shape->GetDimCount();
    int m_id = dim_count0 - 2;
    int k_id = dim_count0 - 1;
    int n_id = dim_count1 - 1;
    if (param.transA) {
        m_id = dim_count0 - 1;
        k_id = dim_count0 - 2;
    }
    if (param.transB) {
        n_id = dim_count1 - 2;
    }
    int64_t M = input_shape->GetDim(m_id);
    if (dim_count1 == 2 && dim_count0 > 2){ // matmul case
        for (int i = 0; i < m_id; i++){
            M *= input_shape->GetDim(i);
        }
    }
    int64_t K     = input_shape->GetDim(k_id);
    int64_t N     = weight_shape->GetDim(n_id);
    cublasLtMatmulDesc_t operationDesc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr, Ddesc = nullptr;

    cublasOperation_t transa = param.transA == true ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transb = param.transB == true ? CUBLAS_OP_T : CUBLAS_OP_N;
    float alpha = 1;
    float beta = 0.f;
    int64_t lda = param.transA ? M : K;
    int64_t ldb = param.transB ? K : N;
    int64_t ldc = N;
    cudaDataType_t dt = CUDA_R_32F;
    if (input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        dt = CUDA_R_16F;
    }

    // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
    // set the transforms for A and B
#if __CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10 >= 11000
    cublasComputeType_t ct = CUBLAS_COMPUTE_32F_FAST_TF32;
    if (input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        ct = CUBLAS_COMPUTE_32F;
    }
    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, ct, dt));
#elif __CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10 > 10000
    cudaDataType_t ct = dt;
    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, ct));
#endif
    // exchange A & B
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transb, sizeof(transb)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transa, sizeof(transa)));
    if(batch == 1) {
        // create matrix descriptors, we are good with the details here so no need to set any extra attributes
        checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, dt, transa == CUBLAS_OP_N ? K : M, transa == CUBLAS_OP_N ? M : K, lda));
        checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, dt, transb == CUBLAS_OP_N ? N : K, transb == CUBLAS_OP_N ? K : N, ldb));
        checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, dt, N, M, ldc));
    } else {
        int64_t stridea = M * K;
        int64_t strideb = K * N;
        int64_t stridec = M * N;
        // create matrix descriptors, we are good with the details here so no need to set any extra attributes
        checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, dt, transa == CUBLAS_OP_N ? K : M, transa == CUBLAS_OP_N ? M : K, lda));
        checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
        checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridea, sizeof(stridea)));

        checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, dt, transb == CUBLAS_OP_N ? N : K, transb == CUBLAS_OP_N ? K : N, ldb));
        checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
        checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideb, sizeof(strideb)));

        checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, dt, N, M, ldc));
        checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
        checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridec, sizeof(stridec)));
    }
    // **************** Request Algos **************
    // create preference handle; here we could use extra attributes to disable tensor ops or to make sure algo selected
    // will work with badly aligned A, B, C; here for simplicity we just assume A,B,C are always well aligned (e.g.
    // directly come from cudaMalloc)
    cublasLtMatmulPreference_t preference = nullptr;
    constexpr int requested_algo = 8;
    int returnedResults                             = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult[requested_algo] = { 0 };
    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
    checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));
    // we just need the best available heuristic to try and run matmul. There is no guarantee this will work, e.g. if A
    // is badly aligned, you can request more (e.g. 32) algos and try to run them one by one until something works
    checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Bdesc, Adesc, Cdesc, Cdesc, preference, requested_algo, heuristicResult, &returnedResults));
    if (returnedResults == 0) {
        checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    // run and evaluate timing
    constexpr int repeatAlgoCheck = 5;
    int bestAlgoIdx = 0;
    float time = 0;
    float bestAlgoTime = 0;
    cudaEvent_t startEvent, stopEvent;
    checkCudaStatus(cudaEventCreate(&startEvent));
    checkCudaStatus(cudaEventCreate(&stopEvent));
    for (int algoIdx = 0; algoIdx < returnedResults; algoIdx++) {
        checkCudaStatus(cudaEventRecord(startEvent, stream));
        for (int checkIdx = 0; checkIdx < repeatAlgoCheck; checkIdx++) {
            checkCublasStatus(cublasLtMatmul(ltHandle,
                                        operationDesc,
                                        (const void*)(&alpha),
                                        weight,
                                        Bdesc,
                                        input,
                                        Adesc,
                                        (const void*)(&beta),
                                        output,
                                        Cdesc,
                                        output,
                                        Cdesc,
                                        &heuristicResult[algoIdx].algo,
                                        workspace,
                                        workspaceSize,
                                        stream));
        }
        checkCudaStatus(cudaEventRecord(stopEvent, stream));
        checkCudaStatus(cudaEventSynchronize(stopEvent));
        checkCudaStatus(cudaEventElapsedTime(&time, startEvent, stopEvent));
        time /= repeatAlgoCheck;

        if (algoIdx == 0 || time < bestAlgoTime) {
            bestAlgoTime = time;
            bestAlgoIdx = algoIdx;
        }
    }
    memcpy(&algo, &heuristicResult[bestAlgoIdx].algo, sizeof(algo));

    if (startEvent) checkCudaStatus(cudaEventDestroy(startEvent));
    if (stopEvent) checkCudaStatus(cudaEventDestroy(stopEvent));
    // descriptors are no longer needed as all GPU work was already enqueued
    if (preference) checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
    if (Ddesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Ddesc));
    if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
    return bestAlgoTime;
}

template <typename T>
__global__ void ppl_matmul_add_bias(int32_t num_elems, int32_t bias_len, T* output, const T* bias) {
    int b_idx= blockIdx.y;
    int index = threadIdx.x + blockIdx.x * blockDim.x; 
    if (index >= num_elems) return;
    int32_t bias_offset = b_idx * bias_len + index % bias_len;
    int32_t output_offset = b_idx * num_elems + index;
    output[output_offset] += bias[bias_offset];
}

ppl::common::RetCode PPLCUDAMatMulCublasForwardImp(
        const cudaStream_t stream,
        const cublasLtHandle_t& ltHandle,
        const GemmKernelParam &param,
        const ppl::common::TensorShape* input_shape,
        const void* input,
        const ppl::common::TensorShape* weight_shape,
        const void* weight,
        const ppl::common::TensorShape* output_shape,
        void* output,
        const ppl::common::TensorShape* bias_shape,
        const void* bias,
        const int64_t batch,
        void* workspace, 
        size_t workspaceSize,
        bool use_heuristic,
        cublasLtMatmulAlgo_t algo) {
#if __CUDACC_VER_MAJOR__ >= 9 && !defined(_WIN64)
    std::call_once(is_less_volta_deivce_onceflag, isLessVolta);
    if (g_is_less_volta_deivce_) {
        use_heuristic = false; // cublas matmul is tested crash on GTX1060, but success on volta, turing, ampere
    }
#endif
    
    auto dim_count0 = input_shape->GetDimCount();
    auto dim_count1 = weight_shape->GetDimCount();
    int m_id = dim_count0 - 2;
    int k_id = dim_count0 - 1;
    int n_id = dim_count1 - 1;
    if (param.transA) {
        m_id = dim_count0 - 1;
        k_id = dim_count0 - 2;
    }
    if (param.transB) {
        n_id = dim_count1 - 2;
    }
    int64_t M = input_shape->GetDim(m_id);
    if (dim_count1 == 2 && dim_count0 > 2){ // matmul case
        for (int i = 0; i < m_id; i++){
            M *= input_shape->GetDim(i);
        }
    }
    int64_t K     = input_shape->GetDim(k_id);
    int64_t N     = weight_shape->GetDim(n_id);
    cublasLtMatmulDesc_t operationDesc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr, Ddesc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;

    cublasOperation_t transa = param.transA == true ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transb = param.transB == true ? CUBLAS_OP_T : CUBLAS_OP_N;
    float alpha = param.alpha;
    float beta = 0.f;
    int lda = param.transA ? M : K;
    int ldb = param.transB ? K : N;
    int ldc = N;
    cudaDataType_t dt = CUDA_R_32F;
    using Dtype = float;
    if (input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        dt = CUDA_R_16F;
        using Dtype = half;
    }

#if __CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10 >= 11000
    cublasComputeType_t ct = CUBLAS_COMPUTE_32F_FAST_TF32;
    if (input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        ct = CUBLAS_COMPUTE_32F;
    }
    // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
    // set the transforms for A and B
    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, ct, dt));
#elif __CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10 > 10000
    cudaDataType_t ct = dt;
    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, ct));
#endif

    bias = reinterpret_cast<const Dtype*>(bias);
    // exchange A & B
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transb, sizeof(transb)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transa, sizeof(transa)));
#if __CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10 >= 11000
    if (bias != nullptr) {
        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
        checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(Dtype*)));
    }
#endif
    if(batch == 1) {
        // create matrix descriptors, we are good with the details here so no need to set any extra attributes
        checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, dt, transa == CUBLAS_OP_N ? K : M, transa == CUBLAS_OP_N ? M : K, lda));
        checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, dt, transb == CUBLAS_OP_N ? N : K, transb == CUBLAS_OP_N ? K : N, ldb));
        checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, dt, N, M, ldc));
    } else {
        int64_t stridea = M * K;
        int64_t strideb = K * N;
        int64_t stridec = M * N;
        // create matrix descriptors, we are good with the details here so no need to set any extra attributes
        checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, dt, transa == CUBLAS_OP_N ? K : M, transa == CUBLAS_OP_N ? M : K, lda));
        checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
        checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridea, sizeof(stridea)));

        checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, dt, transb == CUBLAS_OP_N ? N : K, transb == CUBLAS_OP_N ? K : N, ldb));
        checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
        checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideb, sizeof(strideb)));

        checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, dt, N, M, ldc));
        checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
        checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridec, sizeof(stridec)));
    }

    #if 1
    if (use_heuristic) {
        cublasLtMatmulHeuristicResult_t heurResult;
        cublasStatus_t algoStatus = cublasLtMatmulAlgoCheck(ltHandle,
                                operationDesc,
                                Bdesc,
                                Adesc,
                                Cdesc,
                                Cdesc,
                                &algo, 
                                &heurResult);  
        if (algoStatus != CUBLAS_STATUS_SUCCESS) {
            use_heuristic = false;
        }
    }
    #endif

    checkCublasStatus(cublasLtMatmul(ltHandle,
                                    operationDesc,
                                    (const void*)(&alpha),
                                    weight,
                                    Bdesc,
                                    input,
                                    Adesc,
                                    (const void*)(&beta),
                                    output,
                                    Cdesc,
                                    output,
                                    Cdesc,
                                    use_heuristic ? &algo : nullptr,
                                    workspace,
                                    workspaceSize,
                                    stream));

// cuda 10.1&10.2 need tackle bias alone
#if __CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10 < 11000
    if (bias != nullptr) {
        int64_t num_elems = M * N;
        int64_t block_size = 128;
        int64_t blocks = (num_elems + block_size - 1) / block_size;
        dim3 grid_size(blocks, batch, 1);
        ppl_matmul_add_bias<Dtype><<<grid_size, block_size, 0, stream>>>(num_elems, N, (Dtype*)output, (const Dtype*)bias);
    }
#endif

    // descriptors are no longer needed as all GPU work was already enqueued
    if (preference) checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
    if (Ddesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Ddesc));
    if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
    
    return ppl::common::RC_SUCCESS;
}