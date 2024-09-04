#pragma once

#include <iostream>
#include <cuda_fp16.h>
#include "ppl/kernel/llm/cuda/cublas/gemm.h"
#include "ppl/kernel/llm/cuda/cublas/gemm_algo.h"

#ifdef PPLNN_ENABLE_FP8

#include <cuda_fp8.h>

using fp8_t    = __nv_fp8_e4m3;

#endif

using fp16_t   = half;

#define CUDA_CHECK_RC(call) do { \
    cudaError_t __status = (call); \
    if (__status != cudaSuccess) { \
        fprintf(stderr, "CUDA call failed in function %s at line %d: %s\n", #call, __LINE__, cudaGetErrorString(__status)); \
        return ppl::common::RC_DEVICE_RUNTIME_ERROR; \
    } \
} while (0)

#define CUBLAS_CHECK_RC(call) do { \
    cublasStatus_t __status = (call); \
    if (__status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "CublasLt call failed in function %s: %s\n", #call, cublasLtGetStatusString(__status)); \
        return ppl::common::RC_DEVICE_RUNTIME_ERROR; \
    } \
} while (0)


struct Result {
    float tot_exe_time;
    double avg_exe_time;
    double avg_gflops;
    double avg_gbps;
};


class gemm_test_base {
private:
    cudaEvent_t start_event;
    cudaEvent_t stop_event;

    float tot_exe_time = 0;
    double avg_exe_time = 0;
    double avg_gflops = 0;
    double avg_gbps = 0;

protected:
    int64_t M;
    int64_t N;
    int64_t K;
    int64_t alignment;

    bool transa = false;
    bool transb = true;

    int64_t lda;
    int64_t ldb;
    int64_t ldc;

    int64_t A_num_elements;
    int64_t B_num_elements;
    int64_t C_num_elements;

    int64_t A_num_bytes;
    int64_t B_num_bytes;
    int64_t C_num_bytes;

    size_t A_element_size;
    float B_element_size;
    size_t C_element_size;

    void* A = nullptr;
    void* B = nullptr;
    void* C = nullptr;

    void* host_A = nullptr;
    void* host_B = nullptr;

    double gops;
    double gbs;   

    cudaStream_t stream;
    
    void* bias = nullptr;
    int32_t alpha = 1;
    int32_t beta = 0;
    int64_t workspace_size = 4 * 1024 * 1024;
    void* workspace = nullptr;
    cublasLtMatmulAlgo_t* algo = nullptr;
    ppl::kernel::llm::cuda::cublas::AlgoCache algo_cache;

    virtual ppl::common::RetCode gemm() = 0;

public:
    ppl::common::RetCode init_cuda();
    virtual ppl::common::RetCode init_handle() = 0;
    void init_params(int64_t m, int64_t n, int64_t k);
    virtual ppl::common::RetCode generate_data() = 0;
    ppl::common::RetCode init_device_data();
    ppl::common::RetCode warmup(const int64_t tot_iter);
    ppl::common::RetCode benchmark(const int64_t tot_iter);
    virtual void deallocate_host_mem() = 0;
    ppl::common::RetCode deallocate_cuda_mem();
    Result get_result();
    ppl::common::RetCode destroy_cuda();
    virtual ppl::common::RetCode destroy_handle() = 0;
    virtual ppl::common::RetCode destroy_scale();
    virtual ppl::common::RetCode init_scale();

    virtual ~gemm_test_base() = default;
};


class gemm_test_i8i8 : public gemm_test_base {
protected:
    cublasLtHandle_t gemm_handle; 
    ppl::common::RetCode gemm() override;
      
public:
    gemm_test_i8i8() : gemm_test_base() {
        alignment = 1;
        A_element_size = sizeof(int8_t);
        B_element_size = sizeof(int8_t);
        C_element_size = sizeof(int32_t);
    }

    ppl::common::RetCode init_handle() override;
    ppl::common::RetCode generate_data() override;
    ppl::common::RetCode destroy_handle() override;
    void deallocate_host_mem() override;
};


class gemm_test_i8i8col32 : public gemm_test_i8i8 {
private:
    const bool use_4r4_kernel = true;

protected:
    ppl::common::RetCode gemm() override;

public:
    gemm_test_i8i8col32() : gemm_test_i8i8() {
        alignment = 32;
    }
};


class gemm_test_fp16 : public gemm_test_i8i8 {
protected:
    ppl::common::RetCode gemm() override;

public:
    gemm_test_fp16() : gemm_test_i8i8() {
        alignment = 1;
        A_element_size = sizeof(fp16_t);
        B_element_size = sizeof(fp16_t);
        C_element_size = sizeof(fp16_t);
    };

    ppl::common::RetCode generate_data() override;
    void deallocate_host_mem() override;
};

#ifdef PPLNN_ENABLE_FP8

class gemm_test_fp8 : public gemm_test_fp16 {
protected:
    ppl::common::RetCode gemm() override;

public:
    gemm_test_fp8() : gemm_test_fp16() {
        alignment = 1;
        A_element_size = sizeof(fp8_t);
        B_element_size = sizeof(fp8_t);
        C_element_size = sizeof(fp16_t);
    };

    ppl::common::RetCode generate_data() override;
    void deallocate_host_mem() override;
};

#endif


class gemm_test_w4a16 : public gemm_test_base {
private:
    void* host_weight_scale;
    void* weight_scale;

protected:
    void* gemm_handle = nullptr; 
    ppl::common::RetCode gemm() override;
    
public:
    gemm_test_w4a16() : gemm_test_base() {
        alignment = 128;
        A_element_size = sizeof(fp16_t);
        B_element_size = 0.5;
        C_element_size = sizeof(fp16_t);
    }

    ppl::common::RetCode init_handle() override;
    ppl::common::RetCode generate_data() override;
    ppl::common::RetCode destroy_handle() override;
    ppl::common::RetCode init_scale() override;
    ppl::common::RetCode destroy_scale() override;
    void deallocate_host_mem() override;
};
