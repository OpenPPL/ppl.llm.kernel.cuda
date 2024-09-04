#include <ctime>
#include <cstdlib>
#include <memory>
#include <random>
#include "ppl/kernel/llm/cuda/pmx/i4f16/gemm.h"
#include "gemm_test.h"


#define CHECK_BITS_FP16(value) ((value & 0b0111110000000000) == 0b0111110000000000)
#define CHECK_BITS_FP8(value) ((value & 0b01111000) == 0b01111000)

#define RETCODE_CHECK(call) do { \
    ppl::common::RetCode __status = (call); \
    if (__status != ppl::common::RC_SUCCESS) { \
        return __status; \
    } \
} while (0)

#define CHECK_ALLOC(ptr) \
    if (!(ptr)) { \
        delete[] (ptr); \
        std::cerr << "ERROR! Host memory allocation failed." << std::endl; \
        return ppl::common::RC_HOST_MEMORY_ERROR; \
    }


template <typename T>
void deallocate_mem(void* &ptr) {
    delete[] static_cast<T*>(ptr);
    ptr = nullptr;
}


ppl::common::RetCode gemm_test_base::init_cuda() {
    CUDA_CHECK_RC(cudaStreamCreate(&stream));
    CUDA_CHECK_RC(cudaEventCreate(&start_event));
    CUDA_CHECK_RC(cudaEventCreate(&stop_event));
    return ppl::common::RC_SUCCESS;
}


void gemm_test_base::init_params(int64_t m, int64_t n, int64_t k) {
    M = m;
    N = (n + alignment - 1) / alignment * alignment;
    K = (k + alignment - 1) / alignment * alignment;

    lda = transa ? M : K;
    ldb = transb ? K : N;
    ldc = N;

    A_num_elements = M * K;
    B_num_elements = K * N;
    C_num_elements = M * N;

    A_num_bytes = A_num_elements * A_element_size;
    B_num_bytes = B_num_elements * B_element_size;
    C_num_bytes = C_num_elements * C_element_size;

    gops = (double)M * N * K * 2 / 1e9;
    gbs = (double)(A_num_bytes + B_num_bytes + C_num_bytes) / 1e9;
}


ppl::common::RetCode gemm_test_base::init_device_data() {
    CUDA_CHECK_RC(cudaMalloc(&workspace, workspace_size));
    CUDA_CHECK_RC(cudaMalloc((void**)&A, A_num_bytes));
    CUDA_CHECK_RC(cudaMalloc((void**)&B, B_num_bytes));
    CUDA_CHECK_RC(cudaMalloc((void**)&C, C_num_bytes));

    CUDA_CHECK_RC(cudaMemcpy(A, host_A, A_num_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_RC(cudaMemcpy(B, host_B, B_num_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_RC(cudaDeviceSynchronize());

    return ppl::common::RC_SUCCESS;
}


ppl::common::RetCode gemm_test_base::warmup(const int64_t tot_iter) {
    for (int64_t w = 0; w <= tot_iter; ++w) {
        RETCODE_CHECK(gemm());
    }
    CUDA_CHECK_RC(cudaStreamSynchronize(stream));
    return ppl::common::RC_SUCCESS;
};


ppl::common::RetCode gemm_test_base::benchmark(const int64_t tot_iter) {
    int64_t tot_exe_iter = 0;
    CUDA_CHECK_RC(cudaEventRecord(start_event, stream));
    for (; tot_exe_iter < tot_iter; ++tot_exe_iter) { 
        RETCODE_CHECK(gemm());
    }
    CUDA_CHECK_RC(cudaEventRecord(stop_event, stream));
    CUDA_CHECK_RC(cudaEventSynchronize(stop_event));
    CUDA_CHECK_RC(cudaEventElapsedTime(&tot_exe_time, start_event, stop_event));

    avg_exe_time = tot_exe_time / tot_exe_iter;
    avg_gflops = gops / (avg_exe_time / 1e3);
    avg_gbps = gbs / (avg_exe_time / 1e3);

    return ppl::common::RC_SUCCESS;
};


ppl::common::RetCode gemm_test_base::deallocate_cuda_mem() {
    CUDA_CHECK_RC(cudaFree(A));
    CUDA_CHECK_RC(cudaFree(B));
    CUDA_CHECK_RC(cudaFree(C));
    return ppl::common::RC_SUCCESS;
}


Result gemm_test_base::get_result() {
    return {tot_exe_time, avg_exe_time, avg_gflops, avg_gbps};
}


ppl::common::RetCode gemm_test_base::destroy_cuda() {
    CUDA_CHECK_RC(cudaEventDestroy(start_event));
    CUDA_CHECK_RC(cudaEventDestroy(stop_event));
    CUDA_CHECK_RC(cudaStreamDestroy(stream));
    return ppl::common::RC_SUCCESS;
}


ppl::common::RetCode gemm_test_base::init_scale() {
    return ppl::common::RC_SUCCESS;
}


ppl::common::RetCode gemm_test_base::destroy_scale() {
    return ppl::common::RC_SUCCESS;
}


ppl::common::RetCode gemm_test_i8i8::init_handle() {
    CUBLAS_CHECK_RC(cublasLtCreate(&gemm_handle));
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode gemm_test_i8i8::destroy_handle() {
    CUBLAS_CHECK_RC(cublasLtDestroy(gemm_handle));
    return ppl::common::RC_SUCCESS;
}


ppl::common::RetCode gemm_test_i8i8::generate_data() {
    int8_t* temp_A = new int8_t[A_num_elements];
    int8_t* temp_B = new int8_t[B_num_elements];
    CHECK_ALLOC(temp_A);
    CHECK_ALLOC(temp_B);
    
    for (int64_t i = 0; i < A_num_elements; i++) {
        temp_A[i] = rand() % 256 - 128;
    }
    for (int64_t i = 0; i < B_num_elements; i++) {
        temp_B[i] = rand() % 256 - 128;
    }
    
    host_A = static_cast<void*>(temp_A);
    host_B = static_cast<void*>(temp_B);

    return ppl::common::RC_SUCCESS;
}


void gemm_test_i8i8::deallocate_host_mem() {
    deallocate_mem<int8_t>(host_A);
    deallocate_mem<int8_t>(host_B);
}


ppl::common::RetCode gemm_test_i8i8::gemm() {
    RETCODE_CHECK(gemm_i8i8i32(
        stream,
        gemm_handle,
        algo,
        transa,
        lda,
        ppl::common::DATATYPE_INT8,
        A,
        transb,
        ldb,
        ppl::common::DATATYPE_INT8,
        B,
        bias,
        M,
        N,
        K,
        alpha,
        beta,
        workspace_size,
        workspace,
        &algo_cache,
        ldc,
        ppl::common::DATATYPE_INT32,
        C
    ));

    return ppl::common::RC_SUCCESS;
}


ppl::common::RetCode gemm_test_i8i8col32::gemm() {
    RETCODE_CHECK(ppl::kernel::llm::cuda::cublas::gemm_i8i8i32_col32(
        stream,
        gemm_handle,
        A,
        B,
        M,
        N,
        K,
        use_4r4_kernel,
        C
    ));

    return ppl::common::RC_SUCCESS;
}


ppl::common::RetCode gemm_test_fp16::generate_data() {
    fp16_t* temp_A = new fp16_t[A_num_elements];
    fp16_t* temp_B = new fp16_t[B_num_elements];
    CHECK_ALLOC(temp_A);
    CHECK_ALLOC(temp_B);

    for (int64_t i = 0; i < A_num_elements; i++) {
        int16_t random_int;
        do {
            random_int = rand() % 65536 - 32768;
        } while (CHECK_BITS_FP16(random_int));
        memcpy(&temp_A[i], &random_int, sizeof(int16_t));
    }
    for (int64_t i = 0; i < B_num_elements; i++) {
        int16_t random_int;
        do {
            random_int = rand() % 65536 - 32768;
        } while (CHECK_BITS_FP16(random_int));
        memcpy(&temp_B[i], &random_int, sizeof(int16_t));
    }
    host_A = static_cast<void*>(temp_A);
    host_B = static_cast<void*>(temp_B);

    return ppl::common::RC_SUCCESS;
}


void gemm_test_fp16::deallocate_host_mem() {
    deallocate_mem<fp16_t>(host_A);
    deallocate_mem<fp16_t>(host_B);
}


ppl::common::RetCode gemm_test_fp16::gemm() {
    RETCODE_CHECK(ppl::kernel::llm::cuda::cublas::gemm(
        stream,
        gemm_handle,
        algo,
        transa,
        lda,
        ppl::common::DATATYPE_FLOAT16,
        A,
        transb,
        ldb,
        ppl::common::DATATYPE_FLOAT16,
        B,
        bias,
        M,
        N,
        K,
        alpha,
        beta,
        workspace_size,
        workspace,
        ldc,
        ppl::common::DATATYPE_FLOAT16,
        C
    ));

    return ppl::common::RC_SUCCESS;
}

#ifdef PPLNN_ENABLE_FP8

ppl::common::RetCode gemm_test_fp8::generate_data() {
    fp8_t* temp_A = new fp8_t[A_num_elements];
    fp8_t* temp_B = new fp8_t[B_num_elements];
    CHECK_ALLOC(temp_A);
    CHECK_ALLOC(temp_B);

    for (int64_t i = 0; i < A_num_elements; i++) {
        int8_t random_int;
        do {
            random_int = rand() % 256 - 128;
        } while (CHECK_BITS_FP8(random_int));
        memcpy(&temp_A[i], &random_int, sizeof(int8_t));
    }
    for (int64_t i = 0; i < B_num_elements; i++) {
        int8_t random_int;
        do {
            random_int = rand() % 256 - 128;
        } while (CHECK_BITS_FP8(random_int));
        memcpy(&temp_B[i], &random_int, sizeof(int8_t));
    }
    host_A = static_cast<void*>(temp_A);
    host_B = static_cast<void*>(temp_B);

    return ppl::common::RC_SUCCESS;
}


void gemm_test_fp8::deallocate_host_mem() {
    deallocate_mem<fp8_t>(host_A);
    deallocate_mem<fp8_t>(host_B);
}


ppl::common::RetCode gemm_test_fp8::gemm() {
    RETCODE_CHECK(ppl::kernel::llm::cuda::cublas::gemm_fp8(
        stream,
        gemm_handle,
        algo,
        transa,
        lda,
        ppl::common::DATATYPE_FLOAT8E4M3,
        A,
        transb,
        ldb,
        ppl::common::DATATYPE_FLOAT8E4M3,
        B,
        bias,
        M,
        N,
        K,
        alpha,
        beta,
        workspace_size,
        workspace,
        ldc,
        ppl::common::DATATYPE_FLOAT16,
        C
    ));

    return ppl::common::RC_SUCCESS;
}

#endif


ppl::common::RetCode gemm_test_w4a16::init_handle() {
    gemm_handle = ppl::kernel::llm::cuda::pmx::i4f16::create_gemm_handle();
    if (gemm_handle == nullptr) {
        fprintf(stderr, "pmx::i4f16::create_gemm_handle failed.");
        return ppl::common::RC_INTERNAL_ERROR;
    }
    return ppl::common::RC_SUCCESS;
}


ppl::common::RetCode gemm_test_w4a16::destroy_handle() {
    ppl::kernel::llm::cuda::pmx::i4f16::destory_gemm_handle(gemm_handle);
    return ppl::common::RC_SUCCESS;
}


ppl::common::RetCode gemm_test_w4a16::generate_data() {
    int64_t temp_B_elements = (B_num_bytes + 3) / 4;

    fp16_t* temp_A = new fp16_t[A_num_elements];
    int32_t* temp_B = new int32_t[temp_B_elements];
    CHECK_ALLOC(temp_A);
    CHECK_ALLOC(temp_B);

    for (int64_t i = 0; i < A_num_elements; i++) {
        int16_t random_int;
        do {
            random_int = rand() % 65536 - 32768;
        } while (CHECK_BITS_FP16(random_int));
        memcpy(&temp_A[i], &random_int, sizeof(int16_t));
    }

    for (int64_t i = 0; i < temp_B_elements; i++) {
        temp_B[i] = rand() % 4294967296 - 2147483648;
    }
    host_A = static_cast<void*>(temp_A);
    host_B = static_cast<void*>(temp_B);

    return ppl::common::RC_SUCCESS;
}


void gemm_test_w4a16::deallocate_host_mem() {
    deallocate_mem<fp16_t>(host_A);
    deallocate_mem<int32_t>(host_B);
}


ppl::common::RetCode gemm_test_w4a16::gemm() {
    RETCODE_CHECK(ppl::kernel::llm::cuda::pmx::i4f16::gemm(
        stream,
        gemm_handle,
        A,
        B,
        weight_scale,
        bias,
        M,
        N,
        K,
        workspace_size,
        workspace,
        C));
        
    return ppl::common::RC_SUCCESS;
}


ppl::common::RetCode gemm_test_w4a16::init_scale() {
    int64_t scale_elements = N * (K/128);
    fp16_t* temp_scale = new fp16_t[scale_elements];
    CHECK_ALLOC(temp_scale);

    for (int64_t i = 0; i < scale_elements; i++) {
        int16_t random_int;
        do {
            random_int = rand() % 65536 - 32768;
        } while (CHECK_BITS_FP16(random_int));
        memcpy(&temp_scale[i], &random_int, sizeof(int16_t));
    }
    host_weight_scale = static_cast<void*>(temp_scale);

    int64_t scale_bytes = scale_elements * sizeof(fp16_t);
    CUDA_CHECK_RC(cudaMalloc((void**)&weight_scale, scale_bytes));
    CUDA_CHECK_RC(cudaMemcpy(weight_scale, host_weight_scale, scale_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_RC(cudaDeviceSynchronize());
    return ppl::common::RC_SUCCESS;
}


ppl::common::RetCode gemm_test_w4a16::destroy_scale() {
    CUDA_CHECK_RC(cudaFree(weight_scale));
    deallocate_mem<fp16_t>(host_weight_scale);
    return ppl::common::RC_SUCCESS;
}