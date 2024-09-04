#ifdef PPLNN_ENABLE_FP8

#ifndef __PPL_KERNEL_LLM_CUDA_PMX_F8F8_ROW_PARALLEL_LINEAR_H__
#define __PPL_KERNEL_LLM_CUDA_PMX_F8F8_ROW_PARALLEL_LINEAR_H__

#include "ppl/kernel/llm/cuda/common/general_include.h"

#include "ppl/kernel/llm/cuda/cublas/gemm.h"
#include "ppl/common/cuda/nccl_utils.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx { namespace f8f8 {

ppl::common::RetCode row_parallel_linear(
    const cudaStream_t stream,
    const cublasLtHandle_t& cublaslt_handle,
    const cublasLtMatmulAlgo_t* algo,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* weight_shape,
    const void* weight,
    const ppl::common::TensorShape* bias_shape,
    const void* bias,
    const int64_t in_features,
    const int64_t out_features,
    const ppl::common::NcclParam* nccl_param,
    const bool input_is_parallel,
    void* split_buffer,
    const int64_t cublas_workspace_size,
    void* cublas_workspace,
    const ppl::common::TensorShape* output_shape,
    void* output);


}}}}}}

#endif
#endif