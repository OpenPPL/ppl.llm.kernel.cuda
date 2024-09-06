#ifdef PPLNN_ENABLE_FP8

#include "ppl/kernel/llm/cuda/pmx/f8f8/row_parallel_linear.h"
#include "ppl/common/log.h"

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
    void* output)
{
    if (!input_is_parallel) {
        LOG(ERROR) << "currnetly only support parallel input";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (bias && bias_shape->GetDataType() != ppl::common::DATATYPE_FLOAT16) {
        LOG(ERROR) << "only support fp16 bias";
        return ppl::common::RC_UNSUPPORTED;
    }

    // input (M, K/w)
    // weight (N, K/w)
    // output (M, N)

    const int64_t M = input_shape->CalcElementsToDimensionIncludingPadding(input_shape->GetDimCount() - 1);
    const int64_t N = out_features;
    const int64_t Kw = in_features / nccl_param->size;

    const void* reduce_bias = nccl_param->rank == 0 ? bias : nullptr;

    ppl::common::RetCode status;

    status = ppl::kernel::llm::cuda::cublas::gemm_fp8(
        stream,
        cublaslt_handle,
        algo,
        false,
        Kw,
        input_shape->GetDataType(),
        input,
        true,
        Kw,
        weight_shape->GetDataType(),
        weight,
        reduce_bias,
        M,
        N,
        Kw,
        1.0f,
        0.0f,
        cublas_workspace_size,
        cublas_workspace,
        N,
        output_shape->GetDataType(),
        output);

    if (ppl::common::RC_SUCCESS != status)
        return status;

    if (nccl_param->size > 1) {
        return ppl::common::NcclAllReduceSum<half>(
            (half*)output,
            (half*)output,
            M * N,
            nccl_param,
            stream);
    }

    return ppl::common::RC_SUCCESS;
}

}}}}}}

#endif