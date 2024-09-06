#ifdef PPLNN_ENABLE_FP8

#include "ppl/kernel/llm/cuda/pmx/f8f8/column_parallel_linear.h"
#include "ppl/common/log.h"

#include "cudakernel/memory/transpose.h"
#include "ppl/kernel/llm/cuda/pmx/f8f8/cast.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx { namespace f8f8 {


ppl::common::RetCode column_parallel_linear(
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
    const bool gather_output,
    void* gather_buffer,
    const int64_t cublas_workspace_size,
    void* cublas_workspace,
    const ppl::common::TensorShape* output_shape,
    void* output)
{
    // input (M, K)
    // weight (N/w, K)
    // gemm_output (M, N/w)
    // output (M, N)

    if (bias && bias_shape->GetDataType() != ppl::common::DATATYPE_FLOAT16) {
        LOG(ERROR) << "only support fp16 bias";
        return ppl::common::RC_UNSUPPORTED;
    }

    const int64_t M = input_shape->CalcElementsToDimensionIncludingPadding(input_shape->GetDimCount() - 1);
    const int64_t Nw = out_features / nccl_param->size;
    const int64_t K = in_features;

    void *gemm_output = output;
    if (gather_output && nccl_param->size > 1) {
        gemm_output = (char*)gather_buffer
            + nccl_param->rank * M * Nw * ppl::common::GetSizeOfDataType(output_shape->GetDataType());
    }

    ppl::common::RetCode status;

    status = ppl::kernel::llm::cuda::cublas::gemm_fp8(
        stream,
        cublaslt_handle,
        algo,
        false,
        K,
        input_shape->GetDataType(),
        input,
        true,
        K,
        weight_shape->GetDataType(),
        weight,
        bias,
        M,
        Nw,
        K,
        1.0f,
        0.0f,
        cublas_workspace_size,
        cublas_workspace,
        Nw,
        output_shape->GetDataType(),
        gemm_output);

    if (ppl::common::RC_SUCCESS != status)
        return status;


    if (gather_output && nccl_param->size > 1) {
        status = ppl::common::NcclAllGather<half>(
            (half*)gemm_output,
            (half*)gather_buffer,
            M * Nw,
            nccl_param,
            stream);
        if (ppl::common::RC_SUCCESS != status)
            return status;

        // gather_buffer(w, M, N/w)
        status = PPLCUDATranspose01ForwardImp(
            stream, gather_buffer,
            output_shape->GetDataType(),
            nccl_param->size,
            M,
            Nw,
            output);
        if (ppl::common::RC_SUCCESS != status)
            return status;
    }

    return ppl::common::RC_SUCCESS;
}

}}}}}}

#endif