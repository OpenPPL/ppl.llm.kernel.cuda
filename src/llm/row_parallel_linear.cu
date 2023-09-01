#include "cudakernel/llm/row_parallel_linear.h"
#include "cudakernel/llm/matmul_cublas.h"
#include "cudakernel/memory/transpose.h"

ppl::common::RetCode PPLCUDARowParallelLinearForwardImp(
            const cudaStream_t stream,
            const cublasLtHandle_t& ltHandle,
            const GemmKernelParam &param, 
            ppl::common::NcclParam* nccl_param,
            const ppl::common::TensorShape* input_shape,
            void* input,
            void* split_buffer,
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
            bool input_is_parallel,
            cublasLtMatmulAlgo_t algo)
{
    if(!input_is_parallel) {
        // TODO
        return ppl::common::RC_UNSUPPORTED;
    }

    auto status = PPLCUDAMatMulCublasForwardImp(stream, ltHandle, param, input_shape, input,
                        weight_shape, weight, output_shape, output,
                        bias_shape, bias, batch, workspace, workspaceSize,
                        use_heuristic, algo);

    if (ppl::common::RC_SUCCESS != status)
        return status;

    if (nccl_param->size > 1) {
        int sendcount = output_shape->CalcElementsIncludingPadding();
        return ppl::common::NcclAllReduceSum<half>((half*)output, (half*)output, sendcount, nccl_param, stream);
    }

    return ppl::common::RC_SUCCESS;
}


