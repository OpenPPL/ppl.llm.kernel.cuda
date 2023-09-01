#include "cudakernel/llm/column_parallel_linear.h"
#include "cudakernel/llm/matmul_cublas.h"
#include "cudakernel/memory/transpose.h"

ppl::common::RetCode PPLCUDAColumnParallelLinearForwardImp(
            const cudaStream_t stream,
            const cublasLtHandle_t& ltHandle,
            const GemmKernelParam &param, 
            ppl::common::NcclParam* nccl_param,
            const ppl::common::TensorShape* input_shape,
            const void* input,
            const ppl::common::TensorShape* weight_shape,
            const void* weight,
            const ppl::common::TensorShape* output_shape,
            void* output,
            void* gather_buffer,
            const ppl::common::TensorShape* bias_shape,
            const void* bias,
            void* workspace,
            size_t workspaceSize,
            bool use_heuristic,
            bool gather_output,
            cublasLtMatmulAlgo_t algo)
{

    void *matmul_output = output;
    ppl::common::TensorShape matmul_output_shape = *output_shape;
    int64_t matmul_output_elem_per_part = 0;
    if(gather_output && nccl_param->size > 1) {
        // weight_shape (N/w, K)
        // matmul_output_elem_per_part batch*seqlen*N/w
        matmul_output_shape.SetDim(matmul_output_shape.GetDimCount() - 1, weight_shape->GetDim(0));
        matmul_output_elem_per_part = matmul_output_shape.CalcElementsExcludingPadding();
        matmul_output = (half*)gather_buffer + nccl_param->rank * matmul_output_elem_per_part;
    }

    auto status = PPLCUDAMatMulCublasForwardImp(stream, ltHandle, param, input_shape, input,
                        weight_shape, weight, &matmul_output_shape, matmul_output,
                        bias_shape, bias, 1, workspace, workspaceSize,
                        use_heuristic, algo);

    if (ppl::common::RC_SUCCESS != status)
        return status;

    if(gather_output && nccl_param->size > 1) {
        status = ppl::common::NcclAllGather<half>(
            (half*)matmul_output, (half*)gather_buffer, matmul_output_elem_per_part, nccl_param, stream);
        if (ppl::common::RC_SUCCESS != status)
            return status;

        // gather_buffer(w, batch*seqlen, N/w)
        status = PPLCUDATranspose01ForwardImp(
            stream, gather_buffer, 
            output_shape->GetDataType(), nccl_param->size,
            matmul_output_elem_per_part / weight_shape->GetDim(0),
            weight_shape->GetDim(0), output);
        if (ppl::common::RC_SUCCESS != status)
            return status;
    }

    return ppl::common::RC_SUCCESS;
}


