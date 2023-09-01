#include "cudakernel/llm/parallel_embedding.h"
#include "cudakernel/memory/transpose.h"
#include "cudakernel/common/cuda_check.h"
#include "cudakernel/common/common.cuh"

template<int VPT, int TPB>
__global__ 
void embedding_kernel(const int64_t* indices_buff, const half* weight_buff, half* output, const int64_t emb_dim) {
    int64_t indice = indices_buff[blockIdx.x];
    int64_t weight_idx = indice * emb_dim + threadIdx.x * VPT;
    int64_t output_idx = blockIdx.x * emb_dim + threadIdx.x * VPT;
    copy<sizeof(half) * VPT>(&weight_buff[weight_idx], &output[output_idx]);
}

ppl::common::RetCode PPLCUDAParallelEmbeddingForwardImp(
            const cudaStream_t stream,
            ppl::common::NcclParam* nccl_param,
            const ppl::common::TensorShape* indices_shape,
            const void* indices,
            const ppl::common::TensorShape* weight_shape,
            const void* weight,
            const ppl::common::TensorShape* output_shape,
            void* output,
            const float max_norm,
            const float norm_type,
            void* gather_buffer)
{
    const int64_t emb_size = weight_shape->GetDim(0);
    const int64_t emb_dim = weight_shape->GetDim(1);
    const int64_t num_indices = indices_shape->CalcElementsIncludingPadding();
    constexpr int32_t VPT = 16 / sizeof(half);

    if (nccl_param->size == 1) {
      switch (emb_dim)
      {
        case 4096:
          embedding_kernel<VPT, 4096 / VPT>
            <<<num_indices, 4096 / VPT, 0, stream>>>((int64_t*)indices, (half*)weight, (half*)output, emb_dim);
          break;
        default:
          PPL_CHECK(false, "ParallelEmbedding not support this shape");
      }
    } else {
      int output_elem_per_part = num_indices * emb_dim;

      switch (emb_dim)
      {
        case 4096:
          embedding_kernel<VPT, 4096 / VPT>
            <<<num_indices, 4096 / VPT, 0, stream>>>(
              (int64_t*)indices, (half*)weight,
              (half*)gather_buffer + nccl_param->rank * output_elem_per_part,
              emb_dim
            );
          break;
        case 2560:
          embedding_kernel<VPT, 2560 / VPT>
            <<<num_indices, 2560 / VPT, 0, stream>>>(
              (int64_t*)indices, (half*)weight,
              (half*)gather_buffer + nccl_param->rank * output_elem_per_part,
              emb_dim
            );
          break;
        case 1024:
          embedding_kernel<VPT, 1024 / VPT>
            <<<num_indices, 1024 / VPT, 0, stream>>>(
              (int64_t*)indices, (half*)weight,
              (half*)gather_buffer + nccl_param->rank * output_elem_per_part,
              emb_dim
            );
          break;
        default:
          PPL_CHECK(false, "ParallelEmbedding not support this shape");
      }

      auto status = ppl::common::NcclAllGather<half>(
          (half*)gather_buffer + nccl_param->rank * output_elem_per_part,
          (half*)gather_buffer, output_elem_per_part, nccl_param, stream);
      if (ppl::common::RC_SUCCESS != status)
        return status;

      status = PPLCUDATranspose01ForwardImp(
        stream, gather_buffer, output_shape->GetDataType(),
        nccl_param->size, num_indices, emb_dim, output);
      if (ppl::common::RC_SUCCESS != status)
        return status;
    }

    return ppl::common::RC_SUCCESS;
}