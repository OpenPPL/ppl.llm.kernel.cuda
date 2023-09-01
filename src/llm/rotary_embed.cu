#include "cudakernel/llm/rotary_embed.h"
#include "ppl/common/tensor_shape.h"
#include "cudakernel/common/cuda_check.h"
#include "cudakernel/common/common.cuh"
#include <cuda_fp16.h>


/**
 * Rotary Embedding Cuda impl.
 *
 * @param dim_offset: Represents the position (offset) of the current element in the hidden_dimension.
 * @param seq_offset: Represents the position (offset) of the current element in the sequence dimension.

 * @param hidden_dim: Total size of hidden_dimension.
 * @param theta parameter used to compute freq.
 */
inline __device__
float2 __RotaryEmbedding(
    const int64_t dim_offset, 
    const int64_t seq_offset,
    const int64_t hidden_dim,
    const float theta
){
    // fp16 does not have __sincosf instruction.
    // So we have only fp32 implementation of Rotary Embedding.
    float2 ret = {0.0f, 0.0f};
    const float freq = 1.0f / __powf(theta, (dim_offset / (float)hidden_dim)) * seq_offset;
    __sincosf(freq, &ret.y, &ret.x);
    return ret;
}

template<int TPB>
__global__
void _ApplyRotaryEmbeddingQK(
    const half2 *input_q,
    const half2 *input_k,
    const int64_t size_per_head,
    const int64_t num_heads,
    const int64_t* cu_start_pos_ptr,
    const int64_t start_pos_val,
    const int64_t seqlen,
    const int64_t num_of_elements,
    const float theta,
    const int32_t bypass_k,
    half2 *out_q,
    half2 *out_k
){
    // in this kernel:
    // blockIdx.x is batch_offset.
    // blockIdx.y is seq_offset.
    // blockIdx.z is dim_offset inside a head.
    int64_t start_pos = start_pos_val;
    if (cu_start_pos_ptr != nullptr)
        start_pos = cu_start_pos_ptr[0];
    const int64_t idx = (
        blockIdx.x * size_per_head * num_heads * seqlen + 
        blockIdx.y * size_per_head * num_heads + 
        blockIdx.z * TPB + threadIdx.x);

    if (blockIdx.z * TPB + threadIdx.x < size_per_head * num_heads) {
        const int64_t dim_offset = (blockIdx.z * TPB + threadIdx.x) % size_per_head;
        const int64_t seq_offset = start_pos + blockIdx.y;

        const float2 q = __half22float2(input_q[idx]);
        const float2 b = __RotaryEmbedding(
            dim_offset, seq_offset, size_per_head, theta);

        out_q[idx].x = __float2half(q.x * b.x - q.y * b.y);
        out_q[idx].y = __float2half(q.y * b.x + q.x * b.y);

        if (bypass_k) {
            out_k[idx] = input_k[idx];
        } else {
            const float2 k = __half22float2(input_k[idx]);
            out_k[idx].x = __float2half(k.x * b.x - k.y * b.y);
            out_k[idx].y = __float2half(k.y * b.x + k.x * b.y);
        }
    }
}

ppl::common::RetCode PPLCUDARotaryEmbQKForwardImp(
    cudaStream_t stream,
    const void* input_q,
    const void* input_k,
    const float theta,
    const void* cu_start_pos,
    const int64_t start_pos_val,
    const int32_t type, // 0 for llama, 1 for palm
    const int32_t bypass_k,
    ppl::common::TensorShape* input_shape,
    void* output_q,
    void* output_k)
{
    PPL_CHECK(input_shape->GetDimCount() == 4, "input's dim should be 4"); // [bsz, seqlen, self.n_local_heads, self.head_dim]
    PPL_CHECK(input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16, "RotaryEmb only support fp16");

    const int64_t batchsize       = input_shape->GetDim(0);
    const int64_t seqlen          = input_shape->GetDim(1);
    const int64_t num_heads       = input_shape->GetDim(2);
    const int64_t size_per_head   = input_shape->GetDim(3);
    const int64_t num_of_elements = batchsize * seqlen * num_heads * size_per_head;
    PPL_CHECK(size_per_head % 2 == 0, "size_per_head should be an even number");
    constexpr int32_t VPT = 2;
    constexpr int32_t TPB = 512;

    const dim3 grid = {
        (unsigned int)batchsize, (unsigned int)seqlen,
        (unsigned int)((num_heads * size_per_head / (TPB * VPT)) + (num_heads * size_per_head % (TPB * VPT) != 0))
    };
    // const int32_t grid_size = (num_of_elements / (TPB * VPT)) + (num_of_elements % (TPB * VPT) != 0);


    if(type == 0) {
        _ApplyRotaryEmbeddingQK<TPB>
        <<<grid, TPB, 0, stream>>>
        (
            (const half2*)input_q, (const half2*)input_k, size_per_head / 2, num_heads, 
            (const int64_t*)cu_start_pos, start_pos_val, seqlen, num_of_elements / 2, theta,
            bypass_k, (half2*)output_q, (half2*)output_k
        );
    } else {
        PPL_CHECK(false, "RotaryEmbed only support type 0");
    }
    return ppl::common::RC_SUCCESS;
}

__global__
void _ApplyRotaryEmbeddingQKDynamicbatch(
    const half2 *input_q, //(1,S,H,D)
    const half2 *input_k,
    const int64_t* seqstart_q,
    const int64_t batch,
    const int64_t size_per_head,
    const int64_t num_heads,
    const int64_t* start_pos,
    const float theta,
    const int32_t bypass_k,
    half2 *out_q,
    half2 *out_k
){
    if(blockIdx.y < seqstart_q[blockIdx.x + 1] - seqstart_q[blockIdx.x]) {

        const int64_t batch_id = blockIdx.x;
        const int64_t seq_id = blockIdx.y;

        for(int tid = threadIdx.x; tid < num_heads * size_per_head; tid += blockDim.x) {
            const int64_t head_id = tid / size_per_head;
            const int64_t dim_offset = tid % size_per_head;
            const int64_t idx = (seqstart_q[batch_id] + seq_id) * num_heads * size_per_head + head_id * size_per_head + dim_offset; 
            int64_t seq_offset = seq_id + start_pos[batch_id];

            const float2 q = __half22float2(input_q[idx]);
            const float2 b = __RotaryEmbedding(
                dim_offset, seq_offset, size_per_head, theta);

            out_q[idx].x = __float2half(q.x * b.x - q.y * b.y);
            out_q[idx].y = __float2half(q.y * b.x + q.x * b.y);

            if (bypass_k) {
                out_k[idx] = input_k[idx];
            } else {
                const float2 k = __half22float2(input_k[idx]);
                out_k[idx].x = __float2half(k.x * b.x - k.y * b.y);
                out_k[idx].y = __float2half(k.y * b.x + k.x * b.y);
            }
        }
    }
}

ppl::common::RetCode PPLCUDARotaryEmbQKDynamicBatchForwardImp(
    cudaStream_t stream,
    ppl::common::TensorShape* input_shape,
    const void* input_q, // (S,H,D)
    const void* input_k,
    const void* seqstart_q,
    const float theta,
    ppl::common::TensorShape* start_pos_shape,
    const void* start_pos,
    const int32_t type, // 0 for llama, 1 for palm
    const int32_t bypass_k,
    void* output_q,
    void* output_k,
    const int64_t max_seqlen)
{
    PPL_CHECK(input_shape->GetDimCount() == 3, "dynamic batch input's dim should be 3"); // [fuse_seqlen, self.n_local_heads, self.head_dim]
    PPL_CHECK(input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16, "RotaryEmb only support fp16");
    const int64_t batch = start_pos_shape->GetDim(0);
    const int64_t fuse_seqlen     = input_shape->GetDim(0);
    const int64_t num_heads       = input_shape->GetDim(1);
    const int64_t size_per_head   = input_shape->GetDim(2);
    const int64_t num_of_elements = fuse_seqlen * num_heads * size_per_head;
    PPL_CHECK(size_per_head % 2 == 0, "size_per_head should be an even number");
    int32_t TPB = GetBlockSize(num_of_elements / 2);


    if(type == 0) {
        dim3 grid(batch, max_seqlen);
        _ApplyRotaryEmbeddingQKDynamicbatch
        <<<grid, TPB, 0, stream>>>
        (
            (const half2*)input_q, (const half2*)input_k, (int64_t*)seqstart_q, batch, size_per_head / 2, num_heads, 
            (const int64_t*)start_pos, theta,
            bypass_k, (half2*)output_q, (half2*)output_k
        );
    } else {
        PPL_CHECK(false, "RotaryEmbed only support type 0");
    }
    return ppl::common::RC_SUCCESS;
}
