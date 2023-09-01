#include "xformer_fmha/xformer_fmha.h"
#include "ppl/common/tensor_shape.h"
#include "cudakernel/llm/multi_head_attention.h"
#include "ppl/common/retcode.h"
#include "cudakernel/common/common.cuh"
#include "cudakernel/common/cuda_check.h"
#include <float.h> // need for FLT_MAX

template<int32_t THREAD_GROUP_SIZE, int32_t ELEMENT_NUM>
__device__ inline
float __AttnThreadGroupDot(half *local_Q, half *local_K){
    // Helper function for QK Dot.
    // [TODO] It should be optimized by type fp32x4.

    float qk = 0.0f;
# pragma unroll
    for(int32_t i = 0; i < ELEMENT_NUM; i++) {
        qk += __half2float(local_Q[i]) * __half2float(local_K[i]);
    }
#pragma unroll
    for (int32_t mask = THREAD_GROUP_SIZE / 2; mask >= 1; mask /= 2) {
        qk += __shfl_xor_sync(uint32_t(-1), qk, mask);
    }
    return qk;
}

template<int32_t WPT>
__device__ inline
float __AttnBlockReduceMax(float reducing, float *shared_mem){
    // Helper function for reduce softmax qkmax.
    constexpr int32_t WARP_SIZE = 32;
    const int32_t lane_id = threadIdx.x % WARP_SIZE;
    const int32_t warp_id = threadIdx.x / WARP_SIZE;

# pragma unroll
    for (int32_t mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
        reducing = fmaxf(reducing, __shfl_xor_sync(uint32_t(-1), reducing, mask));
    }

    if (lane_id == 0) {
        shared_mem[warp_id] = reducing;
    }
    __syncthreads();

    if (lane_id < WPT) reducing = shared_mem[lane_id];
    else reducing = -FLT_MAX;

# pragma unroll
    for (int32_t mask = WPT / 2; mask >= 1; mask /= 2) {
        reducing = fmaxf(reducing, __shfl_xor_sync(uint32_t(-1), reducing, mask));
    }

    reducing = __shfl_sync(uint32_t(-1), reducing, 0);
    return reducing;
}

template<int32_t WPT>
__device__ inline
float __AttnBlockReduceSum(float reducing, float *shared_mem){
    // Helper function for reduce softmax exp sum.
    constexpr int32_t WARP_SIZE = 32;
    const int32_t lane_id = threadIdx.x % WARP_SIZE;
    const int32_t warp_id = threadIdx.x / WARP_SIZE;

# pragma unroll
    for (int32_t mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
        reducing += __shfl_xor_sync(uint32_t(-1), reducing, mask);
    }

    if (lane_id == 0) shared_mem[warp_id] = reducing;
    __syncthreads();

    if (lane_id < WPT) reducing = shared_mem[lane_id];

# pragma unroll
    for (int32_t mask = WPT / 2; mask >= 1; mask /= 2) {
        reducing += __shfl_xor_sync(uint32_t(-1), reducing, mask);
    }
    reducing = __shfl_sync(uint32_t(-1), reducing, 0);
    return reducing;
}

template<
    int32_t HEAD_SIZE,
    int32_t THREAD_GROUP_SIZE,        // how many threads inside a group
    int32_t TPB>
__global__
void _DecodingAttention_fp16(
    half* __restrict__ out,          // [context_lens, num_heads, head_size]
    const half* __restrict__ Q,      // [seq_lens, num_heads, head_size]
    const half* __restrict__ K,      // [context_lens, num_heads, head_size]
    const half* __restrict__ V,      // [context_lens, num_heads, head_size]
    const float  attn_scale,
    const int64_t *context_lens_cumsum // something like: [0, 160, 160+2, 160+2+3]
) {
    /***
    * You have to remember that this Kernel was created by a brother on the night of July 20, 2023. On that day,
    * Beijing experienced the strongest rainstorm since the beginning of summer.

    DecodingAttention is a special operator designed specifically for large language models(LLM) decoding. 
    
    It requires that the length of each input Query is always 1, 
        while the Key and Value can have different lengths.

    This operator supports padding removal optimization, meaning that Q, K, and V all need to have their tokens 
        concentrated in one sentence for input, with shapes like Q: [seq_lens, num_heads, head_size], 
        and K: [context_lens, num_heads, head_size].

    Since the Query sentence length is always 1, this operator is literally a fused matrix-vector multiplications operation.
        It does not utilize tensor cores for computation. 

    The calculation logic is divided into three steps: gemv(QK) + softmax(Attention) + gemv(KV). 
        In the provided code, it has already been split into these three parts.
    ***/

    /* --- Decoding Attention Kernel Implementation --- */
    constexpr int64_t WARP_SIZE = 32;                              // warp size
    constexpr int64_t WPT       = TPB / WARP_SIZE;                 // warp per thread block
    constexpr int64_t GPW       = WARP_SIZE / THREAD_GROUP_SIZE;       // thread group per warp
    constexpr int64_t GPT       = WARP_SIZE / THREAD_GROUP_SIZE * WPT; // thread group per thread block

    const int64_t head_idx      = blockIdx.x;
    const int64_t seq_idx       = blockIdx.y;
    const int64_t num_heads     = gridDim.x;
    // const int64_t num_seqs      = gridDim.y;
    constexpr int64_t VEC_SIZE  = 16 / sizeof(half);  // 128 bits

    // ------------------------------------------------ //
    // Step 1. Load Q into Thread Reg.
    constexpr int64_t VEC_LEN = (HEAD_SIZE / VEC_SIZE) / THREAD_GROUP_SIZE;

    static_assert((HEAD_SIZE / THREAD_GROUP_SIZE) % VEC_SIZE == 0);
    static_assert(HEAD_SIZE % THREAD_GROUP_SIZE == 0);

    // The elements in Q, K, and V will be evenly distributed across each thread group.
    half local_Q[VEC_SIZE * VEC_LEN];

    const int64_t warp_id       = threadIdx.x / WARP_SIZE;
    const int64_t warp_lane_id  = threadIdx.x % WARP_SIZE;
    const int64_t group_id      = warp_lane_id / THREAD_GROUP_SIZE;
    const int64_t group_lane_id = warp_lane_id % THREAD_GROUP_SIZE;
    const int64_t token_offset  = context_lens_cumsum[seq_idx];
    const int64_t token_stride  = num_heads * HEAD_SIZE;

    #pragma unroll
    for (int64_t i = 0; i < VEC_LEN; i++) {
        // copy 128(16 * 8) bits from Q to Local Q

        copy<sizeof(half) * VEC_SIZE>(
            &Q[seq_idx * token_stride + head_idx * HEAD_SIZE + 
            (group_lane_id + i * THREAD_GROUP_SIZE) * VEC_SIZE], 
            &local_Q[i * VEC_SIZE]);
    }
    // ------------------------------------------------ //
    // Step 2. Solve QK Dot

    // In the process of handling the QK matrix multiplication, we will divide a complete Thread Warp into several Thread groups.
    // Each thread group reads the entire Query and saves it in registers. 
    // Then, each thread group iterates through the vectors in the Key and performs dot products with the Query. 
    // During this process, a WARP performs multiple vector dot product operations at once. 
    // At the same time, we also record the maximum value of the dot product results for later use in the softmax operation.
    const int64_t context_len = context_lens_cumsum[seq_idx + 1] - context_lens_cumsum[seq_idx];
    extern __shared__ float logits[];
    float qk_max = -FLT_MAX;

    for (int64_t base_id = warp_id * GPW; base_id < context_len; base_id += GPT) {
        half local_K[VEC_SIZE * VEC_LEN];
        int64_t token_id = base_id + group_id;

        // all thread groups within a warp must be launched together.
        if (token_id >= context_len){
            memset(local_K, 0, sizeof(local_K));
        }
        else{
        # pragma unroll
            for (int64_t i = 0; i < VEC_LEN; i++) {
                // copy 128(16 * 8) bits from K to Local K
                copy<sizeof(half) * VEC_SIZE>(
                    &K[(token_offset + token_id) * token_stride + head_idx * HEAD_SIZE + 
                    (group_lane_id + i * THREAD_GROUP_SIZE) * VEC_SIZE], 
                    &local_K[i * VEC_SIZE]);
            }
        }

        // Ready for QK Dot
        float qk_dot = attn_scale * __AttnThreadGroupDot<THREAD_GROUP_SIZE, VEC_LEN * VEC_SIZE>(local_Q, local_K);

        if (group_lane_id == 0 && token_id < context_len) {
            logits[token_id] = qk_dot;
            qk_max = fmaxf(qk_dot, qk_max);
        }
    }

    // ------------------------------------------------ //
    // Step 3. Softmax

    // The process of solving softmax is divided into two stages. 
    // First, we need to reduce qk_max in two dimensions: WARP and ThreadBlock. 
    // Afterward, we use reduced qk_max to perform softmax calculations,
    //    the results will all be stored in shared memory.
    __shared__ float red_smem[WPT];

    // reduce qk_max in thread block and boardcast
    qk_max = __AttnBlockReduceMax<WPT>(qk_max, red_smem);

    // Softmax Kernel Logic Start here
    float exp_sum = 0.0f;
    for (int64_t token_id = threadIdx.x; token_id < context_len; token_id += TPB){
        logits[token_id] -= qk_max;
        logits[token_id] = exp(logits[token_id]);
        exp_sum += logits[token_id];
    }

    // block reduce sum on exp_sum
    // Warp per thread block must be power-of-2 for reducation, check __AttnBlockReduceSum kernel.
    static_assert(WPT == 2 || WPT == 4 || WPT == 8 || WPT == 16 || WPT == 32 || WPT == 64);
    exp_sum = __AttnBlockReduceSum<WPT>(exp_sum, red_smem);

    const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
    for (int64_t token_id = threadIdx.x; token_id < context_len; token_id += TPB) {
        logits[token_id] *= inv_sum;
    }
    __syncthreads(); // Must have this.

    // ------------------------------------------------ //
    // Step 4. Solve logits * V, this part should be carefully optimized. [TODO]
    for(int64_t i = threadIdx.x; i < HEAD_SIZE; i += TPB){
        float local_acc = 0.0f;

        for(int64_t token_id = 0; token_id < context_len; token_id++) {
            float logit = logits[token_id];
            float value = __half2float(V[
                (token_offset + token_id) * token_stride + 
                head_idx * HEAD_SIZE + i
            ]);
            local_acc = fma(value, logit, local_acc);
        }

        out[seq_idx * token_stride + head_idx * HEAD_SIZE + i] = __float2half(local_acc);
    }
}

ppl::common::RetCode PPLCUDAMultiHeadAttentionForwardImp(
            const cudaStream_t stream,
            const cudaDeviceProp& device_prop,
            const ppl::common::TensorShape* query_shape,
            void* query,
            const ppl::common::TensorShape* key_shape,
            void* key,
            void* value,
            const ppl::common::TensorShape* mask_shape,
            const void* mask,
            const ppl::common::TensorShape* output_shape,
            void* output)
{
    int64_t custom_mask_type = 1;
    if(query_shape->GetDim(1) == 1)
        custom_mask_type = 0;
    if(query_shape->GetDim(1) > 1 && query_shape->GetDim(1) != key_shape->GetDim(1)) {
        return ppl::common::RC_UNSUPPORTED;
    }
    return PPLCUDAFMHAForwardImp(
        device_prop, stream,
        query_shape, query,
        key_shape, key,
        key_shape, value,
        mask_shape, mask,
        nullptr, nullptr,
        nullptr, nullptr,
        nullptr, nullptr,
        0, custom_mask_type, 0,
        output_shape, output);
}

ppl::common::RetCode PPLCUDAMultiHeadAttentionDBForwardImp(
            const cudaStream_t stream,
            const cudaDeviceProp& device_prop,
            ppl::common::TensorShape* query_shape, //(S,H,D)
            void* query,
            ppl::common::TensorShape* key_shape,
            void* key,
            void* value,
            ppl::common::TensorShape* seqstart_q_shape,
            void* seqstart_q,
            void* seqstart_k,
            int64_t decoding_batches,
            int64_t max_seqlen,
            int64_t max_kvlen,
            ppl::common::TensorShape* output_shape,
            void* output)
{
    const int64_t q_num = query_shape->GetDim(0);
    const int64_t kv_num = key_shape->GetDim(0);

    const int64_t decodeing_q_num = decoding_batches;
    const int64_t prefill_qkv_num = q_num - decodeing_q_num;
    
    const int64_t num_head = query_shape->GetDim(1);
    const int64_t head_dim = query_shape->GetDim(2);
    const float attn_scale = float(1.0 / std::sqrt(float(head_dim)));

    void* prefill_seqstart_q = ((int64_t*)seqstart_q) + decodeing_q_num;
    void* prefill_seqstart_k = ((int64_t*)seqstart_k) + decodeing_q_num;

    if(decodeing_q_num > 0) {
        const int64_t WARP_SIZE = 32;
        const int64_t TPB = 256;
        const int64_t reduce_shm_size = TPB / WARP_SIZE * sizeof(float);
        const int64_t logits_size = max_kvlen * sizeof(float);
        const int64_t MAX_SHM_SIZE = 48 * 1024;
        if (reduce_shm_size + logits_size <= MAX_SHM_SIZE) {
            const dim3 grid_size = {(unsigned int)num_head, (unsigned int)decodeing_q_num, 1};
            switch (head_dim){
                case 64:
                _DecodingAttention_fp16<64, 4, 256>
                <<<grid_size, 256, logits_size, stream>>>(
                    (half*)(output), (half*)(query), (half*)(key), (half*)(value),
                    attn_scale, (int64_t*)(seqstart_k)
                );
                break;
                case 96:
                _DecodingAttention_fp16<96, 4, 256>
                <<<grid_size, 256, logits_size, stream>>>(
                    (half*)(output), (half*)(query), (half*)(key), (half*)(value),
                    attn_scale, (int64_t*)(seqstart_k)
                );
                break;
                case 128:
                _DecodingAttention_fp16<128, 8, 256>
                <<<grid_size, 256, logits_size, stream>>>(
                    (half*)(output), (half*)(query), (half*)(key), (half*)(value),
                    attn_scale, (int64_t*)(seqstart_k)
                );
                break;
                case 256:
                _DecodingAttention_fp16<256, 16, 256>
                <<<grid_size, 256, logits_size, stream>>>(
                    (half*)(output), (half*)(query), (half*)(key), (half*)(value),
                    attn_scale, (int64_t*)(seqstart_k)
                );
                break;
                default:
                PPL_CHECK(false, "Failed to invoke this Decoding Attention Kernel, Head size is unsupported.");
            }
        } else {
            int64_t custom_mask_type = 0;
            int64_t max_seqlen_ = 1;

            ppl::common::TensorShape decode_query_shape(*query_shape);
            ppl::common::TensorShape decode_key_shape(*key_shape);
            ppl::common::TensorShape decode_seqstart_q_shape(*seqstart_q_shape);

            decode_query_shape.Reshape({1, q_num, num_head, head_dim});
            decode_key_shape.Reshape({1, kv_num, num_head, head_dim});
            decode_seqstart_q_shape.Reshape({decodeing_q_num + 1});

            auto status = PPLCUDAFMHAForwardImp(device_prop, stream,
                &decode_query_shape, query,
                &decode_key_shape, key,
                &decode_key_shape, value,
                nullptr, nullptr,
                &decode_seqstart_q_shape, seqstart_q,
                &decode_seqstart_q_shape, seqstart_k,
                nullptr, nullptr,
                max_seqlen_, custom_mask_type, 0,
                &decode_query_shape, output);
            if (ppl::common::RC_SUCCESS != status)
                return status;
        }
    }

    
    if(prefill_qkv_num > 0) {
        int64_t custom_mask_type = 1;

        ppl::common::TensorShape prefill_query_shape(*query_shape);
        ppl::common::TensorShape prefill_key_shape(*key_shape);
        ppl::common::TensorShape prefill_seqstart_q_shape(*seqstart_q_shape);

        prefill_query_shape.Reshape({1, q_num, num_head, head_dim});
        prefill_key_shape.Reshape({1, kv_num, num_head, head_dim});
        prefill_seqstart_q_shape.Reshape({seqstart_q_shape->GetDim(0) - decodeing_q_num});

        auto status = PPLCUDAFMHAForwardImp(device_prop, stream,
            &prefill_query_shape, query,
            &prefill_key_shape, key,
            &prefill_key_shape, value,
            nullptr, nullptr,
            &prefill_seqstart_q_shape, prefill_seqstart_q,
            &prefill_seqstart_q_shape, prefill_seqstart_k,
            nullptr, nullptr,
            max_seqlen, custom_mask_type, 0,
            &prefill_query_shape, output);

        if (ppl::common::RC_SUCCESS != status)
            return status;
    }

    return ppl::common::RC_SUCCESS;
}
