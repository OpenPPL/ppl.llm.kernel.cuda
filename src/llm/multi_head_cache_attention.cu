#include "xformer_fmha/xformer_fmha.h"
#include "ppl/common/tensor_shape.h"
#include "cudakernel/llm/multi_head_cache_attention.h"
#include "ppl/common/retcode.h"
#include "cudakernel/common/common.cuh"
#include "cudakernel/common/cuda_check.h"
#include <float.h> // need for FLT_MAX


template<int HEAD_SIZE, int VPT> // 8 fp16 occupy 128 bytes, which can be loaded by a single thread at once.
__global__
void _dynamic_group_quantize_db_new(
    const half  *current_key, //(seqlen,H,D)
    const half  *current_value,//(seqlen,H,D)
    const int64_t  *seqstarts,//(b+1)
    const int64_t  *cachestarts,//(b)
    const int64_t  *start_pos, //(b)
    const int64_t num_layer,
    const int64_t layer_idx,
    const int64_t num_head,
    // need to be modified in this kernel
    int8_t        *cache, //(max_token,L,2,H,D) 1457
    half        *scale   //(max_token,L,2,H,D/g) 
){
    if(blockIdx.y < seqstarts[blockIdx.x + 1] - seqstarts[blockIdx.x]) {
        constexpr int64_t thd_per_head_size = HEAD_SIZE / VPT; // 16
        const int64_t batch_id = blockIdx.x;
        const int64_t seq_id = blockIdx.y;

        const int64_t fuse_seq_id = seqstarts[batch_id] + seq_id;

        for(int tid = threadIdx.x; tid < HEAD_SIZE * num_head / VPT; tid += blockDim.x) {
            const int64_t head_id = tid / thd_per_head_size; // 0-31
            const int64_t dim_id = (tid % thd_per_head_size) * VPT; // 0 - 120
            const int64_t scale_dim_id = dim_id / VPT; // 0-15

            half key_in[VPT]; int8_t key_out[VPT];
            half value_in[VPT]; int8_t value_out[VPT];

            const int64_t idx = fuse_seq_id * num_head * HEAD_SIZE + head_id * HEAD_SIZE + dim_id;

            copy<sizeof(half) * VPT>(&current_key[idx], key_in);
            copy<sizeof(half) * VPT>(&current_value[idx], value_in);

            // calculate kv scale
            const half eps = 1e-5f;
            const half fact = 127.f;
            half key_scale = 0.0f;
            half value_scale = 0.0f;

            #pragma unroll
            for(int i = 0; i < VPT; i ++){
                key_scale = key_scale > __habs(key_in[i]) ? key_scale : __habs(key_in[i]);
                value_scale = value_scale > __habs(value_in[i]) ? value_scale : __habs(value_in[i]);
            }
            key_scale = key_scale / fact; 
            value_scale = value_scale / fact;
            key_scale = key_scale > eps ? key_scale : eps;
            value_scale = value_scale > eps ? value_scale : eps;


            int64_t token_index = cachestarts[batch_id] + seq_id + start_pos[batch_id];

            int64_t token_stride = num_layer * 2 * num_head * HEAD_SIZE;
            int64_t layer_stride = 2 * num_head * HEAD_SIZE;
            int64_t cache_kv_stride = num_head * HEAD_SIZE;

            int64_t key_idx = token_index * token_stride + layer_idx * layer_stride + 0 + head_id * HEAD_SIZE + dim_id;
            int64_t value_idx = key_idx + cache_kv_stride;

            int64_t key_scale_idx = (key_idx - dim_id) / VPT + scale_dim_id;
            int64_t value_scale_idx = key_scale_idx + cache_kv_stride / VPT;

            scale[key_scale_idx] = key_scale;
            scale[value_scale_idx] = value_scale;

        #pragma unroll
            for(int i = 0; i < VPT; i ++){
                key_out[i] = (int8_t)__half2short_rn(key_in[i] / key_scale);
                value_out[i] = (int8_t)__half2short_rn(value_in[i] / value_scale);
            }

            copy<sizeof(int8_t) * VPT>(key_out, &cache[key_idx]);
            copy<sizeof(int8_t) * VPT>(value_out, &cache[value_idx]);
        }
    }
}

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
    int32_t TPB,
    int32_t QUANT_GROUP>
__global__
void _DecodingCacheAttention_fp16(
    half* __restrict__ out,          // [context_lens, num_heads, head_size]
    const half* __restrict__ Q,      // [seq_lens, num_heads, head_size]
    const float  attn_scale,
    const int64_t num_layer,
    const int64_t layer_idx,
    const int8_t* cache,              // [max_token, num_layer, 2, num_heads, head_size]
    const half* scale,                // [max_token, num_layer, 2, num_heads, head_size / quant_group(8)]
    const int64_t* cachestarts,
    const int64_t* context_lens_cumsum // something like: [0, 160, 160+2, 160+2+3]
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
    static_assert(QUANT_GROUP == 8);

    const int64_t QUANT_GROUP_SHIFT = 3;

    // The elements in Q, K, and V will be evenly distributed across each thread group.
    half local_Q[VEC_SIZE * VEC_LEN];

    const int64_t warp_id       = threadIdx.x / WARP_SIZE;
    const int64_t warp_lane_id  = threadIdx.x % WARP_SIZE;
    const int64_t group_id      = warp_lane_id / THREAD_GROUP_SIZE;
    const int64_t group_lane_id = warp_lane_id % THREAD_GROUP_SIZE;
    const int64_t token_offset  = cachestarts[seq_idx];
    const int64_t embed_stride  = num_heads * HEAD_SIZE;

    const int64_t token_stride  = num_layer * 2 * num_heads * HEAD_SIZE;
    const int64_t layer_stride  = 2 * num_heads * HEAD_SIZE;
    const int64_t kv_stride = num_heads * HEAD_SIZE;
    const int64_t head_stride =  HEAD_SIZE;

    #pragma unroll
    for (int64_t i = 0; i < VEC_LEN; i++) {
        // copy 128(16 * 8) bits from Q to Local Q

        copy<sizeof(half) * VEC_SIZE>(
            &Q[seq_idx * embed_stride + head_idx * HEAD_SIZE + 
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
        int8_t local_K_quant[VEC_SIZE * VEC_LEN];
        half local_K[VEC_SIZE * VEC_LEN];
        half local_K_scale[VEC_LEN];
        const int64_t token_id = base_id + group_id;

        // all thread groups within a warp must be launched together.
        if (token_id >= context_len){
            memset(local_K, 0, sizeof(local_K));
        }
        else{
            const int64_t key_offset = (token_offset + token_id) * token_stride
                            + layer_idx * layer_stride
                            + head_idx * HEAD_SIZE
                            + group_lane_id * VEC_SIZE;
            #pragma unroll
            for (int64_t i = 0; i < VEC_LEN; i++) {
                // copy 128(16 * 8) bits from K to Local K
                const int64_t key_idx = key_offset + i * THREAD_GROUP_SIZE * VEC_SIZE;
                copy<sizeof(int8_t) * VEC_SIZE>(&cache[key_idx],  &local_K_quant[i * VEC_SIZE]);

                const int64_t key_scale_idx = key_idx >> QUANT_GROUP_SHIFT;
                local_K_scale[i] = scale[key_scale_idx];
            }

            #pragma unroll
            for (int64_t i = 0; i < VEC_LEN; i++) {
                #pragma unroll
                for (int64_t j = 0; j < VEC_SIZE; j++) {
                    local_K[i * VEC_SIZE + j] = local_K_scale[i] * (half)local_K_quant[i * VEC_SIZE + j];
                }
            }

        }

        // Ready for QK Dot
        const float qk_dot = attn_scale * __AttnThreadGroupDot<THREAD_GROUP_SIZE, VEC_LEN * VEC_SIZE>(local_Q, local_K);

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
    // for(int64_t i = threadIdx.x; i < HEAD_SIZE; i += TPB){
    //     float local_acc = 0.0f;

    //     int64_t value_idx = token_offset * token_stride + layer_idx * layer_stride + kv_stride + head_idx * head_stride + i;
    //     for(int64_t token_id = 0; token_id < context_len; token_id++) {
    //         const float logit = logits[token_id];

    //         const int64_t value_scale_idx = value_idx >> QUANT_GROUP_SHIFT;
    //         const float value = (float)cache[value_idx] * __half2float(scale[value_scale_idx]);

    //         local_acc = fma(value, logit, local_acc);

    //         value_idx += token_stride;
    //     }

    //     out[seq_idx * embed_stride + head_idx * HEAD_SIZE + i] = __float2half(local_acc);
    // }



    int8_t local_V_quant[VEC_SIZE * VEC_LEN];
    float local_V[VEC_SIZE * VEC_LEN];
    half local_V_scale[VEC_LEN];

    #pragma unroll
    for(int i = 0; i < VEC_SIZE * VEC_LEN; i++) {
        local_V[i] = 0;
    }

    for (int64_t base_id = warp_id * GPW; base_id < context_len; base_id += GPT) {
        const int64_t token_id = base_id + group_id;
        // all thread groups within a warp must be launched together.
        if (token_id < context_len){
            const int64_t value_offset = (token_offset + token_id) * token_stride
                            + layer_idx * layer_stride
                            + head_idx * head_stride
                            + kv_stride
                            + group_lane_id * VEC_SIZE;
            #pragma unroll
            for (int64_t i = 0; i < VEC_LEN; i++) {
                // copy 128(16 * 8) bits from V to Local V
                const int64_t value_idx = value_offset + i * THREAD_GROUP_SIZE * VEC_SIZE;
                copy<sizeof(int8_t) * VEC_SIZE>(&cache[value_idx],  &local_V_quant[i * VEC_SIZE]);

                const int64_t value_scale_idx = value_idx >> QUANT_GROUP_SHIFT;
                local_V_scale[i] = scale[value_scale_idx];
            }

            #pragma unroll
            for (int64_t i = 0; i < VEC_LEN; i++) {
                #pragma unroll
                for (int64_t j = 0; j < VEC_SIZE; j++) {
                    local_V[i * VEC_SIZE + j] += __half2float(local_V_scale[i]) * (float)local_V_quant[i * VEC_SIZE + j] * logits[token_id];
                }
            }
        }
    }

    #pragma unroll
    for(int i = 0; i < VEC_SIZE * VEC_LEN; i++) {
        #pragma unroll
        for(int mask = THREAD_GROUP_SIZE; mask <= WARP_SIZE >> 1; mask = mask << 1) {
            local_V[i] += __shfl_xor_sync(uint32_t(-1), local_V[i], mask);
        }
    }
    //for now, every warp's each thread group got the partial result inside a warp 
    //we need to add up each warp's first thread group by reusing the logits smem
    __syncthreads();

    for(int64_t i = threadIdx.x; i < HEAD_SIZE; i += TPB){
        logits[i] = 0;
    }

    __syncthreads();

    if(warp_lane_id < THREAD_GROUP_SIZE) {
        #pragma unroll
        for(int i = 0; i < VEC_LEN; i++) {
            #pragma unroll
            for(int j = 0; j < VEC_SIZE; j++) {
                atomicAdd(logits + i * THREAD_GROUP_SIZE * VEC_SIZE + warp_lane_id * VEC_SIZE + j, local_V[i * VEC_SIZE + j]);
            }
        }
    }

    __syncthreads();

    for(int64_t i = threadIdx.x; i < HEAD_SIZE; i += TPB){
        out[seq_idx * embed_stride + head_idx * HEAD_SIZE + i] = logits[i];
    }
}


ppl::common::RetCode PPLCUDAMultiHeadCacheAttentionForwardImp(
            const cudaStream_t stream,
            const cudaDeviceProp& device_prop,
            ppl::common::TensorShape* query_shape, //(S,H,D)
            void* query,
            ppl::common::TensorShape* key_shape,
            void* key,
            void* value,
            ppl::common::TensorShape* seqstart_q_shape,
            const void* seqstart_q,
            const void* seqstart_k,
            const void* start_pos,
            void* cache,
            void* scale,
            const void* cachestarts,
            const int64_t decoding_batches,
            const int64_t max_seqlen,
            const int64_t max_kvlen,
            const int64_t layer_idx,
            const int64_t num_layer,
            ppl::common::TensorShape* output_shape,
            void* output)
{
    const int64_t q_num = query_shape->GetDim(0);
    const int64_t kv_num = key_shape->GetDim(0);

    const int64_t decodeing_q_num = decoding_batches;
    const int64_t prefill_qkv_num = q_num - decodeing_q_num;
    
    const int64_t num_head = query_shape->GetDim(1);
    const int64_t head_dim = query_shape->GetDim(2);
    const int64_t batch = seqstart_q_shape->GetDim(0) - 1;
    const float attn_scale = float(1.0 / std::sqrt(float(head_dim)));

    void* prefill_seqstart_q = ((int64_t*)seqstart_q) + decodeing_q_num;

    {
        dim3 grid(batch, max_seqlen);
        PPL_CHECK(head_dim == 128, "Failed to invoke this dynamic_group_quantize Kernel, Head size is unsupported, only support 128.");
        int32_t block_size = GetBlockSize(num_head * head_dim / 8);
        _dynamic_group_quantize_db_new<128, 8><<<grid, block_size, 0, stream>>>((half*)key, (half*)value, (int64_t*)seqstart_q, (int64_t*)cachestarts, (int64_t*)start_pos,
            num_layer, layer_idx, num_head, (int8_t*)cache, (half*)scale);
    }


    if(decodeing_q_num > 0) {
        const int64_t WARP_SIZE = 32;
        const int64_t TPB = 256;
        const int64_t reduce_shm_size = TPB / WARP_SIZE * sizeof(float);
        const int64_t logits_size = max(max_kvlen * sizeof(float), head_dim * sizeof(float));
        const int64_t MAX_SHM_SIZE = 48 * 1024;

        if (reduce_shm_size + logits_size <= MAX_SHM_SIZE) {
            const dim3 grid_size = {(unsigned int)num_head, (unsigned int)decodeing_q_num, 1};
            switch (head_dim){
                case 64:
                _DecodingCacheAttention_fp16<64, 4, 256, 8>
                <<<grid_size, 256, logits_size, stream>>>(
                    (half*)(output), (half*)(query), attn_scale, num_layer, layer_idx, 
                    (int8_t*)cache, (half*)scale, (int64_t*)cachestarts, (int64_t*)(seqstart_k)
                );
                break;
                case 96:
                _DecodingCacheAttention_fp16<96, 4, 256, 8>
                <<<grid_size, 256, logits_size, stream>>>(
                    (half*)(output), (half*)(query), attn_scale, num_layer, layer_idx, 
                    (int8_t*)cache, (half*)scale, (int64_t*)cachestarts, (int64_t*)(seqstart_k)
                );
                break;
                case 128:
                _DecodingCacheAttention_fp16<128, 8, 256, 8>
                <<<grid_size, 256, logits_size, stream>>>(
                    (half*)(output), (half*)(query), attn_scale, num_layer, layer_idx, 
                    (int8_t*)cache, (half*)scale, (int64_t*)cachestarts, (int64_t*)(seqstart_k)
                );
                break;
                case 256:
                _DecodingCacheAttention_fp16<256, 16, 256, 8>
                <<<grid_size, 256, logits_size, stream>>>(
                    (half*)(output), (half*)(query), attn_scale, num_layer, layer_idx, 
                    (int8_t*)cache, (half*)scale, (int64_t*)cachestarts, (int64_t*)(seqstart_k)
                );
                break;
                default:
                PPL_CHECK(false, "Failed to invoke this Decoding Cache Attention Kernel, Head size is unsupported.");
            }
        } else {
            PPL_CHECK(false, "Decoding Cache Attention shm not enough, cache flash attention is unsupported.");
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

        PPLCUDAFMHAForwardImp(device_prop, stream,
            &prefill_query_shape, query,
            &prefill_key_shape, key,
            &prefill_key_shape, value,
            nullptr, nullptr,
            &prefill_seqstart_q_shape, prefill_seqstart_q,
            &prefill_seqstart_q_shape, prefill_seqstart_q,
            nullptr, nullptr,
            max_seqlen, custom_mask_type, 0,
            &prefill_query_shape, output);

        return ppl::common::RC_SUCCESS;
    } else {
        return ppl::common::RC_SUCCESS;
    }
}
