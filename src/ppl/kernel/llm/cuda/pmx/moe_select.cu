// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "ppl/kernel/llm/cuda/pmx/moe_select.h"
#include "cudakernel/common/common.cuh"
#include "ppl/common/log.h"

#include <cub/cub.cuh>
#include <cuda_fp16.h>

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

static constexpr __device__ half HALF_MAX() { 
    constexpr uint16_t a = 0x7bff; 
    return *((half*)(&a)); 
}

moe_select_config moe_select_prepare(
    const ppl::common::TensorShape* invert_permutation_shape,
    const int64_t num_experts) 
{
    const int64_t expand_tokens = invert_permutation_shape->CalcElementsExcludingPadding();

    moe_select_config config;

    void *d_temp_storage = nullptr;
    int64_t* null_int = nullptr;
    size_t temp_storage_bytes = 0;

    ::cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes, null_int, null_int,
        null_int, null_int, expand_tokens, 0, (int)(log2(num_experts)) + 1);

    config.expert_ids_size = expand_tokens * sizeof(int64_t);
    config.sorted_expert_ids_size = expand_tokens * sizeof(int64_t);
    config.source_row_size = expand_tokens * sizeof(int64_t);
    config.permute_token_idx_size = expand_tokens * sizeof(int64_t);
    config.sort_buffer_size = temp_storage_bytes;

    config.temp_buffer_size = config.expert_ids_size + config.sorted_expert_ids_size + config.source_row_size +
                            config.permute_token_idx_size + config.sort_buffer_size;

    return config;
}

template<int32_t NUM_EXPERTS, int32_t TPB, int32_t VPT>
__global__ 
void moe_topk_softmax_kernel(
    const half* scores,     // [tokens, NUM_EXPERTS] 
    const int64_t tokens,
    const int64_t top_k,
    half* expert_weights,   // [tokens, k] 
    int64_t* expert_ids,    // [tokens, k]
    int64_t* source_row)   // [tokens, k]
{
    constexpr int32_t THREAD_GROUP_SIZE = NUM_EXPERTS * sizeof(half) / 16;
    const int32_t tid = blockIdx.x * TPB + threadIdx.x;
    const int32_t group_id = tid / THREAD_GROUP_SIZE;
    const int32_t thread_group_lane = tid % THREAD_GROUP_SIZE;

    if (tid * VPT >= tokens * NUM_EXPERTS) {
        return;
    }

    static_assert(NUM_EXPERTS >= 8);
    static_assert(THREAD_GROUP_SIZE <= 32);

    constexpr int64_t MAX_K = 8;

    __shared__ float shared_max[TPB * MAX_K];
    __shared__ int32_t shared_idx[TPB * MAX_K];
    register half local_scores[VPT];

    float *local_shm_max = shared_max + threadIdx.x * MAX_K;
    int32_t *local_shm_idx = shared_idx + threadIdx.x * MAX_K;

    copy<sizeof(half) * VPT>(scores + tid * VPT, local_scores);

    // 1. top k
    for (int32_t k_idx = 0; k_idx < top_k; ++k_idx) {
        float local_max = -FLT_MAX;
        int32_t local_idx = 0;

        // max in VPT
        #pragma unroll
        for (int32_t i = 0; i < VPT; ++i) {
            if (local_max < __half2float(local_scores[i])) {
                local_max = __half2float(local_scores[i]);
                local_idx = i + thread_group_lane * VPT;
            }
        }

        // reduce max in thread group
        float reducing_max = local_max;
        int32_t reducing_idx = local_idx;

        #pragma unroll
        for (int32_t mask = THREAD_GROUP_SIZE / 2; mask >= 1; mask /= 2) {
            float receving_max  = __shfl_xor_sync(uint32_t(-1), reducing_max, mask);
            int32_t receving_idx = __shfl_xor_sync(uint32_t(-1), reducing_idx, mask);

            if (reducing_max < receving_max || (reducing_max == receving_max && receving_idx < reducing_idx)) {
                reducing_max = receving_max;
                reducing_idx = receving_idx;
            }
        }

        local_idx = reducing_idx - thread_group_lane * VPT;
        #pragma unroll
        for (int32_t i = 0; i < VPT; ++i) {
            if (local_idx == i)
                local_scores[i] = -HALF_MAX();
        }

        if (thread_group_lane == 0) {
            local_shm_max[k_idx] = reducing_max;
            local_shm_idx[k_idx] = reducing_idx;
        }
    }

    // 2. softmax
    register float softmax_max = -FLT_MAX;
    register float softmax_sum = 0.0f;

    if (thread_group_lane == 0) {
        for (auto i = 0; i < top_k; i++) {
            softmax_max = max(softmax_max, local_shm_max[i]);
        }

        for (auto i = 0; i < top_k; i++) {
            auto exp_scale = exp(local_shm_max[i] - softmax_max);
            local_shm_max[i] = exp_scale;
            softmax_sum += exp_scale;
        }

        for (auto i = 0; i < top_k; i++) {
            expert_weights[group_id * top_k + i] = __float2half(local_shm_max[i] / softmax_sum);
            expert_ids[group_id * top_k + i] = local_shm_idx[i];
            source_row[group_id * top_k + i] = group_id * top_k + i;
        }
    }
}

template<int32_t TPB>
__global__
void moe_topk_softmax_kernel_default(
    const half* scores,     // [tokens, NUM_EXPERTS] 
    const int64_t tokens,
    const int64_t num_experts,
    const int64_t k,
    half* expert_weights,   // [tokens, k] 
    int64_t* expert_ids,
    int64_t* source_row) // [tokens, k]
{
    
    const int32_t tid = blockIdx.x * TPB + threadIdx.x;

    constexpr int32_t MAX_NUM_EXPERTS = 128;
    constexpr int32_t MAX_K = 128;

    for (int32_t offset = tid; offset < tokens; offset += gridDim.x * TPB) {
        half local_scores[MAX_NUM_EXPERTS];
        float local_output[MAX_K];

        const half *score_ptr = scores + offset * num_experts;

        for (int32_t i = 0; i < num_experts; ++i) {
            local_scores[i] = score_ptr[i];
        }

        // 1. top k
        for (int32_t k_idx = 0; k_idx < k; ++k_idx) {
            half local_max = -HALF_MAX();
            int64_t local_idx = 0;
            for (int32_t i = 0; i<num_experts; ++i) {
                if (local_max < local_scores[i]) {
                    local_max = local_scores[i];
                    local_idx = i;
                }
            }
            
            local_output[k_idx] = __half2float(local_max);
            expert_ids[offset * k + k_idx] = local_idx;
            local_scores[local_idx] = -HALF_MAX();
        }
        
        // 2. softmax
        float softmax_max = local_output[0];
        float softmax_sum = 0.0f;

        for (int32_t i = 0; i < k; ++i) {
            softmax_max = max(softmax_max, local_output[i]);
        }

        for (int32_t i = 0; i < k; ++i) {
            softmax_sum += exp(local_output[i] - softmax_max);
        }

        for (int32_t i = 0; i < k; ++i) {
            expert_weights[offset * k + i] = __float2half(exp(local_output[i] - softmax_max) / softmax_sum);
            source_row[offset * k + i] = offset * k + i;
        }
    }
}

void moe_topk_softmax(
    const half* scores, 
    const int64_t tokens, 
    const int64_t dim, 
    const int64_t num_experts, 
    const int64_t num_experts_per_token, 
    half* expert_weights, 
    int64_t* expert_ids, 
    int64_t* source_row,
    const cudaStream_t stream) 
{

    const int64_t num_elem = tokens * num_experts;

    constexpr int32_t TPB = 256;
    constexpr int32_t VPT = 16 / sizeof(half);
    const int32_t BPG = (num_elem + TPB * VPT - 1) / (TPB * VPT);

    if (num_experts_per_token <= 8) {
        switch (num_experts) 
        {
            case 8:
                moe_topk_softmax_kernel<8, TPB, VPT><<<BPG, TPB, 0, stream>>>(
                    scores, 
                    tokens,
                    num_experts_per_token,
                    expert_weights, 
                    expert_ids,
                    source_row);
                break;
            case 16:
                moe_topk_softmax_kernel<16, TPB, VPT><<<BPG, TPB, 0, stream>>>(
                    scores, 
                    tokens,
                    num_experts_per_token,
                    expert_weights, 
                    expert_ids,
                    source_row);
                break;
            case 32:
                moe_topk_softmax_kernel<32, TPB, VPT><<<BPG, TPB, 0, stream>>>(
                    scores, 
                    tokens,
                    num_experts_per_token,
                    expert_weights, 
                    expert_ids,
                    source_row);
                break;
            case 64:
                moe_topk_softmax_kernel<64, TPB, VPT><<<BPG, TPB, 0, stream>>>(
                    scores, 
                    tokens,
                    num_experts_per_token,
                    expert_weights, 
                    expert_ids,
                    source_row);
                break;
            default:
                moe_topk_softmax_kernel_default<TPB><<<BPG, TPB, 0, stream>>>(
                    scores,
                    tokens,
                    num_experts,
                    num_experts_per_token,
                    expert_weights, 
                    expert_ids,
                    source_row);
        }
    } else {
        moe_topk_softmax_kernel_default<TPB><<<BPG, TPB, 0, stream>>>(
            scores,
            tokens,
            num_experts,
            num_experts_per_token,
            expert_weights, 
            expert_ids,
            source_row);
    }
}

void sort_pairs(
    int64_t* key_in,
    int64_t* key_out,
    int64_t* value_in,
    int64_t* value_out,
    void* sort_buffer,
    const int32_t num_key_value_pairs,
    const int64_t sort_buffer_size,
    const int64_t num_experts,
    const cudaStream_t stream)
{
    size_t m_sort_buffer_size = sort_buffer_size;
    cub::DeviceRadixSort::SortPairs(
        sort_buffer, m_sort_buffer_size, key_in, key_out, value_in, value_out,
        num_key_value_pairs, 0, (int)(log2(num_experts)) + 1, stream);
}

__device__ 
int32_t find_right_bound(const int64_t* sorted_indices, const int32_t arr_length, const int32_t target) {
    int32_t low = 0, high = arr_length - 1, target_location = -1;
    while (low <= high) {
        int32_t mid = (low + high) / 2;

        if (sorted_indices[mid] > target) {
            high = mid - 1;
        } else {
            low = mid + 1;
            target_location = mid;
        }
    }
    return target_location + 1;
}

__global__ 
void compute_offset_kernel(
    const int64_t* sorted_expert_indices, 
    const int32_t num_expand_tokens, 
    const int32_t num_experts, 
    int64_t* expert_offset) 
{
    const int32_t expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert >= num_experts) {
        return;
    }

    if (expert == 0) {
        expert_offset[0] = 0;
    }

    expert_offset[expert + 1] = find_right_bound(sorted_expert_indices, num_expand_tokens, expert);
}

void compute_offset(
    const int64_t* sorted_expert_indices, 
    const int32_t num_expand_tokens, 
    const int32_t num_experts, 
    int64_t* expert_offset, 
    const cudaStream_t stream) 
{
    const int32_t threads = std::min(256, num_experts);
    const int32_t blocks = (num_experts + threads - 1) / threads;
    
    compute_offset_kernel<<<blocks, threads, 0, stream>>>(
        sorted_expert_indices, num_expand_tokens, num_experts, expert_offset);
}

__global__ void expand_permute_kernel(
    const half* x, 
    const int64_t* token_idx_dest2source, 
    const int32_t num_tokens, 
    const int32_t cols, 
    const int32_t repeats, 
    half* y, 
    int64_t* token_idx_source2dest) 
{
    const int64_t dest_row = blockIdx.x;

    const int64_t expanded_source_row = token_idx_dest2source[dest_row];
    
    const int64_t source_row = expanded_source_row / repeats;

    if (threadIdx.x == 0) {
        token_idx_source2dest[expanded_source_row] = dest_row;
    }

    const half* source_row_ptr = x + source_row * cols;
    half* dest_row_ptr = y + dest_row * cols;

    for(int32_t col_id = threadIdx.x; col_id < cols; col_id += blockDim.x) {
        dest_row_ptr[col_id] = source_row_ptr[col_id];
    }
}

void expand_permute(
    const half* x, 
    const int64_t* token_idx_dest2source, 
    const int32_t num_tokens,
    const int32_t cols,
    const int32_t repeats,
    half* y, 
    int64_t* token_idx_source2dest, 
    const cudaStream_t stream) 
{
    const int32_t TPB = min(1024, cols);
    const int32_t BPG = num_tokens * repeats;
    expand_permute_kernel<<<BPG, TPB, 0, stream>>>(
        x, token_idx_dest2source, num_tokens, cols, repeats, y, token_idx_source2dest);
}

ppl::common::RetCode moe_select(
    const cudaStream_t stream,
    const ppl::common::TensorShape* x_shape,
    const void* x,
    const ppl::common::TensorShape* scores_shape,
    const void* scores,
    const int64_t num_experts,
    const int64_t num_experts_per_token,
    const moe_select_config& config,
    void* temp_buffer,
    void* x_expand_permute,
    void* expert_weights,
    void* invert_permutation,
    void* expert_offset)
{
    if (scores_shape->GetDim(scores_shape->GetDimCount() - 1) != num_experts) {
        LOG(ERROR) << "scores_shape[-1] != num_experts";
        return ppl::common::RC_OTHER_ERROR;
    }

    if (num_experts_per_token > num_experts) {
        LOG(ERROR) << "num_experts_per_token[" << num_experts_per_token << "] must less or equal with num_experts[" << num_experts <<"]";
        return ppl::common::RC_OTHER_ERROR;
    }

    const int64_t tokens = x_shape->CalcElementsToDimensionExcludingPadding(x_shape->GetDimCount() - 1);

    const int64_t dim = x_shape->GetDim(x_shape->GetDimCount() - 1);

    void* expert_ids = temp_buffer;
    void* sorted_expert_ids = (void*)((char*)temp_buffer + config.expert_ids_size);
    void* source_row = (void*)((char*)temp_buffer + config.expert_ids_size + config.sorted_expert_ids_size);
    void* permute_token_idx = (void*)((char*)temp_buffer + config.expert_ids_size +
                                config.sorted_expert_ids_size + config.source_row_size);
    void* sort_buffer = (void*)((char*)temp_buffer + config.expert_ids_size + config.sorted_expert_ids_size +
                                config.source_row_size +  config.permute_token_idx_size);

    moe_topk_softmax(
        (const half*)scores, tokens, dim, num_experts,
        num_experts_per_token, (half*)expert_weights,
        (int64_t*)expert_ids, (int64_t*)source_row, stream);

    const int32_t num_expand_tokens = tokens * num_experts_per_token;

    sort_pairs(
        (int64_t*)expert_ids, (int64_t*)sorted_expert_ids, (int64_t*)source_row, (int64_t*)permute_token_idx,
        sort_buffer, num_expand_tokens, config.sort_buffer_size, num_experts, stream);

    compute_offset(
        (const int64_t*)sorted_expert_ids, num_expand_tokens,
        num_experts, (int64_t*)expert_offset, stream);

    expand_permute(
        (const half*)x, (const int64_t*)permute_token_idx,
        tokens, dim, num_experts_per_token,
        (half*)x_expand_permute, (int64_t*)invert_permutation, stream);

    return ppl::common::RC_SUCCESS;
}

}}}}}