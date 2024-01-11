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

template<int32_t NUM_EXPERTS, int32_t TPB, int32_t VPT>
__global__ 
void moe_topk_softmax_kernel(
    const half* scores,     // [tokens, NUM_EXPERTS] 
    const int64_t tokens,
    const int64_t k,
    half* expert_weights,   // [tokens, k] 
    int64_t* expert_ids,    // [tokens, k]
    int64_t* permute_token_idx)   // [tokens, k]
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

    register float _maximum[MAX_K];
    register int64_t _indices[MAX_K];
    register half local_scores[VPT];

    copy<sizeof(half) * VPT>(scores + tid * VPT, local_scores);

    // 1. top k
    for(int k_idx = 0; k_idx < k; ++k_idx) {        
        half local_max = -HALF_MAX();
        int64_t local_idx = 0;

        // max in VPT
        #pragma unroll
        for(int i = 0; i < VPT; ++i) {
            if (local_max < local_scores[i]) {
                local_max = local_scores[i];
                local_idx = i + thread_group_lane * VPT;
            }
        }

        // reduce max in thread group
        half reducing_max = local_max;
        int64_t reducing_idx = local_idx;
        #pragma unroll
        for(int mask = THREAD_GROUP_SIZE / 2; mask >=1; mask /= 2) {
            half receving_max  = __shfl_xor_sync(uint32_t(-1), reducing_max, mask);
            int64_t receving_idx = __shfl_xor_sync(uint32_t(-1), reducing_idx, mask);

            if (reducing_max < receving_max || (reducing_max == receving_max && receving_idx < reducing_idx)) {
                reducing_max = receving_max;
                reducing_idx = receving_idx;
            }
        }

        local_idx = reducing_idx - thread_group_lane * VPT;
        if (local_idx >= 0 && local_idx < VPT) {
            local_scores[local_idx] = -HALF_MAX();    
        }

        if(thread_group_lane == 0){
            _maximum[k_idx] = __half2float(reducing_max);
            _indices[k_idx] = reducing_idx;
        }
    }

    // 2. softmax
    register float softmax_max = __half2float(-HALF_MAX());
    register float softmax_sum = 0.0f;
    register float* local_softmax = _maximum;
    
    if (thread_group_lane == 0) {
        for(auto i = 0; i < k; i++){
            softmax_max = max(softmax_max, local_softmax[i]);
        }

        for(auto i = 0; i < k; i++){
            softmax_sum += exp(local_softmax[i] - softmax_max);
        }

        for(auto i=0; i < k; i++) {
            expert_weights[group_id * k + i] = __float2half(exp(local_softmax[i] - softmax_max) / softmax_sum);
            expert_ids[group_id * k + i] = _indices[i];
            permute_token_idx[group_id * k + i] = group_id * k + i;
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
    int64_t* permute_token_idx) // [tokens, k]
{
    
    const int32_t tid = blockIdx.x * TPB + threadIdx.x;

    constexpr int MAX_NUM_EXPERTS = 128;
    constexpr int MAX_K = 128;

    for(int offset = tid; offset < tokens; offset += gridDim.x * TPB) {
        half local_scores[MAX_NUM_EXPERTS];
        float local_output[MAX_K];

        const half* _scores = scores + offset * num_experts;

        for(int i=0; i<num_experts; ++i) {
            local_scores[i] = _scores[i];
        }

        // 1. top k
        for (int k_idx = 0; k_idx < k; ++k_idx) {
            half local_max = -HALF_MAX();
            int64_t local_idx = 0;
            for(int i=0; i<num_experts; ++i) {
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

        for(int i=0; i<k; ++i) {
            softmax_max = max(softmax_max, local_output[i]);
        }
        for(int i=0; i<k; ++i) {
            softmax_sum += exp(local_output[i] - softmax_max);
        }

        for(int i=0; i<k; ++i) {
            expert_weights[offset * k + i] = __float2half(exp(local_output[i] - softmax_max) / softmax_sum);
            permute_token_idx[offset * k + i] = offset * k + i;
        }
    }
}

void moe_topk_softmax(const half* scores, const int64_t tokens, const int64_t dim, const int64_t num_experts, const int64_t num_experts_per_token, half* expert_weights, int64_t* expert_ids, int64_t* permute_token_idx, cudaStream_t stream) {

    const int64_t num_elem = tokens * num_experts;

    constexpr int TPB = 256;
    constexpr int VPT = 16 / sizeof(half);
    const int BPG = (num_elem / VPT + TPB - 1) / TPB;

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
                    permute_token_idx);
                break;
            case 16:
                moe_topk_softmax_kernel<16, TPB, VPT><<<BPG, TPB, 0, stream>>>(
                    scores, 
                    tokens,
                    num_experts_per_token,
                    expert_weights, 
                    expert_ids,
                    permute_token_idx);
                break;
            case 32:
                moe_topk_softmax_kernel<32, TPB, VPT><<<BPG, TPB, 0, stream>>>(
                    scores, 
                    tokens,
                    num_experts_per_token,
                    expert_weights, 
                    expert_ids,
                    permute_token_idx);
                break;
            case 64:
                moe_topk_softmax_kernel<64, TPB, VPT><<<BPG, TPB, 0, stream>>>(
                    scores, 
                    tokens,
                    num_experts_per_token,
                    expert_weights, 
                    expert_ids,
                    permute_token_idx);
                break;
            default:
                moe_topk_softmax_kernel_default<TPB><<<BPG, TPB, 0, stream>>>(
                    scores,
                    tokens,
                    num_experts,
                    num_experts_per_token,
                    expert_weights, 
                    expert_ids,
                    permute_token_idx);
        }
    } else {
        moe_topk_softmax_kernel_default<TPB><<<BPG, TPB, 0, stream>>>(
            scores,
            tokens,
            num_experts,
            num_experts_per_token,
            expert_weights, 
            expert_ids,
            permute_token_idx);
    }
}

void sort_pairs(int64_t* key, int64_t* value, void* sort_buffer, const int num_key_value_pairs, size_t sort_buffer_size, const int64_t num_experts, const cudaStream_t stream) {
    cub::DeviceRadixSort::SortPairs(sort_buffer, sort_buffer_size, key, key, value, value, num_key_value_pairs, 0, (int)(log2(num_experts)) + 1, stream);
}

moe_select_config moe_select_prepare(const ppl::common::TensorShape* invert_permutation_shape, const int64_t num_experts) {

    const int64_t expand_tokens = invert_permutation_shape->CalcElementsExcludingPadding();

    moe_select_config config;

    void *d_temp_storage = nullptr;
    int64_t* null_int = nullptr;
    size_t temp_storage_bytes = 0;

    ::cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, null_int, null_int, null_int, null_int, expand_tokens, 0, (int)(log2(num_experts)) + 1);

    config.expert_ids_size = expand_tokens * sizeof(int64_t);
    config.permute_token_idx_size = expand_tokens * sizeof(int64_t);
    config.sort_buffer_size = temp_storage_bytes;

    return config;
}

__device__ 
int findTotalEltsLeqTarget(const int64_t* sorted_indices, const int arr_length, const int target) {
    int64_t low = 0, high = arr_length - 1, target_location = -1;
    while (low <= high) {
        int64_t mid = (low + high) / 2;

        if (sorted_indices[mid] > target) {
            high = mid - 1;
        }
        else {
            low = mid + 1;
            target_location = mid;
        }
    }
    return target_location + 1;
}

__global__ 
void compute_offset_kernel(const int64_t* sorted_expert_indices, const int num_expand_tokens, const int num_experts, int64_t* expert_offset) {

    const int expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert >= num_experts) {
        return;
    }

    if (expert == 0) {
        expert_offset[0] = 0;
    }

    expert_offset[expert + 1] = findTotalEltsLeqTarget(sorted_expert_indices, num_expand_tokens, expert);
}

void compute_offset(const int64_t* sorted_expert_indices, const int num_expand_tokens, const int num_experts, int64_t* expert_offset, const cudaStream_t stream) {
    const int threads = std::min(256, num_experts);
    const int blocks = (num_experts + threads - 1) / threads;
    
    compute_offset_kernel<<<blocks, threads, 0, stream>>>(sorted_expert_indices, num_expand_tokens, num_experts, expert_offset);
}

__global__ void expand_permute_kernel(const half* x, const int64_t* token_idx_dest2source, const int num_tokens, int cols, const int repeats, half* y, int64_t* token_idx_source2dest) {
    const int64_t dest_row = blockIdx.x;

    const int64_t expanded_source_row = token_idx_dest2source[dest_row];
    
    const int64_t source_row = expanded_source_row / repeats;

    if (threadIdx.x == 0) {
        token_idx_source2dest[expanded_source_row] = dest_row;
    }

    const half* source_row_ptr = x + source_row * cols;
    half* dest_row_ptr = y + dest_row * cols;

    for(int col_id = threadIdx.x; col_id < cols; col_id += blockDim.x) {
        dest_row_ptr[col_id] = source_row_ptr[col_id];
    }
}

void expand_permute(const half* x, const int64_t* token_idx_dest2source, int num_tokens, int cols, int repeats, half* y, int64_t* token_idx_source2dest, const cudaStream_t stream) {
    const int TPB = min(1024, cols);
    const int BPG = num_tokens * repeats;
    expand_permute_kernel<<<BPG, TPB, 0, stream>>>(x, token_idx_dest2source, num_tokens, cols, repeats, y, token_idx_source2dest);
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
    
    if (ppl::common::DATATYPE_FLOAT16 != x_shape->GetDataType()) {
        LOG(ERROR) << "moe_select only support fp16, but got ["<< x_shape->GetDataType() << "]";
        return ppl::common::RC_UNSUPPORTED;
    }
    
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
    void* permute_token_idx = (void*)((char*)temp_buffer + config.expert_ids_size);
    void* sort_buffer = (void*)((char*)temp_buffer + config.expert_ids_size + config.permute_token_idx_size);

    moe_topk_softmax((const half*)scores, tokens, dim, num_experts, num_experts_per_token, (half*)expert_weights, (int64_t*)expert_ids, (int64_t*)permute_token_idx, stream);

    const int num_expand_tokens = tokens * num_experts_per_token;

    sort_pairs((int64_t*)expert_ids, (int64_t*)permute_token_idx, sort_buffer, num_expand_tokens, config.sort_buffer_size, num_experts, stream);

    compute_offset((const int64_t*)expert_ids, num_expand_tokens, num_experts, (int64_t*)expert_offset, stream);

    expand_permute((const half*)x, (const int64_t*)permute_token_idx, tokens, dim, num_experts_per_token, (half*)x_expand_permute, (int64_t*)invert_permutation, stream);

    return ppl::common::RC_SUCCESS;
}

}}}}}