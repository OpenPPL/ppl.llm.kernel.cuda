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

#include "ppl/kernel/llm/cuda/pmx/rotary_position_embedding.h"
#include "ppl/common/log.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

/**
 * Rotary Embedding Cuda impl.
 *
 * @param dim_dix: Represents the position (offset) of the current element in the hidden_dimension.
 * @param pos_idx: Represents the position (offset) of the current element in the sequence position.

 * @param head_dim: Total size of hidden_dimension.
 * @param theta parameter used to compute freq.
 */
inline __device__
float2 rotary_position_embedding_coeff(
    const int64_t dim_dix, 
    const int64_t pos_idx,
    const int64_t head_dim,
    const float theta)
{
    // fp16 does not have __sincosf instruction.
    // So we have only fp32 implementation of Rotary Embedding.
    float2 ret = {0.0f, 0.0f};
    const float freq = 1.0f / __powf(theta, (dim_dix / (float)head_dim)) * pos_idx;
    __sincosf(freq, &ret.y, &ret.x);
    return ret;
}

template<typename T, int TPB>
__global__ void rotary_position_embedding_kernel(
    const half2 *query,
    const half2 *key,
    const int64_t* cu_start_pos,
    const int64_t start_pos,
    const float theta,
    const int32_t bypass_key,
    const int64_t rotary_dim,
    const int64_t num_heads,
    const int64_t num_key_heads,
    const int64_t head_dim,
    const int64_t seqlen,
    const int64_t query_stride_s,
    const int64_t key_stride_s,
    const int64_t rotated_query_stride_s,
    const int64_t rotated_key_stride_s,
    half2 *rotated_query,
    half2 *rotated_key)
{
    // in this kernel:
    // blockIdx.x is batch_dix.
    // blockIdx.y is seq_idx.
    // blockIdx.z is block_idx over num_heads * head_dim.
    int64_t _start_pos = start_pos;
    if (cu_start_pos != nullptr)
        _start_pos = cu_start_pos[0];
    const int64_t tid = blockIdx.z * TPB + threadIdx.x;

    if (tid < head_dim * num_heads) {
        const int64_t head_idx = tid / head_dim;
        const int64_t dim_dix = tid % head_dim;
        const int64_t pos_idx = _start_pos + blockIdx.y;

        const int64_t token_idx = (blockIdx.x * seqlen + blockIdx.y);
        const int64_t q_idx = token_idx * query_stride_s + tid;
        const int64_t k_idx = token_idx * key_stride_s + tid;
        const int64_t rq_idx = token_idx * rotated_query_stride_s + tid;
        const int64_t rk_idx = token_idx * rotated_key_stride_s + tid;

        if (dim_dix >= rotary_dim) {
            rotated_query[rq_idx] = query[q_idx];
            if (head_idx < num_key_heads)
                rotated_key[rk_idx] = key[k_idx];
        } else {
            const float2 q = __half22float2(query[q_idx]);

            const float2 b = rotary_position_embedding_coeff(
                dim_dix, pos_idx, head_dim, theta);

            rotated_query[rq_idx] = {
                __float2half(q.x * b.x - q.y * b.y),
                __float2half(q.y * b.x + q.x * b.y)
            };

            if (head_idx < num_key_heads) {
                if (bypass_key) {
                    rotated_key[rk_idx] = key[k_idx];
                } else {
                    const float2 k = __half22float2(key[k_idx]);

                    rotated_key[rk_idx] = {
                        __float2half(k.x * b.x - k.y * b.y),
                        __float2half(k.y * b.x + k.x * b.y)
                    };
                }
            }
        }
    }
}

template <typename T>
ppl::common::RetCode rotary_position_embedding(
    cudaStream_t stream,
    const ppl::common::TensorShape* query_shape,
    const void* query, // (batch, seqlen, ..., head_dim), dim[2] is the leading dim of heads
    const ppl::common::TensorShape* key_shape,
    const void* key,  // (batch, seqlen, ..., head_dim), dim[2] is the leading dim of heads
    const void* cu_start_pos, // cuda mem start_pos
    const int64_t start_pos, // cpu mem start_pos
    const float theta,
    const int32_t bypass_key,
    const int64_t rotary_dim,
    const int64_t num_heads,
    const int64_t num_key_heads,
    const ppl::common::TensorShape* rotated_query_shape,
    void* rotated_query, // (batch, seqlen, ..., head_dim), dim[2] is the leading dim of heads
    const ppl::common::TensorShape* rotated_key_shape,
    void* rotated_key) // (batch, seqlen, ..., head_dim), dim[2] is the leading dim of heads
{
    if (query_shape->GetDataType() != ppl::common::DATATYPE_FLOAT16) {
        LOG(ERROR) << "only support fp16";
        return ppl::common::RC_UNSUPPORTED;
    }
    if (query_shape->GetDimCount() != 4) {
        LOG(ERROR) << "query's dim should be 4";
        return ppl::common::RC_INVALID_VALUE;
    }
    if (key_shape->GetDimCount() != 4) {
        LOG(ERROR) << "key's dim should be 4";
        return ppl::common::RC_INVALID_VALUE;
    }
    if (rotated_query_shape->GetDimCount() != 4) {
        LOG(ERROR) << "rotated_query's dim should be 4";
        return ppl::common::RC_INVALID_VALUE;
    }
    if (rotated_key_shape->GetDimCount() != 4) {
        LOG(ERROR) << "rotated_key's dim should be 4";
        return ppl::common::RC_INVALID_VALUE;
    }

    const int64_t batch           = query_shape->GetDim(0);
    const int64_t seqlen          = query_shape->GetDim(1);
    const int64_t head_dim        = query_shape->GetDim(3);

    if (head_dim % 2 != 0) {
        LOG(ERROR) << "head_dim should be an even number";
        return ppl::common::RC_INVALID_VALUE;
    }
    if (rotary_dim % 2 != 0) {
        LOG(ERROR) << "rotary_dim should be an even number";
        return ppl::common::RC_INVALID_VALUE;
    }

    constexpr int32_t VPT = 2;
    constexpr int32_t TPB = 512;

    const dim3 grid = {
        (unsigned int)batch, (unsigned int)seqlen,
        (unsigned int)((num_heads * head_dim / (TPB * VPT)) + (num_heads * head_dim % (TPB * VPT) != 0))
    };

    rotary_position_embedding_kernel<T, TPB><<<grid, TPB, 0, stream>>>(
        (const half2*)query,
        (const half2*)key,
        (const int64_t*)cu_start_pos,
        start_pos,
        theta,
        bypass_key,
        rotary_dim / 2,
        num_heads,
        num_key_heads,
        head_dim / 2,
        seqlen,
        query_shape->GetDim(2) * head_dim / 2,
        key_shape->GetDim(2) * head_dim / 2,
        rotated_query_shape->GetDim(2) * head_dim / 2,
        rotated_key_shape->GetDim(2) * head_dim / 2,
        (half2*)rotated_query,
        (half2*)rotated_key
    );
    return ppl::common::RC_SUCCESS;
}

template <typename T>
__global__ void dynamic_batching_rotary_position_embedding_kernel(
    const half2 *query,
    const half2 *key,
    const int64_t* start_pos,
    const int64_t* seqstarts,
    const float theta,
    const int32_t bypass_key,
    const int64_t rotary_dim,
    const int64_t num_heads,
    const int64_t num_key_heads,
    const int64_t head_dim,
    const int64_t query_stride_s,
    const int64_t key_stride_s,
    const int64_t rotated_query_stride_s,
    const int64_t rotated_key_stride_s,
    half2 *rotated_query,
    half2 *rotated_key) {
    if (blockIdx.y < seqstarts[blockIdx.x + 1] - seqstarts[blockIdx.x]) {
        const int64_t batch_idx = blockIdx.x;
        const int64_t seq_idx = blockIdx.y;

        const int64_t token_idx = seqstarts[batch_idx] + seq_idx;
        const int64_t pos_idx = seq_idx + start_pos[batch_idx];
        auto q_ptr = query + token_idx * query_stride_s;
        auto k_ptr = key + token_idx * key_stride_s;
        auto rq_ptr = rotated_query + token_idx * rotated_query_stride_s;
        auto rk_ptr = rotated_key + token_idx * rotated_key_stride_s;

        for (int64_t tid = threadIdx.x;
            tid < num_heads * head_dim;
            tid += blockDim.x)
        {
            const int64_t head_idx = tid / head_dim;
            const int64_t dim_dix = tid % head_dim;

            if (dim_dix >= rotary_dim) {
                rq_ptr[tid] = q_ptr[tid];
                if (head_idx < num_key_heads)
                    rk_ptr[tid] = k_ptr[tid];
            } else {
                const float2 q = __half22float2(q_ptr[tid]);

                const float2 b = rotary_position_embedding_coeff(
                    dim_dix, pos_idx, head_dim, theta);

                rq_ptr[tid] = {
                    __float2half(q.x * b.x - q.y * b.y),
                    __float2half(q.y * b.x + q.x * b.y)
                };

                if (head_idx < num_key_heads) {
                    if (bypass_key) {
                        rk_ptr[tid] = k_ptr[tid];
                    } else {
                        const float2 k = __half22float2(k_ptr[tid]);

                        rk_ptr[tid] = {
                            __float2half(k.x * b.x - k.y * b.y),
                            __float2half(k.y * b.x + k.x * b.y)
                        };
                    }
                }
            }
        }
    }
}

template <typename T>
ppl::common::RetCode dynamic_batching_rotary_position_embedding(
    cudaStream_t stream,
    const ppl::common::TensorShape* query_shape,
    const void* query, // (seqstarts[-1], ..., head_dim), dim[1] is the leading dim of heads
    const ppl::common::TensorShape* key_shape,
    const void* key, // (seqstarts[-1], ..., head_dim), dim[1] is the leading dim of heads
    const void* start_pos,
    const void* seqstarts,
    const float theta,
    const int32_t bypass_key,
    const int64_t rotary_dim,
    const int64_t batch,
    const int64_t num_heads,
    const int64_t num_key_heads,
    const int64_t max_seqlen,
    const ppl::common::TensorShape* rotated_query_shape,
    void* rotated_query, // (seqstarts[-1], ..., head_dim), dim[1] is the leading dim of heads
    const ppl::common::TensorShape* rotated_key_shape,
    void* rotated_key) // (seqstarts[-1], ..., head_dim), dim[1] of shape may not be num_heads
{
    if (query_shape->GetDataType() != ppl::common::DATATYPE_FLOAT16) {
        LOG(ERROR) << "only support fp16";
        return ppl::common::RC_UNSUPPORTED;
    }
    if (query_shape->GetDimCount() != 3) {
        LOG(ERROR) << "query's dim should be 3";
        return ppl::common::RC_INVALID_VALUE;
    }
    if (key_shape->GetDimCount() != 3) {
        LOG(ERROR) << "key's dim should be 3";
        return ppl::common::RC_INVALID_VALUE;
    }
    if (rotated_query_shape->GetDimCount() != 3) {
        LOG(ERROR) << "rotated_query's dim should be 3";
        return ppl::common::RC_INVALID_VALUE;
    }
    if (rotated_key_shape->GetDimCount() != 3) {
        LOG(ERROR) << "rotated_key's dim should be 3";
        return ppl::common::RC_INVALID_VALUE;
    }

    const int64_t head_dim = query_shape->GetDim(2);

    if (head_dim % 2 != 0) {
        LOG(ERROR) << "head_dim should be an even number";
        return ppl::common::RC_INVALID_VALUE;
    }
    if (rotary_dim % 2 != 0) {
        LOG(ERROR) << "rotary_dim should be an even number";
        return ppl::common::RC_INVALID_VALUE;
    }

    const int32_t TPB = GetBlockSize(query_shape->GetDim(0) * num_heads * head_dim / 2);
    const dim3 grid(batch, max_seqlen);
    dynamic_batching_rotary_position_embedding_kernel<T><<<grid, TPB, 0, stream>>>(
        (const half2*)query,
        (const half2*)key,
        (const int64_t*)start_pos,
        (const int64_t*)seqstarts,
        theta,
        bypass_key,
        rotary_dim / 2,
        num_heads,
        num_key_heads,
        head_dim / 2,
        query_shape->GetDim(1) * head_dim / 2,
        key_shape->GetDim(1) * head_dim / 2,
        rotated_query_shape->GetDim(1) * head_dim / 2,
        rotated_key_shape->GetDim(1) * head_dim / 2,
        (half2*)rotated_query,
        (half2*)rotated_key
    );
    
    return ppl::common::RC_SUCCESS;
}

FUNCTION_REGISTER(rotary_position_embedding)
FUNCTION_REGISTER(dynamic_batching_rotary_position_embedding);

}}}}}
