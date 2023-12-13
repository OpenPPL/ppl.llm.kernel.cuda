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

#include "ppl/kernel/llm/cuda/pmx/rotary_2d_position_embedding.h"
#include "ppl/common/log.h"

#include "cudakernel/common/common.cuh"

#include <cuda_fp16.h>

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

/**
 * Rotary Embedding Cuda impl.
 *
 * @param dim_dix: Represents the position (offset) of the current element in the hidden_dimension.
 * @param pos_idx: Represents the position (offset) of the current element in the sequence position.

 * @param head_dim: Total size of hidden_dimension.
 * @param theta parameter used to compute freq.
 pytorch code is as follow:
freqs = (1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float, device=query.device)[: (dim // 2)] / dim)))
t = torch.arange(position, position + seqlen, dtype=torch.float, device=query.device)
freqs_cis = torch.outer(t, freqs)
cos, sin = freqs_cis.cos().unsqueeze(1), freqs_cis.sin().unsqueeze(1)  # (seqlen, 1, dim / 2)
 * 
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

ppl::common::RetCode rotary_2d_position_embedding(
    cudaStream_t stream,
    const ppl::common::TensorShape* query_shape,
    const void* query, // (batch, seqlen, ..., head_dim), dim[2] is the leading dim of heads
    const ppl::common::TensorShape* key_shape,
    const void* key,  // (batch, seqlen, ..., head_dim), dim[2] is the leading dim of heads
    const void* cu_start_pos,
    const void* first_seqlen,
    const void* pad_len,
    const int64_t start_pos,
    const float theta,
    const int32_t bypass_key,
    const int64_t num_heads,
    const int64_t num_key_heads,
    const ppl::common::TensorShape* rotated_query_shape,
    void* rotated_query, // (batch, seqlen, ..., head_dim), dim[2] is the leading dim of heads
    const ppl::common::TensorShape* rotated_key_shape,
    void* rotated_key) // (batch, seqlen, ..., head_dim), dim[2] is the leading dim of heads
{
    LOG(ERROR) << "static rotary 2d position embedding currently not supported";
    return ppl::common::RC_UNSUPPORTED;
}

struct dynamic_batching_rotary_2d_position_embedding_kernel_param {
    half2 *query;
    half2 *key;
    int64_t* start_pos;
    int64_t* seqstarts;
    float theta;
    int64_t num_heads;
    int64_t num_key_heads;
    int64_t head_dim;
    int64_t* first_seqlen;
    int64_t query_stride_s;
    int64_t key_stride_s;
    int64_t rotated_query_stride_s;
    int64_t rotated_key_stride_s;
    half2 *rotated_query;
    half2 *rotated_key;
};

template<int32_t TPB, bool bypass_key>
__global__
void dynamic_batching_rotary_2d_position_embedding_kernel(
    dynamic_batching_rotary_2d_position_embedding_kernel_param p
) {
    if (blockIdx.y < p.seqstarts[blockIdx.x + 1] - p.seqstarts[blockIdx.x]) {

        const int64_t batch_idx = blockIdx.x;
        const int64_t seq_idx = blockIdx.y;
        const int64_t tid = blockIdx.z * TPB + threadIdx.x;

        if (tid < p.num_heads * p.head_dim) {

            const int64_t token_idx = p.seqstarts[batch_idx] + seq_idx;
            int64_t pos_idx1 = -1, pos_idx2 = -1;
            int64_t seqlen = p.seqstarts[blockIdx.x + 1] - p.seqstarts[blockIdx.x];

            if (seqlen > 1) {   // prefill
                pos_idx1 = seq_idx;
                pos_idx2 = 0;
                if (pos_idx1 == seqlen - 1) {
                    pos_idx1 = seqlen - 2;
                    pos_idx2 = 1;
                }
            } else {    // decoding
                pos_idx1 = p.first_seqlen[batch_idx] - 2;
                pos_idx2 = p.start_pos[batch_idx] - p.first_seqlen[batch_idx] + 2;
            }

            auto q_ptr = p.query + token_idx * p.query_stride_s;
            auto k_ptr = p.key + token_idx * p.key_stride_s;
            auto rq_ptr = p.rotated_query + token_idx * p.rotated_query_stride_s;
            auto rk_ptr = p.rotated_key + token_idx * p.rotated_key_stride_s;

            const int64_t head_idx = tid / p.head_dim;
            const int64_t dim_idx = tid % p.head_dim;
            
            float2 b;
            if (dim_idx < p.head_dim / 2) { // 做pos1的emb
                b = rotary_position_embedding_coeff(
                    dim_idx, pos_idx1, p.head_dim / 2, p.theta);    // (cos, sin)
            } else if (dim_idx >= p.head_dim / 2 && dim_idx < p.head_dim) {
                b = rotary_position_embedding_coeff(
                    dim_idx - p.head_dim / 2, pos_idx2, p.head_dim / 2, p.theta);
            }

            const float2 q = __half22float2(q_ptr[tid]);

            rq_ptr[tid] = {
                __float2half(q.x * b.x - q.y * b.y),   // (x_a * cos - x_b * sin)
                __float2half(q.y * b.x + q.x * b.y)    // (x_b * cos + x_a * sin)
            };
            // rotary k
            if (head_idx < p.num_key_heads) {
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

ppl::common::RetCode dynamic_batching_rotary_2d_position_embedding(
    cudaStream_t stream,
    const ppl::common::TensorShape* query_shape,
    const void* query, // (seqstarts[-1], ..., head_dim), dim[1] is the leading dim of heads
    const ppl::common::TensorShape* key_shape,
    const void* key, // (seqstarts[-1], ..., head_dim), dim[1] is the leading dim of heads
    const void* start_pos,
    const void* seqstarts,
    const void* first_seqlen,
    const float theta,
    const int32_t bypass_key,
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

    if (head_dim % 4 != 0) {
        LOG(ERROR) << "head_dim should be multiples of 4";
        return ppl::common::RC_INVALID_VALUE;
    }

    dynamic_batching_rotary_2d_position_embedding_kernel_param p = {
        (half2 *)query,
        (half2 *)key,
        (int64_t*) start_pos,
        (int64_t*) seqstarts,
        theta,
        num_heads,
        num_key_heads,
        head_dim / 2,
        (int64_t*)first_seqlen,
        query_shape->GetDim(1) * head_dim / 2,
        key_shape->GetDim(1) * head_dim / 2,
        rotated_query_shape->GetDim(1) * head_dim / 2,
        rotated_key_shape->GetDim(1) * head_dim / 2,
        (half2 *)rotated_query,
        (half2 *)rotated_key,
    };

    const int32_t TPB = 256;
    const dim3 grid(batch, max_seqlen, (num_heads * head_dim / 2 + TPB - 1) / TPB);
    if (bypass_key) {
        dynamic_batching_rotary_2d_position_embedding_kernel<TPB, true><<<grid, TPB, 0, stream>>>(p);
    } else {
        dynamic_batching_rotary_2d_position_embedding_kernel<TPB, false><<<grid, TPB, 0, stream>>>(p);
    }

    return ppl::common::RC_SUCCESS;
}
}}}}}

