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

#include "ppl/kernel/llm/cuda/pmx/multi_head_attention.h"
#include "ppl/common/log.h"

#include "ppl/kernel/llm/cuda/xformer/fmha.h"
#include "cudakernel/common/common.cuh"

#include <cuda_fp16.h>

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

ppl::common::RetCode multi_head_attention(
    const cudaStream_t stream,
    const cudaDeviceProp& device_prop,
    const ppl::common::TensorShape* query_shape,
    const void* query, // (B, Sq, Hq ,D)
    const ppl::common::TensorShape* key_shape,
    const void* key, // (B, Skv, Hkv ,D)
    const ppl::common::TensorShape* value_shape,
    const void* value, // (B, Skv, Hkv ,D)
    const ppl::common::TensorShape* attn_mask_shape,
    const void* attn_mask, // (Sq, Skv), (B, Hq, Sq, Skv)
    const bool is_causal,
    const int64_t num_heads,
    const int64_t num_kv_heads,
    const int64_t head_dim,
    const ppl::common::TensorShape* output_shape,
    void* output) // (B, Sq, Hq ,D)
{
    if (query_shape->GetDim(2) != num_heads) {
        LOG(ERROR) << "currnetly do not support leading dim on query's num_heads";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (output_shape->GetDim(2) != num_heads) {
        LOG(ERROR) << "currnetly do not support leading dim on output's num_heads";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (key_shape->GetDim(2) != num_kv_heads) {
        LOG(ERROR) 
            << "currnetly do not support leading dim on current_key's num_kv_heads";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (value_shape->GetDim(2) != num_kv_heads) {
        LOG(ERROR) 
            << "currnetly do not support leading dim on current_value's num_kv_heads";
        return ppl::common::RC_UNSUPPORTED;
    }

    const int64_t custom_mask_type = is_causal ? 2 : 0;
    const float attn_scale = float(1.0 / std::sqrt(float(head_dim)));

    constexpr int64_t VPT = 8;

    const int64_t batch = query_shape->GetDim(0);
    const int64_t max_seqlen = query_shape->GetDim(1);
    const int64_t max_kvlen = key_shape->GetDim(1);

    const int64_t q_stride_s = query_shape->GetDim(2) * head_dim;
    const int64_t k_stride_s = key_shape->GetDim(2) * head_dim;
    const int64_t v_stride_s = value_shape->GetDim(2) * head_dim;
    const int64_t o_stride_s = output_shape->GetDim(2) * head_dim;

    const int64_t q_stride_b = max_seqlen * q_stride_s;
    const int64_t k_stride_b = max_kvlen * k_stride_s;
    const int64_t v_stride_b = max_kvlen * v_stride_s;

    int64_t mask_stride_b = 0;
    int64_t mask_stride_h = 0;
    int64_t mask_stride_s = 0;
    if (attn_mask && attn_mask_shape->CalcElementsExcludingPadding() > 0) {
        if (attn_mask_shape->GetDimCount() == 4) {
            mask_stride_b = attn_mask_shape->GetDim(1) * attn_mask_shape->GetDim(2) * attn_mask_shape->GetDim(3);
            mask_stride_h = attn_mask_shape->GetDim(2) * attn_mask_shape->GetDim(3);
            mask_stride_s = attn_mask_shape->GetDim(3);
        } else if (attn_mask_shape->GetDimCount() == 3) {
            mask_stride_h = attn_mask_shape->GetDim(1) * attn_mask_shape->GetDim(2);
            mask_stride_s = attn_mask_shape->GetDim(2);
        } else if (attn_mask_shape->GetDimCount() == 2) {
            mask_stride_s = attn_mask_shape->GetDim(1);
        } else {
            LOG(ERROR) << "attn_mask must be 2d, 3d or 4d";
            return ppl::common::RC_UNSUPPORTED;
        }
        if (mask_stride_s % VPT != 0) {
            LOG(ERROR) << "last dimension of attn_mask must be aligned with " << VPT;
            return ppl::common::RC_UNSUPPORTED;
        }
    }

    return llm::cuda::xformer::fmha(
        stream,
        device_prop,
        query_shape->GetDataType(),
        query,
        key,
        value,
        attn_mask,
        nullptr,
        nullptr,
        batch,
        q_stride_b, q_stride_s, head_dim,
        k_stride_b, k_stride_s, head_dim,
        v_stride_b, v_stride_s, head_dim,
        mask_stride_b, mask_stride_s, mask_stride_h,
        o_stride_s,
        max_seqlen,
        max_kvlen,
        num_heads,
        num_kv_heads,
        head_dim,
        custom_mask_type,
        attn_scale,
        output
    );
}

ppl::common::RetCode dynamic_batching_multi_head_attention(
    const cudaStream_t stream,
    const cudaDeviceProp& device_prop,
    const ppl::common::TensorShape* query_shape,
    const void* query, // (Sq, ..., D)
    const ppl::common::TensorShape* key_shape,
    const void* key, // (Skv, ..., D)
    const ppl::common::TensorShape* value_shape,
    const void* value, // (Skv, ..., D)
    const ppl::common::TensorShape* attn_mask_shape,
    const void* attn_mask, // (seqstarts[-1], aligned(kvstarts[-1], 8)), (num_heads, seqstarts[-1], aligned(kvstarts[-1], 8))
    const void* seqstarts, // (B + 1)
    const void* kvstarts, // (B + 1)
    const bool is_causal,
    const int64_t batch,
    const int64_t decoding_batches,
    const int64_t max_seqlen,
    const int64_t max_kvlen,
    const int64_t num_heads,
    const int64_t num_kv_heads,
    const int64_t head_dim,
    const ppl::common::TensorShape* output_shape,
    void* output) // (Sq, ..., D)
{
    if (query_shape->GetDim(1) != num_heads) {
        LOG(ERROR) << "currnetly do not support leading dim on query's num_heads";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (output_shape->GetDim(1) != num_heads) {
        LOG(ERROR) << "currnetly do not support leading dim on output's num_heads";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (key_shape->GetDim(1) != num_kv_heads) {
        LOG(ERROR)
            << "currnetly do not support leading dim on current_key's num_kv_heads";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (value_shape->GetDim(1) != num_kv_heads) {
        LOG(ERROR)
            << "currnetly do not support leading dim on current_value's num_kv_heads";
        return ppl::common::RC_UNSUPPORTED;
    }

    constexpr int64_t VPT = 8;

    const int64_t custom_mask_type = is_causal ? 2 : 0;
    const float attn_scale = float(1.0 / std::sqrt(float(head_dim)));

    const int64_t q_stride_s = query_shape->GetDim(1) * head_dim;
    const int64_t k_stride_s = key_shape->GetDim(1) * head_dim;
    const int64_t v_stride_s = value_shape->GetDim(1) * head_dim;
    const int64_t o_stride_s = output_shape->GetDim(1) * head_dim;

    int64_t mask_stride_s = 0;
    int64_t mask_stride_h = 0;
    if (attn_mask && attn_mask_shape->CalcElementsExcludingPadding() > 0) {
        if (attn_mask_shape->GetDimCount() == 3) {
            mask_stride_h = attn_mask_shape->GetDim(1) * attn_mask_shape->GetDim(2);
            mask_stride_s = attn_mask_shape->GetDim(2);
        } else if (attn_mask_shape->GetDimCount() == 2) {
            mask_stride_s = attn_mask_shape->GetDim(1);
        } else {
            LOG(ERROR) << "attn_mask must be 2d or 3d";
            return ppl::common::RC_UNSUPPORTED;
        }
        if (mask_stride_s % VPT != 0) {
            LOG(ERROR) << "last dimension of attn_mask must be aligned with " << VPT;
            return ppl::common::RC_UNSUPPORTED;
        }
    }

    return llm::cuda::xformer::fmha(
        stream,
        device_prop,
        query_shape->GetDataType(),
        query,
        key,
        value,
        attn_mask,
        seqstarts,
        kvstarts,
        batch,
        0, q_stride_s, head_dim,
        0, k_stride_s, head_dim,
        0, v_stride_s, head_dim,
        0, mask_stride_s, mask_stride_h,
        o_stride_s,
        max_seqlen,
        max_kvlen,
        num_heads,
        num_kv_heads,
        head_dim,
        custom_mask_type,
        attn_scale,
        output
    );
}

}}}}}
