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

#include "ppl/kernel/llm/cuda/pmx/multi_head_cache_attention.h"
#include "ppl/common/log.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

static constexpr int32_t UNIFORM_PAGE_SIZE = 128;

ppl::common::RetCode dynamic_batching_multi_head_cache_attention::prepare(
    const cudaStream_t stream,
    const cudaDeviceProp& device_prop,
    const ppl::common::TensorShape* query_shape,
    const void* query, // (Sq, ..., D)
    const ppl::common::TensorShape* current_key_shape,
    const void* current_key, // (Skv, ..., D)
    const ppl::common::TensorShape* current_value_shape,
    const void* current_value, // (Skv, ..., D)
    const ppl::common::TensorShape* attn_mask_shape,
    const void* attn_mask, // (seqstarts[-1], aligned(kvstarts[-1], 8)), (num_heads, seqstarts[-1], aligned(kvstarts[-1], 8))
    const void* seqstarts, // (B + 1)
    const void* kvstarts, // (B + 1)
    const void* cachestarts, // (B)
    const void* start_pos, // (B)
    const void* alibi_slopes, // (num_head)
    const bool is_causal,
    const int64_t batch,
    const int64_t decoding_batches,
    const int64_t max_seqlen,
    const int64_t max_kvlen,
    const int64_t layer_idx,
    const int64_t num_layer,
    const int64_t num_heads,
    const int64_t num_kv_heads,
    const int64_t head_dim,
    const int32_t cache_mode,
    const int64_t page_size,
    const int64_t cache_stride_s,
    const int64_t cache_stride_l,
    const int64_t cache_stride_h,
    const int64_t cache_stride_kv,
    const int64_t cachestarts_stride_b,
    void* cache, // int8 (S, L, 2, KVH, D), (L, KVH, S, 2, D)
    void* scale, // float16 (S, L, 2, KVH, D/8), (L, KVH, S, 2, D/8)
    const ppl::common::TensorShape* output_shape,
    void* output) // (S, .., D)
{
    if (query_shape->GetDim(1) != num_heads) {
        LOG(ERROR) << "currnetly do not support leading dim on query's num_heads";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (output_shape->GetDim(1) != num_heads) {
        LOG(ERROR) << "currnetly do not support leading dim on output's num_heads";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (current_key_shape->GetDim(1) != num_kv_heads) {
        LOG(ERROR)
            << "currnetly do not support leading dim on current_key's num_kv_heads";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (current_value_shape->GetDim(1) != num_kv_heads) {
        LOG(ERROR)
            << "currnetly do not support leading dim on current_value's num_kv_heads";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (cache_mode == 1) {
        if (page_size != UNIFORM_PAGE_SIZE) {
            LOG(ERROR) << "currently only support page_size == UNIFORM_PAGE_SIZE";
            return ppl::common::RC_UNSUPPORTED;
        }
    }

    constexpr int64_t WARP_SIZE = 32;
    constexpr int64_t TPB = 256;
    constexpr int64_t VPT = 8;

    int64_t num_kv_repeats = num_heads / num_kv_heads;
    if ((num_kv_heads * num_kv_repeats) != num_heads) {
        LOG(ERROR) << "only support num_heads % num_kv_heads == 0.";
        return ppl::common::RC_UNSUPPORTED;
    }

    const int64_t prefill_batches = batch - decoding_batches;
    const float attn_scale = float(1.0 / std::sqrt(float(head_dim)));

    const int64_t q_stride_s = query_shape->GetDim(1) * head_dim;
    const int64_t k_stride_s = current_key_shape->GetDim(1) * head_dim;
    const int64_t v_stride_s = current_value_shape->GetDim(1) * head_dim;
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

    // ------------------- heuristic begin -------------------------------------

    // some basic data
    const int32_t mhca_head_blocks = num_heads;
    const int32_t gqca_head_block_size = 8;
    const int32_t gqca_head_blocks =
        (num_kv_repeats + gqca_head_block_size - 1) / gqca_head_block_size * num_kv_heads;
    const int32_t mhca_total_blocks = mhca_head_blocks * decoding_batches;
    const int32_t gqca_total_blocks = gqca_head_blocks * decoding_batches;
    const int32_t multi_processor_count = device_prop.multiProcessorCount;

    // liangjiexin: By the measurement on LLAMA 70B with A100 40G,
    //              I suggest not to use gqca when sm has not been occupied a lot.
    //              And gqca also increase the reduce time of multi block reduction stage.
    //              So we have to tune the multi block policy for gqca in the feature.
    bool use_infinity_gqca = decoding_batches > 0 && 
        (mhca_total_blocks > multi_processor_count * 0.6f) &&
        (num_kv_repeats == 4 || num_kv_repeats == 6 || num_kv_repeats == 8 || num_kv_repeats == 16);
    const int32_t decoding_total_blocks = (use_infinity_gqca ? gqca_total_blocks : mhca_total_blocks);
    // Get multi block enable thresholds
    // LOL. e is a magic!!! I just want the curve be smoother.
    const int32_t kv_length_threshold = 512;
    const float kv_length_scale = std::max(
        std::min(1.0f, expf(-1024.0f / max_kvlen + 1)), // scale for kvlen < 1024
        logf(max_kvlen / 1024.0f)); // scale for kvlen >= 1024 * e
    const float multi_block_threshold = multi_processor_count * kv_length_scale;

    // Get multi block size, by measurement
    int32_t decoding_multi_block_size = 1;
    if (decoding_batches > 0 &&
        decoding_total_blocks < multi_block_threshold &&
        max_kvlen >= kv_length_threshold) {
        while (decoding_multi_block_size < TPB / (head_dim / VPT)) {
            decoding_multi_block_size <<= 1;
        }
    }

    // Get TPB by decoding_total_blocks
    int32_t decoding_threads_per_block = TPB;
    if (decoding_total_blocks < multi_processor_count * 0.9f &&
        decoding_multi_block_size == 1 &&
        decoding_batches > 0) {
        int32_t num_blocks_per_sm = -1;
        // register usage of each kernel measured by mayimin
        if (use_infinity_gqca) {
            num_blocks_per_sm = head_dim == 64 ? 2 : 1;
            if (head_dim == 128 && (attn_mask || num_kv_repeats > 4)) {
                num_blocks_per_sm = 2;
            }
        } else {
            num_blocks_per_sm = head_dim == 96 ? 4 : 5;
        }
        int32_t block_size_factor =
            (multi_processor_count * num_blocks_per_sm +
                decoding_total_blocks - 1) / decoding_total_blocks;
        block_size_factor = std::min<int64_t>(block_size_factor, num_blocks_per_sm);
        decoding_threads_per_block = std::min<int64_t>(TPB * block_size_factor, 1024);
        if (decoding_threads_per_block >= 1024 && max_kvlen > 512) {
            decoding_threads_per_block = 1024;
        } else if (decoding_threads_per_block >= 512 && max_kvlen > 256) {
            decoding_threads_per_block = 512;
        } else {
            decoding_threads_per_block = 256;
        }
    }

    // Get decoding shared memory size
    bool use_infinity_mhca = false;
    int64_t decoding_shm_size = 0;
    if (decoding_batches > 0 && !use_infinity_gqca) {
        const int64_t WPT = decoding_threads_per_block / WARP_SIZE;
        const int64_t reduce_shm_size = decoding_threads_per_block / WARP_SIZE * sizeof(float);
        const int64_t max_multi_block_kvlen =
            (max_kvlen * sizeof(float) + decoding_multi_block_size - 1) / decoding_multi_block_size;
        const int64_t logits_size = max(max_multi_block_kvlen, WPT * head_dim * sizeof(float));
        decoding_shm_size = reduce_shm_size + logits_size;
        // use infinity mhca when shm is not enough
        use_infinity_mhca = decoding_shm_size > (int64_t)device_prop.sharedMemPerBlockOptin;
        if (use_infinity_mhca) {
            decoding_shm_size = 0;
            decoding_threads_per_block = std::min<int64_t>(decoding_threads_per_block, 512);
        }
    }

    // ------------------- heuristic end ---------------------------------------

    cfg.device_prop = const_cast<cudaDeviceProp*>(&device_prop);
    cfg.datatype = query_shape->GetDataType();

    cfg.query = const_cast<void*>(query);
    cfg.current_key = const_cast<void*>(current_key);
    cfg.current_value = const_cast<void*>(current_value);
    cfg.attn_mask = const_cast<void*>(attn_mask);

    cfg.seqstarts = const_cast<void*>(seqstarts);
    cfg.kvstarts = const_cast<void*>(kvstarts);
    cfg.cachestarts = const_cast<void*>(cachestarts);
    cfg.start_pos = const_cast<void*>(start_pos);
    cfg.alibi_slopes = const_cast<void*>(alibi_slopes);

    cfg.cache = cache;
    cfg.scale = scale;

    cfg.output = output;

    cfg.is_causal = is_causal;
    cfg.batch = batch;
    cfg.decoding_batches = decoding_batches;
    cfg.max_seqlen = max_seqlen;
    cfg.max_kvlen = max_kvlen;
    cfg.layer_idx =layer_idx;
    cfg.num_layer = num_layer;
    cfg.num_heads = num_heads;
    cfg.num_kv_heads = num_kv_heads;
    cfg.head_dim = head_dim;
    cfg.cache_mode = cache_mode;
    cfg.page_size = page_size;
    cfg.cache_stride_s = cache_stride_s;
    cfg.cache_stride_l = cache_stride_l;
    cfg.cache_stride_h = cache_stride_h;
    cfg.cache_stride_kv = cache_stride_kv;
    cfg.cachestarts_stride_b = cachestarts_stride_b;

    cfg.prefill_batches = prefill_batches;
    cfg.q_stride_s = q_stride_s;
    cfg.k_stride_s = k_stride_s;
    cfg.v_stride_s = v_stride_s;
    cfg.o_stride_s = o_stride_s;

    cfg.mask_stride_s = mask_stride_s;
    cfg.mask_stride_h = mask_stride_h;

    cfg.attn_scale = attn_scale;
    cfg.num_kv_repeats = num_kv_repeats;
    cfg.use_infinity_gqca = use_infinity_gqca;
    cfg.use_infinity_mhca = use_infinity_mhca;

    cfg.decoding_threads_per_block = decoding_threads_per_block;
    cfg.decoding_shm_size = decoding_shm_size;
    cfg.decoding_multi_block_size = decoding_multi_block_size;
    if (cfg.decoding_multi_block_size > 1) {
        cfg.decoding_multi_block_output_size = decoding_batches * num_heads * head_dim * decoding_multi_block_size * sizeof(int16_t);
        cfg.decoding_multi_block_sum_size = decoding_batches * num_heads * decoding_multi_block_size * sizeof(float);
        cfg.decoding_multi_block_counter_size = decoding_batches * num_heads * sizeof(int32_t);
    }

    cfg.temp_buffer_size
        = cfg.decoding_multi_block_output_size
        + cfg.decoding_multi_block_sum_size
        + cfg.decoding_multi_block_counter_size;
    cfg.temp_buffer = nullptr;

    return ppl::common::RC_SUCCESS;
}

}}}}}
