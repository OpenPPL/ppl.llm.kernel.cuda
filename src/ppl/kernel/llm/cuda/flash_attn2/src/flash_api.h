#pragma once

#include "cuda_runtime.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace flash_attn2 {

void run_fmha_fwd(
    const cudaStream_t stream,
    const cudaDeviceProp& device_prop,
    // const ppl::common::datatype_t datatype, // no dtype, only fp16!
    const void* query,
    const void* key,
    const void* value,
    const void* optional_attn_mask,
    const void* optional_seqstart_q,
    const void* optional_seqstart_k,
    const void* optional_block_table,       // block table for paged attention
    const void* optional_cache_batch_idx,   // int32
    const void* optional_k_quant_scale,
    const void* optional_v_quant_scale,
    const void* optional_alibi_slopes_ptr,
    const int64_t batch,
    const int64_t query_stride_b,
    const int64_t query_stride_s,
    const int64_t query_stride_h,
    const int64_t key_stride_b,
    const int64_t key_stride_s,
    const int64_t key_stride_h,
    const int64_t value_stride_b,
    const int64_t value_stride_s,
    const int64_t value_stride_h,
    const int64_t mask_stride_b,
    const int64_t mask_stride_s,
    const int64_t mask_stride_h,
    const int64_t alibi_slopes_stride_b,
    const int64_t output_stride_b,
    const int64_t output_stride_s,
    const int64_t output_stride_h,
    const int64_t max_seqlen,
    const int64_t max_kvlen,
    const int64_t num_heads,
    const int64_t num_kv_heads,
    // const int64_t head_dim,
    const int64_t qk_head_dim,
    const int64_t v_head_dim,
    const bool is_causal,
    const bool is_mla,
    const int64_t cache_mode,          // 0 for normal, 1 for paged attention
    const int64_t page_block_size,
    const int64_t block_table_batch_stride,
    const int64_t quant_bit,           // 0 for no quant, 8 for 8bit int
    const int64_t quant_group,
    const float attn_scale,
    void* output);

}}}}}
