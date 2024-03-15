#pragma once

#include "cuda_runtime.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace flash_attn2 {

void run_fmha_fwd(
    const cudaStream_t stream,
    const cudaDeviceProp& device_prop,
    // const ppl::common::datatype_t datatype, // currently no dtype, only fp16!
    const void* query,
    const void* key,
    const void* value,
    const void* optional_attn_mask,
    const void* optional_seqstart_q, // (B + 1)
    const void* optional_seqstart_k, // (B + 1)
    const int64_t batch,
    const int64_t query_stride_b, // 0 if dynamic batch
    const int64_t query_stride_s,
    const int64_t query_stride_h,
    const int64_t key_stride_b, // 0 if dynamic batch
    const int64_t key_stride_s,
    const int64_t key_stride_h,
    const int64_t value_stride_b, // 0 if dynamic batch
    const int64_t value_stride_s,
    const int64_t value_stride_h,
    const int64_t mask_stride_b, //
    const int64_t mask_stride_s,
    const int64_t mask_stride_h,
    const int64_t output_stride_s,
    const int64_t max_seqlen,
    const int64_t max_kvlen, // unused if dynamic batch
    const int64_t num_heads,
    const int64_t num_kv_heads,
    const int64_t head_dim,
    const int64_t is_causal, //
    const float attn_scale,
    const void* alibi_slopes_ptr,
    const int64_t alibi_slopes_batch_stride,
    void* output);

}}}}}
