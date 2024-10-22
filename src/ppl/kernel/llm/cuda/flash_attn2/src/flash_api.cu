/******************************************************************************
 * orginal Copyright (c) 2023, Tri Dao.
 *
 * Modifications are made to incorporate into ppl.nn.llm:
 *  1. remove pytorch/aten dependency
 *  2. no dropout, not needed in ppl.nn.llm
 *  3. never write back softmax score
 *  4. using 64bit index if necessary
 ******************************************************************************/

#include <cutlass/numeric_types.h>

#include "flash.h"
#include "static_switch.h"
#include <assert.h>

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace flash_attn2 {

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel=false) {
    // remove bf16 for compilation speed
    //FP16_SWITCH(!params.is_bf16, [&] {
    using elem_type = cutlass::half_t;
    if (params.is_mla) {
        assert(params.qk_d == 192);
        assert(params.num_splits <= 1 && !force_split_kernel);
        MLA_HEADDIM_SWITCH(params.qk_d, [&] {
            run_mha_fwd_mla_<elem_type, kQKHeadDim, 0, 0>(params, stream);
        });
    } else {
        HEADDIM_SWITCH(params.qk_d, [&] {
            QUANTBIT_SWITCH(params.quant_bit, [&] {
                constexpr static int QuantGroup = QuantBit;
                if (params.num_splits <= 1 && !force_split_kernel) {  // If we don't set it num_splits == 0
                    run_mha_fwd_<elem_type, kQKHeadDim, QuantBit, QuantGroup>(params, stream);
                } else {
                    run_mha_fwd_splitkv_dispatch<elem_type, kQKHeadDim, QuantBit, QuantGroup>(params, stream);
                }
            });
        });
    }
}

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
    void* output)
{
    if (!is_mla) {
        assert(qk_head_dim == v_head_dim);
    }

    Flash_fwd_params params = {};

    params.dprops = &device_prop;

    params.q_ptr = const_cast<void*>(query);
    params.k_ptr = const_cast<void*>(key);
    params.v_ptr = const_cast<void*>(value);
    params.o_ptr = output;

    // set bias ptr
    params.bias_ptr = const_cast<void*>(optional_attn_mask);

    // set scale ptr
    params.k_scale_ptr = const_cast<void*>(optional_k_quant_scale);
    params.v_scale_ptr = const_cast<void*>(optional_v_quant_scale);

    params.is_bf16 = false; // TODO

    params.q_row_stride  = query_stride_s;
    params.q_head_stride = query_stride_h;
    params.k_row_stride  = key_stride_s;
    params.k_head_stride = key_stride_h;
    params.v_row_stride  = value_stride_s;
    params.v_head_stride = value_stride_h;
    params.o_row_stride  = output_stride_s;
    params.o_head_stride = output_stride_h;

    // 0 for dynamic batch
    params.q_batch_stride = query_stride_b;
    params.k_batch_stride = key_stride_b;
    params.v_batch_stride = value_stride_b;
    params.o_batch_stride = output_stride_b; // same as q?

    // bias strides
    params.bias_batch_stride   = mask_stride_b;
    params.bias_head_stride    = mask_stride_h;
    params.bias_seqlenq_stride = mask_stride_s;

    // only for varlen
    // cu_seqlens_q = nullptr for ordinary fmha
    params.cu_seqlens_q =(int64_t*) const_cast<void*>(optional_seqstart_q);
    params.cu_seqlens_k =(int64_t*) const_cast<void*>(optional_seqstart_k);

    // P = softmax(QK^T)
    params.p_ptr = nullptr; //p_d;

    // softmax_lse is not necessary for attention forward pass!
    // size_t softmax_lse_size = get_softmax_lse_size(max_seqlen, batch, num_heads);
    // void** p_lse;
    // cudaMalloc((void**)&p_lse, softmax_lse_size*2);
    params.softmax_lse_ptr = nullptr; //p_lse;

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int qk_head_dim_rounded = round_multiple((int)qk_head_dim, 32);
    const int v_head_dim_rounded = round_multiple((int)v_head_dim, 32);
    const int max_seqlen_rounded = round_multiple((int)max_seqlen, 128);
    const int max_kvlen_rounded = round_multiple((int)max_kvlen, 128);

    // Set the dimensions.
    params.b = batch;
    params.h = num_heads;
    params.h_k = num_kv_heads;
    params.h_h_k_ratio = num_heads / num_kv_heads;
    params.seqlen_q = max_seqlen;
    params.seqlen_k = max_kvlen;
    params.seqlen_q_rounded = max_seqlen_rounded;
    params.seqlen_k_rounded = max_kvlen_rounded;
    params.qk_d = qk_head_dim;
    params.v_d = v_head_dim;
    // params.d_rounded = head_dim_rounded;
    params.qk_d_rounded = qk_head_dim_rounded;
    params.v_d_rounded = v_head_dim_rounded;

    params.scale_softmax = attn_scale;
    params.scale_softmax_log2 = attn_scale * M_LOG2E;
    params.scale_bias = 1.0f / params.scale_softmax;

    params.is_causal = is_causal;
    params.is_seqlens_k_cumulative = true;

    params.block_table = const_cast<int64_t*>(reinterpret_cast<const int64_t*>(optional_block_table));
    params.block_table_batch_stride = block_table_batch_stride;
    params.page_block_size = page_block_size;

    params.cache_batch_idx = const_cast<int64_t*>(reinterpret_cast<const int64_t*>(optional_cache_batch_idx));

    // If window_size != (-1, -1), implements sliding window local attention. Query at position i
    // will only attend to keys between
    // [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.
    params.window_size_left = -1;
    params.window_size_right = -1;
    if (is_causal)
    {
        params.window_size_right = 0;
    }

    // TODO: alibi_slope;
    params.alibi_slopes_ptr = const_cast<void*>(optional_alibi_slopes_ptr);
    params.alibi_slopes_batch_stride = alibi_slopes_stride_b;

    //
    bool paged_KVCache = cache_mode==1 && optional_block_table!=nullptr;
    if (paged_KVCache)
        params.num_splits = 1;

    // kv quant group
    params.quant_bit = quant_bit;
    params.quant_group = quant_group;
    // mla
    params.is_mla = is_mla;
    run_mha_fwd(params, stream, paged_KVCache);
}

}}}}}
