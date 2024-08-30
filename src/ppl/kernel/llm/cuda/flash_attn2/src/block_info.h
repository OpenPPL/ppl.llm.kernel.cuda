/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

namespace flash {

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool Varlen=true>
struct BlockInfo {

    template<typename Params>
    __device__ BlockInfo(const Params &params, const int bidb)
        : sum_s_q(!Varlen || params.cu_seqlens_q == nullptr ? -1 : params.cu_seqlens_q[bidb])
        , sum_s_k(!Varlen || params.cu_seqlens_k == nullptr || !params.is_seqlens_k_cumulative ? -1 : params.cu_seqlens_k[bidb])
        , actual_seqlen_q(!Varlen || params.cu_seqlens_q == nullptr ? params.seqlen_q : params.cu_seqlens_q[bidb + 1] - sum_s_q)
        // If is_seqlens_k_cumulative, then seqlen_k is cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb].
        // Otherwise it's cu_seqlens_k[bidb], i.e., we use cu_seqlens_k to store the sequence lengths of K.
        , seqlen_k_cache(!Varlen || params.cu_seqlens_k == nullptr ? params.seqlen_k : (params.is_seqlens_k_cumulative ? params.cu_seqlens_k[bidb + 1] - sum_s_k : params.cu_seqlens_k[bidb]))
        , actual_seqlen_k(params.seqused_k ? params.seqused_k[bidb] : seqlen_k_cache + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew))
        {
        }

    template <typename index_t>
    __forceinline__ __device__ index_t q_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
        return sum_s_q == -1 ? bidb * batch_stride : uint32_t(sum_s_q) * row_stride;
    }

    template <typename index_t>
    __forceinline__ __device__ index_t k_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
        return sum_s_k == -1 ? bidb * batch_stride : uint32_t(sum_s_k) * row_stride;
    }

    // start idx for attn_bias of current batch&head
    // shape: (batch, num_heads,     seqlen,     kvlen)    (static batch)
    //        ( N.A., num_heads, sum_seqlen, sum_kvlen)    (dynamic batch, block-diagonal w.r.t batch)
    // stride (   s2,        s1,         s0,         1)
    // For dynamic batch, attn_bias may look like as follows (per head):
    //      xxx0000     batch0
    //      xxx0000
    //      000yyyy     batch1
    //      000yyyy
    //      000yyyy
    //      ...
    template <typename index_t>
    __forceinline__ __device__ index_t bias_offset(
        const index_t batch_stride,
        const index_t head_stride,
        const index_t seqlenq_stride,
        const int bidb,
        const int bidh) const {
        if (sum_s_q == -1 && sum_s_k == -1)
            return bidb * batch_stride + bidh * head_stride;
        else
            return (bidh * head_stride + uint32_t(sum_s_q) * seqlenq_stride + uint32_t(sum_s_k));
    }

    // get bias shape (last 2 dim)
    // shape: (batch, num_heads,     seqlen,     kvlen)    (static batch)
    //        ( N.A., num_heads, sum_seqlen, sum_kvlen)    (dynamic batch, block-diagonal w.r.t batch)
    // CHECK: bias shape is not used at all?
    template <typename index_t>
    __forceinline__ __device__ void get_bias_shape(
        const index_t seqlen,
        const index_t kvlen,
        index_t& qsize,
        index_t& ksize) const {
        if (sum_s_q == -1 && sum_s_k == -1)
        {
            qsize = seqlen;
            ksize = kvlen;
        }
        else
        {
            qsize = sum_s_q;
            ksize = sum_s_k;
        }
    }

    const int sum_s_q;
    const int sum_s_k;
    const int actual_seqlen_q;
    // We have to have seqlen_k_cache declared before actual_seqlen_k, otherwise actual_seqlen_k is set to 0.
    const int seqlen_k_cache;
    const int actual_seqlen_k;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace flash
