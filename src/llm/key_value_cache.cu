#include "cudakernel/llm/key_value_cache.h"
#include "ppl/common/tensor_shape.h"
#include "cudakernel/common/common.cuh"
#include <cuda_fp16.h>

/*
similar kernel like _dynamic_group_quantize_fp16_to_int8, with dynamic batch & (page_attention)

dynamic batch: key & value input is like (S,H,D), seqstart_k is (b+1) tensor contain each batch's start seqid
(page_attention: page_kvcache & page kvscale is two vector of pointer (batch, max_seqlen), each pointer's size is (L,2,H,D))

cachestarts : contain each batch's start token index in cache & scale
num_of_elements_in : current_key's element cnt
num_of_elements_out : key's element cnt
max_token : the token num carried by kvcache memory system

"S -> sum(each batch's seqlen), L -> num_layer, H -> head_size, D -> size_per_head"
*/
template<int TPB, int VPT> // 8 fp16 occupy 128 bytes, which can be loaded by a single thread at once.
__global__
void _dynamic_group_quantize_db(
    const half  *current_key, //(seqlen,H,D)
    const half  *current_value,//(seqlen,H,D)
    const int64_t  *seqstarts,//(b+1)
    const int64_t  *cachestarts,//(b)
    const int64_t  *start_pos, //(b)
    const int64_t num_of_elements, // seqlen * H * D,
    const int64_t num_layer,
    const int64_t layer_idx,
    const int64_t batch,
    const int64_t num_head,
    const int64_t head_dim,
    // need to be modified in this kernel
    int8_t        *cache, //(max_token,L,2,H,D) 1457
    half        *scale   //(max_token,L,2,H,D/g) 
){
    const int64_t idx = (blockIdx.x * TPB + threadIdx.x) * VPT;
    // int64_t max_cache_idx = 1457 * 32 * 2 * 4096;
    // int64_t max_scale_idx = 1457 * 32 * 2 * 4096 / VPT;
    half key_in[VPT]; int8_t key_out[VPT];
    half value_in[VPT]; int8_t value_out[VPT];

    if (idx < num_of_elements) {
        copy<sizeof(half) * VPT>(&current_key[idx], key_in);
        copy<sizeof(half) * VPT>(&current_value[idx], value_in);


        int64_t stride_s = num_head * head_dim;
        int64_t stride_h = head_dim;

        int64_t fuse_seq_id = idx / stride_s;
        int64_t head_id = (idx - fuse_seq_id * stride_s) / stride_h;
        int64_t dim_id = idx % stride_h;
        int64_t scale_dim_id = dim_id / VPT;

        int64_t batch_id = 0;
        int64_t seq_id = 0;

        for(int i = 1; i <= batch; i++) {
            if(fuse_seq_id < seqstarts[i]) {
                batch_id = i - 1;
                seq_id = fuse_seq_id - seqstarts[i - 1];
                break;
            }
        }

        // calculate kv scale
        const half eps = 1e-5f;
        const half fact = 127.f;
        half key_scale = 0.0f;
        half value_scale = 0.0f;

    #pragma unroll
        for(int i = 0; i < VPT; i ++){
            key_scale = key_scale > __habs(key_in[i]) ? key_scale : __habs(key_in[i]);
            value_scale = value_scale > __habs(value_in[i]) ? value_scale : __habs(value_in[i]);
        }
        key_scale = key_scale / fact; 
        value_scale = value_scale / fact;
        key_scale = key_scale > eps ? key_scale : eps;
        value_scale = value_scale > eps ? value_scale : eps;


        int64_t token_index = cachestarts[batch_id] + seq_id + start_pos[batch_id];

        int64_t token_stride = num_layer * 2 * stride_s;
        int64_t layer_stride = 2 * stride_s;
        int64_t cache_kv_stride = stride_s;

        int64_t key_idx = token_index * token_stride + layer_idx * layer_stride + 0 + head_id * stride_h + dim_id;
        int64_t value_idx = key_idx + cache_kv_stride;

        int64_t key_scale_idx = (key_idx - dim_id) / VPT + scale_dim_id;
        int64_t value_scale_idx = key_scale_idx + cache_kv_stride / VPT;
        scale[key_scale_idx] = key_scale;
        scale[value_scale_idx] = value_scale;

    #pragma unroll
        for(int i = 0; i < VPT; i ++){
            key_out[i] = (int8_t)__half2short_rn(key_in[i] / key_scale);
            value_out[i] = (int8_t)__half2short_rn(value_in[i] / value_scale);
        }

        copy<sizeof(int8_t) * VPT>(key_out, &cache[key_idx]);
        copy<sizeof(int8_t) * VPT>(value_out, &cache[value_idx]);
    }
    
}

template<int TPB, int VPT> // 8 fp16 occupy 128 bytes, which can be loaded by a single thread at once.
__global__
void _dynamic_group_dequantize_db(
    const int8_t   *cache, //(max_token,L,2,H,D) 1457
    const half     *scale,   //(max_token,L,2,H,D/g) 
    const int64_t  *kvstarts,      //(b+1) seqstarts for output_key&output_value
    const int64_t  *cachestarts,//(b)
    const int64_t num_of_elements, // seqlen * H * D,
    const int64_t num_layer,
    const int64_t layer_idx,
    const int64_t batch,
    const int64_t num_head,
    const int64_t head_dim,
    // need to be modified in this kernel
    half* output_key,              //
    half* output_value            //
){
    const int64_t idx = (blockIdx.x * TPB + threadIdx.x) * VPT;

    int8_t key_in_deq[VPT]; half key_out_deq[VPT];
    int8_t value_in_deq[VPT]; half value_out_deq[VPT];


    if (idx < num_of_elements){

        int64_t stride_s = num_head * head_dim;
        int64_t stride_h = head_dim;

        int64_t fuse_seq_id = idx / stride_s;
        int64_t head_id = (idx - fuse_seq_id * stride_s) / stride_h;
        int64_t dim_id = idx % stride_h;
        int64_t scale_dim_id = dim_id / VPT;

        int64_t batch_id = 0;
        int64_t seq_id = 0;

        for(int i = 1; i <= batch; i++) {
            if(fuse_seq_id < kvstarts[i]) {
                batch_id = i - 1;
                seq_id = fuse_seq_id - kvstarts[i - 1];
                break;
            }
        }

        int64_t token_index = cachestarts[batch_id] + seq_id;

        int64_t token_stride = num_layer * 2 * stride_s;
        int64_t layer_stride = 2 * stride_s;
        int64_t cache_kv_stride = stride_s;

        int64_t key_idx = token_index * token_stride + layer_idx * layer_stride + 0 + head_id * stride_h + dim_id;
        int64_t value_idx = key_idx + cache_kv_stride;

        int64_t key_scale_idx = (key_idx - dim_id) / VPT + scale_dim_id;
        int64_t value_scale_idx = key_scale_idx + cache_kv_stride / VPT;

        copy<sizeof(int8_t) * VPT>(&cache[key_idx], key_in_deq);
        copy<sizeof(int8_t) * VPT>(&cache[value_idx], value_in_deq);

    #pragma unroll
        for(int i = 0; i < VPT; i ++){
            key_out_deq[i] = (half)key_in_deq[i] * scale[key_scale_idx];
            value_out_deq[i] = (half)value_in_deq[i] * scale[value_scale_idx];
        }
        copy<sizeof(half) * VPT>(key_out_deq, &output_key[idx]);
        copy<sizeof(half) * VPT>(value_out_deq, &output_value[idx]);
    }

}

template<int TPB, int VPT> // 8 fp16 occupy 128 bytes, which can be loaded by a single thread at once.
__global__
void _dynamic_group_quantize_fp16_to_int8(
    const half  *key, //(b,seqlen,32,128)
    const half  *value,//(b,seqlen,32,128)
    const int64_t num_of_elements,
    const int64_t num_layer,
    const int64_t start_pos,
    const int64_t layer_idx,
    const int64_t batch,
    const int64_t max_seqlen,
    const int64_t seqlen,
    const int64_t num_head,
    const int64_t head_dim,
    int8_t        *cache, //(5,32,2,347,32,128)
    half        *scale_out //(5,32,2,347,32,16)
){
    const int64_t idx = (blockIdx.x * TPB + threadIdx.x) * VPT;
    half key_in[VPT]; int8_t key_out[VPT];
    half value_in[VPT]; int8_t value_out[VPT];

    if (idx < num_of_elements){
        int64_t stride_b = seqlen * num_head * head_dim;
        int64_t stride_s = num_head * head_dim;
        int64_t stride_h = head_dim;

        int64_t bid = idx / stride_b;
        int64_t seq_id = (idx - bid * stride_b) / stride_s;
        int64_t head_id = (idx - bid * stride_b - seq_id * stride_s) / stride_h;
        int64_t dim_id = idx % stride_h;
        int64_t scale_dim_id = dim_id / VPT;

        copy<sizeof(half) * VPT>(&key[idx], key_in);
        copy<sizeof(half) * VPT>(&value[idx], value_in);
        const half eps = 1e-5f;
        const half fact = 127.f;
        half key_scale = 0.0f;
        half value_scale = 0.0f;

    #pragma unroll
        for(int i = 0; i < VPT; i ++){
            key_scale = key_scale > __habs(key_in[i]) ? key_scale : __habs(key_in[i]);
            value_scale = value_scale > __habs(value_in[i]) ? value_scale : __habs(value_in[i]);
        }
        key_scale = key_scale / fact; 
        value_scale = value_scale / fact;
        key_scale = key_scale > eps ? key_scale : eps;
        value_scale = value_scale > eps ? value_scale : eps;
        // float inv_key_s = 1.0f / key_scale;
        // float inv_value_s = 1.0f / value_scale;

        int64_t cache_batch_stride = num_layer * 2 * max_seqlen * stride_s;
        int64_t cache_layer_stride = 2 * max_seqlen * stride_s;
        int64_t cache_kv_stride = max_seqlen * stride_s;

        int64_t key_idx = bid * cache_batch_stride + layer_idx * cache_layer_stride + 0 + (start_pos + seq_id) * stride_s + head_id * stride_h + dim_id;
        int64_t value_idx = key_idx + cache_kv_stride;

        int64_t key_scale_idx = (key_idx - dim_id) / VPT + scale_dim_id;
        int64_t value_scale_idx = key_scale_idx + cache_kv_stride / VPT;

        scale_out[key_scale_idx] = key_scale;
        scale_out[value_scale_idx] = value_scale;

    #pragma unroll
        for(int i = 0; i < VPT; i ++){
            key_out[i] = (int8_t)__half2short_rn(key_in[i] / key_scale);
            value_out[i] = (int8_t)__half2short_rn(value_in[i] / value_scale);
        }
        copy<sizeof(int8_t) * VPT>(key_out, &cache[key_idx]);
        copy<sizeof(int8_t) * VPT>(value_out, &cache[value_idx]);
    }
}

template<int TPB, int VPT>
__global__
void _dynamic_group_dequantize_int8_to_fp16(
    const int8_t  *cache, // (b,32,2,max_t,32,128)
    const half  *scale, // (b,32,2,max_t,32,16)
    const int64_t  num_of_elements,
    const int64_t num_layer,
    const int64_t start_pos,
    const int64_t layer_idx,
    const int64_t batch,
    const int64_t max_seqlen,
    const int64_t seqlen,
    const int64_t num_head,
    const int64_t head_dim,
    half        *output_key,  //(b,start_pos + seqlen,32,128)
    half        *output_value
){
    const int64_t idx = (blockIdx.x * TPB + threadIdx.x) * VPT;
    int8_t key_in[VPT]; half key_out[VPT];
    int8_t value_in[VPT]; half value_out[VPT];


    if (idx < num_of_elements){

        int64_t stride_b = (start_pos + seqlen) * num_head * head_dim;
        int64_t stride_s = num_head * head_dim;
        int64_t stride_h = head_dim;

        int64_t bid = idx / stride_b;
        int64_t seq_id = (idx - bid * stride_b) / stride_s;
        int64_t head_id = (idx - bid * stride_b - seq_id * stride_s) / stride_h;
        int64_t dim_id = idx % stride_h;
        int64_t scale_dim_id = dim_id / VPT;

        int64_t cache_batch_stride = num_layer * 2 * max_seqlen * stride_s;
        int64_t cache_layer_stride = 2 * max_seqlen * stride_s;
        int64_t cache_kv_stride = max_seqlen * stride_s;

        int64_t key_idx = bid * cache_batch_stride + layer_idx * cache_layer_stride + 0 + seq_id * stride_s + head_id * stride_h + dim_id;
        int64_t value_idx = key_idx + cache_kv_stride;

        int64_t key_scale_idx = (key_idx - dim_id) / VPT + scale_dim_id;
        int64_t value_scale_idx = key_scale_idx + cache_kv_stride / VPT;

        copy<sizeof(int8_t) * VPT>(&cache[key_idx], key_in);
        copy<sizeof(int8_t) * VPT>(&cache[value_idx], value_in);
        

    #pragma unroll
        for(int i = 0; i < VPT; i ++){
            key_out[i] = (half)key_in[i] * scale[key_scale_idx];
            value_out[i] = (half)value_in[i] * scale[value_scale_idx];
        }
        copy<sizeof(half) * VPT>(key_out, &output_key[idx]);
        copy<sizeof(half) * VPT>(value_out, &output_value[idx]);
    }
}

ppl::common::RetCode PPLCUDAKeyValueCacheForwardImp(
    cudaStream_t stream,
    ppl::common::TensorShape* key_shape,
    const void* key,
    const void* value,
    const int64_t start_pos,
    const int64_t num_layer,
    const int64_t layer_idx,
    ppl::common::TensorShape* kvcache_shape,
    void* kvcache, // (b,32,2,max_t,32,128)
    void* kvscale, // (b,32,2,max_t,32,16)
    void* key_out,   //[b, start_pos + seqlen, h, d]
    void* value_out)
{
    
    constexpr int32_t TPB = 256;
    constexpr int32_t VPT = 16 / sizeof(half);
    const int64_t batch = key_shape->GetDim(0);
    const int64_t seqlen = key_shape->GetDim(1);
    const int64_t num_heads = key_shape->GetDim(2);
    const int64_t size_per_head = key_shape->GetDim(3);
    const int64_t max_seqlen = kvcache_shape->GetDim(3);

    const int64_t num_of_elements_in = batch * seqlen * num_heads * size_per_head;
    const int64_t num_of_elements_out = batch * (start_pos + seqlen) * num_heads * size_per_head;

    const int64_t grid_size_in = num_of_elements_in / VPT / TPB + (num_of_elements_in % (VPT * TPB) != 0);
    const int64_t grid_size_out = num_of_elements_out / VPT / TPB + (num_of_elements_out % (VPT * TPB) != 0);

    dim3 block(TPB);
    
    _dynamic_group_quantize_fp16_to_int8<TPB, VPT><<<grid_size_in, block, 0, stream>>>((half*)key, (half*)value, num_of_elements_in, num_layer, start_pos, layer_idx, batch,
                                max_seqlen, seqlen, num_heads, size_per_head, (int8_t*)kvcache, (half*)kvscale);

    _dynamic_group_dequantize_int8_to_fp16<TPB, VPT><<<grid_size_out, block, 0, stream>>>((int8_t*)kvcache, (half*)kvscale, num_of_elements_out, num_layer, start_pos, layer_idx, batch,
                                max_seqlen, seqlen, num_heads, size_per_head, (half*)key_out, (half*)value_out);
      
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode PPLCUDAKeyValueCacheDBForwardImp(
    cudaStream_t stream,
    ppl::common::TensorShape* input_kv_shape,
    const void* current_key, // (seqlen,h,d)
    const void* current_value,
    const void* seqstarts,//(b+1)
    const void* kvstarts,
    const void* cachestarts,//(b)
    ppl::common::TensorShape* start_pos_shape,
    const void* start_pos,
    void* cache, //(max_token,L,2,H,D) 
    void* scale, //(max_token,L,2,H,D/g) 
    const int64_t layer_idx,
    const int64_t num_layer,
    ppl::common::TensorShape* output_kv_shape,
    void* key,   //[sum(start_pos) + seqlen, h, d]
    void* value)
{

    constexpr int32_t TPB = 256;
    constexpr int32_t VPT = 16 / sizeof(half);

    const int64_t batch = start_pos_shape->GetDim(0);
    const int64_t input_seqlen = input_kv_shape->GetDim(0);
    const int64_t output_seqlen = output_kv_shape->GetDim(0);
    const int64_t num_heads = output_kv_shape->GetDim(1);
    const int64_t size_per_head = output_kv_shape->GetDim(2);

    const int64_t num_of_elements_in = input_seqlen * num_heads * size_per_head;
    const int64_t num_of_elements_out = output_seqlen * num_heads * size_per_head;

    const int64_t grid_size_quant = num_of_elements_in / VPT / TPB + (num_of_elements_in % (VPT * TPB) != 0);
    const int64_t grid_size_dequant = num_of_elements_out / VPT / TPB + (num_of_elements_in % (VPT * TPB) != 0);

    dim3 block(TPB);
    _dynamic_group_quantize_db<TPB, VPT><<<grid_size_quant, block, 0, stream>>>((half*)current_key, (half*)current_value, (int64_t*)seqstarts, (int64_t*)cachestarts, (int64_t*)start_pos,
            num_of_elements_in, num_layer, layer_idx, batch, num_heads, size_per_head, (int8_t*)cache, (half*)scale);

    _dynamic_group_dequantize_db<TPB, VPT><<<grid_size_dequant, block, 0, stream>>>((int8_t*)cache, (half*)scale, (int64_t*)kvstarts, (int64_t*)cachestarts,
            num_of_elements_out, num_layer, layer_idx, batch, num_heads, size_per_head, (half*)key, (half*)value);

      
    return ppl::common::RC_SUCCESS;
}
