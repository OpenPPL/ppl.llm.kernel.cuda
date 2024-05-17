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

#include "ppl/kernel/llm/cuda/pmx/sample.h"
#include "ppl/common/log.h"

#include "cudakernel/common/common.cuh"

#include <cuda_runtime.h>
#include <float.h>
#include <cub/cub.cuh>

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

using fp32_t = float;

struct SortingPair {
    fp32_t value;
    int32_t index;
    __device__ SortingPair(fp32_t value, int32_t index): value(value), index(index) {}
};

template<typename Dtype, int32_t TPB>
__device__ __host__ inline
int32_t pad_vocab(int32_t vocab_size)
{
    // for vector load/store
    constexpr int32_t VPT = 16 / sizeof(Dtype);
    return vocab_size + (TPB * VPT) - vocab_size % (TPB * VPT);
}

template <int32_t TPB, int32_t VPT>
struct CachedVocabStorage {
    fp32_t* local_storage;
    const fp32_t *data_ptr;
    int32_t index;

    __device__ inline CachedVocabStorage(
        const fp32_t *data_ptr,
        fp32_t *local_storage,
        const int64_t base_idx)
    {
        this->data_ptr      = data_ptr;
        this->local_storage = local_storage;
        this->index         = base_idx;
    }

    __device__ inline fp32_t Pop(int32_t local_selection)
    {
        if(local_selection % VPT == 0)
            copy<sizeof(float) * VPT>(data_ptr + index + local_selection, local_storage);
        return local_storage[local_selection % VPT];
    }
};

template<int32_t WPT>
__device__ inline
fp32_t sample_block_reduce_max(fp32_t reducing, fp32_t *shared_mem)
{
    // Helper function for reduce max.
    constexpr int32_t WARP_SIZE = 32;
    const int32_t lane_id = threadIdx.x % WARP_SIZE;
    const int32_t warp_id = threadIdx.x / WARP_SIZE;

    for (int32_t mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
        reducing = fmaxf(reducing, __shfl_xor_sync(uint32_t(-1), reducing, mask));
    }

    if (lane_id == 0) {
        shared_mem[warp_id] = reducing;
    }
    __syncthreads();

    if (lane_id < WPT) reducing = shared_mem[lane_id];
    else reducing = -FLT_MAX;

# pragma unroll
    for (int32_t mask = WPT / 2; mask >= 1; mask /= 2) {
        reducing = fmaxf(reducing, __shfl_xor_sync(uint32_t(-1), reducing, mask));
    }

    reducing = __shfl_sync(uint32_t(-1), reducing, 0);
    return reducing;
}

template<int32_t WPT>
__device__ inline
SortingPair sample_block_reduce_max_with_index(fp32_t reducing, int32_t index, void *shared_mem)
{
    // Helper function for reduce max.
    constexpr int32_t WARP_SIZE = 32;
    const int32_t lane_id = threadIdx.x % WARP_SIZE;
    const int32_t warp_id = threadIdx.x / WARP_SIZE;

    int32_t* shared_mem_i32 = reinterpret_cast<int32_t*>(shared_mem);
    fp32_t* shared_mem_fp32 = reinterpret_cast<fp32_t*>(shared_mem) + WPT;

    fp32_t reducing_value = reducing, receving_value;
    int32_t reducing_index = index, receving_index;

    for (int32_t mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
        receving_value = __shfl_xor_sync(uint32_t(-1), reducing_value, mask);
        receving_index = __shfl_xor_sync(uint32_t(-1), reducing_index, mask);

        if (receving_value > reducing_value){
            reducing_value = receving_value;
            reducing_index = receving_index;
        }
    }

    if (lane_id == 0) {
        shared_mem_fp32[warp_id] = reducing_value;
        shared_mem_i32[warp_id] = reducing_index;
    }
    __syncthreads();

    if (lane_id < WPT) {
        reducing_value = shared_mem_fp32[lane_id];
        reducing_index = shared_mem_i32[lane_id];
    }
    else {
        reducing_value = -FLT_MAX;
        reducing_index = -1;
    }

# pragma unroll
    for (int32_t mask = WPT / 2; mask >= 1; mask /= 2) {
        receving_value = __shfl_xor_sync(uint32_t(-1), reducing_value, mask);
        receving_index = __shfl_xor_sync(uint32_t(-1), reducing_index, mask);

        if (receving_value > reducing_value){
            reducing_value = receving_value;
            reducing_index = receving_index;
        }
    }

    reducing_value = __shfl_sync(uint32_t(-1), reducing_value, 0);
    reducing_index = __shfl_sync(uint32_t(-1), reducing_index, 0);
    return SortingPair(reducing_value, reducing_index);
}


template<int32_t WPT>
__device__ inline
fp32_t sample_block_reduce_sum(fp32_t reducing, fp32_t *shared_mem)
{
    // Helper function for reduce sum.
    constexpr int32_t WARP_SIZE = 32;
    const int32_t lane_id = threadIdx.x % WARP_SIZE;
    const int32_t warp_id = threadIdx.x / WARP_SIZE;

# pragma unroll
    for (int32_t mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
        reducing += __shfl_xor_sync(uint32_t(-1), reducing, mask);
    }

    if (lane_id == 0) shared_mem[warp_id] = reducing;
    __syncthreads();

    if (lane_id < WPT) reducing = shared_mem[lane_id];

# pragma unroll
    for (int32_t mask = WPT / 2; mask >= 1; mask /= 2) {
        reducing += __shfl_xor_sync(uint32_t(-1), reducing, mask);
    }
    reducing = __shfl_sync(uint32_t(-1), reducing, 0);
    return reducing;
}

struct SamplePrefixOp
{
    // Powered by TRT
    fp32_t running_total;

    __device__ SamplePrefixOp(fp32_t running_total) : running_total(running_total) {}

    __device__ fp32_t operator() (fp32_t block_aggregate) {
        fp32_t old_prefix = running_total;
        running_total += block_aggregate;
        return old_prefix;
    }
};

template<int32_t TPB, int32_t VPT>
__global__
void sample_topk_topp_default_kernel(
    const fp32_t __restrict__ *logits,      // [num_batches, batch_stride]
    const fp32_t *temperatures,             // [num_batches]
    const fp32_t *top_p,                    // [num_batches]
    const fp32_t *rnd,                      // [num_batches]
    const int32_t vocab_size,
    const int32_t batch_stride,
    const int32_t top_k_val,
    const fp32_t top_p_val,
    const fp32_t rnd_val,
    fp32_t *padded_logits,                  // [num_batches, padded_vocab_size]
    int32_t *output)                        // [num_batches, 1])
{
    constexpr int32_t WARP_SIZE = 32;
    constexpr int32_t WPT = TPB / WARP_SIZE; // warp per thread block.

    const int64_t batch_id            = blockIdx.x;
    const int64_t batch_offset        = batch_id * batch_stride;
    const int64_t padded_batch_offset = batch_id * pad_vocab<fp32_t, TPB>(vocab_size);
    const fp32_t  temperature         = temperatures ? max(abs(temperatures[batch_id]) + 1e-7, 0.01): 1.0f; // temperature 最低 0.01

    extern __shared__ fp32_t topk_shm[];
    fp32_t *topk_sums = topk_shm;
    int32_t *topk_idxs = reinterpret_cast<int32_t*>(&topk_shm[top_k_val]);
    __shared__ fp32_t reducing_memory[WPT * 2];

    fp32_t max_val = -FLT_MAX;
    int32_t max_idx = -1;
    float local_sum = 1.f;
    float local_top_p = top_p == nullptr ? top_p_val : top_p[batch_id];

    for (int32_t vocab_base = threadIdx.x * VPT; vocab_base < vocab_size; vocab_base += TPB * VPT) {
        fp32_t local_vals[VPT];

        #pragma unroll
        for (int32_t vec_idx = 0; vec_idx < VPT; vec_idx++) {
            int32_t vocab_idx = vocab_base + vec_idx;
            // TODO: optimize branch (vocab_idx < vocab_size)
            local_vals[vec_idx] = vocab_idx < vocab_size
                ? logits[batch_offset + vocab_idx]
                : -FLT_MAX;
            if (local_vals[vec_idx] > max_val) {
                max_val = local_vals[vec_idx];
                max_idx = vocab_idx;
            }
        }

        // TODO: optimize branch
        if (local_top_p != 0.f)
            copy<VPT * sizeof(fp32_t)>(local_vals, &padded_logits[padded_batch_offset + vocab_base]);
    }
    SortingPair max_pair = sample_block_reduce_max_with_index<WPT>(max_val, max_idx, reducing_memory);
    max_val = max_pair.value;
    max_idx = max_pair.index;

    if (local_top_p == 0.f) {
        if (threadIdx.x == 0) {
            output[batch_id] = max_idx;
        }
    } else {
        if (threadIdx.x == 0) {
            topk_sums[0] = 1.f;
            topk_idxs[0] = max_idx;
            padded_logits[padded_batch_offset + max_idx] = -FLT_MAX;
        }
        __syncthreads();

        for (int32_t i = 1; i < top_k_val; i++) {
            fp32_t local_max_val = -FLT_MAX;
            int32_t local_max_idx = -1;
            for (int32_t vocab_base = threadIdx.x * VPT; vocab_base < vocab_size; vocab_base += TPB * VPT) {
                fp32_t local_vals[VPT];
                copy<VPT * sizeof(fp32_t)>(&padded_logits[padded_batch_offset + vocab_base], local_vals);

                #pragma unroll
                for (int32_t vec_idx = 0; vec_idx < VPT; vec_idx++) {
                    if (local_vals[vec_idx] > local_max_val) {
                        local_max_val = local_vals[vec_idx];
                        local_max_idx = vocab_base + vec_idx;
                    }
                }
            }
            SortingPair p = sample_block_reduce_max_with_index<WPT>(local_max_val, local_max_idx, reducing_memory);
            local_max_val = p.value;
            local_max_idx = p.index;

            local_max_val = exp((local_max_val - max_val) / temperature);
            local_sum += local_max_val;

            if (threadIdx.x == 0) {
                topk_sums[i] = local_sum;
                topk_idxs[i] = local_max_idx;
                padded_logits[padded_batch_offset + local_max_idx] = -FLT_MAX;
            }
        }

        __syncthreads();

        local_sum = __fdividef(1.f, local_sum + 1e-6f);
        const fp32_t top_p_selection = local_top_p * (rnd == nullptr ? rnd_val : rnd[batch_id]);
        fp32_t prob_sum = (threadIdx.x < top_k_val) ? topk_sums[threadIdx.x] * local_sum : 1.f;
        int32_t count = __syncthreads_count((int32_t)(prob_sum >= top_p_selection));

        if (threadIdx.x == min(TPB - count, TPB - 1)) {
            output[batch_id] = topk_idxs[threadIdx.x];
        }
    }
}


template<int32_t TPB, int32_t LPT, int32_t KPT>
__global__
void sample_topk_topp_radix_select_kernel(
    const fp32_t __restrict__ *logits,      // [num_batches, batch_stride]
    const fp32_t *temperatures,             // [num_batches]
    const fp32_t *top_p,                    // [num_batches]
    const fp32_t *rnd,                      // [num_batches]
    const int32_t vocab_size,
    const int32_t batch_stride,
    const int32_t top_k_val,
    const fp32_t top_p_val,
    const fp32_t rnd_val,
    int32_t *output)                        // [num_batches, 1])
{
    constexpr int32_t WARP_SIZE = 32;
    constexpr int32_t WPT = TPB / WARP_SIZE; // warp per thread block.
    constexpr int32_t LOAD_LEN = LPT - KPT;

    const int64_t batch_id            = blockIdx.x;
    const int64_t batch_offset        = batch_id * batch_stride;
    const fp32_t  temperature         = temperatures ? max(abs(temperatures[batch_id]) + 1e-7, 0.01): 1.0f; // temperature 最低 0.01

    __shared__ fp32_t reducing_memory[WPT * 2];

    fp32_t local_vals[LPT];
    int32_t local_idxs[LPT];
    #pragma unroll
    for (int32_t i = 0; i < KPT; i++) {
        int32_t vocab_idx = threadIdx.x + i * TPB;
        if (vocab_idx < vocab_size) {
            local_vals[i] = logits[batch_offset + vocab_idx];
            local_idxs[i] = vocab_idx;
        } else {
            local_vals[i] = -FLT_MAX;
            local_idxs[i] = -1;
        }
    }

    typedef cub::BlockRadixSort<fp32_t, TPB, LPT, int32_t> BlockRadixSort;
    __shared__ typename BlockRadixSort::TempStorage sort_storage;

    for (int32_t vocab_base = threadIdx.x + KPT * TPB; vocab_base < vocab_size; vocab_base += TPB * LOAD_LEN) {
        #pragma unroll
        for (int32_t i = 0; i < LOAD_LEN; i++) {
            int32_t vocab_idx = vocab_base + i * TPB;
            // TODO: optimize branch
            if (vocab_idx < vocab_size) {
                local_vals[i + KPT] = logits[batch_offset + vocab_idx];
                local_idxs[i + KPT] = vocab_idx;
            } else {
                local_vals[i + KPT] = -FLT_MAX;
                local_idxs[i + KPT] = -1;
            }
        }
        __syncthreads();
        BlockRadixSort(sort_storage).SortDescendingBlockedToStriped(local_vals, local_idxs);
    }

    float local_top_p = (top_p == nullptr ? top_p_val : top_p[batch_id]);
    if (local_top_p == 0.f) {
        if (threadIdx.x == 0) {
            output[batch_id] = local_idxs[0];
        }
    } else {
        if (threadIdx.x == 0) {
            reducing_memory[0] = local_vals[0];
        }
        __syncthreads();

        float local_sum = 0.f;
        fp32_t max_val = reducing_memory[0];

        for (int32_t i = 0; i < KPT; i++) {
            if (threadIdx.x + i * TPB < top_k_val) {
                local_vals[i] = exp((local_vals[i] - max_val) / temperature);
                local_sum += local_vals[i];
            }
        }

        local_sum = sample_block_reduce_sum<WPT>(local_sum, reducing_memory);
        local_sum = __fdividef(1.f, local_sum + 1e-6f);

        typedef cub::BlockScan<float, TPB> BlockScan;
        __shared__ typename BlockScan::TempStorage scan_storage;
        SamplePrefixOp prefix_op(0);

        const fp32_t top_p_selection = local_top_p * (rnd == nullptr ? rnd_val : rnd[batch_id]);
        fp32_t prob_sum = 0.f;
        int32_t count = 0;
        int32_t select_idx;

        for (int32_t i = 0; i < KPT; i++) {
            int32_t top_idx = threadIdx.x + i * TPB;
            fp32_t prob = (top_idx < top_k_val) ? local_vals[i] * local_sum : 1.f;
            BlockScan(scan_storage).InclusiveSum(prob, prob_sum, prefix_op);
            count = __syncthreads_count((int32_t)(prob_sum >= top_p_selection));
            select_idx = local_idxs[i];
            if (count != 0) {
                break;
            }
        }

        if (threadIdx.x == min(TPB - count, TPB - 1)) {
            output[batch_id] = select_idx;
        }
    }
}


template<int32_t TPB, int32_t VPT, int32_t TILE>
__global__
void flash_sample_top_p_kernel(
    const fp32_t __restrict__ *logits,      // [num_batches, batch_stride]
    const fp32_t *temperatures,             // [num_batches]
    const fp32_t *top_p,                    // [num_batches]
    const fp32_t *rnd,                      // [num_batches]
    const int32_t vocab_size,
    const int32_t batch_stride,
    const fp32_t top_p_val,
    const fp32_t rnd_val,
    fp32_t *sorted_value,                   // [num_batches, padded_vocab_size]
    int32_t *sorted_order,                  // [num_batches, padded_vocab_size]
    int32_t *output)                        // [num_batches])
{
    /*
        这是一个投机取巧版本的 Sample Topp 实现，我想它应该是一个非常快的版本。

        Sample Topp 操作要分成几个部分来完成：

            首先要执行一次排序，由于访存无法被合并，这次排序操作会很慢。

            而后要执行 softmax，这意味着三次访存(分别统计 global_max, global_sum, 以及执行 softmax 计算)

            然后你要对 global_sum 乘以一个 [0, 1] 之间的随机数，去执行 sampling 操作，这里我选择使用"接受拒绝采样"

        万幸的是，我们总是可以假设经过充分训练的模型是收敛的，其输出的词表概率应当是"十分尖锐"的——大部分词应该都没什么出现概率，采样过程可以忽略他们。

        因此，我们不执行完整的排序过程，我们采用 局部排序 + flash attention + 多路赢者树归并 的方式实现这个 kernel

        这种实现下的 Sample Topp 比单独执行一次排序还要快。

        这个 kernel 的性能不是稳定的，概率分布越不均衡它越快。但是如果概率分布是均匀的，这个 Kernel 的性能可能会差。
    */
    /* Radix Sort + Softmax + Sampling */

    // allocate shared mem
    constexpr int32_t WARP_SIZE = 32;
    constexpr int32_t WPT = TPB / WARP_SIZE; // warp per thread block.
    typedef cub::BlockRadixSort<fp32_t, TPB, VPT, int32_t> BlockRadixSort;
    __shared__ typename BlockRadixSort::TempStorage temp_storage;

    __shared__ fp32_t tile_softmax_m[TPB];
    __shared__ fp32_t tile_softmax_l[TPB];
    __shared__ fp32_t reducing_memory[WPT * 2];
    fp32_t sorting_keys[VPT]; int32_t sorting_values[VPT];

    tile_softmax_m[threadIdx.x] = 0.0f;
    tile_softmax_l[threadIdx.x] = -FLT_MAX;
    __syncthreads();

    // Stage 1. Block Internal Sort & Reducing Max
    const int64_t batch_id            = blockIdx.x;
    const int64_t batch_offset        = batch_id * batch_stride;
    const int64_t padded_batch_offset = batch_id * pad_vocab<fp32_t, TPB>(vocab_size);
    const fp32_t  temperature         = temperatures ? max(abs(temperatures[batch_id]) + 1e-7, 0.01): 1.0f; // temperature 最低 0.01

    for(int32_t block_base_idx = 0; block_base_idx < vocab_size; block_base_idx += TPB * VPT){
        fp32_t local_max = -FLT_MAX, local_sum = 0.0;
        // process multiple elements at once
        const int32_t thread_local_idx = block_base_idx + threadIdx.x;

        # pragma unroll
        for(int32_t internal_loop_idx = 0; internal_loop_idx < VPT; internal_loop_idx++) {
            const int32_t vocab_idx = thread_local_idx + internal_loop_idx * TPB;
            const int64_t load_idx  = batch_offset + vocab_idx;

            if(vocab_idx < vocab_size) {
                const fp32_t value = logits[load_idx] / temperature;
                sorting_keys[internal_loop_idx]   = - value;      // 倒序排序
                sorting_values[internal_loop_idx] = vocab_idx;
                local_max = local_max > value ? local_max : value;
            } else {
                sorting_keys[internal_loop_idx]   = FLT_MAX;
                sorting_values[internal_loop_idx] = -1;
            }
        }

        // Block Reduce max
        __syncthreads();
        local_max = sample_block_reduce_max<WPT>(local_max, reducing_memory);

        // calling Cub::RaidxSort, 这个东西耗时最多，我也不知道咋优化他
        __syncthreads();
        BlockRadixSort(temp_storage).Sort(sorting_keys, sorting_values);

        // 将每一个 block 中的数据写回内存, 此处数据的写回必须合并访存，否则性能很烂
        const int64_t vocab_idx = block_base_idx + threadIdx.x * VPT;
        const int64_t write_idx = vocab_idx + padded_batch_offset;

        # pragma unroll
        for(int32_t internal_loop_idx = 0; internal_loop_idx < VPT; internal_loop_idx++){
            sorting_keys[internal_loop_idx] = exp(- sorting_keys[internal_loop_idx] - local_max);
            local_sum += sorting_keys[internal_loop_idx]; // softmax in this tile
        }
        copy<VPT * sizeof(fp32_t)>(sorting_keys,    &sorted_value[write_idx]);
        copy<VPT * sizeof(int32_t)>(sorting_values, &sorted_order[write_idx]);

        local_sum = sample_block_reduce_sum<WPT>(local_sum, reducing_memory);

        // write block softmax result to shared memory
        // following logic is inspired by flash-attention
        if (threadIdx.x == 0){
            const int32_t tile_idx = block_base_idx / (TPB * VPT);
            tile_softmax_l[tile_idx] = local_max;
            tile_softmax_m[tile_idx] = local_sum;
        }
    }

    fp32_t global_sum = 0.0f;
    fp32_t global_max = -FLT_MAX;
    // flash-attention reduce max, reduce sum
    global_max = sample_block_reduce_max<WPT>(tile_softmax_l[threadIdx.x], reducing_memory);
    __syncthreads();

    global_sum = sample_block_reduce_sum<WPT>(
        tile_softmax_m[threadIdx.x] * exp(tile_softmax_l[threadIdx.x] - global_max),
        reducing_memory);

    fp32_t top_p_selection_rnd = global_sum * (top_p == nullptr ? top_p_val : top_p[batch_id]) * (rnd == nullptr ? rnd_val : rnd[batch_id]);

    // multi way merge-sort & sample top_p
    // 后面这里的采样过程可以进一步优化，但是好像正常来讲不会采样非常多次
    // 需要注意，这是接受拒绝采样，且进行随机访存，如果迟迟无法结束采样，这个 kernel 性能会很差
    // 届时此处的优化将至关重要
    int32_t *selection_slot = reinterpret_cast<int32_t*>(tile_softmax_m);
    selection_slot[threadIdx.x] = 0;
    __syncthreads();

    int64_t base_selection_idx = threadIdx.x * VPT * TPB;
    const fp32_t scale_factor = exp(tile_softmax_l[threadIdx.x] - global_max);

    auto _local_storage = sorting_keys; // 复用一下
    CachedVocabStorage<TPB, VPT> cached_store(
        sorted_value, _local_storage, padded_batch_offset + base_selection_idx);

    for (int32_t selected = 0; selected < vocab_size; selected++) {
        fp32_t selecting_value = -FLT_MAX;
        int32_t select_thread_idx = 0;
        int32_t local_selection = selection_slot[threadIdx.x];

        if (base_selection_idx + local_selection < vocab_size && local_selection < VPT * TPB){
            selecting_value = cached_store.Pop(local_selection);
            selecting_value = selecting_value * scale_factor; // flash softmax
        }

        // block mergesort
        SortingPair p = sample_block_reduce_max_with_index<WPT>(selecting_value, threadIdx.x, reducing_memory);
        select_thread_idx = p.index;
        selecting_value = p.value;
        top_p_selection_rnd -= selecting_value;

        if (threadIdx.x == 0) {
            if (top_p_selection_rnd <= 0.0f) {
                // sampling success, write to output.
                output[batch_id] = sorted_order[
                    padded_batch_offset +
                    select_thread_idx * VPT * TPB +
                    selection_slot[select_thread_idx]
                ];
            }
            selection_slot[select_thread_idx] ++;
        }
        __syncthreads();
        if (top_p_selection_rnd <= 0.0f)
            return;
    }
}

template<int32_t TPB>
__global__
void sample_argmax_kernel(
    const fp32_t* __restrict__ logits, // [batch, batch_stride]
    const int32_t vocab_size,
    const int32_t batch_stride,
    int32_t* output)                   // [batch, 1]
{
    const int64_t batch_id = blockIdx.x;
    int32_t selection_idx = 0;
    fp32_t selecting_value = -FLT_MAX;

    for(int32_t idx = threadIdx.x; idx < vocab_size; idx += TPB) {
        // fp32_t loading = __half2float(logits[batch_id * vocab_size + idx]);
        fp32_t loading = logits[batch_id * batch_stride + idx];
        if (loading > selecting_value) {
            selecting_value = loading;
            selection_idx   = idx;
        }
    }

    // initilize shared memory
    constexpr int32_t WARP_SIZE = 32;
    constexpr int32_t WPT = TPB / WARP_SIZE;
    __shared__ int32_t buffer[TPB];
    __shared__ fp32_t red_smem[WPT * 2];
    buffer[threadIdx.x] = selection_idx;
    __syncthreads();

    SortingPair p = sample_block_reduce_max_with_index<WPT>(selecting_value, threadIdx.x, red_smem);
    if (threadIdx.x == 0)
        output[batch_id] = buffer[p.index];
}

int64_t flash_sample_top_p_get_workspace_size(
    int32_t batch,
    int32_t vocab_size)
{
    return int64_t(batch) * pad_vocab<fp32_t, 256>(vocab_size) * (sizeof(float) + sizeof(int32_t));
}

ppl::common::RetCode flash_sample_topp(
    cudaStream_t stream,
    const float* logits, // (batch, batch_stride)
    const float* temperatures, // (batch)
    const float* top_p, // (batch)
    const float* rnd, // (batch)
    const int32_t num_batches,
    const int32_t vocab_size,
    const int32_t batch_stride,
    const float top_p_val,
    const float rnd_val,
    void *workspace,
    int32_t* output) // (batch)
{
    /* Flash Sample Topp

    The FlashSampleTopp function is a high-performance Topp sampling implementation that integrates the functionalities of
        Sorting, Softmax, and Topp sampling.

     sorted_value  {num_batches, pad_vocab<fp32_t, 256>(vocab_size)}

     sorted_index = {num_batches, pad_vocab<fp32_t, 256>(vocab_size)}

    */

    float* sorted_value = (float*)workspace; // (batch, padded_vocab_size)
    int32_t* sorted_index = (int32_t*)sorted_value + pad_vocab<fp32_t, 256>(vocab_size) * sizeof(float); // (batch, padded_vocab_size)

    if (vocab_size <= 32768) {
        flash_sample_top_p_kernel<256, 4, 32>
        <<<num_batches, 256, 0, stream>>>(
            logits, temperatures, top_p,
            rnd, vocab_size, batch_stride,
            top_p_val, rnd_val,
            sorted_value,
            sorted_index,
            output
        );
    } else if (vocab_size <= 262144) {
        flash_sample_top_p_kernel<256, 4, 256>
        <<<num_batches, 256, 0, stream>>>(
            logits, temperatures, top_p,
            rnd, vocab_size, batch_stride,
            top_p_val, rnd_val,
            sorted_value,
            sorted_index,
            output
        );
    } else {
        LOG(ERROR) << "only supporte vocab_size <= 262144, vocab_size = " << vocab_size;
        return ppl::common::RC_UNSUPPORTED;
    }

    return ppl::common::RC_SUCCESS;
}

int64_t sample_topk_topp_get_workspace_size(int32_t batch, int32_t vocab_size, int32_t top_k_val) {
    int64_t buffer_size = 0;
    if (top_k_val != 1 && top_k_val <= 12) {
        buffer_size = int64_t(batch) * pad_vocab<fp32_t, 256>(vocab_size) * sizeof(float);
    }
    return buffer_size;
}

ppl::common::RetCode sample_topk_topp(
    cudaStream_t stream,
    const float* logits, // (batch, batch_stride)
    const float* temperatures, // (batch)
    const float* top_p, // (batch)
    const float* rnd, // (batch)
    const int32_t num_batches,
    const int32_t vocab_size,
    const int32_t batch_stride,
    const int32_t top_k_val,
    const float top_p_val,
    const float rnd_val,
    void *workspace,
    int32_t* output) // (batch)
{
    constexpr int32_t TPB = 256;
    constexpr int32_t VPT = 4;
    constexpr int32_t LPT = 32;     // sort length per thread

    float* padded_logits = (float*)workspace; // (batch, padded_vocab_szie)

    if (top_k_val == 1 || (!top_p && top_p_val == 0.0f)) {
        sample_argmax_kernel<TPB>
        <<<num_batches, TPB, 0, stream>>>(
            logits, vocab_size, batch_stride, output
        );
    } else if (top_k_val <= 12) {
        sample_topk_topp_default_kernel<TPB, VPT>
        <<<num_batches, TPB, top_k_val * 8, stream>>>(
            logits, temperatures, top_p, rnd,
            vocab_size, batch_stride,
            top_k_val, top_p_val, rnd_val,
            padded_logits, output
        );
    } else if (top_k_val <= 256) {
        sample_topk_topp_radix_select_kernel<TPB, LPT, 1>
        <<<num_batches, TPB, 0, stream>>>(
            logits, temperatures, top_p, rnd,
            vocab_size, batch_stride,
            top_k_val, top_p_val, rnd_val,
            output
        );
    } else if (top_k_val <= 512) {
        sample_topk_topp_radix_select_kernel<TPB, LPT, 2>
        <<<num_batches, TPB, 0, stream>>>(
            logits, temperatures, top_p, rnd,
            vocab_size, batch_stride,
            top_k_val, top_p_val, rnd_val,
            output
        );
    } else if (top_k_val <= 768) {
        sample_topk_topp_radix_select_kernel<TPB, LPT, 3>
        <<<num_batches, TPB, 0, stream>>>(
            logits, temperatures, top_p, rnd,
            vocab_size, batch_stride,
            top_k_val, top_p_val, rnd_val,
            output
        );
    } else if (top_k_val <= 1024) {
        sample_topk_topp_radix_select_kernel<TPB, LPT, 4>
        <<<num_batches, TPB, 0, stream>>>(
            logits, temperatures, top_p, rnd,
            vocab_size, batch_stride,
            top_k_val, top_p_val, rnd_val,
            output
        );
    } else {
        LOG(ERROR) << "only supporte top_k <= 1024, top_k = " << top_k_val;
        return ppl::common::RC_UNSUPPORTED;
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode sample_argmax(
    cudaStream_t stream,
    const float* logits, // (batch, batch_stride)
    const int32_t num_batches,
    const int32_t vocab_size,
    const int32_t batch_stride,
    int32_t* output) // (batch)
{
    sample_argmax_kernel<256><<<num_batches, 256, 0, stream>>>(logits, vocab_size, batch_stride, output);

    return ppl::common::RC_SUCCESS;
}

}}}}}
