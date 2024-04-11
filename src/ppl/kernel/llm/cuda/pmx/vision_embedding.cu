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

#include "ppl/kernel/llm/cuda/pmx/vision_embedding.h"
#include "ppl/common/log.h"

#include <cuda_fp16.h>

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

__global__ void input_embedding_kernel0_fp16(
    const half* patch_embeds,
    const half* cls_emb_weight,
    const half* pos_emb_weight,
    int32_t hidden_dim,
    half* output)
{
    int32_t thread_id = threadIdx.x;
    if (thread_id >= hidden_dim) {
        return;
    }

    half patch_value;
    int32_t index;
    if (blockIdx.x == 0) {  // cls
        patch_value = cls_emb_weight[thread_id];
    }
    else {                  // patch
        index = (blockIdx.y * (gridDim.x - 1) + (blockIdx.x - 1)) * hidden_dim + thread_id;
        patch_value = patch_embeds[index];
    }

    int32_t position_index = blockIdx.x * hidden_dim + thread_id;
    half position_value = pos_emb_weight[position_index];
    half value = position_value + patch_value;

    index = (blockIdx.y * gridDim.x + blockIdx.x) * hidden_dim + thread_id;
    output[index] = value;
}

__global__ void input_embedding_kernel1_fp16(
    const half* patch_embeds,
    const half* cls_emb_weight,
    const half* pos_emb_weight,
    int32_t hidden_dim,
    half* output)
{
    int32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= hidden_dim) {
        return;
    }

    half patch_value;
    int32_t index;
    if (blockIdx.y == 0) {  // cls
        patch_value = cls_emb_weight[thread_id];
    }
    else {                  // patch
        index = (blockIdx.z * (blockDim.y - 1) + (blockIdx.y - 1)) * hidden_dim + thread_id;
        patch_value = patch_embeds[index];
    }

    int32_t position_index = blockIdx.y * hidden_dim + thread_id;
    half position_value = pos_emb_weight[position_index];
    half value = position_value + patch_value;

    index = (blockIdx.z * gridDim.y + blockIdx.y) * hidden_dim + thread_id;
    output[index] = value;
}

void combine_embedding(
    const half* patch_embeds,   // [batch_size, grid * grid, hidden_dim]
    const half* cls_emb_weight, // [hidden_dim]
    const half* pos_emb_weight, // [num_positions, hidden_dim]
    int32_t batch_size,
    int32_t num_positions,
    int32_t hidden_dim,
    half* output)
{
    dim3 grid, block;
    if (hidden_dim <= 1024) {
        grid.x = num_positions;
        grid.y = batch_size;
        if (hidden_dim <= 64) {
            block.x = 64;
        }
        else if (hidden_dim <= 128) {
            block.x = 128;
        }
        else if (hidden_dim <= 256) {
            block.x = 256;
        }
        else if (hidden_dim <= 512) {
            block.x = 512;
        }
        else {
            block.x = 1024;
        }
        input_embedding_kernel0_fp16<<<grid, block>>>(patch_embeds, cls_emb_weight,
            pos_emb_weight, hidden_dim, output);
    }
    else {
        block.x = 1024;
        grid.x = (hidden_dim + 1023) >> 10;
        grid.y = num_positions;
        grid.z = batch_size;
        input_embedding_kernel1_fp16<<<grid, block>>>(patch_embeds, cls_emb_weight,
            pos_emb_weight, hidden_dim, output);
    }
}

ppl::common::RetCode vision_embedding(
    const cudaStream_t stream,
    cudnnHandle_t cudnn_handle,
    cudnnTensorDescriptor_t image_desc,
    const void* images,
    cudnnFilterDescriptor_t filter_desc,
    const void* patch_emb_weight,  // weight of convolution filter
    cudnnConvolutionDescriptor_t conv_desc,
    void* workspace,
    size_t workspace_size,
    cudnnTensorDescriptor_t patch_desc0,
    void* patch_embeds0,  // output of cudnnConvolutionForward()
    cudnnTensorDescriptor_t patch_desc1,
    void* patch_embeds1,  // output of cudnnTransformTensor()
    const void* cls_emb_weight,  // [hidden_dim]
    const void* pos_emb_weight,  // [num_positions * hidden_dim]
    int32_t grid,
    int32_t batch_size,
    const int32_t hidden_dim,
    void* output)
{
    cudnnStatus_t status;
    float alpha = 1.f;
    float beta = 1.f;
    status = cudnnConvolutionForward(cudnn_handle, &alpha, image_desc,
                 images, filter_desc, patch_emb_weight, conv_desc,
                 CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, workspace,
                 workspace_size, &beta, patch_desc0, patch_embeds0);
    if (status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to execute convolution with error: " << cudnnGetErrorString(status);
        return ppl::common::RC_OTHER_ERROR;
    }

    status = cudnnTransformTensor(cudnn_handle, &alpha, patch_desc0,
                 patch_embeds0, &beta, patch_desc1, patch_embeds1);
    if (status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to transpose convolution output with error: " << cudnnGetErrorString(status);
        return ppl::common::RC_OTHER_ERROR;
    }

    int32_t num_positions = grid * grid + 1;
    combine_embedding((const half*)patch_embeds1, (const half*)cls_emb_weight,
                      (const half*)pos_emb_weight, batch_size,
                      num_positions, hidden_dim, (half*)output);

    return ppl::common::RC_SUCCESS;
}

}}}}}
