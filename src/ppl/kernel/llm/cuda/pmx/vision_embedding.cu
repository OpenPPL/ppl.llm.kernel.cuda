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

#ifdef PPLNN_CUDA_ENABLE_CUDNN

#include "ppl/kernel/llm/cuda/pmx/vision_embedding.h"
#include "ppl/common/log.h"

#include <cuda_fp16.h>

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

__global__ void input_embedding_small_dims_kernel_fp16(
    const half* patch_embeds,
    const half* class_weight,
    const half* position_weight,
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
        patch_value = class_weight[thread_id];
    }
    else {                  // patch
        index = (blockIdx.y * (gridDim.x - 1) + (blockIdx.x - 1)) * hidden_dim + thread_id;
        patch_value = patch_embeds[index];
    }

    int32_t position_index = blockIdx.x * hidden_dim + thread_id;
    half position_value = position_weight[position_index];
    half value = position_value + patch_value;

    index = (blockIdx.y * gridDim.x + blockIdx.x) * hidden_dim + thread_id;
    output[index] = value;
}

__global__ void input_embedding_kernel_fp16(
    const half* patch_embeds,
    const half* class_weight,
    const half* position_weight,
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
        patch_value = class_weight[thread_id];
    }
    else {                  // patch
        index = (blockIdx.z * (blockDim.y - 1) + (blockIdx.y - 1)) * hidden_dim + thread_id;
        patch_value = patch_embeds[index];
    }

    int32_t position_index = blockIdx.y * hidden_dim + thread_id;
    half position_value = position_weight[position_index];
    half value = position_value + patch_value;

    index = (blockIdx.z * gridDim.y + blockIdx.y) * hidden_dim + thread_id;
    output[index] = value;
}

void combine_embedding(
    const half* patch_embeds,     // [batch_size, grid * grid, hidden_dim]
    const half* class_weight,     // [hidden_dim]
    const half* position_weight,  // [num_positions, hidden_dim]
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
        input_embedding_small_dims_kernel_fp16<<<grid, block>>>(patch_embeds, class_weight,
            position_weight, hidden_dim, output);
    }
    else {
        block.x = 1024;
        grid.x = (hidden_dim + 1023) >> 10;
        grid.y = num_positions;
        grid.z = batch_size;
        input_embedding_kernel_fp16<<<grid, block>>>(patch_embeds, class_weight,
            position_weight, hidden_dim, output);
    }
}

ppl::common::RetCode vision_embedding_preprocessing(
    vision_embedding_config& config)
{
    config.grid = config.image_size / config.patch_size;
    config.cudnn_status = cudnnCreateTensorDescriptor(&config.image_desc);
    if (config.cudnn_status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to create the tensor descriptor of image with error: " << cudnnGetErrorString(config.cudnn_status);
        return ppl::common::RC_OTHER_ERROR;
    }
    config.cudnn_status = cudnnCreateTensorDescriptor(&config.patch_nchw_desc);
    if (config.cudnn_status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to create the tensor descriptor of the convolution output with error: " << cudnnGetErrorString(config.cudnn_status);
        return ppl::common::RC_OTHER_ERROR;
    }
    config.cudnn_status = cudnnCreateTensorDescriptor(&config.patch_nhwc_desc);
    if (config.cudnn_status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to create the tensor descriptor of the transposed convolution output with error: " << cudnnGetErrorString(config.cudnn_status);
        return ppl::common::RC_OTHER_ERROR;
    }
    config.cudnn_status = cudnnSetTensor4dDescriptor(config.image_desc, CUDNN_TENSOR_NCHW,
             CUDNN_DATA_HALF, config.batch_size, config.image_channel, config.image_size, config.image_size);
    if (config.cudnn_status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to set the tensor descriptor of image with error: " << cudnnGetErrorString(config.cudnn_status);
        return ppl::common::RC_OTHER_ERROR;
    }
    config.cudnn_status = cudnnSetTensor4dDescriptor(config.patch_nchw_desc, CUDNN_TENSOR_NCHW,
             CUDNN_DATA_HALF, config.batch_size, config.hidden_dim, config.grid, config.grid);
    if (config.cudnn_status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to set the tensor descriptor of the convolution output with error: " << cudnnGetErrorString(config.cudnn_status);
        return ppl::common::RC_OTHER_ERROR;
    }
    config.cudnn_status = cudnnSetTensor4dDescriptor(config.patch_nhwc_desc, CUDNN_TENSOR_NHWC,
             CUDNN_DATA_HALF, config.batch_size, config.hidden_dim, config.grid, config.grid);
    if (config.cudnn_status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to set the tensor descriptor of the transposed convolution output with error: " << cudnnGetErrorString(config.cudnn_status);
        return ppl::common::RC_OTHER_ERROR;
    }

    config.cudnn_status = cudnnCreateFilterDescriptor(&config.filter_desc);
    if (config.cudnn_status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to create the filter descriptor with error: " << cudnnGetErrorString(config.cudnn_status);
        return ppl::common::RC_OTHER_ERROR;
    }
    config.cudnn_status = cudnnSetFilter4dDescriptor(config.filter_desc, CUDNN_DATA_HALF,
             CUDNN_TENSOR_NCHW, config.hidden_dim, config.image_channel, config.patch_size, config.patch_size);
    if (config.cudnn_status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to set the filter descriptor with error: " << cudnnGetErrorString(config.cudnn_status);
        return ppl::common::RC_OTHER_ERROR;
    }

    if (config.bias_term) {
        config.cudnn_status = cudnnCreateTensorDescriptor(&config.bias_desc);
        if (config.cudnn_status != CUDNN_STATUS_SUCCESS) {
            LOG(ERROR) << "failed to create the tensor descriptor of the convolution bias with error: " << cudnnGetErrorString(config.cudnn_status);
            return ppl::common::RC_OTHER_ERROR;
        }
        config.cudnn_status = cudnnSetTensor4dDescriptor(config.bias_desc, CUDNN_TENSOR_NCHW,
                CUDNN_DATA_HALF, 1, config.hidden_dim, 1, 1);
        if (config.cudnn_status != CUDNN_STATUS_SUCCESS) {
            LOG(ERROR) << "failed to set the tensor descriptor of the convolution bias with error: " << cudnnGetErrorString(config.cudnn_status);
            return ppl::common::RC_OTHER_ERROR;
        }
        config.cudnn_status = cudnnCreateActivationDescriptor(&config.act_desc);
        if (config.cudnn_status != CUDNN_STATUS_SUCCESS) {
            LOG(ERROR) << "failed to create the activation descriptor of the convolution with error: " << cudnnGetErrorString(config.cudnn_status);
            return ppl::common::RC_OTHER_ERROR;
        }
        config.cudnn_status = cudnnSetActivationDescriptor(config.act_desc, CUDNN_ACTIVATION_IDENTITY,
                CUDNN_NOT_PROPAGATE_NAN, 0.f);
        if (config.cudnn_status != CUDNN_STATUS_SUCCESS) {
            LOG(ERROR) << "failed to set the activation descriptor of the convolution with error: " << cudnnGetErrorString(config.cudnn_status);
            return ppl::common::RC_OTHER_ERROR;
        }
    }

    config.cudnn_status = cudnnCreateConvolutionDescriptor(&config.conv_desc);
    if (config.cudnn_status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to create the convolution descriptor with error: " << cudnnGetErrorString(config.cudnn_status);
        return ppl::common::RC_OTHER_ERROR;
    }
    config.cudnn_status = cudnnSetConvolution2dDescriptor(config.conv_desc, 0, 0, config.patch_size, config.patch_size, 1, 1,
             CUDNN_CONVOLUTION, CUDNN_DATA_HALF);
    if (config.cudnn_status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to set the convolution descriptor with error: " << cudnnGetErrorString(config.cudnn_status);
        return ppl::common::RC_OTHER_ERROR;
    }
    config.cudnn_status = cudnnSetConvolutionMathType(config.conv_desc, CUDNN_TENSOR_OP_MATH);
    if (config.cudnn_status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to set the convolution math type with error: " << cudnnGetErrorString(config.cudnn_status);
        return ppl::common::RC_OTHER_ERROR;
    }

    size_t workspace_size;
    config.cudnn_status = cudnnGetConvolutionForwardWorkspaceSize(config.cudnn_handle, config.image_desc,
                          config.filter_desc, config.conv_desc, config.patch_nchw_desc,
                          CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, &workspace_size);
    if (config.cudnn_status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to get the workspace size of convolution with error: " << cudnnGetErrorString(config.cudnn_status);
        return ppl::common::RC_OTHER_ERROR;
    }
    config.patch_embeds_size = config.batch_size * config.hidden_dim * config.grid * config.grid * sizeof(half);
    config.patch_embeds_size = ((config.patch_embeds_size + 127) >> 7) << 7;
    config.conv_workspace_size = workspace_size;
    config.total_buffer_size = config.patch_embeds_size * 2 + config.conv_workspace_size;

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode vision_embedding(
    const cudaStream_t stream,
    vision_embedding_config& config,
    const void* pixel_values,    // [batch_size, image_channel, image_size, image_size]
    const void* class_weight,    // [hidden_dim]
    const void* patch_weight,    // [hidden_dim, channels, patch_size, patch_size]
    const void* position_weight, // [num_positions, hidden_dim]
    const void* patch_bias,      // [hidden_dim]
    void* output_embeddings)     // [batch_size, grid*grid + 1, hidden_dim]
{
    void* patch_embeds_nchw = config.buffer_addr;
    void* patch_embeds_nhwc = patch_embeds_nchw + config.patch_embeds_size;
    void* conv_workspace = patch_embeds_nhwc + config.patch_embeds_size;

    float alpha = 1.f;
    float beta = 1.f;
    if (config.bias_term) {
        float alpha1 = 1.f;
        float alpha2 = 0.f;
        config.cudnn_status = cudnnConvolutionBiasActivationForward(config.cudnn_handle, &alpha1, config.image_desc,
                            pixel_values, config.filter_desc, patch_weight, config.conv_desc,
                            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, conv_workspace,
                            config.conv_workspace_size, &alpha2, config.patch_nchw_desc, patch_embeds_nchw, config.bias_desc, patch_bias, config.act_desc, config.patch_nchw_desc, patch_embeds_nchw);
        if (config.cudnn_status != CUDNN_STATUS_SUCCESS) {
            LOG(ERROR) << "failed to execute convolution with bias with error: " << cudnnGetErrorString(config.cudnn_status);
            return ppl::common::RC_OTHER_ERROR;
        }

    } else {
        config.cudnn_status = cudnnConvolutionForward(config.cudnn_handle, &alpha, config.image_desc,
                            pixel_values, config.filter_desc, patch_weight, config.conv_desc,
                            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, conv_workspace,
                            config.conv_workspace_size, &beta, config.patch_nchw_desc, patch_embeds_nchw);
        if (config.cudnn_status != CUDNN_STATUS_SUCCESS) {
            LOG(ERROR) << "failed to execute convolution with error: " << cudnnGetErrorString(config.cudnn_status);
            return ppl::common::RC_OTHER_ERROR;
        }
    }
    config.cudnn_status = cudnnTransformTensor(config.cudnn_handle, &alpha, config.patch_nchw_desc,
                        patch_embeds_nchw, &beta, config.patch_nhwc_desc, patch_embeds_nhwc);
    if (config.cudnn_status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to transpose the convolution output with error: " << cudnnGetErrorString(config.cudnn_status);
        return ppl::common::RC_OTHER_ERROR;
    }

    int32_t num_positions = config.grid * config.grid + 1;
    combine_embedding((const half*)patch_embeds_nhwc, (const half*)class_weight,
                      (const half*)position_weight, config.batch_size,
                      num_positions, config.hidden_dim, (half*)output_embeddings);

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode vision_embedding_postprocessing(
    vision_embedding_config& config)
{
    config.cudnn_status = cudnnDestroyTensorDescriptor(config.image_desc);
    if (config.cudnn_status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to destroy the tensor descriptor of the image with error: " << cudnnGetErrorString(config.cudnn_status);
        return ppl::common::RC_OTHER_ERROR;
    }
    config.cudnn_status = cudnnDestroyTensorDescriptor(config.patch_nchw_desc);
    if (config.cudnn_status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to destroy the tensor descriptor of the convolution output with error: " << cudnnGetErrorString(config.cudnn_status);
        return ppl::common::RC_OTHER_ERROR;
    }
    config.cudnn_status = cudnnDestroyTensorDescriptor(config.patch_nhwc_desc);
    if (config.cudnn_status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to destroy the tensor descriptor of the transposed convolution output with error: " << cudnnGetErrorString(config.cudnn_status);
        return ppl::common::RC_OTHER_ERROR;
    }
    if (config.bias_term) {
        config.cudnn_status = cudnnDestroyTensorDescriptor(config.bias_desc);
        if (config.cudnn_status != CUDNN_STATUS_SUCCESS) {
            LOG(ERROR) << "failed to destroy the tensor descriptor of the convolution bias with error: " << cudnnGetErrorString(config.cudnn_status);
            return ppl::common::RC_OTHER_ERROR;
        }
        config.cudnn_status = cudnnDestroyActivationDescriptor(config.act_desc);
        if (config.cudnn_status != CUDNN_STATUS_SUCCESS) {
            LOG(ERROR) << "failed to destroy the activation descriptor of the convolution with error: " << cudnnGetErrorString(config.cudnn_status);
            return ppl::common::RC_OTHER_ERROR;
        }

    }
    config.cudnn_status = cudnnDestroyFilterDescriptor(config.filter_desc);
    if (config.cudnn_status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to destroy the filter descriptor with error: " << cudnnGetErrorString(config.cudnn_status);
        return ppl::common::RC_OTHER_ERROR;
    }
    config.cudnn_status = cudnnDestroyConvolutionDescriptor(config.conv_desc);
    if (config.cudnn_status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to destroy the convolution descriptor with error: " << cudnnGetErrorString(config.cudnn_status);
        return ppl::common::RC_OTHER_ERROR;
    }

    return ppl::common::RC_SUCCESS;
}

}}}}}

#endif
