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
#include <cudnn.h>

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
    const ppl::common::TensorShape* image_shape,  // [n, c, h, w]
    const void* images,
    const void* cls_emb_weight,  // [hidden_dim]
    const ppl::common::TensorShape* patch_emb_shape,  // [hidden_dim, c, patch_size, patch_size]
    const void* patch_emb_weight,
    const void* pos_emb_weight,  // [num_positions * hidden_dim]
    const int32_t hidden_dim,
    const int32_t image_size,
    const int32_t patch_size,
    const ppl::common::TensorShape* output_shape, // [batch_size, (image_size / patch_size) * (image_size / patch_size) + 1, hidden_dim]
    void* output)
{
    if (image_shape->GetDataType() != ppl::common::DATATYPE_FLOAT16) {
        LOG(ERROR) << "vision_embedding image only support fp16, but got ["<< image_shape->GetDataType() << "]";
        return ppl::common::RC_UNSUPPORTED;
    }
    if (patch_emb_shape->GetDataType() != ppl::common::DATATYPE_FLOAT16) {
        LOG(ERROR) << "vision_embedding kernel only support fp16, but got ["<< patch_emb_shape->GetDataType() << "]";
        return ppl::common::RC_UNSUPPORTED;
    }
    if (output_shape->GetDataType() != ppl::common::DATATYPE_FLOAT16) {
        LOG(ERROR) << "vision_embedding output only support fp16, but got ["<< output_shape->GetDataType() << "]";
        return ppl::common::RC_UNSUPPORTED;
    }
    if (images == nullptr) {
        LOG(ERROR) << "invalid pointer of the image data.";
        return ppl::common::RC_INVALID_VALUE;
    }
    if (cls_emb_weight == nullptr) {
        LOG(ERROR) << "invalid pointer of the class embedding weight.";
        return ppl::common::RC_INVALID_VALUE;
    }
    if (patch_emb_weight == nullptr) {
        LOG(ERROR) << "invalid pointer of the filter weight.";
        return ppl::common::RC_INVALID_VALUE;
    }
    if (pos_emb_weight == nullptr) {
        LOG(ERROR) << "invalid pointer of the position embedding weight.";
        return ppl::common::RC_INVALID_VALUE;
    }
    if (output == nullptr) {
        LOG(ERROR) << "invalid pointer of the output.";
        return ppl::common::RC_INVALID_VALUE;
    }

    // image convolution
    cudnnStatus_t status;
    cudnnHandle_t cudnn_handle;
    status = cudnnCreate(&cudnn_handle);
    if (status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to create a cudnn handle with error: " << status;
        return ppl::common::RC_OTHER_ERROR;
    }

    const int32_t batch_size = image_shape->GetDim(0);
    const int32_t image_channel = image_shape->GetDim(1);
    const int32_t image_size1 = image_shape->GetDim(2);
    if (image_size != image_size1) {
        LOG(ERROR) << "invalid image size.";
        return ppl::common::RC_INVALID_VALUE;
    }
    const int32_t grid = image_size / patch_size;
    cudnnTensorDescriptor_t image_desc, patch_desc0, patch_desc1;
    status = cudnnCreateTensorDescriptor(&image_desc);
    if (status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to create the tensor descriptor of image with error: " << status;
        return ppl::common::RC_OTHER_ERROR;
    }
    status = cudnnCreateTensorDescriptor(&patch_desc0);
    if (status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to create the tensor descriptor of convolution output with error: " << status;
        return ppl::common::RC_OTHER_ERROR;
    }
    status = cudnnCreateTensorDescriptor(&patch_desc1);
    if (status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to create the tensor descriptor of transposed convolution output with error: " << status;
        return ppl::common::RC_OTHER_ERROR;
    }
    status = cudnnSetTensor4dDescriptor(image_desc, CUDNN_TENSOR_NCHW,
             CUDNN_DATA_HALF, batch_size, image_channel, image_size, image_size);
    if (status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to set the tensor descriptor of input with error: " << status;
        return ppl::common::RC_OTHER_ERROR;
    }
    status = cudnnSetTensor4dDescriptor(patch_desc0, CUDNN_TENSOR_NCHW,
             CUDNN_DATA_HALF, batch_size, hidden_dim, grid, grid);
    if (status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to set the tensor descriptor of convolution output with error: " << status;
        return ppl::common::RC_OTHER_ERROR;
    }
    status = cudnnSetTensor4dDescriptor(patch_desc1, CUDNN_TENSOR_NHWC,
             CUDNN_DATA_HALF, batch_size, hidden_dim, grid, grid);
    if (status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to set the tensor descriptor of transposed convolution output with error: " << status;
        return ppl::common::RC_OTHER_ERROR;
    }

    cudnnFilterDescriptor_t kernel_desc;
    status = cudnnCreateFilterDescriptor(&kernel_desc);
    if (status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to create the filter descriptor with error: " << status;
        return ppl::common::RC_OTHER_ERROR;
    }
    status = cudnnSetFilter4dDescriptor(kernel_desc, CUDNN_DATA_HALF,
             CUDNN_TENSOR_NCHW, hidden_dim, image_channel, patch_size, patch_size);
    if (status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to set the filter descriptor with error: " << status;
        return ppl::common::RC_OTHER_ERROR;
    }

    cudnnConvolutionDescriptor_t conv;
    status = cudnnCreateConvolutionDescriptor(&conv);
    if (status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to create the convolution descriptor with error: " << status;
        return ppl::common::RC_OTHER_ERROR;
    }
    status = cudnnSetConvolution2dDescriptor(conv, 0, 0, patch_size, patch_size, 1, 1,
             CUDNN_CONVOLUTION, CUDNN_DATA_HALF);
    if (status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to set the convolution descriptor with error: " << status;
        return ppl::common::RC_OTHER_ERROR;
    }
    status = cudnnSetConvolutionMathType(conv, CUDNN_TENSOR_OP_MATH);
    if (status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to set the convolution math type with error: " << status;
        return ppl::common::RC_OTHER_ERROR;
    }
/* // come here
    cudnnConvolutionFwdAlgoPerf_t algos[3];
    int returnedAlgoCount;
    status = cudnnGetConvolutionForwardAlgorithm_v7(cudnn_handle, image_desc,
             kernel_desc, conv, patch_desc0, 3, &returnedAlgoCount, algos);
    if (status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to get the convolution algorithm with error: " << status;
        return ppl::common::RC_OTHER_ERROR;
    } */

    size_t workspace_size;
    status = cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle, image_desc,
             kernel_desc, conv, patch_desc0,
             CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
             &workspace_size);
    if (status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to get the workspace size of convolution with error: " << status;
        return ppl::common::RC_OTHER_ERROR;
    }

    size_t size = batch_size * hidden_dim * grid * grid * sizeof(half);
    void* workspace;
    cudaMalloc(&workspace, workspace_size);
    void* patch_embeds0;
    void* patch_embeds1;
    cudaMalloc(&patch_embeds0, size);
    cudaMalloc(&patch_embeds1, size);

    float alpha = 1.f;
    float beta = 1.f;
    status = cudnnConvolutionForward(cudnn_handle, &alpha, image_desc,
                 images, kernel_desc, patch_emb_weight, conv,
                 CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, workspace,
                 workspace_size, &beta, patch_desc0, patch_embeds0);
    if (status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to execute convolution with error: " << status;
        return ppl::common::RC_OTHER_ERROR;
    }

    status = cudnnTransformTensor(cudnn_handle, &alpha, patch_desc0,
                 patch_embeds0, &beta, patch_desc1, patch_embeds1);
    if (status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to transpose convolution output with error: " << status;
        return ppl::common::RC_OTHER_ERROR;
    }

    int32_t num_positions = grid * grid + 1;
    combine_embedding((const half*)patch_embeds1, (const half*)cls_emb_weight,
                      (const half*)pos_emb_weight, batch_size,
                      num_positions, hidden_dim, (half*)output);

    cudaFree(workspace);
    cudaFree(patch_embeds0);
    cudaFree(patch_embeds1);
    status = cudnnDestroyTensorDescriptor(image_desc);
    if (status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to destroy the cudnn tensor descriptor image_desc with error: " << status;
        return ppl::common::RC_OTHER_ERROR;
    }
    status = cudnnDestroyTensorDescriptor(patch_desc0);
    if (status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to destroy the cudnn tensor descriptor patch_desc0 with error: " << status;
        return ppl::common::RC_OTHER_ERROR;
    }
    status = cudnnDestroyTensorDescriptor(patch_desc1);
    if (status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to destroy the cudnn tensor descriptor patch_desc1 with error: " << status;
        return ppl::common::RC_OTHER_ERROR;
    }
    status = cudnnDestroyFilterDescriptor(kernel_desc);
    if (status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to destroy the cudnn filter descriptor kernel_desc with error: " << status;
        return ppl::common::RC_OTHER_ERROR;
    }
    status = cudnnDestroyConvolutionDescriptor(conv);
    if (status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to destroy the cudnn convolution descriptor conv with error: " << status;
        return ppl::common::RC_OTHER_ERROR;
    }
    status = cudnnDestroy(cudnn_handle);
    if (status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to destroy the cudnn handle with error: " << status;
        return ppl::common::RC_OTHER_ERROR;
    }

    return ppl::common::RC_SUCCESS;
}

}}}}}
