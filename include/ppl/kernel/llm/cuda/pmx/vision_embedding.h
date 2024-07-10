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

#ifndef __PPL_KERNEL_LLM_CUDA_PMX_VISION_EMBEDDING_H__
#define __PPL_KERNEL_LLM_CUDA_PMX_VISION_EMBEDDING_H__

#ifdef PPLNN_CUDA_ENABLE_CUDNN

#include <cudnn.h>

#include "ppl/kernel/llm/cuda/common/general_include.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

struct vision_embedding_config {
    int32_t hidden_dim;
    int32_t batch_size;
    int32_t image_channel;
    int32_t image_size;
    int32_t patch_size;
    int32_t grid;
    int32_t bias_term;
    size_t total_buffer_size;
    size_t conv_workspace_size;
    size_t patch_embeds_size;
    void* buffer_addr;
    cudnnStatus_t cudnn_status;
    cudnnHandle_t cudnn_handle;
    cudnnTensorDescriptor_t image_desc;
    cudnnTensorDescriptor_t patch_nchw_desc;
    cudnnTensorDescriptor_t patch_nhwc_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnTensorDescriptor_t bias_desc;
    cudnnConvolutionDescriptor_t conv_desc;
};

ppl::common::RetCode vision_embedding_preprocessing(
    vision_embedding_config& config);

ppl::common::RetCode vision_embedding(
    const cudaStream_t stream,
    vision_embedding_config& config,
    const void* images,
    const void* patch_weight,
    const void* patch_bias,
    const void* class_weight,
    const void* position_weight,
    void* output);

ppl::common::RetCode vision_embedding_postprocessing(
    vision_embedding_config& config);

}}}}}

#endif

#endif
