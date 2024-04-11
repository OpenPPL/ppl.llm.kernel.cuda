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

#include <cudnn.h>

#include "ppl/kernel/llm/cuda/common/general_include.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

ppl::common::RetCode vision_embedding(
    const cudaStream_t stream,
    cudnnHandle_t cudnn_handle,
    cudnnTensorDescriptor_t image_desc,
    const void* images,
    cudnnFilterDescriptor_t filter_desc,
    const void* patch_emb_weight,
    cudnnConvolutionDescriptor_t conv_desc,
    void* workspace,
    size_t workspace_size,
    cudnnTensorDescriptor_t patch_desc0,
    void* patch_embeds0,
    cudnnTensorDescriptor_t patch_desc1,
    void* patch_embeds1,
    const void* cls_emb_weight,
    const void* pos_emb_weight,
    int32_t grid,
    int32_t batch_size,
    const int32_t hidden_dim,
    void* output);

}}}}}

#endif