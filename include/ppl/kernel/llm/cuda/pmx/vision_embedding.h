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

#include "ppl/kernel/llm/cuda/common/general_include.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

ppl::common::RetCode vision_embedding(
    const cudaStream_t stream,
    const ppl::common::TensorShape* image_shape, //[batch_size, channels, image_size, image_size]
    const void* images,
    const void* cls_emb_weight,    // [hidden_dim]
    const ppl::common::TensorShape* patch_emb_shape, // [hidden_dim, channels, patch_size, patch_size]
    const void* patch_emb_weight,
    const void* pos_emb_weight,  // [num_positions * hidden_dim]
    const int32_t hidden_dim,
    const int32_t image_size,
    const int32_t patch_size,
    const ppl::common::TensorShape* output_shape,  // [batch_size, (image_size / patch_size) * (image_size / patch_size) + 1, hidden_dim]
    void* output);

}}}}}

#endif