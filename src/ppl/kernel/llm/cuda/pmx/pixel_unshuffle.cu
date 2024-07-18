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

#include "ppl/kernel/llm/cuda/pmx/pixel_unshuffle.h"
#include "cudakernel/common/common.cuh"
#include "ppl/common/log.h"

#include "type.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

struct pixel_unshuffle_fp16_kernel_param {
    fp16_t* input;              // [N, H*r, W*r, C]
    int64_t scale_factor;
    int64_t n_stride;
    int64_t W;
    int64_t C;
    fp16_t* output;             // [N, H, W, C*r*r]
};

template <int32_t VPT, int32_t TPB>
__global__
void pixel_unshuffle_fp16_kernel(
    pixel_unshuffle_fp16_kernel_param p
) {
    const int64_t n_id     = blockIdx.x;
    const int64_t n_offset = blockIdx.y * TPB * VPT + threadIdx.x * VPT;

    const int64_t group_size   = p.C * p.scale_factor;
    const int64_t group_sid    = n_offset / group_size;
    const int64_t group_offset = n_offset % group_size;

    const int64_t w_did = group_sid % p.W;
    const int64_t h_did = group_sid / (p.W * p.scale_factor);
    const int64_t hr_id = group_sid / p.W % p.scale_factor;

    const int64_t group_did = h_did * p.W * p.scale_factor + w_did * p.scale_factor + hr_id;

    copy<VPT * sizeof(fp16_t)>(
        &p.input[n_id * p.n_stride + n_offset],
        &p.output[n_id * p.n_stride + group_did * group_size + group_offset]
    );
};

ppl::common::RetCode pixel_unshuffle(
    const cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,              // [N, H*r, W*r, C]
    const int64_t scale_factor,
    void* output                    // [N, H, W, C*r*r]
) {
    if (input_shape->GetDataType() != ppl::common::DATATYPE_FLOAT16) {
        LOG(ERROR) << "pixel unshuffle only support fp16, but got ["<< input_shape->GetDataType() << "]";
        return ppl::common::RC_UNSUPPORTED;
    }

    const int64_t IN_SHAPE_DIM = input_shape->GetDimCount();
    if (IN_SHAPE_DIM != 3 && IN_SHAPE_DIM != 4) {
        LOG(ERROR) << "pixel unshuffle input must be 3d or 4d";
        return ppl::common::RC_UNSUPPORTED;
    }

    const int64_t N = IN_SHAPE_DIM == 4 ? input_shape->GetDim(0) : 1;
    const int64_t H = input_shape->GetDim(IN_SHAPE_DIM - 3) / scale_factor;
    const int64_t W = input_shape->GetDim(IN_SHAPE_DIM - 2) / scale_factor;
    const int64_t C = input_shape->GetDim(IN_SHAPE_DIM - 1);
    const int64_t n_stride = input_shape->CalcElementsExcludingPadding() / N;

    constexpr int64_t TPB = 256;    // thread_per_block
    constexpr int64_t VPT = 16 / sizeof(fp16_t);

    if (C * scale_factor % VPT) {
        LOG(ERROR) << "C * scale_factor in pixel unshuffle input must be aligned to 128bit";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (n_stride % (TPB * VPT)) {
        LOG(ERROR) << "n_stride in pixel unshuffle input must be aligned to TPB * VPT";
        return ppl::common::RC_UNSUPPORTED;
    }

    const int64_t batch_size = n_stride / TPB / VPT;
    const dim3 grid(N, batch_size, 1);

    pixel_unshuffle_fp16_kernel_param p = {
        (fp16_t*) input,
        scale_factor,
        n_stride,
        W,
        C,
        (fp16_t*) output
    };

    pixel_unshuffle_fp16_kernel<VPT, TPB>
        <<<grid, TPB, 0, stream>>>(p);
    return ppl::common::RC_SUCCESS;
}

}}}}}