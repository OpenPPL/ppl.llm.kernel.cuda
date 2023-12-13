#ifndef __PPL_KERNEL_LLM_CUDA_PMX_LAYERNORM_H__
#define __PPL_KERNEL_LLM_CUDA_PMX_LAYERNORM_H__

#include "ppl/kernel/llm/cuda/common/general_include.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

ppl::common::RetCode layer_norm(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const void* weight,
    const void* bias,
    const void* skip_in,
    const int32_t axis,
    const bool elementwise_affine,
    const float eps, 
    const bool skip_term,
    void* output,
    void* skip_out);

}}}}}

#endif