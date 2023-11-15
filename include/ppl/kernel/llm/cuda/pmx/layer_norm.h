#ifndef __PPL_KERNEL_LLM_CUDA_PMX_LAYERNORM_H__
#define __PPL_KERNEL_LLM_CUDA_PMX_LAYERNORM_H__

#include "ppl/kernel/llm/cuda/common/general_include.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

ppl::common::RetCode layer_norm(
    cudaStream_t stream,
    ppl::common::TensorShape* input_shape,
    const void* input,
    const void* scale,
    const void* shift,
    const void* skip_in,
    void* output,
    void* skip_out,
    int64_t normalize_shape,
    bool elementwise_affine,
    float eps, 
    bool skip_term);

}}}}}

#endif