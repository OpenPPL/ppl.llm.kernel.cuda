#ifndef __PPLCUDA_FMHA_H_
#define __PPLCUDA_FMHA_H_
#include "autogen/cutlassF.h"
#include "kernel_forward.h"
#include "ppl/common/tensor_shape.h"
#include "ppl/common/types.h"
#include "ppl/common/retcode.h"
#include "ppl/common/log.h"
#include <cuda_runtime.h>                                                      

ppl::common::RetCode PPLCUDAFMHAForwardImp(
    const cudaDeviceProp& device_prop,
    const cudaStream_t stream,
    const ppl::common::TensorShape* query_shape,
    const void* query,
    const ppl::common::TensorShape* key_shape,
    const void* key,
    const ppl::common::TensorShape* value_shape,
    const void* value,
    const ppl::common::TensorShape* attn_mask_shape,
    const void* attn_mask,
    const ppl::common::TensorShape* seqstart_q_shape,
    const void* seqstart_q,
    const ppl::common::TensorShape* seqstart_k_shape,
    const void* seqstart_k,
    const ppl::common::TensorShape* seqlen_k_shape,
    const void* seqlen_k,
    const int64_t max_seqlen,
    const int64_t custom_mask_type,
    const double scale,
    const ppl::common::TensorShape* output_shape,
    void* output);

#endif // __PPLCUDA_FMHA_H_