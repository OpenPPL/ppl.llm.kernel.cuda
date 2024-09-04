#ifdef PPLNN_ENABLE_FP8

#include "ppl/kernel/llm/cuda/pmx/f8f8/cast.h"

#include "ppl/common/log.h"
#include "cudakernel/common/common.cuh"

#include <cuda_fp8.h>
using float8 = __nv_fp8_storage_t;

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx { namespace f8f8 {

template<int32_t TPB>
__global__
void cast_fp16_kernel(
    const half* input, // [batch, quant_dim]
    const int64_t batch,
    const int64_t quant_dim,
    float8* output // [batch, quant_dim]
)
{
    constexpr int32_t VPT       = 16;
    constexpr int32_t V8PT      = VPT / sizeof(float8);
    constexpr int32_t V16PT     = VPT / sizeof(half);
    const int64_t batch_id      = blockIdx.x;
    const int64_t batch_offset  = batch_id * quant_dim;

    MatrixLayoutHelper<MATRIX_LAYOUT_ROW_MAJOR> idx_hlp;
    idx_hlp.Init(batch, quant_dim);

    for (int64_t idx = threadIdx.x * V8PT; idx < quant_dim; idx += TPB * V8PT) {
        half local_in[V8PT];
        float8 local_out[V8PT];
        copy<VPT>(&input[batch_offset + idx], local_in);
        copy<VPT>(&input[batch_offset + idx + V16PT], &local_in[V16PT]);

        #pragma unroll
        for (int32_t vidx = 0; vidx < V8PT; vidx += 1) {
            local_out[vidx] = __nv_cvt_halfraw_to_fp8(
                *reinterpret_cast<__half_raw*>(&local_in[vidx]),
                __NV_SATFINITE,
                __NV_E4M3
            );
        }
        copy<VPT>(local_out, &output[idx_hlp.GetOffset(batch_id, idx)]);
    }
}


ppl::common::RetCode cast_fp16(
    cudaStream_t stream,
    const void* input, // fp16, [batch, quant_dim]
    const int64_t batch,
    const int64_t quant_dim,
    void* casted // fp8, [batch, quant_dim]
)
{
    constexpr int32_t TPB = 256;
    cast_fp16_kernel<TPB><<<batch, TPB, 0, stream>>>(
        (const half*)input, batch, quant_dim, (float8*)casted
    );

    return ppl::common::RC_SUCCESS;
}

}}}}}} 

#endif