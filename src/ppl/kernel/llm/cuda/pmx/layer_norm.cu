#include "ppl/kernel/llm/cuda/pmx/layer_norm.h"
#include "cudakernel/common/common.cuh"
#include "ppl/common/log.h"

#include <cuda_fp16.h>

__device__ static inline float2 operator+(const float2& a, const float2& b) {
    return {a.x + b.x, a.y + b.y};
}

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

template <int VPT, int TPB, bool HAS_AFFINE>
__global__
void layer_norm_kernel_fp16(
    const half *x,
    const half *weight,
    const half *bias,
    const float eps,
    const int64_t normalize_shape,
    half *output
)
{
    const int64_t idx = normalize_shape * blockIdx.x + threadIdx.x * VPT;
    half inLocal[VPT]; half weightLocal[VPT]; half biasLocal[VPT];

    copy<sizeof(half) * VPT>(&x[idx], inLocal);
    float2 loc = {0.f, 0.f}; // accumulator
    float r_normalize_shape = 1.0f / (normalize_shape);
    float value = 0.0f;

#pragma unroll
    for (int32_t it = 0; it < VPT; it++) {
        value = __half2float(inLocal[it]);
        loc.x += value * r_normalize_shape;
        loc.y += value * value * r_normalize_shape;
    }

    const float2 reduced = BlockAllReduce<SumOp, float2, TPB>(loc);

    __shared__ float mu;     // mean
    __shared__ float rsigma; // std
    if (threadIdx.x == 0) {
        mu = reduced.x;
        rsigma = rsqrtf(reduced.y - mu * mu + eps);
    }

    __syncthreads();

    half outLocal[VPT];
    if (HAS_AFFINE) {
        copy<sizeof(half) * VPT>(&bias[threadIdx.x * VPT], biasLocal);
        copy<sizeof(half) * VPT>(&weight[threadIdx.x * VPT], weightLocal);
    }

#pragma unroll
    for (int32_t it = 0; it < VPT; it++) {
        if (HAS_AFFINE) {
            outLocal[it] = __float2half((__half2float(inLocal[it]) - mu) * rsigma * __half2float(weightLocal[it]) + __half2float(biasLocal[it]));
        } else {
            outLocal[it] = __float2half((__half2float(inLocal[it]) - mu) * rsigma);
        }
    }
    copy<sizeof(half) * VPT>(outLocal, &output[idx]);
};

template <int TPB, bool HAS_AFFINE>
__global__
void layer_norm_kernel_fp16_default(
    const half *x,
    const half *weight,
    const half *bias,
    const float eps,
    const int64_t normalize_shape,
    half *output
)
{
    const half* cur_in = x + normalize_shape * blockIdx.x;
    half* cur_out = output + normalize_shape * blockIdx.x;
    
    float2 loc = {0.f, 0.f}; // accumulator
    float r_normalize_shape = 1.0f / (normalize_shape);
    float value = 0.0f;

    for (int idx = threadIdx.x; idx < normalize_shape; idx += TPB) {
        value = __half2float(cur_in[idx]);
        loc.x += value * r_normalize_shape;
        loc.y += value * value * r_normalize_shape;
    }

    const float2 reduced = BlockAllReduce<SumOp, float2, TPB>(loc);

    __shared__ float mu;     // mean
    __shared__ float rsigma; // std

    if (threadIdx.x == 0) {
        mu = reduced.x;
        rsigma = rsqrt(reduced.y - mu * mu + eps);
    }

    __syncthreads();

    for (int idx=threadIdx.x; idx < normalize_shape; idx += TPB) {
        if (HAS_AFFINE) {
            cur_out[idx] = __float2half((__half2float(cur_in[idx]) - mu) * rsigma * __half2float(weight[idx]) + __half2float(bias[idx]));
        } else {
            cur_out[idx] = __float2half((__half2float(cur_in[idx]) - mu) * rsigma);
        }
    }
}

template <int VPT, int TPB, bool HAS_AFFINE, bool HAS_SKIP_IN>
__global__
void skip_layer_norm_kernel_fp16(
    const half *x,
    const half *weight,
    const half *bias,
    const half *skip_in,
    const float eps,
    const int64_t normalize_shape,
    half *output,
    half *skip_out
)
{
    const int64_t idx = normalize_shape * blockIdx.x + threadIdx.x * VPT;
    half inLocal[VPT]; half weightLocal[VPT]; half biasLocal[VPT];

    copy<sizeof(half) * VPT>(&x[idx], inLocal);
    if (HAS_SKIP_IN) {
        copy<sizeof(half) * VPT>(&skip_in[idx], biasLocal);
    }

// step 1. compute x + skip_in
#pragma unroll
    for (int32_t it = 0; it < VPT; it++){
        if (HAS_SKIP_IN) {
            inLocal[it] = inLocal[it] + biasLocal[it];
        }
    }
    copy<sizeof(half) * VPT>(inLocal, &skip_out[idx]);


// step 2: compute mean and var
    float2 loc = {0.f, 0.f}; // accumulator
    float r_normalize_shape = 1.0f / (normalize_shape);
    float value = 0.0f;

#pragma unroll
    for (int32_t it = 0; it < VPT; it++) {
        value = __half2float(inLocal[it]);
        loc.x += value * r_normalize_shape;
        loc.y += value * value * r_normalize_shape;
    }

    const float2 reduced = BlockAllReduce<SumOp, float2, TPB>(loc);

    __shared__ float mu;     // mean
    __shared__ float rsigma; // std.dev.
    if (threadIdx.x == 0) {
        mu = reduced.x;
        rsigma = rsqrt(reduced.y - mu * mu + eps);
    }
    __syncthreads();

    half outLocal[VPT];
    if (HAS_AFFINE) {
        copy<sizeof(half) * VPT>(&bias[threadIdx.x * VPT], biasLocal);
        copy<sizeof(half) * VPT>(&weight[threadIdx.x * VPT], weightLocal);
    }
#pragma unroll
    for (int32_t it = 0; it < VPT; it++) {
        if (HAS_AFFINE) {
            outLocal[it] = __float2half((__half2float(inLocal[it]) - mu) * rsigma * __half2float(weightLocal[it]) + __half2float(biasLocal[it]));
        } else {
            outLocal[it] = __float2half((__half2float(inLocal[it]) - mu) * rsigma);
        }
    }

    copy<sizeof(half) * VPT>(outLocal, &output[idx]);
}


template <int TPB, bool HAS_AFFINE, bool HAS_SKIP_IN>
__global__
void skip_layer_norm_kernel_fp16_default(
    const half *x,
    const half *weight,
    const half *bias,
    const half *skip_in,
    const float eps,
    const int64_t normalize_shape,
    half *output,
    half *skip_out
)
{
    const half* cur_in = x + normalize_shape * blockIdx.x;
    const half* cur_skip_in = skip_in + normalize_shape * blockIdx.x;
    half* cur_out = output + normalize_shape * blockIdx.x;
    half* cur_skip_out = skip_out + normalize_shape * blockIdx.x;

    // step 1: compute x + skip_in
    for (int idx = threadIdx.x; idx < normalize_shape; idx += TPB) {
        if (HAS_SKIP_IN) {
            cur_skip_out[idx] = cur_in[idx] + cur_skip_in[idx];
        } else {
            cur_skip_out[idx] = cur_in[idx];
        }
    }
    
    // step 2: compute means and var
    float2 loc = {0.f, 0.f}; // accumulator
    float r_normalize_shape = 1.0f / (normalize_shape);
    float value = 0.0f;

    for (int idx = threadIdx.x; idx < normalize_shape; idx += TPB) {
        value = __half2float(cur_skip_out[idx]);
        loc.x += value * r_normalize_shape;
        loc.y += value * value * r_normalize_shape;
    }

    const float2 reduced = BlockAllReduce<SumOp, float2, TPB>(loc);

    __shared__ float mu;     // mean
    __shared__ float rsigma; // std

    if (threadIdx.x == 0) {
        mu = reduced.x;
        rsigma = rsqrt(reduced.y - mu * mu + eps);
    }

    __syncthreads();

    for (int idx = threadIdx.x; idx < normalize_shape; idx += TPB) {
        if (HAS_AFFINE) {
            cur_out[idx] = __float2half((__half2float(cur_skip_out[idx]) - mu) * rsigma * __half2float(weight[idx]) + __half2float(bias[idx]));
        } else {
            cur_out[idx] = __float2half((__half2float(cur_skip_out[idx]) - mu) * rsigma);
        }
    }
}

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
    void* skip_out)
{

    if(input_shape->GetDataType() != ppl::common::DATATYPE_FLOAT16) {
      LOG(ERROR) << "LayerNorm only support fp16, but got ["<< input_shape->GetDataType() << "]";
    }
    constexpr int32_t VPT = 16 / sizeof(half);

    const int32_t real_axis = axis < 0 ? input_shape->GetDimCount() + axis : axis;
    const int64_t normalize_shape = input_shape->CalcElementsFromDimensionExcludingPadding(real_axis);
    const int64_t grid_size = input_shape->CalcElementsToDimensionExcludingPadding(real_axis);

    if (skip_term) {
        if (elementwise_affine) {
            if (skip_in) {
                switch (normalize_shape)
                {
                case 768:
                    skip_layer_norm_kernel_fp16<VPT, 768 / VPT, true, true>
                    <<<grid_size, 768 / VPT, 0, stream>>>(
                        (const half*)input,
                        (const half*)weight,
                        (const half*)bias,
                        (const half*)skip_in,
                        eps, 
                        normalize_shape, 
                        (half*)output,
                        (half*)skip_out);
                    break;
                case 1024:
                    skip_layer_norm_kernel_fp16<VPT, 1024 / VPT, true, true>
                    <<<grid_size, 1024 / VPT, 0, stream>>>(
                        (const half*)input,
                        (const half*)weight,
                        (const half*)bias,
                        (const half*)skip_in,
                        eps, 
                        normalize_shape, 
                        (half*)output,
                        (half*)skip_out);
                    break;
                case 2048:
                    skip_layer_norm_kernel_fp16<VPT, 2048 / VPT, true, true>
                    <<<grid_size, 2048 / VPT, 0, stream>>>(
                        (const half*)input,
                        (const half*)weight,
                        (const half*)bias,
                        (const half*)skip_in,
                        eps, 
                        normalize_shape, 
                        (half*)output,
                        (half*)skip_out);
                    break;
                case 4096:
                    skip_layer_norm_kernel_fp16<VPT, 4096 / VPT, true, true>
                    <<<grid_size, 4096 / VPT, 0, stream>>>(
                        (const half*)input,
                        (const half*)weight,
                        (const half*)bias,
                        (const half*)skip_in,
                        eps, 
                        normalize_shape, 
                        (half*)output,
                        (half*)skip_out);
                    break;
                default:
                    skip_layer_norm_kernel_fp16_default<512, true, true>
                    <<<grid_size, 512, 0, stream>>>(
                        (const half*)input,
                        (const half*)weight,
                        (const half*)bias,
                        (const half*)skip_in,
                        eps,
                        normalize_shape,
                        (half*)output,
                        (half*)skip_out);
                }
            } else {
                switch (normalize_shape)
                {
                case 768:
                    skip_layer_norm_kernel_fp16<VPT, 768 / VPT, true, false>
                    <<<grid_size, 768 / VPT, 0, stream>>>(
                        (const half*)input,
                        (const half*)weight,
                        (const half*)bias,
                        (const half*)skip_in,
                        eps, 
                        normalize_shape, 
                        (half*)output,
                        (half*)skip_out);
                    break;
                case 1024:
                    skip_layer_norm_kernel_fp16<VPT, 1024 / VPT, true, false>
                    <<<grid_size, 1024 / VPT, 0, stream>>>(
                        (const half*)input,
                        (const half*)weight,
                        (const half*)bias,
                        (const half*)skip_in,
                        eps, 
                        normalize_shape, 
                        (half*)output,
                        (half*)skip_out);
                    break;
                case 2048:
                    skip_layer_norm_kernel_fp16<VPT, 2048 / VPT, true, false>
                    <<<grid_size, 2048 / VPT, 0, stream>>>(
                        (const half*)input,
                        (const half*)weight,
                        (const half*)bias,
                        (const half*)skip_in,
                        eps, 
                        normalize_shape, 
                        (half*)output,
                        (half*)skip_out);
                    break;
                case 4096:
                    skip_layer_norm_kernel_fp16<VPT, 4096 / VPT, true, false>
                    <<<grid_size, 4096 / VPT, 0, stream>>>(
                        (const half*)input,
                        (const half*)weight,
                        (const half*)bias,
                        (const half*)skip_in,
                        eps, 
                        normalize_shape, 
                        (half*)output,
                        (half*)skip_out);
                    break;
                default:
                    skip_layer_norm_kernel_fp16_default<512, true, false>
                    <<<grid_size, 512, 0, stream>>>(
                        (const half*)input,
                        (const half*)weight,
                        (const half*)bias,
                        (const half*)skip_in,
                        eps,
                        normalize_shape,
                        (half*)output,
                        (half*)skip_out);
                }
            }
        } else {
            if (skip_in) {
                switch (normalize_shape)
                {
                case 768:
                    skip_layer_norm_kernel_fp16<VPT, 768 / VPT, false, true>
                    <<<grid_size, 768 / VPT, 0, stream>>>(
                        (const half*)input,
                        (const half*)weight,
                        (const half*)bias,
                        (const half*)skip_in,
                        eps, 
                        normalize_shape, 
                        (half*)output,
                        (half*)skip_out);
                    break;
                case 1024:
                    skip_layer_norm_kernel_fp16<VPT, 1024 / VPT, false, true>
                    <<<grid_size, 1024 / VPT, 0, stream>>>(
                        (const half*)input,
                        (const half*)weight,
                        (const half*)bias,
                        (const half*)skip_in,
                        eps, 
                        normalize_shape, 
                        (half*)output,
                        (half*)skip_out);
                    break;
                case 2048:
                    skip_layer_norm_kernel_fp16<VPT, 2048 / VPT, false, true>
                    <<<grid_size, 2048 / VPT, 0, stream>>>(
                        (const half*)input,
                        (const half*)weight,
                        (const half*)bias,
                        (const half*)skip_in,
                        eps, 
                        normalize_shape, 
                        (half*)output,
                        (half*)skip_out);
                    break;
                case 4096:
                    skip_layer_norm_kernel_fp16<VPT, 4096 / VPT, false, true>
                    <<<grid_size, 4096 / VPT, 0, stream>>>(
                        (const half*)input,
                        (const half*)weight,
                        (const half*)bias,
                        (const half*)skip_in,
                        eps, 
                        normalize_shape, 
                        (half*)output,
                        (half*)skip_out);
                    break;
                default:
                    skip_layer_norm_kernel_fp16_default<512, false, true>
                    <<<grid_size, 512, 0, stream>>>(
                        (const half*)input,
                        (const half*)weight,
                        (const half*)bias,
                        (const half*)skip_in,
                        eps,
                        normalize_shape,
                        (half*)output,
                        (half*)skip_out);
                }
            } else {
                switch (normalize_shape)
                {
                case 768:
                    skip_layer_norm_kernel_fp16<VPT, 768 / VPT, false, false>
                    <<<grid_size, 768 / VPT, 0, stream>>>(
                        (const half*)input,
                        (const half*)weight,
                        (const half*)bias,
                        (const half*)skip_in,
                        eps, 
                        normalize_shape, 
                        (half*)output,
                        (half*)skip_out);
                    break;
                case 1024:
                    skip_layer_norm_kernel_fp16<VPT, 1024 / VPT, false, false>
                    <<<grid_size, 1024 / VPT, 0, stream>>>(
                        (const half*)input,
                        (const half*)weight,
                        (const half*)bias,
                        (const half*)skip_in,
                        eps, 
                        normalize_shape, 
                        (half*)output,
                        (half*)skip_out);
                    break;
                case 2048:
                    skip_layer_norm_kernel_fp16<VPT, 2048 / VPT, false, false>
                    <<<grid_size, 2048 / VPT, 0, stream>>>(
                        (const half*)input,
                        (const half*)weight,
                        (const half*)bias,
                        (const half*)skip_in,
                        eps, 
                        normalize_shape, 
                        (half*)output,
                        (half*)skip_out);
                    break;
                case 4096:
                    skip_layer_norm_kernel_fp16<VPT, 4096 / VPT, false, false>
                    <<<grid_size, 4096 / VPT, 0, stream>>>(
                        (const half*)input,
                        (const half*)weight,
                        (const half*)bias,
                        (const half*)skip_in,
                        eps, 
                        normalize_shape, 
                        (half*)output,
                        (half*)skip_out);
                    break;
                default:
                    skip_layer_norm_kernel_fp16_default<512, false, false>
                    <<<grid_size, 512, 0, stream>>>(
                        (const half*)input,
                        (const half*)weight,
                        (const half*)bias,
                        (const half*)skip_in,
                        eps,
                        normalize_shape,
                        (half*)output,
                        (half*)skip_out);
                }
            }
        }

    } else {
        if (elementwise_affine) {        
            switch (normalize_shape)
            {
            case 768:
                layer_norm_kernel_fp16<VPT, 768 / VPT, true>
                <<<grid_size, 768 / VPT, 0, stream>>>(
                    (const half*)input,
                    (const half*)weight,
                    (const half*)bias,
                    eps, 
                    normalize_shape, 
                    (half*)output);
                break;
            case 1024:
                layer_norm_kernel_fp16<VPT, 1024 / VPT, true>
                <<<grid_size, 1024 / VPT, 0, stream>>>(
                    (const half*)input,
                    (const half*)weight,
                    (const half*)bias,
                    eps, 
                    normalize_shape, 
                    (half*)output);
                break;
            case 2048:
                layer_norm_kernel_fp16<VPT, 2048 / VPT, true>
                <<<grid_size, 2048 / VPT, 0, stream>>>(
                    (const half*)input,
                    (const half*)weight,
                    (const half*)bias,
                    eps, 
                    normalize_shape, 
                    (half*)output);
                break;
            case 4096:
                layer_norm_kernel_fp16<VPT, 4096 / VPT, true>
                <<<grid_size, 4096 / VPT, 0, stream>>>(
                    (const half*)input,
                    (const half*)weight,
                    (const half*)bias,
                    eps, 
                    normalize_shape, 
                    (half*)output);
                break;
            default:
                layer_norm_kernel_fp16_default<512, true>
                <<<grid_size, 512, 0, stream>>> (
                    (const half*)input,
                    (const half*)weight,
                    (const half*)bias,
                    eps, 
                    normalize_shape, 
                    (half*)output);
            }
        } else {
            switch (normalize_shape)
            {
            case 768:
                layer_norm_kernel_fp16<VPT, 768 / VPT, false>
                <<<grid_size, 768 / VPT, 0, stream>>>(
                    (const half*)input,
                    (const half*)weight,
                    (const half*)bias,
                    eps, 
                    normalize_shape, 
                    (half*)output);
                break;
            case 1024:
                layer_norm_kernel_fp16<VPT, 1024 / VPT, false>
                <<<grid_size, 1024 / VPT, 0, stream>>>(
                    (const half*)input,
                    (const half*)weight,
                    (const half*)bias,
                    eps, 
                    normalize_shape, 
                    (half*)output);
                break;
            case 2048:
                layer_norm_kernel_fp16<VPT, 2048 / VPT, false>
                <<<grid_size, 2048 / VPT, 0, stream>>>(
                    (const half*)input,
                    (const half*)weight,
                    (const half*)bias,
                    eps, 
                    normalize_shape, 
                    (half*)output);
                break;
            case 4096:
                layer_norm_kernel_fp16<VPT, 4096 / VPT, false>
                <<<grid_size, 4096 / VPT, 0, stream>>>(
                    (const half*)input,
                    (const half*)weight,
                    (const half*)bias,
                    eps, 
                    normalize_shape, 
                    (half*)output);
                break;
            default:
                layer_norm_kernel_fp16_default<512, false>
                <<<grid_size, 512, 0, stream>>> (
                    (const half*)input,
                    (const half*)weight,
                    (const half*)bias,
                    eps, 
                    normalize_shape, 
                    (half*)output);
            }
        } 
    }
    return ppl::common::RC_SUCCESS;
}

}}}}}
