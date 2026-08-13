// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/*
 * Fused MoE finalize + shared-output add (bf16 output, SM>=90 for PDL).
 *
 * Forked from flashinfer's ``finalizeKernel`` and ``finalizeKernelVecLoad``
 * (trtllm_fused_moe_dev_kernel.cu:639 and :803), stripped of the MoE
 * backend's KernelParams / UsePdl templating, and extended with an
 * optional shared_output residual add on the epilogue side.
 *
 * For each token t, computes:
 *     out[t] = Σ_k expert_weights[t, k] * gemm2_out[permuted_idx(t, k)]
 *            + shared_output[t]                      // if non-null
 *
 * Eliminates the native PyTorch ``routed + shared_output`` add (and the
 * separate ``*= routed_scaling_factor`` kernel when applicable) from
 * ``DeepseekV3MoE.forward``, and gives the downstream allreduce+rmsnorm
 * a clean PDL handoff.
 *
 * Expert-weight dtype is templated on ``TypeExpW`` so we accept both bf16
 * and fp32 topk weights. The trtllm deferred-finalize path always feeds bf16
 * (the trtllm-gen routing kernel emits bf16 for every routing method); fp32
 * is kept for callers that produce topk weights in fp32.
 *
 * Expert-weight scale convention: in our target backends
 * (flashinfer trtllm nvfp4 + unquantized), ``apply_routed_scaling_factor_on_output``
 * is True, so the routed scaling factor is already folded into
 * ``expert_weights`` at topk time. This kernel does not apply any
 * additional scale.
 */

#include <cutlass/array.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "tvm_ffi_utils.h"
#include <cuda_runtime.h>

namespace sglang {

using BF16 = cutlass::bfloat16_t;

constexpr int FINALIZE_THREADS_PER_BLOCK = 256;
constexpr int MAX_TOPK = 64;

// ---------------------------------------------------------------------------
// General kernel — one CTA per (hidden_chunk, token). Picks up small-to-mid
// workloads where the block count fits in a few waves.
// ---------------------------------------------------------------------------
template <typename TypeExpW>
__global__ void moeFinalizeKernel(
    int numTokens,
    int hiddenDim,
    int hiddenDimPadded,
    int topK,
    BF16 const* __restrict__ inPtr,
    int const* __restrict__ expandedIdxToPermutedIdx,
    TypeExpW const* __restrict__ expertWeightsPtr,
    BF16 const* __restrict__ sharedBiasPtr,
    BF16* __restrict__ outPtr,
    float const* __restrict__ accInPtr,
    float* __restrict__ accOutPtr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
#endif

  for (int64_t tokenIdx = blockIdx.y; tokenIdx < numTokens; tokenIdx += gridDim.y) {
    for (int64_t hiddenIdx = threadIdx.x + blockDim.x * blockIdx.x; hiddenIdx < hiddenDim;
         hiddenIdx += blockDim.x * gridDim.x) {
      float acc = 0.0f;
      for (int k = 0; k < topK; k++) {
        int64_t const expandedIdx = tokenIdx * topK + k;
        int64_t const permutedIdx = expandedIdxToPermutedIdx[expandedIdx];
        if (permutedIdx == -1) {
          continue;
        }
        float const scale = static_cast<float>(expertWeightsPtr[expandedIdx]);
        float const val = static_cast<float>(inPtr[permutedIdx * hiddenDimPadded + hiddenIdx]);
        acc += scale * val;
      }
      if (accInPtr != nullptr) {
        acc += accInPtr[tokenIdx * hiddenDim + hiddenIdx];
      }
      if (sharedBiasPtr != nullptr) {
        acc += static_cast<float>(sharedBiasPtr[tokenIdx * hiddenDim + hiddenIdx]);
      }
      if (accOutPtr != nullptr) {
        accOutPtr[tokenIdx * hiddenDim + hiddenIdx] = acc;
      } else {
        outPtr[tokenIdx * hiddenDim + hiddenIdx] = static_cast<BF16>(acc);
      }
    }
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// ---------------------------------------------------------------------------
// Vectorized-load kernel — one CTA per token, 128-bit loads, topK unrolled.
// Better at prefill shapes where the general kernel's block count saturates
// many waves and the indirect gather from gemm2_out dominates.
// ---------------------------------------------------------------------------

__device__ inline float4 vectorizedLoadPtx(float4 const* ptr) {
  float4 ret;
  asm volatile("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w)
               : "l"(ptr));
  return ret;
}

template <int TopKUnrollFactor>
struct IdxPackedTraits;
template <>
struct IdxPackedTraits<1> {
  using Packed = int;
};
template <>
struct IdxPackedTraits<2> {
  using Packed = int2;
};
template <>
struct IdxPackedTraits<4> {
  using Packed = int4;
};

template <typename TypeExpW, int TopKUnrollFactor>
__global__ void moeFinalizeKernelVecLoad(
    int numTokens,
    int hiddenDim,
    int hiddenDimPadded,
    int topK,
    BF16 const* __restrict__ inPtr,
    int const* __restrict__ expandedIdxToPermutedIdx,
    TypeExpW const* __restrict__ expertWeightsPtr,
    BF16 const* __restrict__ sharedBiasPtr,
    BF16* __restrict__ outPtr,
    float const* __restrict__ accInPtr,
    float* __restrict__ accOutPtr) {
  static_assert(
      TopKUnrollFactor == 1 || TopKUnrollFactor == 2 || TopKUnrollFactor == 4, "TopKUnrollFactor must be 1, 2, or 4");
  using IdxPackedType = typename IdxPackedTraits<TopKUnrollFactor>::Packed;
  using IdxArrayType = cutlass::Array<int, TopKUnrollFactor>;
  using ScaleArrayType = cutlass::Array<TypeExpW, TopKUnrollFactor>;

  // 128 bits per thread → 8 bf16 elements.
  constexpr int FINALIZE_ELEM_PER_THREAD = 8;
  using InputElem = cutlass::Array<BF16, FINALIZE_ELEM_PER_THREAD>;
  using OutputElem = cutlass::Array<BF16, FINALIZE_ELEM_PER_THREAD>;
  using ComputeElem = cutlass::Array<float, FINALIZE_ELEM_PER_THREAD>;

  int64_t const tokenIdx = blockIdx.x;
  int64_t const startOffset = threadIdx.x;
  int64_t const stride = FINALIZE_THREADS_PER_BLOCK;
  int64_t const numElemsInPaddedCol = hiddenDimPadded / FINALIZE_ELEM_PER_THREAD;
  int64_t const numElemsInCol = hiddenDim / FINALIZE_ELEM_PER_THREAD;

  // Stage the per-token (topK/unroll) indices + scales into smem.
  __shared__ ScaleArrayType scaleArrSmem[MAX_TOPK / TopKUnrollFactor];
  __shared__ IdxArrayType permutedIdxArrSmem[MAX_TOPK / TopKUnrollFactor];

  for (int kChunkIdx = threadIdx.x; kChunkIdx < topK / TopKUnrollFactor; kChunkIdx += blockDim.x) {
    int64_t const expandedIdx = tokenIdx * topK + kChunkIdx * TopKUnrollFactor;
    auto const permutedIdxPacked =
        reinterpret_cast<IdxPackedType const*>(expandedIdxToPermutedIdx)[expandedIdx / TopKUnrollFactor];
    permutedIdxArrSmem[kChunkIdx] = *reinterpret_cast<IdxArrayType const*>(&permutedIdxPacked);
#pragma unroll
    for (int ki = 0; ki < TopKUnrollFactor; ++ki) {
      scaleArrSmem[kChunkIdx][ki] = expertWeightsPtr[expandedIdx + ki];
    }
  }

  // outPtr is null when this launch produces an fp32 partial instead.
  auto* outElemPtr = outPtr != nullptr ? reinterpret_cast<OutputElem*>(outPtr + tokenIdx * hiddenDim) : nullptr;
  auto const* inElemPtr = reinterpret_cast<InputElem const*>(inPtr);
  auto const* sharedElemPtr =
      sharedBiasPtr != nullptr ? reinterpret_cast<InputElem const*>(sharedBiasPtr + tokenIdx * hiddenDim) : nullptr;
  // fp32 accumulator rows: 8 floats per thread-element = two 128-bit accesses.
  auto const* accInRow = accInPtr != nullptr ? accInPtr + tokenIdx * hiddenDim : nullptr;
  auto* accOutRow = accOutPtr != nullptr ? accOutPtr + tokenIdx * hiddenDim : nullptr;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
#endif
  __syncthreads();

  for (int elemIndex = startOffset; elemIndex < numElemsInCol; elemIndex += stride) {
    ComputeElem threadOutput;
    threadOutput.fill(0.0f);

    for (int kChunkIdx = 0; kChunkIdx < topK / TopKUnrollFactor; kChunkIdx++) {
      IdxArrayType permutedIdxArr = permutedIdxArrSmem[kChunkIdx];
      InputElem inputElemArr[TopKUnrollFactor];
#pragma unroll
      for (int ki = 0; ki < TopKUnrollFactor; ++ki) {
        int const permutedIdx = permutedIdxArr[ki];
        if (permutedIdx == -1) {
          continue;
        }
        auto const* inputPermutedPtr = inElemPtr + permutedIdx * numElemsInPaddedCol;
        float4 input = vectorizedLoadPtx(reinterpret_cast<float4 const*>(&inputPermutedPtr[elemIndex]));
        inputElemArr[ki] = *reinterpret_cast<InputElem const*>(&input);
      }
      ScaleArrayType scaleArr = scaleArrSmem[kChunkIdx];
#pragma unroll
      for (int ki = 0; ki < TopKUnrollFactor; ++ki) {
        int const permutedIdx = permutedIdxArr[ki];
        if (permutedIdx == -1) {
          continue;
        }
        float const scale = static_cast<float>(scaleArr[ki]);
        cutlass::NumericArrayConverter<float, BF16, FINALIZE_ELEM_PER_THREAD> toFloat;
        ComputeElem expertResult = toFloat(inputElemArr[ki]);
#pragma unroll
        for (int e = 0; e < FINALIZE_ELEM_PER_THREAD; ++e) {
          threadOutput[e] += scale * expertResult[e];
        }
      }
    }

    if (accInRow != nullptr) {
      float const* accSlot = accInRow + elemIndex * FINALIZE_ELEM_PER_THREAD;
      float4 lo = vectorizedLoadPtx(reinterpret_cast<float4 const*>(accSlot));
      float4 hi = vectorizedLoadPtx(reinterpret_cast<float4 const*>(accSlot + 4));
      threadOutput[0] += lo.x;
      threadOutput[1] += lo.y;
      threadOutput[2] += lo.z;
      threadOutput[3] += lo.w;
      threadOutput[4] += hi.x;
      threadOutput[5] += hi.y;
      threadOutput[6] += hi.z;
      threadOutput[7] += hi.w;
    }

    if (sharedElemPtr != nullptr) {
      float4 shared = vectorizedLoadPtx(reinterpret_cast<float4 const*>(&sharedElemPtr[elemIndex]));
      InputElem sharedElem = *reinterpret_cast<InputElem const*>(&shared);
      cutlass::NumericArrayConverter<float, BF16, FINALIZE_ELEM_PER_THREAD> toFloat;
      ComputeElem sharedFloat = toFloat(sharedElem);
#pragma unroll
      for (int e = 0; e < FINALIZE_ELEM_PER_THREAD; ++e) {
        threadOutput[e] += sharedFloat[e];
      }
    }

    if (accOutRow != nullptr) {
      float* accSlot = accOutRow + elemIndex * FINALIZE_ELEM_PER_THREAD;
#pragma unroll
      for (int e = 0; e < FINALIZE_ELEM_PER_THREAD; ++e) {
        accSlot[e] = threadOutput[e];
      }
    } else {
      cutlass::NumericArrayConverter<BF16, float, FINALIZE_ELEM_PER_THREAD> toBF16;
      outElemPtr[elemIndex] = toBF16(threadOutput);
    }
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// ---------------------------------------------------------------------------
// Typed dispatch
// ---------------------------------------------------------------------------
template <typename TypeExpW>
void dispatchFinalize(
    int numTokens,
    int hiddenDim,
    int hiddenDimPadded,
    int topK,
    BF16 const* inPtr,
    int const* expandedIdxPtr,
    void const* weightsPtrVoid,
    BF16 const* sharedPtr,
    BF16* outPtr,
    float const* accInPtr,
    float* accOutPtr,
    bool useVecLoad,
    cudaStream_t stream,
    cudaLaunchAttribute const* attrs,
    int numAttrs) {
  auto const* weightsPtr = static_cast<TypeExpW const*>(weightsPtrVoid);
  constexpr int kNumThreads = 256;

  if (!useVecLoad) {
    int const numBlocksX = (hiddenDim + kNumThreads - 1) / kNumThreads;
    int const numBlocksY = std::min(8192, numTokens);
    cudaLaunchConfig_t config;
    config.gridDim = dim3(numBlocksX, numBlocksY);
    config.blockDim = dim3(kNumThreads);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    config.numAttrs = numAttrs;
    config.attrs = const_cast<cudaLaunchAttribute*>(attrs);

    cudaLaunchKernelEx(
        &config,
        moeFinalizeKernel<TypeExpW>,
        numTokens,
        hiddenDim,
        hiddenDimPadded,
        topK,
        inPtr,
        expandedIdxPtr,
        weightsPtr,
        sharedPtr,
        outPtr,
        accInPtr,
        accOutPtr);
    return;
  }

  auto launch = [&](auto unroll_tag) {
    constexpr int UNROLL = decltype(unroll_tag)::value;
    cudaLaunchConfig_t config;
    config.gridDim = dim3(numTokens);
    config.blockDim = dim3(FINALIZE_THREADS_PER_BLOCK);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    config.numAttrs = numAttrs;
    config.attrs = const_cast<cudaLaunchAttribute*>(attrs);
    cudaLaunchKernelEx(
        &config,
        moeFinalizeKernelVecLoad<TypeExpW, UNROLL>,
        numTokens,
        hiddenDim,
        hiddenDimPadded,
        topK,
        inPtr,
        expandedIdxPtr,
        weightsPtr,
        sharedPtr,
        outPtr,
        accInPtr,
        accOutPtr);
  };
  // Match flashinfer's LAUNCH_TOPK_EXPW dispatch order.
  if (topK % 4 == 0) {
    launch(std::integral_constant<int, 4>{});
  } else if (topK % 2 == 0) {
    launch(std::integral_constant<int, 2>{});
  } else {
    launch(std::integral_constant<int, 1>{});
  }
}

}  // namespace sglang

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------
// ``acc_in`` / ``acc_out`` are fp32 [numTokens, hiddenDim] partial-sum buffers,
// either of which may be empty. They let one token's top-k sum be split across
// several launches (disjoint expert slices) while keeping the accumulation in
// fp32 throughout, so the result rounds to bf16 exactly once -- as it would in
// a single unsplit call. Empty acc_out means "produce the bf16 result now".
static void finalizeImpl(
    TensorView out,
    TensorView gemm2_out,
    TensorView expanded_idx_to_permuted_idx,
    TensorView expert_weights,
    TensorView shared_output,
    TensorView acc_in,
    TensorView acc_out,
    int64_t top_k,
    bool enable_pdl) {
  TVM_FFI_ICHECK_EQ(gemm2_out.ndim(), 2) << "gemm2_out must be 2-D [totalNumPaddedTokens, hiddenDimPadded]";
  TVM_FFI_ICHECK_EQ(expanded_idx_to_permuted_idx.ndim(), 1);
  TVM_FFI_ICHECK_EQ(expert_weights.ndim(), 2) << "expert_weights must be 2-D [numTokens, topK]";

  bool const hasAccOut = acc_out.numel() > 0;
  bool const hasAccIn = acc_in.numel() > 0;
  // Dimensions come from whichever destination this launch writes.
  TensorView const& shape_src = hasAccOut ? acc_out : out;
  TVM_FFI_ICHECK_EQ(shape_src.ndim(), 2) << "destination must be 2-D [numTokens, hiddenDim]";

  int const numTokens = int(shape_src.size(0));
  int const hiddenDim = int(shape_src.size(1));
  int const hiddenDimPadded = int(gemm2_out.size(1));
  TVM_FFI_ICHECK_LE(top_k, sglang::MAX_TOPK);
  TVM_FFI_ICHECK_EQ(expanded_idx_to_permuted_idx.size(0), numTokens * top_k);
  TVM_FFI_ICHECK_EQ(expert_weights.size(0), numTokens);
  TVM_FFI_ICHECK_EQ(expert_weights.size(1), top_k);

  bool const hasShared = shared_output.numel() > 0;
  if (hasShared) {
    TVM_FFI_ICHECK_EQ(shared_output.ndim(), 2);
    TVM_FFI_ICHECK_EQ(shared_output.size(0), numTokens);
    TVM_FFI_ICHECK_EQ(shared_output.size(1), hiddenDim);
  }

  if (hasAccIn) {
    TVM_FFI_ICHECK_EQ(acc_in.ndim(), 2);
    TVM_FFI_ICHECK_EQ(acc_in.size(0), numTokens);
    TVM_FFI_ICHECK_EQ(acc_in.size(1), hiddenDim);
  }

  auto const* inPtr = static_cast<sglang::BF16 const*>(gemm2_out.data_ptr());
  auto const* expandedIdxPtr = static_cast<int const*>(expanded_idx_to_permuted_idx.data_ptr());
  auto const* sharedPtr = hasShared ? static_cast<sglang::BF16 const*>(shared_output.data_ptr()) : nullptr;
  auto* outPtr = hasAccOut ? nullptr : static_cast<sglang::BF16*>(out.data_ptr());
  auto const* accInPtr = hasAccIn ? static_cast<float const*>(acc_in.data_ptr()) : nullptr;
  auto* accOutPtr = hasAccOut ? static_cast<float*>(acc_out.data_ptr()) : nullptr;

  cudaSetDevice(shape_src.device().device_id);
  cudaStream_t const stream = get_stream(shape_src.device());

  // Dispatch heuristic (matches flashinfer): few waves → general kernel,
  // many waves → vectorized. The 1184 threshold comes from 148 SMs × 8
  // blocks/SM on Blackwell.
  constexpr int kNumThreads = 256;
  int const numBlocksX = (hiddenDim + kNumThreads - 1) / kNumThreads;
  int const numBlocksY = std::min(8192, numTokens);
  bool const useVecLoad = (numBlocksX * numBlocksY) >= 1184 && (hiddenDim % 8 == 0) && (hiddenDimPadded % 8 == 0);

  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;

  auto ew_dtype = expert_weights.dtype();
  if (ew_dtype == DLDataType{kDLFloat, 32, 1}) {
    sglang::dispatchFinalize<float>(
        numTokens,
        hiddenDim,
        hiddenDimPadded,
        int(top_k),
        inPtr,
        expandedIdxPtr,
        expert_weights.data_ptr(),
        sharedPtr,
        outPtr,
        accInPtr,
        accOutPtr,
        useVecLoad,
        stream,
        attrs,
        1);
  } else if (ew_dtype == DLDataType{kDLBfloat, 16, 1}) {
    sglang::dispatchFinalize<sglang::BF16>(
        numTokens,
        hiddenDim,
        hiddenDimPadded,
        int(top_k),
        inPtr,
        expandedIdxPtr,
        expert_weights.data_ptr(),
        sharedPtr,
        outPtr,
        accInPtr,
        accOutPtr,
        useVecLoad,
        stream,
        attrs,
        1);
  } else {
    TVM_FFI_ICHECK(false) << "expert_weights dtype must be float32 or bfloat16";
  }

  cudaError_t const err = cudaGetLastError();
  TVM_FFI_ICHECK(err == cudaSuccess) << "moe_finalize_fuse_shared launch failed: " << cudaGetErrorString(err);
}

// ``acc_in`` / ``acc_out`` follow the same empty-tensor-means-absent convention
// as ``shared_output``: pass an empty acc_in on the first expert slice and an
// empty acc_out on the last (which applies shared_output and writes bf16 into
// ``out``). Both empty reproduces the original single-launch behaviour.
void moe_finalize_fuse_shared(
    TensorView out,
    TensorView gemm2_out,
    TensorView expanded_idx_to_permuted_idx,
    TensorView expert_weights,
    TensorView shared_output,
    TensorView acc_in,
    TensorView acc_out,
    int64_t top_k,
    bool enable_pdl) {
  finalizeImpl(
      out, gemm2_out, expanded_idx_to_permuted_idx, expert_weights, shared_output, acc_in, acc_out, top_k, enable_pdl);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_finalize_fuse_shared, moe_finalize_fuse_shared);
