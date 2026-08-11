// Per-expert demand counters for KT margin routing, in one launch.
//
// The swap driver cannot work without these: promotion needs demand for
// non-resident experts, demotion needs traffic served by resident ones. The
// measurement is not optional. Its COST was.
//
// The torch-op form ran roughly eleven kernels per layer per step --
// clamp_min, to(int64), three to(int32), four bitwise ops and three
// scatter_add_ -- about 920 launches per decode step over 92 layers. Decode
// profiling (runs/meta/phaseP.sh) put that at ~5.2% of GPU time and removing
// it moved the step 22.102 -> 19.932 ms, so it was ~10% of decode. The
// arithmetic is trivial; the launches were the whole bill.
//
// Semantics are preserved exactly, including the case the torch form allowed
// where a slot is both insisted and overridden (independent adds, not an
// if/else chain). Counters are integers and atomicAdd is exact, so the
// accumulated values are bit-identical regardless of order -- and routing is
// untouched, so output must stay byte-identical. That is the gate.

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck, div_ceil

#include <sgl_kernel/utils.cuh>  // For LaunchKernel

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

constexpr int kMarginCounterThreads = 128;

/// \brief Fold one forward's routed slots into the three per-expert counters.
///
/// \param insist_count    [E] int32, accumulated in place
/// \param override_count  [E] int32, accumulated in place
/// \param resident_count  [E] int32, accumulated in place
/// \param topk_ids        [n_slots] int64, ORIGINAL router ids (pre-override)
/// \param insist          [n_slots] uint8, slot kept its CPU-resident expert
/// \param overridden      [n_slots] uint8, slot was substituted
__global__ void kt_margin_counters_kernel(
    int32_t* __restrict__ insist_count,
    int32_t* __restrict__ override_count,
    int32_t* __restrict__ resident_count,
    const int64_t* __restrict__ topk_ids,
    const uint8_t* __restrict__ insist,
    const uint8_t* __restrict__ overridden,
    uint32_t n_slots,
    uint32_t n_experts) {
  const uint32_t stride = gridDim.x * blockDim.x;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_slots; i += stride) {
    int64_t e = topk_ids[i];
    const bool routed = e >= 0;
    // clamp_min(0), matching the torch form exactly -- INCLUDING its latent
    // bug. That form clamped the INDEX to 0 but applied the routed mask only
    // to the resident counter, so a masked (-1) slot credits expert 0 in
    // insist/override and is excluded only from resident. That is phantom
    // demand and could promote the wrong expert, but fixing it here would
    // smuggle a behaviour change into a performance change; the point of this
    // kernel is to be provably inert. Filed separately.
    if (!routed) e = 0;
    // Out of range would have faulted the torch scatter_add_. Refuse rather
    // than corrupt a neighbouring counter.
    if (static_cast<uint64_t>(e) >= n_experts) continue;
    const bool ins = insist[i] != 0;
    const bool ovr = overridden[i] != 0;
    // Independent adds, deliberately not if/else: the torch form scattered
    // insist and override separately, so a slot flagged both counted in both.
    // Preserving that keeps the counters bit-identical rather than merely
    // equivalent under an assumption about the override kernel.
    if (ins) atomicAdd(&insist_count[e], 1);
    if (ovr) atomicAdd(&override_count[e], 1);
    if (routed && !ins && !ovr) atomicAdd(&resident_count[e], 1);
  }
}

struct KtMarginCounters {
  static void run(tvm::ffi::TensorView insist_count, tvm::ffi::TensorView override_count,
                  tvm::ffi::TensorView resident_count, tvm::ffi::TensorView topk_ids,
                  tvm::ffi::TensorView insist, tvm::ffi::TensorView overridden) {
    using namespace host;

    SymbolicDevice device_;
    device_.set_options<kDLCUDA>();

    SymbolicSize NE = {"num_experts"};
    TensorMatcher({NE})
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(insist_count)
        .verify(override_count)
        .verify(resident_count);

    SymbolicSize NS = {"num_slots"};
    TensorMatcher({NS}).with_dtype<int64_t>().with_device(device_).verify(topk_ids);
    // uint8 rather than bool: the matcher maps C++ bool to uint8 while torch
    // reports dtype bool, so the caller passes a free .view(torch.uint8).
    TensorMatcher({NS}).with_dtype<uint8_t>().with_device(device_).verify(insist).verify(overridden);

    const uint32_t n_slots = static_cast<uint32_t>(NS.unwrap());
    const uint32_t n_experts = static_cast<uint32_t>(NE.unwrap());
    if (n_slots == 0) return;

    const size_t grid = div_ceil(static_cast<size_t>(n_slots), static_cast<size_t>(kMarginCounterThreads));
    const DLDevice device = device_.unwrap();

    LaunchKernel(grid, kMarginCounterThreads, device)(
        kt_margin_counters_kernel,
        static_cast<int32_t*>(insist_count.data_ptr()),
        static_cast<int32_t*>(override_count.data_ptr()),
        static_cast<int32_t*>(resident_count.data_ptr()),
        static_cast<const int64_t*>(topk_ids.data_ptr()),
        static_cast<const uint8_t*>(insist.data_ptr()),
        static_cast<const uint8_t*>(overridden.data_ptr()),
        n_slots,
        n_experts);
  }
};

}  // namespace
