// Can KERNELS carry the CPU-expert data path instead of Memcpy nodes?
//
// Measured on the shipping doorbell path at margin 10 (copy_cost.py, 8xB200,
// 92 layers, BS1):
//
//   staging D2H   7,360 B   3.26 us   (0.13 us of transfer at 55 GB/s)
//   result H2D    7,168 B   3.52 us   (0.13 us of transfer)
//   merge add               1.50 us
//
// 25x the transfer time, so the cost is the OPERATION, not the payload. And
// every layer pays all three whether or not a single token routed to a CPU
// expert -- at margin 10, none ever does.
//
// A Memcpy node cannot be skipped: a captured graph bakes its size and
// addresses and replays it unconditionally. That is what sent us to CUDA
// conditional nodes, which reject both signalling mechanisms we have (stream
// memops and cudaLaunchHostFunc alike, both cudaErrorInvalidValue at
// capture_end).
//
// But a graph freezes TOPOLOGY, not BEHAVIOUR. A kernel node reads memory at
// replay time and may branch on it. So the data path can become conditional
// with no conditional node at all -- if a kernel moving these bytes over PCIe
// is competitive with the DMA engine. That is what this measures, and the
// answer is not obvious: zero-copy stores are slower per byte, and 25x
// headroom is a reason to test, not a result.
//
// Three kernels, all taking a device predicate they return early on:
//
//   write   device buffer -> mapped host memory   (replaces the staging D2H)
//   merge   mapped host memory -> accumulate into a device tensor
//           (replaces the result H2D AND the separate merge add, fusing two
//            operations into one)
//   touch   predicate only, no data (floor for what a skipped layer costs)

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck

#include <sgl_kernel/utils.cuh>  // For LaunchKernel

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

constexpr int kZcThreads = 256;

/// \brief Copy `n_vec` 16-byte units from device memory to mapped host memory.
///
/// uint4 because the staging block is 7,360 B = 460 uint4 exactly, and wide
/// stores are what make a PCIe write path bearable at all.
__global__ void kt_zc_write_kernel(
    const uint4* __restrict__ src,
    uint4* __restrict__ dst_host,
    int32_t n_vec,
    const int32_t* __restrict__ flag) {
  if (*flag == 0) return;
  int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t stride = gridDim.x * blockDim.x;
  for (; i < n_vec; i += stride) dst_host[i] = src[i];
}

/// \brief dst += src_host, reading straight from mapped host memory.
///
/// This is the fusion: today the CPU's result crosses in a Memcpy H2D and is
/// then added by a second kernel. One kernel can read across the bus and
/// accumulate in the same pass.
__global__ void kt_zc_merge_kernel(
    const __nv_bfloat16* __restrict__ src_host,
    __nv_bfloat16* __restrict__ dst,
    int32_t n,
    const int32_t* __restrict__ flag) {
  if (*flag == 0) return;
  int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t stride = gridDim.x * blockDim.x;
  for (; i < n; i += stride) dst[i] = __hadd(dst[i], src_host[i]);
}

/// \brief Read the predicate and do nothing else: the cost of a skipped layer.
__global__ void kt_zc_touch_kernel(const int32_t* __restrict__ flag, int32_t* __restrict__ sink) {
  if (*flag == 0) return;
  if (threadIdx.x == 0) *sink = 1;
}

}  // namespace

struct KtZeroCopy {
  /// \brief Allocate page-locked host memory the DEVICE can address directly.
  ///
  /// cudaHostAllocMapped is the point: torch's pinned allocator gives memory
  /// the DMA engine can reach, which is not the same as memory a kernel can
  /// dereference. Returns the host pointer; pair it with device_ptr().
  static int64_t alloc_mapped(int64_t nbytes) {
    using namespace host;
    void* h = nullptr;
    RuntimeCheck(cudaHostAlloc(&h, static_cast<size_t>(nbytes), cudaHostAllocMapped) == cudaSuccess,
                 "cudaHostAlloc(mapped) failed");
    return reinterpret_cast<int64_t>(h);
  }

  /// \brief The device-side address of a mapped host allocation.
  static int64_t device_ptr(int64_t host_ptr) {
    using namespace host;
    void* d = nullptr;
    RuntimeCheck(cudaHostGetDevicePointer(&d, reinterpret_cast<void*>(host_ptr), 0) == cudaSuccess,
                 "cudaHostGetDevicePointer failed");
    return reinterpret_cast<int64_t>(d);
  }

  static void free_mapped(int64_t host_ptr) {
    using namespace host;
    RuntimeCheck(cudaFreeHost(reinterpret_cast<void*>(host_ptr)) == cudaSuccess, "cudaFreeHost failed");
  }

  /// \brief Kernel-side staging copy: `src` (device) -> `dst_dev` (mapped host).
  static void write(tvm::ffi::TensorView src, int64_t dst_dev, tvm::ffi::TensorView flag, int64_t blocks) {
    using namespace host;
    const int64_t nbytes = src.numel() * ((src.dtype().bits + 7) / 8);
    RuntimeCheck(nbytes % 16 == 0, "kt_zc write: source must be a multiple of 16 bytes, got ", nbytes);
    SymbolicDevice device_;
    SymbolicSize kOne = {"one"};
    TensorMatcher({kOne}).with_dtype<int32_t>().with_device(device_).verify(flag);
    LaunchKernel(static_cast<int>(blocks), kZcThreads, device_.unwrap())(
        kt_zc_write_kernel,
        static_cast<const uint4*>(src.data_ptr()),
        reinterpret_cast<uint4*>(dst_dev),
        static_cast<int32_t>(nbytes / 16),
        static_cast<const int32_t*>(flag.data_ptr()));
  }

  /// \brief Kernel-side result + merge: `dst` (device) += `src_dev` (mapped host).
  static void merge(int64_t src_dev, tvm::ffi::TensorView dst, tvm::ffi::TensorView flag, int64_t blocks) {
    using namespace host;
    SymbolicDevice device_;
    SymbolicSize kOne = {"one"};
    TensorMatcher({kOne}).with_dtype<int32_t>().with_device(device_).verify(flag);
    LaunchKernel(static_cast<int>(blocks), kZcThreads, device_.unwrap())(
        kt_zc_merge_kernel,
        reinterpret_cast<const __nv_bfloat16*>(src_dev),
        static_cast<__nv_bfloat16*>(dst.data_ptr()),
        static_cast<int32_t>(dst.numel()),
        static_cast<const int32_t*>(flag.data_ptr()));
  }

  /// \brief Predicate-only launch: the floor cost of a layer with no CPU work.
  static void touch(tvm::ffi::TensorView flag, tvm::ffi::TensorView sink) {
    using namespace host;
    SymbolicDevice device_;
    SymbolicSize kOne = {"one"};
    TensorMatcher({kOne}).with_dtype<int32_t>().with_device(device_).verify(flag);
    LaunchKernel(1, kZcThreads, device_.unwrap())(
        kt_zc_touch_kernel,
        static_cast<const int32_t*>(flag.data_ptr()),
        static_cast<int32_t*>(sink.data_ptr()));
  }
};
