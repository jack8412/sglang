// KT hybrid-MoE CPU-branch elision: skip a layer's whole CPU-expert branch
// device-side when nothing in the batch routes to a CPU-resident expert.
//
// Under margin routing a large share of layer-steps route entirely to
// GPU-resident experts. kt-kernel's inline-empty check already makes the
// POLLER cheap for those, but the GPU still pays the entire branch: the
// staging D2H, the doorbell round trip, the result H2D and the merge-add.
// Measured on 8xB200 (c0_conditional_gate): 22.58 us/layer unconditional,
// falling to 15.05 at 38% empty and 8.24 at 100% empty.
//
// Two entities live here:
//
//   KtCpuBranchFlag  computes the predicate: 1 if any routed slot names an
//                    expert this rank does NOT hold on the GPU.
//   KtCondNode       opens and closes a CUDA conditional (IF) node whose body
//                    is captured on a separate stream.
//
// WHY THE NODE IS BUILT WITH THE DRIVER API. sglang captures with
// torch.cuda.CUDAGraph, and torch has no notion of conditional nodes. It does
// not need one: cuStreamGetCaptureInfo interrogates the DRIVER about the
// capture in progress, so the node can be spliced into the graph torch is
// building without torch's participation -- the same door sglang's own
// cuda_graph_dedup_mixin already goes through to drive raw_graph via
// cuda.bindings. Verified end to end in c0b_torch_compose.py: a torch-captured
// graph whose body runs or is skipped per replay from device state.

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck

#include <sgl_kernel/utils.cuh>  // For LaunchKernel

#include <cuda.h>
#include <cuda_runtime.h>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

constexpr int kBranchFlagThreads = 128;

/// \brief flag[0] = 1 if any routed slot names a non-GPU-resident expert.
///
/// One block: the reduction is over qlen*k slots, which is 128 at BS8 top-16
/// and never large enough at decode to want a grid. A second launch to zero
/// the flag would cost more than the scan.
__global__ void kt_cpu_branch_flag_kernel(
    int32_t* __restrict__ flag,
    const int64_t* __restrict__ topk_ids,
    const uint8_t* __restrict__ gpu_mask,
    uint32_t n_slots,
    uint32_t n_experts) {
  __shared__ int32_t s_any;
  if (threadIdx.x == 0) s_any = 0;
  __syncthreads();

  int32_t local = 0;
  for (uint32_t i = threadIdx.x; i < n_slots; i += kBranchFlagThreads) {
    const int64_t e = topk_ids[i];
    // A negative id is a masked-out slot and names no expert. An out-of-range
    // id cannot be proven resident, so it takes the CPU branch rather than
    // reading past the mask.
    if (e < 0) continue;
    if (static_cast<uint64_t>(e) >= n_experts || gpu_mask[e] == 0) {
      local = 1;
      break;
    }
  }
  if (local) atomicOr(&s_any, 1);
  __syncthreads();
  if (threadIdx.x == 0) *flag = s_any;
}

/// \brief Sets a conditional node's predicate from device memory at replay.
///
/// The value must come from device memory, not from a captured constant: a
/// node captured in a CUDA graph writes whatever was recorded, so a host-side
/// predicate would freeze the branch at capture time and every replay would
/// take the same path.
__global__ void kt_set_conditional_kernel(cudaGraphConditionalHandle handle,
                                          const int32_t* __restrict__ flag) {
  cudaGraphSetConditional(handle, static_cast<unsigned int>(*flag));
}

struct KtCpuBranchFlag {
  /// \param flag      [1] int32, device -- receives 0 or 1
  /// \param topk_ids  [qlen, k] int64, device -- routed expert ids
  /// \param gpu_mask  [num_experts] uint8 view of a bool tensor -- nonzero if
  ///                  GPU-resident here. uint8 because the matcher maps C++ bool
  ///                  to uint8 while torch reports dtype bool; the caller passes a
  ///                  free .view(torch.uint8) rather than a copy.
  static void run(tvm::ffi::TensorView flag, tvm::ffi::TensorView topk_ids, tvm::ffi::TensorView gpu_mask) {
    using namespace host;

    SymbolicDevice device_;
    device_.set_options<kDLCUDA>();

    SymbolicSize kOne = {"one"};
    TensorMatcher({kOne}).with_dtype<int32_t>().with_device(device_).verify(flag);
    RuntimeCheck(kOne.unwrap() == 1, "kt_cpu_branch_flag: flag must hold exactly one element, got ",
                 kOne.unwrap());

    SymbolicSize NE = {"num_experts"};
    TensorMatcher({NE}).with_dtype<uint8_t>().with_device(device_).verify(gpu_mask);

    const uint32_t n_slots = static_cast<uint32_t>(topk_ids.numel());
    const uint32_t n_experts = static_cast<uint32_t>(NE.unwrap());
    const DLDevice device = device_.unwrap();

    LaunchKernel(1, kBranchFlagThreads, device)(
        kt_cpu_branch_flag_kernel,
        static_cast<int32_t*>(flag.data_ptr()),
        static_cast<const int64_t*>(topk_ids.data_ptr()),
        static_cast<const uint8_t*>(gpu_mask.data_ptr()),
        n_slots,
        n_experts);
  }
};

/// No-op host callback for the capture probe above.
static void CUDART_CB kt_noop_host_cb(void*) {}

struct KtCondNode {
  /// \brief Splice an IF node into the capture in progress and open its body.
  ///
  /// Work issued on `body_stream` between begin() and end() lands in the IF
  /// body and executes only when `flag` is non-zero at replay.
  ///
  /// \param main_stream  the stream torch is capturing (raw cudaStream_t)
  /// \param body_stream  a stream NOT already capturing; carries the body
  /// \param flag         [1] int32 device predicate, read at replay
  static void begin(int64_t main_stream, int64_t body_stream, tvm::ffi::TensorView flag) {
    using namespace host;

    cudaStream_t ms = reinterpret_cast<cudaStream_t>(main_stream);
    cudaStream_t bs = reinterpret_cast<cudaStream_t>(body_stream);

    cudaStreamCaptureStatus status;
    cudaGraph_t graph = nullptr;
    const cudaGraphNode_t* deps = nullptr;
    const cudaGraphEdgeData* edges = nullptr;
    size_t n_deps = 0;
    RuntimeCheck(cudaStreamGetCaptureInfo(ms, &status, nullptr, &graph, &deps, &edges, &n_deps) == cudaSuccess,
                 "kt_cond_begin: cudaStreamGetCaptureInfo failed");
    // Outside a capture there is no graph to splice into, and the body would
    // silently execute unconditionally -- the branch would look like it works
    // while never skipping anything.
    RuntimeCheck(status == cudaStreamCaptureStatusActive,
                 "kt_cond_begin requires an active capture on the main stream");

    cudaGraphConditionalHandle handle;
    RuntimeCheck(cudaGraphConditionalHandleCreate(&handle, graph, 0, cudaGraphCondAssignDefault) == cudaSuccess,
                 "kt_cond_begin: cudaGraphConditionalHandleCreate failed");

    kt_set_conditional_kernel<<<1, 1, 0, ms>>>(handle, static_cast<const int32_t*>(flag.data_ptr()));

    // Re-read the dependency set: the predicate kernel just extended it, and
    // the IF must depend on that kernel or it could be evaluated first.
    RuntimeCheck(cudaStreamGetCaptureInfo(ms, &status, nullptr, &graph, &deps, &edges, &n_deps) == cudaSuccess,
                 "kt_cond_begin: cudaStreamGetCaptureInfo (post-predicate) failed");

    cudaGraphNodeParams params = {};
    params.type = cudaGraphNodeTypeConditional;
    params.conditional.handle = handle;
    params.conditional.type = cudaGraphCondTypeIf;
    params.conditional.size = 1;
    cudaGraphNode_t node;
    RuntimeCheck(cudaGraphAddNode(&node, graph, deps, edges, n_deps, &params) == cudaSuccess,
                 "kt_cond_begin: cudaGraphAddNode(conditional) failed");
    RuntimeCheck(cudaStreamUpdateCaptureDependencies(ms, &node, nullptr, 1, cudaStreamSetCaptureDependencies) ==
                     cudaSuccess,
                 "kt_cond_begin: cudaStreamUpdateCaptureDependencies failed");

    RuntimeCheck(cudaStreamBeginCaptureToGraph(bs, params.conditional.phGraph_out[0], nullptr, nullptr, 0,
                                               cudaStreamCaptureModeRelaxed) == cudaSuccess,
                 "kt_cond_begin: cudaStreamBeginCaptureToGraph(body) failed");
  }

  /// \brief Test-only: enqueue a no-op host callback on `stream`.
  ///
  /// The doorbell's stream memops cannot be captured into a conditional
  /// body (measured: cudaErrorInvalidValue at capture_end). The host-node
  /// transport signals with cudaLaunchHostFunc instead, and whether THAT can
  /// live in an IF body decides whether dropping the doorbell unblocks
  /// branch elision or leaves it blocked for both transports.
  static void noop_host_func(int64_t stream) {
    using namespace host;
    RuntimeCheck(cudaLaunchHostFunc((cudaStream_t)stream, kt_noop_host_cb, nullptr) == cudaSuccess,
                 "cudaLaunchHostFunc failed");
  }

  /// \brief Close the IF body opened by begin().
  static void end(int64_t body_stream) {
    using namespace host;

    cudaGraph_t body = nullptr;
    RuntimeCheck(cudaStreamEndCapture(reinterpret_cast<cudaStream_t>(body_stream), &body) == cudaSuccess,
                 "kt_cond_end: cudaStreamEndCapture(body) failed");
  }
};

}  // namespace
