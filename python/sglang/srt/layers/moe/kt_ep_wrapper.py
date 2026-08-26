# SPDX-License-Identifier: Apache-2.0
"""
KT Expert Parallelism Wrapper for MoE layers.

This module provides a generic wrapper that enables CPU-GPU expert parallelism
for any MoE quantization method. It coordinates parallel execution of GPU experts
(using any quantization method) and CPU experts (using AMX/AVX instructions).
"""

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Dict, List, Optional

import torch
import torch.distributed as dist

from sglang.srt.distributed import get_tp_group
from sglang.srt.runtime_context import get_buffer, get_parallel, get_stream
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from sglang.srt.utils import get_compiler_backend

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.server_args import ServerArgs

try:
    from kt_kernel import KTMoEWrapper, generate_gpu_experts_masks

    KTRANSFORMERS_AVAILABLE = True
except ImportError:
    KTRANSFORMERS_AVAILABLE = False

# Activations this wrapper accepts (fail-closed allow-list; the kt-kernel side
# keeps the mirrored method-level allow-list below).
KT_ALLOWED_ACTIVATIONS = frozenset({"silu", "situ"})

# Mirror of kt-kernel's SITU_SUPPORTED_METHODS (0.6.1-k3 build): backends whose
# act_fn implements the Kimi-K3 SiTU epilogue. Keep in sync with the wheel.
KT_SITU_SUPPORTED_METHODS = frozenset(
    {
        "AMXINT4",
        "AMXINT8",
        "RAWINT4",
        "FP8",
        "BF16",
        "FP8_PERCHANNEL",
        "GPTQ_INT4",
        "MXFP4",
        "MXFP8",
    }
)

# Canonical KTMoEWrapper ctor contract. The CPU-side test mock builds its
# signature from these constants, and the real wheel is checked against them at
# import time, so mock and wheel cannot drift (Session-2 amendment C1).
KTMOE_WRAPPER_BASE_CTOR_PARAMS = frozenset(
    {
        "layer_idx",
        "num_experts",
        "num_experts_per_tok",
        "hidden_size",
        "moe_intermediate_size",
        "gpu_experts_mask",
        "cpuinfer_threads",
        "threadpool_count",
        "weight_path",
        "chunked_prefill_size",
        "max_deferred_experts_per_token",
        "method",
        "numa_nodes",
        "swiglu_limit",
        "swiglu_alpha",
    }
)
KTMOE_WRAPPER_SITU_CTOR_PARAMS = frozenset({"situ_beta", "situ_linear_beta"})
# Cold-only residency is the only expert-allocation path this build has, so the
# wrapper passes this on every construction. It stays OUT of the base contract
# on purpose: base is enforced with an ImportError at import time, while a
# wheel without cold-only support gets a clear config-time refusal instead --
# which is the difference between a readable message and a TypeError deep in
# scheduler init. Named here so the CPU test mock cannot drift from it.
KTMOE_WRAPPER_COLD_ONLY_CTOR_PARAMS = frozenset({"cold_only_cpu_experts"})

KT_WHEEL_SUPPORTS_SITU = False
# Capability probe rather than a hard requirement: an older wheel still
# serves, it just cannot hold the cold set only. Checked at config time so
# the failure is a clear message instead of a TypeError deep in scheduler
# init -- which is exactly how this first presented.
KT_WHEEL_SUPPORTS_COLD_ONLY = False
if KTRANSFORMERS_AVAILABLE:
    import inspect as _inspect

    _kt_ctor_params = frozenset(
        _inspect.signature(KTMoEWrapper.__new__).parameters
    ) - {"cls"}
    _kt_missing = KTMOE_WRAPPER_BASE_CTOR_PARAMS - _kt_ctor_params
    if _kt_missing:
        raise ImportError(
            f"Installed kt_kernel KTMoEWrapper ctor lacks {sorted(_kt_missing)}; "
            f"the sglang KT integration requires the >=0.6.1 wrapper contract."
        )
    KT_WHEEL_SUPPORTS_SITU = KTMOE_WRAPPER_SITU_CTOR_PARAMS <= _kt_ctor_params
    KT_WHEEL_SUPPORTS_COLD_ONLY = "cold_only_cpu_experts" in _kt_ctor_params


logger = logging.getLogger(__name__)

# Global cache for GPU experts masks (initialized once per session)


@contextmanager
def _scoped_layer_num_local_experts(layer: torch.nn.Module, num_experts: int):
    """Temporarily present the GPU-resident expert count as the layer's
    num_local_experts.

    The KT wrapper allocates the GPU method's weights for only the resident
    subset via the ``num_experts`` argument, but the pin's native
    ``Mxfp4MoEMethod`` (Kimi-K3) sizes its parameters from
    ``layer.num_local_experts`` / ``layer.num_experts`` instead — with the
    full count (896) that
    over-allocates ~40% VRAM and OOMs at construction. The override must be
    scoped: outside the wrapped-method delegations, ``num_local_experts``
    keeps global semantics (the weight loader's early bounds check must see
    the full count or non-contiguous masks would drop experts pre-remap)."""
    original_local = layer.num_local_experts
    original_global = layer.num_experts
    layer.num_local_experts = num_experts
    layer.num_experts = num_experts
    try:
        yield
    finally:
        layer.num_local_experts = original_local
        layer.num_experts = original_global



@dataclass
class KTConfig:
    """Configuration for KTransformers heterogeneous computing CPU part.

    Args:
        layer_idx: Layer index in the model
        gpu_experts_mask: Boolean tensor of shape [num_experts] indicating which experts are on GPU
        cpuinfer_threads: Number of CPU inference threads
        threadpool_count: Number of thread pools for CPU computation
        numa_nodes: Optional explicit NUMA node ids for each KT threadpool
        weight_path: Path to CPU quantized weights
        chunked_prefill_size: Chunk size for prefill computation
        method: CPU computation method (e.g., "int4")
        num_layers: Total number of layers in the model (optional)
        split_prefill_min_tokens: Smallest forward worth paying the
            cold-expert stream for (--kt-expert-split-prefill-min-tokens)
        routing_margin: Per-token budget for GPU-preferred routing overrides,
            as a share of the token's own mixture weight in [0, 1]
            (0.0 = substitute nothing, bit-exact routing, demand/hit counters
            still accumulate; >= 1.0 = no bound, i.e. full override)
        routing_full_override: Override EVERY CPU-resident pick, making the
            layer's CPU path provably dead so it can be skipped statically
    """

    layer_idx: int
    gpu_experts_mask: torch.Tensor  # bool tensor of shape [num_experts]
    cpuinfer_threads: int
    threadpool_count: int
    weight_path: str
    chunked_prefill_size: int
    max_deferred_experts_per_token: int
    method: str
    numa_nodes: Optional[List[int]] = None
    num_layers: Optional[int] = None
    routing_margin: float = 0.0
    routing_full_override: bool = False
    transport: str = "hostnode"
    transport_pollers: int = 2
    expert_swap_transitions: int = 0
    expert_swap_max: int = 4
    expert_swap_hysteresis: float = 2.0
    split_prefill: bool = False
    split_prefill_token_tile: int = 0
    split_prefill_min_tokens: int = 4096


# Every wrapped MoE layer, in construction order, so the swap driver can walk
# them.  Mutated in place only -- no module rebinding.
_KT_EP_METHODS = []
_KT_SWAP_STATE = {"eager_forwards": 0, "windows": 0, "swaps": 0}

# Split-slice full-expert prefill: every wrapped MoE layer that armed the path,
# in construction order, plus the shared cold store and prefetch pipeline built
# once all layers have loaded.  Mutated in place only -- never rebound -- and
# cleared exclusively by reset_split_prefill(), so a second engine in the same
# process cannot inherit stale layers.  (Keying a reset on layer_idx == 0 would
# not work: K3's early layers are dense, so layer 0 is never an MoE layer.)
_KT_SPLIT_PREFILL_LAYERS = []
_KT_SPLIT_PREFILL_STATE = {
    "pipeline": None,
}

# Resident trtllm-gen parameter names.  K3's native Mxfp4MoEMethod consumes
# the trtllm-gen shuffled layout, and its
# ``process_weights_after_loading`` rebinds exactly these four attributes to
# the shuffled weight stacks and interleaved fp8-viewed scale stacks
# (mxfp4.py L827-830).  The swap machinery addresses resident rows by these
# names.
_MXFP4_TRTLLM_RESIDENT_PARAM_NAMES = (
    "w13_weight",
    "w13_weight_scale",
    "w2_weight",
    "w2_weight_scale",
)


class SharedStagingBuffer:
    """Global shared staging buffer for CPU expert input across all MoE layers.

    This avoids allocating a separate staging buffer per layer, which would
    consume significant GPU memory (chunked_prefill_size * hidden_size * N_layers).
    Instead, all layers share a single buffer since MoE layers are processed
    sequentially, not in parallel.
    """

    def __init__(
        self,
        max_tokens: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.max_tokens = max_tokens
        self.hidden_size = hidden_size
        self.buffer = torch.empty(
            (max_tokens, hidden_size),
            dtype=dtype,
            device=device,
        )
        buffer_size_mb = self.buffer.numel() * self.buffer.element_size() / 1024**2
        logger.info(
            f"[KT] Created shared staging buffer: {buffer_size_mb:.1f} MiB "
            f"(shape={self.buffer.shape}, dtype={dtype})"
        )

    def get_slice(self, num_tokens: int) -> torch.Tensor:
        """Get a slice of the buffer for the given number of tokens."""
        assert num_tokens <= self.max_tokens, (
            f"Batch size {num_tokens} exceeds staging buffer max size {self.max_tokens}"
        )
        return self.buffer[:num_tokens]


def get_or_create_shared_staging_buffer(
    max_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> SharedStagingBuffer:
    """Get or create the process-shared staging buffer (resources tier)."""
    return get_buffer(
        "kt_staging",
        lambda: SharedStagingBuffer(
            max_tokens=max_tokens,
            hidden_size=hidden_size,
            dtype=dtype,
            device=device,
        ),
    )


def _all_tp_ranks_succeeded(local_success: bool) -> bool:
    """True only if EVERY TP rank reports success.

    A MIN all-reduce, so one rank's failure disables the feature everywhere.
    The arena/rank-write paths need a decision every rank takes identically:
    a rank that armed while its peers did not would write rows nobody else
    expects, and the divergence is silent.
    """
    if not dist.is_initialized() or get_parallel().tp_size == 1:
        return local_success
    status = torch.tensor([int(local_success)], dtype=torch.int32, device="cpu")
    dist.all_reduce(status, op=dist.ReduceOp.MIN, group=get_tp_group().cpu_group)
    return bool(status.item())


def generate_front_loading_masks(
    num_layers: int,
    num_experts: int,
    num_gpu_experts: int,
    first_k_dense_replace: int,
    moe_layer_freq: int,
) -> torch.Tensor:
    """Generate masks by filling layers from first MoE layer onwards.

    Args:
        num_layers: Total number of layers in the model
        num_experts: Number of experts per layer
        num_gpu_experts: Total number of GPU experts to allocate
        first_k_dense_replace: Layer index where MoE layers start
        moe_layer_freq: Frequency of MoE layers (e.g., 1 = every layer, 2 = every other layer)

    Returns:
        Boolean mask tensor of shape [num_layers, num_experts]
    """
    masks = torch.zeros(num_layers, num_experts, dtype=torch.bool, device="cpu")
    remaining = num_gpu_experts

    for layer_idx in range(num_layers):
        is_moe = layer_idx >= first_k_dense_replace and layer_idx % moe_layer_freq == 0
        if not is_moe:
            # Dense layer - set all True (bypass KT wrapper)
            masks[layer_idx, :] = True
        elif remaining > 0:
            # MoE layer - allocate GPU experts
            num_for_this_layer = min(remaining, num_experts)
            masks[layer_idx, :num_for_this_layer] = True
            remaining -= num_for_this_layer

    return masks


def generate_layer_concentrated_masks(
    num_layers: int,
    num_experts: int,
    num_cpu_layers: int,
    first_k_dense_replace: int,
    moe_layer_freq: int,
) -> torch.Tensor:
    """Whole-layer placement: every layer is fully GPU-resident (all-True,
    which the wrapping predicate leaves unwrapped) except ``num_cpu_layers``
    evenly spaced MoE layers that are fully CPU-resident (all-False).

    The dense prefix and any non-MoE layers stay all-True by construction.
    Rationale: the hybrid per-layer CPU round-trip is paid per LAYER, not per
    expert — concentrating the CPU work into a few all-CPU layers removes the
    round-trip from every other layer entirely.
    """
    masks = torch.ones(num_layers, num_experts, dtype=torch.bool)
    moe_layers = [
        layer_idx
        for layer_idx in range(num_layers)
        if layer_idx >= first_k_dense_replace and layer_idx % moe_layer_freq == 0
    ]
    num_cpu = max(0, min(num_cpu_layers, len(moe_layers)))
    if num_cpu == 0:
        return masks
    positions = sorted(
        {
            min(int((k + 0.5) * len(moe_layers) / num_cpu), len(moe_layers) - 1)
            for k in range(num_cpu)
        }
    )
    if len(positions) != num_cpu:
        raise ValueError(
            f"layer_concentrated spacing collapsed: {num_cpu} CPU layers over "
            f"{len(moe_layers)} MoE layers produced {len(positions)} slots"
        )
    for position in positions:
        masks[moe_layers[position]] = False
    return masks


def generate_uniform_masks(
    num_layers: int,
    num_experts: int,
    num_gpu_experts: int,
    first_k_dense_replace: int,
    moe_layer_freq: int,
) -> torch.Tensor:
    """Generate masks with equal GPU experts per MoE layer.

    Args:
        num_layers: Total number of layers in the model
        num_experts: Number of experts per layer
        num_gpu_experts: Total number of GPU experts to allocate
        first_k_dense_replace: Layer index where MoE layers start
        moe_layer_freq: Frequency of MoE layers

    Returns:
        Boolean mask tensor of shape [num_layers, num_experts]
    """
    masks = torch.zeros(num_layers, num_experts, dtype=torch.bool, device="cpu")

    # Identify MoE layers
    moe_layers = [
        i for i in range(num_layers)
        if i >= first_k_dense_replace and i % moe_layer_freq == 0
    ]
    num_moe_layers = len(moe_layers)

    if num_moe_layers == 0:
        return masks

    # Distribute GPU experts evenly
    experts_per_layer = num_gpu_experts // num_moe_layers
    remainder = num_gpu_experts % num_moe_layers

    for idx, layer_idx in enumerate(moe_layers):
        # First 'remainder' layers get one extra expert
        num_for_this_layer = experts_per_layer + (1 if idx < remainder else 0)
        num_for_this_layer = min(num_for_this_layer, num_experts)
        masks[layer_idx, :num_for_this_layer] = True

    # Set non-MoE layers to all True
    for layer_idx in range(num_layers):
        if layer_idx < first_k_dense_replace or layer_idx % moe_layer_freq != 0:
            masks[layer_idx, :] = True

    return masks


def generate_random_masks(
    num_layers: int,
    num_experts: int,
    num_gpu_experts: int,
    first_k_dense_replace: int,
    moe_layer_freq: int,
    seed: int = 42,
) -> torch.Tensor:
    """Generate masks by randomly selecting GPU experts (fixed seed).

    Args:
        num_layers: Total number of layers in the model
        num_experts: Number of experts per layer
        num_gpu_experts: Total number of GPU experts to allocate
        first_k_dense_replace: Layer index where MoE layers start
        moe_layer_freq: Frequency of MoE layers
        seed: Random seed for reproducibility

    Returns:
        Boolean mask tensor of shape [num_layers, num_experts]
    """
    masks = torch.zeros(num_layers, num_experts, dtype=torch.bool, device="cpu")

    # Collect all MoE (layer, expert) positions
    moe_positions = []
    for layer_idx in range(num_layers):
        is_moe = layer_idx >= first_k_dense_replace and layer_idx % moe_layer_freq == 0
        if is_moe:
            for expert_idx in range(num_experts):
                moe_positions.append((layer_idx, expert_idx))

    # Randomly select positions
    if len(moe_positions) > 0:
        rng = torch.Generator(device='cpu')
        rng.manual_seed(seed)
        num_to_select = min(num_gpu_experts, len(moe_positions))
        selected_indices = torch.randperm(len(moe_positions), generator=rng, device='cpu')[:num_to_select]

        for idx in selected_indices:
            layer_idx, expert_idx = moe_positions[idx]
            masks[layer_idx, expert_idx] = True

    # Set non-MoE layers to all True
    for layer_idx in range(num_layers):
        if layer_idx < first_k_dense_replace or layer_idx % moe_layer_freq != 0:
            masks[layer_idx, :] = True

    return masks


class _ReapStage:
    """One decode step's REAP measurement, staged across layers.

    ``||f||`` does not decompose across TP ranks: each holds a slice of the
    INTERMEDIATE axis, so a rank only ever sees a partial ``f`` and squaring it
    there drops the cross terms. Squares add only across DISJOINT COORDINATES.
    So every layer parks its partial vectors here and one reduce-scatter turns
    the contraction split into a coordinate split: each rank comes back holding
    a subset of ROWS at full width, already summed, and can norm them exactly.

    One collective for the whole step, not one per layer -- at these sizes a
    collective is latency-bound, so 92 small ones cost several times what one
    large one does.

    Sized and allocated at init. Allocating on first use would take the buffers
    from the CUDA graph's private pool, because the first decode-shaped forward
    is the capture itself.
    """

    def __init__(
        self, *, num_layers, num_experts, top_k, hidden, max_tokens, width, device
    ):
        self.num_layers = num_layers
        # Tokens a captured decode graph runs per request: 1, or the verify
        # width when a draft is in play.
        self.token_width = width
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden = hidden
        self.max_tokens = max_tokens

        # max_tokens leads so that vec[:T] is contiguous, which reduce_scatter
        # requires. Layers write a strided [T, top_k, H] slice into it.
        shape = (max_tokens, num_layers, top_k)
        self.vec = torch.zeros(*shape, hidden, dtype=torch.bfloat16, device=device)
        self.ids = torch.zeros(*shape, dtype=torch.int64, device=device)
        self.weights = torch.zeros(*shape, dtype=torch.float32, device=device)
        self.valid = torch.zeros(*shape, dtype=torch.bool, device=device)

        n = max_tokens * num_layers * top_k
        world = get_tp_group().world_size
        self._rs = torch.zeros(n // world, hidden, dtype=torch.bfloat16, device=device)
        self._norms = torch.zeros(n, dtype=torch.float32, device=device)

        self.num = torch.zeros(num_layers, num_experts, dtype=torch.float32, device=device)
        self.count = torch.zeros(num_layers, num_experts, dtype=torch.int32, device=device)
        # kt's half. Written on rank 0 only -- kt lives there -- and broadcast
        # once per window, so the per-step path carries no extra collective.
        self.kt_num = torch.zeros_like(self.num)
        self.kt_count = torch.zeros_like(self.count)
        # kt writes one norm per (layer, token, slot) it computed, and zero for
        # the slots it does not own. Pinned so the per-step copy is a DMA.
        kt_norms = torch.zeros(num_layers, max_tokens, top_k, dtype=torch.float32)
        # Pinned so kt's writes reach the device as a DMA. Not available
        # without CUDA, which is only the case under test.
        self.kt_norms = kt_norms.pin_memory() if torch.cuda.is_available() else kt_norms
        self._row_offset = (
            torch.arange(num_layers, device=device, dtype=torch.int64) * num_experts
        ).view(1, num_layers, 1)

    def stage(self, *, layer_idx, gemm2_out, permuted_idx, weights, served_ids):
        """Park one layer's per-expert outputs. Runs inside the decode graph."""
        tokens = served_ids.shape[0]
        if tokens > self.max_tokens or layer_idx >= self.num_layers:
            return
        idx = permuted_idx.reshape(tokens, self.top_k).to(torch.int64)
        # A slot this launch did not compute -- a CPU expert, masked to -1
        # before dispatch -- has no row to read. Read row 0 and zero it, rather
        # than branching on it: the shapes must stay static under capture.
        valid = idx >= 0
        rows = gemm2_out[torch.where(valid, idx, 0)]
        self.vec[:tokens, layer_idx] = torch.where(valid.unsqueeze(-1), rows, 0)
        # Unconditionally, including the slots this launch did not compute:
        # those are exactly the ones kt served, and the fold below needs their
        # real expert id. `valid` is what keeps them out of the GPU's half.
        self.ids[:tokens, layer_idx] = served_ids.to(torch.int64).clamp_min(0)
        self.weights[:tokens, layer_idx] = weights.reshape(tokens, self.top_k).float()
        self.valid[:tokens, layer_idx] = valid

    def combine(self, real_tokens: int) -> None:
        """Reduce, norm and accumulate one decode step. Runs OUTSIDE the graph.

        ``real_tokens`` is the batch's true size, not the padded one a captured
        replay ran at. Padding rows carry the previous replay's ids, and
        counting them would corrupt the very denominator REAP exists for.
        """
        tokens = min(int(real_tokens), self.max_tokens)
        if tokens <= 0:
            return
        flat = self.vec[:tokens].reshape(-1, self.hidden)
        world = get_tp_group().world_size
        n = flat.shape[0]
        if world > 1:
            out = self._rs[: n // world]
            dist.reduce_scatter_tensor(out, flat, group=get_tp_group().device_group)
            # Each rank now owns whole ROWS of the true sum, so the norm it
            # takes is the real ||f||, not a partial.
            local = out.float().pow(2).sum(-1).sqrt()
            norms = self._norms[:n]
            dist.all_gather_into_tensor(norms, local, group=get_tp_group().device_group)
        else:
            norms = flat.float().pow(2).sum(-1).sqrt()

        norms = norms.view(tokens, self.num_layers, self.top_k)
        valid = self.valid[:tokens]
        contrib = torch.where(valid, self.weights[:tokens] * norms, 0.0)
        index = (self.ids[:tokens] + self._row_offset).reshape(-1)
        self.num.view(-1).scatter_add_(0, index, contrib.reshape(-1))
        self.count.view(-1).scatter_add_(0, index, valid.reshape(-1).to(torch.int32))
        self._combine_kt(tokens, index, valid)

    def _combine_kt(self, tokens: int, index: torch.Tensor, gpu_valid: torch.Tensor) -> None:
        """Fold the experts kt served into rank 0's half.

        The GPU cannot see these: a non-resident expert -- every promotion
        candidate -- never runs there. kt reports the norm of the summed vector
        per (token, slot); the weight and the id are the ones already staged,
        so the two halves are the same statistic by construction.
        """
        if get_tp_group().rank_in_group != 0:
            return
        kt = self.kt_norms[:, :tokens].to(self.num.device, non_blocking=True)
        # [L, T, k] as kt writes it, [T, L, k] as everything else is laid out.
        kt = kt.permute(1, 0, 2)
        # A slot the GPU served is not kt's, and a slot kt skipped reads 0.
        served = (~gpu_valid) & (kt > 0.0)
        contrib = torch.where(served, self.weights[:tokens] * kt, 0.0)
        self.kt_num.view(-1).scatter_add_(0, index, contrib.reshape(-1))
        self.kt_count.view(-1).scatter_add_(0, index, served.reshape(-1).to(torch.int32))

    def take_window(self):
        """This window's totals, and reset. Zeroing is what keeps the
        accumulators from growing until a float add stops landing."""
        num = self.num.to(torch.float64) + self.kt_num.to(torch.float64)
        count = self.count.to(torch.float64) + self.kt_count.to(torch.float64)
        self.num.zero_()
        self.count.zero_()
        self.kt_num.zero_()
        self.kt_count.zero_()
        return num, count


def reap_note_batch(is_decode: bool, num_requests: int) -> None:
    """Combine the PREVIOUS decode step, then remember this one.

    Called at the top of run_batch, before this batch's forward is enqueued, so
    the staging buffers still hold the previous forward's output. A prefill in
    between simply overwrites staging that is never read: only decode is
    measured, because split prefill runs every expert and so says nothing about
    which ones decode will need resident.
    """
    stage = reap_stage()
    if stage is None:
        return
    pending = _KT_REAP_PENDING.get("tokens", 0)
    if pending:
        reap_combine_decode_step(pending)
    _KT_REAP_PENDING["tokens"] = (
        num_requests * stage.token_width if is_decode else 0
    )


def reap_combine_decode_step(real_tokens: int) -> None:
    """Fold one decode step's staged measurement in. Called by the scheduler.

    Outside the captured graph on purpose. A replay runs no Python, so this is
    the only place a collective can be issued per step without being recorded
    into the graph -- and raw torch.distributed collectives are not
    capture-safe anyway (see pynccl_wrapper).

    ``real_tokens`` is the batch's true size. A captured replay pads up to its
    capture size and sglang deliberately leaves the padded tail of input_ids
    unzeroed, so those rows carry the previous replay's tokens and produce real
    routing. Counting them would corrupt the denominator REAP exists for.
    """
    stage = reap_stage()
    if stage is not None:
        stage.combine(real_tokens)


_KT_REAP_PENDING: Dict[str, int] = {}


def _take_reap_window(stage: Optional["_ReapStage"]):
    """This window's totals per layer, agreed by every rank.

    Returns CPU tensors, or None when nothing is being measured.
    """
    if stage is None:
        return None
    num, count = stage.take_window()
    group = get_tp_group()
    if group.world_size > 1:
        # kt's contribution exists on rank 0 alone, and ranks that disagreed on
        # the scores would build different swap plans and let the placements
        # diverge. One broadcast per window, not per step.
        dist.broadcast(num, src=group.first_rank, group=group.device_group)
        dist.broadcast(count, src=group.first_rank, group=group.device_group)
    return num.cpu(), count.cpu()


def reap_stage() -> Optional["_ReapStage"]:
    """The process-wide REAP stage, or None when placement is not adaptive."""
    return get_buffer("kt_reap_stage", dict).get("stage")


def _init_reap_stage(server_args: "ServerArgs", hf_config, num_layers: int) -> None:
    """Allocate the REAP stage once, at init, before any graph capture.

    Every input is read from ServerArgs and the model config, which are
    identical on every rank. Nothing here may consult a rank-0-only object such
    as the kt wrapper: a stage that exists on one rank only would leave its
    collective unmatched and hang the others.
    """
    holder = get_buffer("kt_reap_stage", dict)
    if "stage" in holder:
        return
    holder["stage"] = None
    if server_args.expert_swap_transitions <= 0:
        return

    hidden = getattr(hf_config, "routed_expert_hidden_size", None)
    top_k = getattr(hf_config, "num_experts_per_token", None)
    num_experts = (
        getattr(hf_config, "n_routed_experts", None)
        or getattr(hf_config, "num_local_experts", None)
        or getattr(hf_config, "num_experts", None)
    )
    if not hidden or not top_k or not num_experts:
        logger.warning("[kt-reap] model config lacks the expert geometry; scoring off")
        return

    decode_bs = server_args.cuda_graph_config.decode.bs
    # A captured decode graph runs one token per request only without
    # speculation; with a draft it runs the whole verify width, and a stage
    # sized for the batch alone would silently refuse every forward.
    width = getattr(server_args, "speculative_num_draft_tokens", None) or 1
    max_tokens = max(decode_bs) * width if decode_bs else 0
    if max_tokens <= 0:
        logger.warning("[kt-reap] no decode capture sizes; scoring off")
        return

    holder["stage"] = _ReapStage(
        num_layers=num_layers,
        num_experts=num_experts,
        top_k=top_k,
        hidden=hidden,
        max_tokens=max_tokens,
        width=width,
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    logger.info(
        "[kt-reap] scoring armed: %d layers x %d experts, top_k %d, hidden %d, "
        "max decode batch %d",
        num_layers, num_experts, top_k, hidden, max_tokens,
    )


def _init_kt_gpu_experts_masks(server_args: "ServerArgs") -> Optional[torch.Tensor]:
    """Initialize GPU experts masks from activation frequency data.

    Args:
        server_args: Global server arguments

    Returns:
        Masks tensor of shape [num_layers, num_experts], or None if KT not configured
    """
    holder = get_buffer("kt_gpu_experts_masks", dict)
    if "masks" in holder:
        return holder["masks"]

    # Get model config (unwrap VL configs that nest the text model config)
    # Not get_model_config(): its lazy cache assigns on the instance, which
    # the published-ServerArgs read-only guard forbids when the cache is cold
    # (e.g. under override_server_args in tests). Construct directly instead.
    from sglang.srt.configs.model_config import ModelConfig

    hf_config = ModelConfig.from_server_args(server_args).hf_config

    # fix for kimi-k2.5 models where text_config holds the actual config
    if getattr(hf_config, "text_config", None) is not None:
        hf_config = hf_config.text_config

    num_layers = getattr(hf_config, "num_hidden_layers", None)
    # Try different attribute names for num_experts
    num_experts = getattr(hf_config, "num_local_experts", None)
    if num_experts is None:
        num_experts = getattr(hf_config, "num_experts", None)
    if num_experts is None:
        num_experts = getattr(hf_config, "n_routed_experts", None)

    if num_layers is None or num_experts is None:
        logger.warning(
            "Could not determine num_layers or num_experts from model config."
        )
        return None

    # Get first_k_dense_replace to identify which layers are MoE layers
    first_k_dense_replace = getattr(hf_config, "first_k_dense_replace", 0) or 0
    moe_layer_freq = getattr(hf_config, "moe_layer_freq", 1)

    # NEW (2026-04-29): V4-Flash has hash-MoE layers at the front (num_hash_layers,
    # typically 3) which the HF DeepseekV3Config treats as first_k_dense_replace=3
    # by default. But hash layers DO have routed experts (n_routed_experts=256)
    # — they are NOT dense. Letting generate_uniform_masks set masks[0..2,:] = True
    # for hash layers makes the kt_ep_wrapper send all 256 experts to a GPU MoE
    # that only loaded num_gpu_experts_per_layer worth of weights, triggering
    # the fused_moe Hidden size mismatch assert. Subtract num_hash_layers so
    # hash layers are correctly classified as MoE for mask purposes.
    # Origin: kt-sglang 耦合 (V4-Flash hash-MoE handling in kt_ep_wrapper).
    num_hash_layers = getattr(hf_config, "num_hash_layers", 0) or 0
    if num_hash_layers > 0:
        first_k_dense_replace = max(0, first_k_dense_replace - num_hash_layers)

    # Normalize list-form moe_layer_freq (e.g., MiMo-V2-Flash: [0, 1, 1, ...])
    # to standard (first_k_dense_replace, moe_layer_freq=1) form
    if isinstance(moe_layer_freq, list):
        # Find first MoE layer index from the mask
        first_moe = next((i for i, v in enumerate(moe_layer_freq) if v), 0)
        first_k_dense_replace = max(first_k_dense_replace or 0, first_moe)
        moe_layer_freq = 1

    # Count actual MoE layers
    num_moe_layers = sum(
        1 for i in range(num_layers)
        if i >= first_k_dense_replace and i % moe_layer_freq == 0
    )
    total_experts = num_moe_layers * num_experts
    logger.debug(
        "[kt-mask] num_layers=%d num_experts=%d first_k_dense_replace=%s (type=%s) "
        "moe_layer_freq=%s (type=%s) computed_num_moe_layers=%d "
        "hf_config_class=%s.%s num_hash_layers=%s n_hash_layers=%s",
        num_layers, num_experts,
        first_k_dense_replace, type(first_k_dense_replace).__name__,
        moe_layer_freq, type(moe_layer_freq).__name__,
        num_moe_layers,
        type(hf_config).__module__, type(hf_config).__name__,
        getattr(hf_config, 'num_hash_layers', '<missing>'),
        getattr(hf_config, 'n_hash_layers', '<missing>'),
    )

    # Determine num_gpu_experts (total across all layers)
    if server_args.kt_gpu_experts_ratio is not None:
        # Use ratio to calculate total GPU experts
        num_gpu_experts = int(total_experts * server_args.kt_gpu_experts_ratio)
        if server_args.kt_num_gpu_experts is not None:
            logger.warning(
                f"--kt-gpu-experts-ratio={server_args.kt_gpu_experts_ratio} is set, "
                f"ignoring --kt-num-gpu-experts={server_args.kt_num_gpu_experts}. "
                f"Actual total GPU experts: {num_gpu_experts} "
                f"(= {total_experts} total experts × {server_args.kt_gpu_experts_ratio})"
            )
        else:
            logger.info(
                f"Using kt_gpu_experts_ratio={server_args.kt_gpu_experts_ratio}, "
                f"total GPU experts: {num_gpu_experts} "
                f"(= {total_experts} total experts × {server_args.kt_gpu_experts_ratio})"
            )
    elif server_args.kt_num_gpu_experts is not None:
        # kt_num_gpu_experts is per-layer, multiply by num_moe_layers
        num_gpu_experts = server_args.kt_num_gpu_experts * num_moe_layers
        logger.info(
            f"Using kt_num_gpu_experts={server_args.kt_num_gpu_experts} per layer, "
            f"total GPU experts: {num_gpu_experts} "
            f"(= {server_args.kt_num_gpu_experts} × {num_moe_layers} MoE layers)"
        )
    elif server_args.kt_expert_placement_strategy == "layer_concentrated":
        # Whole-layer placement is derived from kt_num_cpu_layers alone;
        # the per-layer expert count knobs do not apply.
        num_gpu_experts = 0
    else:
        logger.warning("Either kt_num_gpu_experts or kt_gpu_experts_ratio is required but not set.")
        return None

    # Get GPU expert placement strategy
    strategy = server_args.kt_expert_placement_strategy

    # Generate masks based on strategy
    tp_rank = get_parallel().tp_rank

    if strategy == "frequency":
        # Load activation frequency from init_expert_location if it's a .pt file
        init_loc = server_args.init_expert_location
        has_activation_freq = init_loc and init_loc.endswith(".pt")

        if has_activation_freq:
            logger.info("Loading activation frequency from %s", init_loc)
            loaded_data = torch.load(init_loc, map_location="cpu", weights_only=True)
            # Handle both dict format (from ExpertDistributionRecorder) and raw tensor
            if isinstance(loaded_data, dict):
                if "logical_count" in loaded_data:
                    activation_counts = loaded_data["logical_count"]
                else:
                    raise ValueError(
                        f"Loaded dict does not contain 'logical_count' key. "
                        f"Available keys: {list(loaded_data.keys())}"
                    )
            else:
                activation_counts = loaded_data
            # Expected shape: [buffer_size, num_layers, num_experts]
            if activation_counts.dim() != 3:
                raise ValueError(
                    f"Expected activation counts tensor with 3 dims [buffer_size, num_layers, num_experts], "
                    f"got {activation_counts.dim()} dims with shape {activation_counts.shape}"
                )
            _, file_num_layers, file_num_experts = activation_counts.shape
            if file_num_layers != num_layers:
                raise ValueError(
                    f"Activation counts num_layers ({file_num_layers}) doesn't match "
                    f"model num_layers ({num_layers})"
                )
            if file_num_experts != num_experts:
                raise ValueError(
                    f"Activation counts num_experts ({file_num_experts}) doesn't match "
                    f"model num_experts ({num_experts})"
                )
            # Sum across buffer_size (dim0) to get total activation counts per expert
            activation_freq = activation_counts.sum(dim=0).float()  # [num_layers, num_experts]
            logger.info("Using frequency-based strategy with activation frequency data")
        else:
            # No activation frequency file, use zeros (uniform distribution)
            logger.warning(
                "Using frequency-based strategy WITHOUT activation frequency data "
                "(uniform distribution fallback)"
            )
            activation_freq = torch.zeros(num_layers, num_experts, dtype=torch.float32)
            # For layers that are actually MoE layers, set uniform distribution
            for layer_idx in range(num_layers):
                if layer_idx >= first_k_dense_replace and layer_idx % moe_layer_freq == 0:
                    activation_freq[layer_idx, :] = 1.0

        # Generate masks on rank 0
        if tp_rank == 0:
            masks = generate_gpu_experts_masks(activation_freq, num_gpu_experts)
            # For non-MoE layers, set all experts to GPU
            for layer_idx in range(num_layers):
                if layer_idx < first_k_dense_replace or layer_idx % moe_layer_freq != 0:
                    masks[layer_idx, :] = True
        else:
            masks = torch.zeros(num_layers, num_experts, dtype=torch.bool, device="cpu")

    elif strategy == "front-loading":
        if tp_rank == 0:
            logger.info("Using front-loading strategy for GPU expert placement")
            masks = generate_front_loading_masks(
                num_layers, num_experts, num_gpu_experts,
                first_k_dense_replace, moe_layer_freq
            )
        else:
            masks = torch.zeros(num_layers, num_experts, dtype=torch.bool, device="cpu")

    elif strategy == "uniform":
        if tp_rank == 0:
            logger.info("Using uniform strategy for GPU expert placement")
            masks = generate_uniform_masks(
                num_layers, num_experts, num_gpu_experts,
                first_k_dense_replace, moe_layer_freq
            )
        else:
            masks = torch.zeros(num_layers, num_experts, dtype=torch.bool, device="cpu")

    elif strategy == "random":
        if tp_rank == 0:
            logger.info("Using random strategy for GPU expert placement (seed=42)")
            masks = generate_random_masks(
                num_layers, num_experts, num_gpu_experts,
                first_k_dense_replace, moe_layer_freq, seed=42
            )
        else:
            masks = torch.zeros(num_layers, num_experts, dtype=torch.bool, device="cpu")

    elif strategy == "layer_concentrated":
        if tp_rank == 0:
            logger.info(
                "Using layer_concentrated strategy: %d fully-CPU MoE layers, "
                "the rest fully GPU-resident (unwrapped)",
                server_args.kt_num_cpu_layers,
            )
            masks = generate_layer_concentrated_masks(
                num_layers, num_experts, server_args.kt_num_cpu_layers,
                first_k_dense_replace, moe_layer_freq
            )
        else:
            masks = torch.zeros(num_layers, num_experts, dtype=torch.bool, device="cpu")

    else:
        raise ValueError(f"Unknown kt_expert_placement_strategy: {strategy}")

    if dist.is_initialized():
        dist.broadcast(masks, src=0, group=get_tp_group().cpu_group)

    holder["masks"] = masks

    # Log per-layer GPU expert counts (rank 0 only, MoE layers only)
    if tp_rank == 0:
        per_layer_gpu_experts = masks.sum(dim=1).cpu().tolist()
        for layer_idx, num_gpu in enumerate(per_layer_gpu_experts):
            is_moe_layer = (
                layer_idx >= first_k_dense_replace
                and layer_idx % moe_layer_freq == 0
            )
            # Only log for actual MoE layers
            if is_moe_layer:
                logger.info(
                    "KT GPU experts: layer %d (MoE) has %d GPU experts",
                    layer_idx,
                    int(num_gpu),
                )

        # Count total GPU experts only for actual MoE layers
        total_moe_gpu_experts = sum(
            masks[i].sum().item()
            for i in range(num_layers)
            if i >= first_k_dense_replace and i % moe_layer_freq == 0
        )
        num_moe_layers = sum(
            1 for i in range(num_layers)
            if i >= first_k_dense_replace and i % moe_layer_freq == 0
        )
        logger.info(
            "Generated KT GPU experts masks using '%s' strategy: %d MoE layers (out of %d total layers) x %d experts, "
            "total GPU experts in MoE layers = %d",
            strategy, num_moe_layers, num_layers, num_experts, total_moe_gpu_experts
        )

    return holder["masks"]


def create_kt_config_from_server_args(
    server_args: "ServerArgs", layer_idx: int
) -> Optional[KTConfig]:
    """Create KTConfig from ServerArgs if KT is configured.

    Args:
        server_args: Global server arguments
        layer_idx: Layer index in the model

    Returns:
        KTConfig if KT is configured and not disabled, None otherwise
    """
    # Check if KT EP wrapper is disabled (e.g., for draft models in speculative decoding)
    from sglang.srt.layers.moe.utils import is_kt_ep_wrapper_disabled

    if is_kt_ep_wrapper_disabled():
        return None

    if server_args.kt_weight_path is None:
        return None

    # Get GPU experts masks (initializes if needed)
    masks = _init_kt_gpu_experts_masks(server_args)
    if masks is None:
        return None

    # Get num_layers from model config (unwrap VL configs)
    # Not get_model_config(): its lazy cache assigns on the instance, which
    # the published-ServerArgs read-only guard forbids when the cache is cold
    # (e.g. under override_server_args in tests). Construct directly instead.
    from sglang.srt.configs.model_config import ModelConfig

    hf_config = ModelConfig.from_server_args(server_args).hf_config
    if hasattr(hf_config, "text_config"):
        hf_config = hf_config.text_config
    num_layers = getattr(hf_config, "num_hidden_layers", None)
    if num_layers is not None:
        _init_reap_stage(server_args, hf_config, num_layers)

    # NOTE: hash-layer skip experiment was tried here (return None when
    # layer_idx < num_hash_layers); it didn't help because the underlying
    # fused_moe shape-mismatch in V4 hash MoE happens with or without KT wrap.
    # Reverted; root cause is in V4 MoE weight layout vs sglang fused_moe.

    # Get mask for this specific layer
    gpu_experts_mask = masks[layer_idx]

    if bool(gpu_experts_mask.all()):
        # Fully GPU-resident layer: leave the plain quant method unwrapped —
        # no CPU submit, no staging, no per-layer round-trip.  The loader
        # tolerates per-layer absence.
        return None

    return KTConfig(
        layer_idx=layer_idx,
        gpu_experts_mask=gpu_experts_mask,
        cpuinfer_threads=server_args.kt_cpuinfer,
        threadpool_count=server_args.kt_threadpool_count,
        numa_nodes=server_args.kt_numa_nodes,
        weight_path=server_args.kt_weight_path,
        chunked_prefill_size=server_args.chunked_prefill_size,
        method=server_args.kt_method,
        max_deferred_experts_per_token=server_args.kt_max_deferred_experts_per_token,
        num_layers=num_layers,
        routing_margin=server_args.kt_routing_margin,
        routing_full_override=server_args.kt_routing_full_override,
        transport=server_args.kt_transport,
        transport_pollers=server_args.kt_transport_pollers,
        expert_swap_transitions=server_args.kt_expert_swap_transitions,
        expert_swap_max=server_args.kt_expert_swap_max,
        expert_swap_hysteresis=server_args.kt_expert_swap_hysteresis,
        split_prefill=server_args.kt_expert_split_prefill,
        split_prefill_token_tile=server_args.kt_expert_split_prefill_token_tile,
        split_prefill_min_tokens=server_args.kt_expert_split_prefill_min_tokens,
    )


@torch.compile(dynamic=True, backend=get_compiler_backend())
def mask_and_remap_expert_ids(
    topk_ids: torch.Tensor,
    gpu_experts_mask: torch.Tensor,
    logical_to_gpu_index: torch.Tensor,
) -> torch.Tensor:
    """Mask CPU expert IDs and remap GPU expert IDs to weight indices.

    This function:
    1. Sets CPU expert IDs (gpu_experts_mask=False) to -1 so GPU kernel skips them
    2. Remaps GPU expert IDs to GPU weight indices (0 to num_gpu_experts-1)

    Args:
        topk_ids: Tensor of shape [num_tokens, top_k] containing logical expert IDs
        gpu_experts_mask: Boolean tensor of shape [num_experts] where True indicates GPU expert
        logical_to_gpu_index: Int tensor of shape [num_experts] mapping logical ID to GPU index

    Returns:
        Remapped topk_ids tensor with GPU indices for GPU experts, -1 for CPU experts
    """
    is_gpu_expert = gpu_experts_mask[topk_ids]
    # For GPU experts: remap to GPU weight index; for CPU experts: set to -1
    remapped_ids = torch.where(is_gpu_expert, logical_to_gpu_index[topk_ids], -1)
    return remapped_ids


def _weight_budget_override_slots(
    cpu_routed: torch.Tensor,
    topk_weights: torch.Tensor,
    routed: torch.Tensor,
    budget,
) -> torch.Tensor:
    """CPU-resident slots to substitute: smallest weight first, until the
    token's substituted SHARE of its own mixture would exceed ``budget``.

    A layer computes ``y = sum_j w_j * f_{e_j}(x)``, so replacing the expert in
    slot j costs exactly ``w_j * ||f_sub(x) - f_orig(x)||`` -- linear in that
    slot's weight, and independent of the router logit that put it there.
    Bounding the summed weight of the substituted slots therefore bounds how
    far the layer's output can move, per token, whatever the routing looks
    like.

    This REPLACED a router-logit rule ("override when the pick leads the best
    unselected resident by less than the margin"), which bounded nothing.  Its
    comparison point sat at the top-k selection boundary by construction --
    with 620 residents and ~11 of a token's 16 picks already resident, the best
    unselected resident is around global rank 17-20 -- so leads for lower-ranked
    slots were near zero whatever the placement.  The consequence was inverted
    aggression: on a flat-routing token every lead collapsed and the whole CPU
    tail was substituted (~31% of the mixture), precisely when the model is
    blending many experts and substitution costs most, while a peaked token had
    ~2% of its mass substituted.  It also compared in raw-logit space although
    the router selects in ``sigmoid(logit) + bias`` space and weights in
    ``sigmoid(logit)`` space, where one fixed logit gap spans a 55x range of
    score gaps -- so a single scalar meant different things per token and per
    layer, across all 92 of them.

    Smallest-first is the greedy that maximises how many slots leave the slow
    CPU path for a fixed error budget, so the budget is spent where it buys the
    most speed.  WHICH slots are chosen does not change the bound (it is linear
    in the total substituted weight), so a separate per-slot ceiling would slow
    the path down without tightening anything and is deliberately absent.
    ``cumsum <= budget`` already implies every substituted slot carries at most
    ``budget`` on its own.

    The budget is a SHARE: it is compared against the token's own routed weight
    sum rather than against 1.0.  K3 renormalises only when
    ``config.moe_renormalize`` says so, and ``routed_scaling_factor`` may or may
    not already be folded into the weights depending on
    ``apply_routed_scaling_factor_on_output`` (kimi_k3.py:433-455) -- neither of
    which is visible from here.  Dividing by the token's own total makes one
    threshold mean the same thing under all of them, and across all 92 layers.
    """
    w = topk_weights.float()
    zero = torch.zeros_like(w)
    w_routed = torch.where(routed, w, zero)
    # clamp_min keeps an all-unrouted row from dividing by zero; such a row
    # holds no cpu_routed slot to substitute anyway, so the value is unused.
    total = w_routed.sum(dim=-1, keepdim=True).clamp_min(1e-20)

    # Non-candidates sort last and add nothing to the running total, so the
    # greedy walks CPU-resident slots in weight order whatever slot order the
    # router emitted -- this route runs ``topk(..., sorted=False)``
    # (topk.py:1320-1325), so slot order carries no information at all.
    key = torch.where(cpu_routed, w_routed, torch.full_like(w, float("inf")))
    order = torch.argsort(key, dim=-1)
    key_sorted = torch.gather(key, -1, order)
    cand = torch.isfinite(key_sorted)
    spend = torch.cumsum(torch.where(cand, key_sorted, zero), dim=-1)

    # ``>= 1.0`` is the documented no-bound endpoint and has to hold EXACTLY,
    # so it saturates rather than relying on the comparison: ``spend``
    # accumulates in ascending weight order while ``total`` sums in slot order,
    # and on a token whose entire routed set is CPU-resident the two are the
    # same quantity summed differently. They can disagree by an ulp, which
    # strands the largest slot as an insist and re-ranks every stand-in after
    # it.
    if isinstance(budget, torch.Tensor):
        # Per-token budgets. The ``> 0`` term carries the count-only contract
        # PER TOKEN: with a scalar budget the caller enforces "0.0 means route
        # exactly" by skipping the rewrite for the whole batch, which cannot
        # express one request at 0.0 beside another at 0.10. Without this term
        # such a request would still have its zero-weight slots substituted.
        b = budget.reshape(-1, 1)
        within = (spend <= b * total) | (b >= 1.0)
        take_sorted = cand & within & (b > 0.0)
    else:
        within = cand if budget >= 1.0 else (spend <= budget * total)
        take_sorted = cand & within

    # Back to slot order. int8 rather than a bool scatter: bool ``scatter`` is
    # not uniformly lowered by the compile backends this file runs under.
    take = torch.zeros_like(key_sorted, dtype=torch.int8).scatter(
        -1, order, take_sorted.to(torch.int8)
    )
    return take != 0


def _reweight_for_new_expert_set(
    topk_weights: torch.Tensor,
    routed: torch.Tensor,
    override_slots: torch.Tensor,
    orig_logit: torch.Tensor,
    sub_logit: torch.Tensor,
) -> torch.Tensor:
    """Router weights for the post-substitution set of experts.

    Once a CPU-resident pick is dropped and a resident stands in for it, the
    layer is a mixture over a DIFFERENT set of experts, so the weights must be
    the router's weights for THAT set: every member at its own gate,
    renormalised across the set.  This is REAP's rule -- recompute the top-k
    weights with the dropped expert excluded -- applied per token and per layer
    instead of once at prune time.

    Letting the stand-in inherit the weight of the expert it replaced is what
    this exists to avoid.  The stand-in sits BELOW the top-k cut, so inheriting
    runs a below-cut expert at an above-cut weight; and an expert's output is
    fitted to the gate it trains under, since the loss only ever sees the
    product ``g_k * f_k`` and the two co-adapt.  Each instance is small, but the
    sign is always the same and there are 92 layers to accumulate it.

    Computed as a ratio on the existing weights rather than by re-running the
    router over a masked score vector.  The two give the same numbers -- the
    surviving picks keep their scores and only lose competitors, so a fresh
    top-k returns exactly this set -- but the ratio needs neither the router's
    normaliser (never passed down here) nor any assumption about whether
    ``moe_renormalize`` was on or ``routed_scaling_factor`` is already folded
    in, and it leaves every slot in place, which the packed-topk and doorbell
    staging paths downstream would rather it did.

    Renormalising back to the row's ORIGINAL routed sum rather than to 1.0 is
    what preserves both of those, for the same reason the budget is a share.
    With no overrides every ratio is exactly 1.0, both sums are the same
    expression over the same values, and the weights come back bit-identical --
    which is what keeps the count-only contract exact.
    """
    w = topk_weights.float()
    zero = torch.zeros_like(w)
    # sigmoid IS the gate on this route: K3 routes through DSv3 noaux_tc, which
    # scores with sigmoid and adds the correction bias for SELECTION only, so
    # the weight of expert i is sigma(logit_i) over the selected set's sum
    # (topk.py:1287-1350). Ratios of sigmoids therefore need no bias term.
    g_orig = torch.sigmoid(orig_logit).clamp_min(1e-20)
    g_sub = torch.sigmoid(sub_logit)
    ratio = torch.where(override_slots, g_sub / g_orig, torch.ones_like(w))

    scaled = torch.where(routed, w * ratio, zero)
    old_sum = torch.where(routed, w, zero).sum(dim=-1, keepdim=True)
    new_sum = scaled.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    renormed = scaled * (old_sum / new_sum)
    # Unrouted slots keep whatever they held: their weight is never read, and
    # rewriting it would only invite a downstream consumer to start trusting it.
    return torch.where(routed, renormed, w).to(topk_weights.dtype)


def _margin_override_topk_ids_impl(
    topk_ids: torch.Tensor,
    router_logits: torch.Tensor,
    topk_weights: torch.Tensor,
    correction_bias: Optional[torch.Tensor],
    gpu_experts_mask: torch.Tensor,
    budget,
    full_override: bool = False,
) -> tuple:
    """GPU-preferred top-k rewrite (SPEC-MARGIN-ROUTING P1).

    For each routed slot holding a CPU-resident expert, decide whether to
    rewrite it to a GPU-resident stand-in (an "override") or leave it on the
    slow CPU path (an "insist").  The slots are chosen by
    ``_weight_budget_override_slots``: smallest mixture weight first, until the
    token's substituted share would exceed ``budget``.

    ``budget`` is either a float (the server default for every token) or a
    [num_tokens] tensor of per-request overrides, in units of the token's own
    mixture weight: 0.0 substitutes nothing (count-only), >= 1.0 places no
    bound at all and so coincides with ``full_override``.

    The i-th overridden slot of a token takes the token's i-th best unselected
    resident, so multiple overrides in one token land on distinct experts.  The
    weights are then recomputed for the resulting set of experts by
    ``_reweight_for_new_expert_set`` -- REAP's rule: drop the CPU expert and
    take the router's weights over the set that remains, each member at its own
    gate.  The substitute does NOT inherit the weight of the expert it
    replaced.

    Stand-ins are ranked in the router's own SELECTION space,
    ``sigmoid(logit) + correction_bias``, so the promoted expert is the one the
    router itself would pick next out of the resident pool.  ``correction_bias``
    is bound by the model (kimi_k3._bind_kt_correction_bias) and is None for a
    router that has none, in which case the ranking falls back to the raw logit
    -- the same order, since sigmoid is monotone.

    Dropping the bias here was measurably wrong, not a nicety: it is a
    load-balancing term the model TRAINED under, so ranking without it
    systematically promotes the experts the router was taught to under-select.
    Measured on random routing, 56% of tokens received a different stand-in at
    a bias sd of 0.05 and 77% at 0.15, with mean gate differences of 0.06 and
    0.18 -- because stand-ins come from the top-k boundary, where scores are
    packed tightly enough for a small bias to reorder them.

    Pure tensor ops over the LIVE ``gpu_experts_mask`` -- CUDA-graph
    capturable, and replays follow in-place mask updates (same contract as
    ``make_placement_aware_deferred_selector``).

    Returns:
        (new_topk_ids, new_topk_weights, insist_slots, override_slots) -- the
        masks are per-slot bools aligned with the ORIGINAL topk_ids (true
        router preference), and both tensors must be applied together: ids
        without weights would run the new expert set at the old set's weights.
    """
    safe_ids = topk_ids.clamp_min(0).to(torch.int64)
    routed = topk_ids >= 0
    cpu_routed = ~gpu_experts_mask[safe_ids] & routed

    logits = router_logits.float()
    neg_inf = float("-inf")
    if correction_bias is None:
        # No bias to apply, and sigmoid is monotone, so the raw logit already
        # gives the router's order. Skipping the sigmoid keeps this path
        # exactly as cheap as it was.
        sel_scores = logits
    else:
        sel_scores = torch.sigmoid(logits) + correction_bias.to(logits.dtype)
    resident_scores = sel_scores.masked_fill(~gpu_experts_mask.unsqueeze(0), neg_inf)
    # An already-selected expert (resident picks included) is not an
    # alternative: substituting a duplicate would only re-weight it.
    resident_scores = resident_scores.scatter(-1, safe_ids, neg_inf)

    k = topk_ids.shape[-1]
    alt_scores, alt_ids = torch.topk(resident_scores, k=k, dim=-1)
    # Without a real resident alternative there is nothing to override to, and
    # dropping this rail would let topk hand back an expert from the -inf pool
    # -- i.e. a CPU expert again.
    finite_alt = torch.isfinite(alt_scores[:, :1])
    if full_override:
        # Static Python bool: Dynamo specializes it into its own graph, so the
        # budget greedy is compiled out rather than run against a sentinel.
        override_slots = cpu_routed & finite_alt
    else:
        override_slots = (
            _weight_budget_override_slots(cpu_routed, topk_weights, routed, budget)
            & finite_alt
        )

    # Rank overridden slots within each token, then drop any slot whose
    # assigned alternative is -inf (fewer unselected residents than
    # overrides — degenerate layers only; production keeps residents >> k).
    # Dropping here can only REDUCE the substituted mass, so the per-token
    # bound survives it.
    alt_rank = (torch.cumsum(override_slots.to(torch.int64), dim=-1) - 1).clamp_min(0)
    override_slots = override_slots & torch.isfinite(
        torch.gather(alt_scores, -1, alt_rank)
    )
    insist_slots = cpu_routed & ~override_slots

    substitute = torch.gather(alt_ids, -1, alt_rank)
    new_topk_ids = torch.where(
        override_slots, substitute.to(topk_ids.dtype), topk_ids
    )
    # The stand-in's WEIGHT comes from its own gate, sigmoid(logit), with the
    # bias excluded exactly as the router excludes it from topk_weights. So the
    # logit is gathered fresh here rather than read off alt_scores, which now
    # holds SELECTION scores -- reusing them would fold the load-balancing term
    # into the mixture weight, which the router never does.
    new_topk_weights = _reweight_for_new_expert_set(
        topk_weights,
        routed,
        override_slots,
        torch.gather(logits, -1, safe_ids),
        torch.gather(logits, -1, substitute),
    )
    return new_topk_ids, new_topk_weights, insist_slots, override_slots


margin_override_topk_ids = torch.compile(
    dynamic=True, backend=get_compiler_backend()
)(_margin_override_topk_ids_impl)


def make_placement_aware_deferred_selector(gpu_experts_mask_cuda: torch.Tensor):
    """Build a deferral selector that only defers CPU-resident experts.

    Drop-in replacement for the kt wheel's ``select_deferred_experts``
    (same ``(expert_ids, expert_scores, protected_k) -> (immediate,
    deferred)`` contract, ``-1``-masked tensors).  The wheel's default
    protects the top-``protected_k`` by routing score and defers the rest
    regardless of placement; with most experts GPU-resident the deferred
    slots then fall mostly on experts the CPU skips anyway and deferral
    saves almost nothing.  This selector defers the ``topk - protected_k``
    LOWEST-score experts among the token's CPU-resident ones, moving real
    CPU work off the per-layer critical path.

    Reads the live per-layer ``gpu_experts_mask_cuda`` (updated in place by
    dynamic promotion) with pure tensor ops, so it is CUDA-graph capturable
    and follows mask updates at replay.
    """

    def select(
        expert_ids: torch.Tensor,
        expert_scores: torch.Tensor,
        protected_k: int,
    ):
        topk = expert_ids.shape[-1]
        defer_budget = topk - max(0, min(int(protected_k), topk))
        if defer_budget <= 0:
            return expert_ids, None
        safe_ids = expert_ids.clamp_min(0)
        cpu_routed = ~gpu_experts_mask_cuda[safe_ids] & (expert_ids >= 0)
        candidate_scores = expert_scores.masked_fill(
            ~cpu_routed, float("inf")
        )
        defer_slots = torch.topk(
            candidate_scores, k=min(defer_budget, topk), dim=-1, largest=False
        ).indices
        picked_scores = torch.gather(candidate_scores, -1, defer_slots)
        deferred_mask = torch.zeros_like(expert_ids, dtype=torch.bool)
        # Tokens with fewer CPU-routed experts than the budget picked +inf
        # placeholders — keep those slots immediate.
        deferred_mask.scatter_(-1, defer_slots, torch.isfinite(picked_scores))
        immediate_ids = expert_ids.masked_fill(deferred_mask, -1)
        deferred_ids = expert_ids.masked_fill(~deferred_mask, -1)
        return immediate_ids, deferred_ids

    return select


class KTEPWrapperMethod(FusedMoEMethodBase):
    """Wrapper for any MoE quantization method to enable CPU-GPU expert parallelism.

    This wrapper coordinates parallel execution of:
    - GPU experts (identified by gpu_experts_mask=True) using any quantization method
    - CPU experts (identified by gpu_experts_mask=False) using AMX/AVX instructions

    The wrapper implements the submit-compute-sync pattern:
    1. Submit CPU expert computation (non-blocking)
    2. Execute GPU expert computation in parallel
    3. Synchronize and merge CPU+GPU results

    Example:
        # Wrap any GPU method with AMX/AVX CPU expert support
        gpu_method = CompressedTensorsWNA16MoEMethod(quant_config, prefix)
        kt_config = KTConfig(layer_idx=0, gpu_experts_mask=mask, ...)
        method = KTEPWrapperMethod(gpu_method, kt_config)
    """

    # Warn once per process, not once per layer per step.
    _kt_counter_fallback_warned: bool = False
    _reap_seam_warned: bool = False
    # Keyed by CAUSE, not a single bool, and per-process rather than per-layer.
    # A bool is burned by the first emission -- which under decode-graph capture
    # happens before any request exists -- and then every real request is
    # dropped in silence. 92 copies of one cause still say nothing extra.
    _per_request_budget_warned: set = set()

    # Tag for quant_method_registry.is_wrapped_method() — set as a class
    # attribute so isinstance-style checks in deepseek_v2 / glm4_moe work
    # without importing this module.
    _quant_wrapper_id = "kt_ep"

    def __init__(
        self,
        gpu_method: FusedMoEMethodBase,
        kt_config: KTConfig,
    ):
        """Initialize the KT EP wrapper.

        Args:
            gpu_method: The quantization method to use for GPU experts
            kt_config: Configuration for KT CPU expert computation
        """
        if not KTRANSFORMERS_AVAILABLE:
            raise ImportError(
                "kt_kernel is not installed. To use KTransformers EP wrapper, please install kt_kernel."
            )

        self.gpu_method = gpu_method
        self.kt_config = kt_config

        self.gpu_experts_mask = kt_config.gpu_experts_mask  # bool tensor [num_experts], on CPU
        self.num_gpu_experts = int(self.gpu_experts_mask.sum().item())
        self.override_num_local_experts = True
        self.gpu_method.num_gpu_experts = self.num_gpu_experts
        self.tp_rank = get_parallel().tp_rank
        # Debug/kill-switch env knobs, snapshotted once (read on the hot path).
        # Doorbell transport: one slot per (layer, BATCH SIZE), assigned on
        # this layer's first forward at each size. Not one per layer:
        # KExpertsCPUBuffer keys its rings by batch size, so a single
        # per-layer slot would point the poller at another tier's buffers for
        # every size but the first -- silently, since the shapes match.
        self._db_enabled = kt_config.transport == "doorbell"
        self._db_slots: Dict[int, int] = {}
        # Packed staging on the host-node path: pack activations, ids and
        # weights into one block on the MAIN stream before forking, then a
        # single D2H after it. The three separate copies it replaces were all
        # issued on the CPU stream AFTER the fork, so the dispatch reached the
        # poller only once they landed, which exposes the whole CPU latency
        # because no GPU work is left to hide behind.
        self._fused_enabled = kt_config.transport == "hostnode"
        # Per-expert demand counters are maintained only where something reads
        # them: the swap policy, and the full-override falsification check
        # (which needs to see an insist survive a routing that claims none can).
        # They cost real decode time, so "always on" is not free.
        self._counters_enabled = bool(
            kt_config.expert_swap_transitions > 0 or kt_config.routing_full_override
        )
        # Margin routing (SPEC-MARGIN-ROUTING P1). None = off, bit-exact.
        # A per-token budget: the share of a token's own mixture weight that
        # substitution may move, at each layer.
        self._margin = kt_config.routing_margin
        # The router's noaux_tc selection bias, bound by the model after
        # construction (kimi_k3._bind_kt_correction_bias) because it lives on
        # the gate module and does not travel with the dispatch output. None
        # for a router without one; the stand-in search then ranks by logit,
        # which is the same order.
        self.correction_bias: Optional[torch.Tensor] = None
        self._full_override = kt_config.routing_full_override
        # Armed in create_weights once num_gpu_experts and top_k are known.
        self._skip_cpu_path = False
        # Split-slice full-expert prefill.  Armed in create_weights once the
        # cold store and pipeline exist; _split_prefill_ready gates the hot
        # path so a partially-built config cannot half-enter it.
        self._split_prefill = kt_config.split_prefill
        self._split_prefill_ready = False
        # Set on the wrapper-owning rank once kt loads this layer's CPU
        # weights; the kt-RAM expert source needs it to turn logical expert ids
        # into physical buffer slots.
        self._kt_physical_to_logical = None
        # Break-even against the CPU-expert path, NOT the chunk size. The
        # split path costs a fixed per-forward stream (every cold expert lands
        # once however many tokens the chunk holds), so it wins above roughly
        # stream_seconds x cpu_tokens_per_second. Gating on
        # chunked_prefill_size instead would qualify only an exactly-full
        # chunk, dropping every remainder onto the slow path.
        self._split_prefill_threshold = kt_config.split_prefill_min_tokens
        # Cap the MoE's per-call transients by running it in token tiles. Both
        # scale with tokens -- the gemm2 buffer the kernel sizes for all
        # T*top_k slots, and the fp32 accumulator -- while a token's output
        # depends only on its own row, so tiling changes no value. 0 disables.
        self._split_prefill_token_tile = kt_config.split_prefill_token_tile or None
        self._cold_pipeline = None
        self._cold_scalars = None
        self._margin_insist_count: Optional[torch.Tensor] = None
        self._margin_override_count: Optional[torch.Tensor] = None
        self._margin_format_warned = False
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[kt-wrap-init] tp_rank=%d layer_idx=%s num_gpu_experts=%d "
                "mask_sum=%d mask_shape=%s gpu_method=%s",
                self.tp_rank,
                kt_config.layer_idx,
                self.num_gpu_experts,
                int(self.gpu_experts_mask.sum().item()),
                tuple(self.gpu_experts_mask.shape),
                type(gpu_method).__name__,
            )

        # Mapping tables for non-contiguous GPU expert allocation (CPU tensors)
        # Used by weight_loader to remap expert_id when loading weights
        gpu_expert_indices = torch.where(self.gpu_experts_mask)[0]
        self.logical_to_gpu_index = torch.full(
            (len(self.gpu_experts_mask),), -1, dtype=torch.int32
        )
        self.logical_to_gpu_index[gpu_expert_indices] = torch.arange(
            len(gpu_expert_indices), dtype=torch.int32
        )
        self.gpu_index_to_logical = gpu_expert_indices.to(torch.int32)

        # Split-slice prefill (expert_split_moe) addresses ALL experts in one
        # slot space: residents keep their dense indices [0, num_gpu), and the
        # cold experts follow at [num_gpu, num_experts) in ascending logical
        # order -- the same torch.where ordering the residents get, so both
        # halves derive from one pass over the mask.  Unlike
        # logical_to_gpu_index this is a bijection: no -1, nothing dropped.
        cold_expert_indices = torch.where(~self.gpu_experts_mask)[0]
        self.logical_to_slot = torch.empty(
            len(self.gpu_experts_mask), dtype=torch.int32
        )
        self.logical_to_slot[gpu_expert_indices] = torch.arange(
            len(gpu_expert_indices), dtype=torch.int32
        )
        self.logical_to_slot[cold_expert_indices] = torch.arange(
            len(gpu_expert_indices),
            len(gpu_expert_indices) + len(cold_expert_indices),
            dtype=torch.int32,
        )
        self.cold_index_to_logical = cold_expert_indices.to(torch.int32)

        # CUDA tensors for inference (will be set in create_weights)
        self.gpu_experts_mask_cuda = None
        self.logical_to_gpu_index_cuda = None
        self.logical_to_slot_cuda = None

        self._full_init_args = None
        self.wrapper: Optional[KTMoEWrapper] = None

        # Dual-stream parallelism: cpu_stream for CPU expert operations,
        # main stream for GPU computation (initialized in create_weights)
        self._cpu_stream: Optional[torch.cuda.Stream] = None
        self._sync_done_event: Optional[torch.cuda.Event] = None  # CPU computation done

        # Shared staging buffer reference (initialized in create_weights, shared across all layers)
        self._shared_staging_buffer: Optional[SharedStagingBuffer] = None
        self._staging_buffer_max_size: int = kt_config.chunked_prefill_size or 8192

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weights for both GPU and CPU experts.

        Args:
            layer: The MoE layer module
            num_experts: Total number of experts (GPU + CPU)
            hidden_size: Hidden dimension size
            intermediate_size_per_partition: Intermediate size per TP partition
            params_dtype: Data type for parameters
            **extra_weight_attrs: Additional weight attributes
        """
        self.global_num_experts = num_experts
        self._full_init_args = (
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
        )

        # Get required parameters from layer object
        # top_k: number of experts selected per token
        num_experts_per_tok = layer.top_k

        # intermediate_size_full: full intermediate size before TP partitioning
        intermediate_size_full = (
            layer.intermediate_size_per_partition * layer.moe_tp_size
        )

        layer_max_deferred = self.kt_config.max_deferred_experts_per_token or 0
        if (
            self.kt_config.max_deferred_experts_per_token is not None
            and self.kt_config.num_layers is not None
            and self.kt_config.layer_idx == self.kt_config.num_layers - 1
        ):
            layer_max_deferred = 0

        # 1. Create weights for GPU experts using the wrapped method
        # GPU weights are indexed by gpu_index (0 to num_gpu_experts-1), not logical expert ID
        # The mapping logical_to_gpu_index is used to remap IDs during weight loading and inference
        with _scoped_layer_num_local_experts(layer, self.num_gpu_experts):
            self.gpu_method.create_weights(
                layer=layer,
                num_experts=self.num_gpu_experts,
                hidden_size=hidden_size,
                intermediate_size_per_partition=intermediate_size_per_partition,
                params_dtype=params_dtype,
                **extra_weight_attrs,
            )

        # Move mask and mapping tables to GPU for inference
        target_device = next(layer.parameters()).device
        self.gpu_experts_mask_cuda = self.gpu_experts_mask.to(device=target_device)
        self.logical_to_gpu_index_cuda = self.logical_to_gpu_index.to(device=target_device)
        self.logical_to_slot_cuda = self.logical_to_slot.to(device=target_device)

        # Full override arms the static CPU-path skip, but only where the
        # override is PROVABLE: each displaced slot needs its own distinct
        # unselected resident expert, so a layer must hold at least top_k of
        # them.  Below that the op legitimately declines some overrides, the
        # surviving CPU picks get -1'd by mask_and_remap, and with the CPU
        # path skipped their contribution would vanish silently.  Raise, never
        # warn: the failure mode is wrong numbers, not a crash.
        if self._full_override:
            if self.num_gpu_experts < num_experts_per_tok:
                raise ValueError(
                    f"--kt-routing-full-override needs at least top_k="
                    f"{num_experts_per_tok} GPU-resident experts per layer to "
                    f"guarantee every CPU-resident pick can be replaced, but "
                    f"layer {self.kt_config.layer_idx} has "
                    f"{self.num_gpu_experts}. Raise --kt-num-gpu-experts / "
                    f"--kt-gpu-experts-ratio, or drop full override."
                )
            self._skip_cpu_path = True
            logger.info(
                "[kt-margin] layer=%s full override armed: %d/%d experts "
                "GPU-resident, CPU round-trip skipped (staging/submit/sync/"
                "merge)",
                self.kt_config.layer_idx,
                self.num_gpu_experts,
                num_experts,
            )

        # Margin-routing counters: persistent per-layer buffers, accumulated
        # in-place (scatter_add_) so decode CUDA-graph replays keep counting.
        # Cumulative since launch; indexed by ORIGINAL (pre-override) ids.
        #
        # insist + override = demand for a NON-resident expert (promotion
        # candidates); resident_hit = demand actually served on GPU (its
        # inverse picks demotion victims).  Together they are the whole input
        # to the swap policy, so both sides of a swap decision come from the
        # same forward passes and need no extra instrumentation.
        #
        # They are NOT free: three scatter_add_ and several bitwise kernels
        # per layer per step, baked into the captured graph so they run every
        # step forever whether or not anything reads them. Maintained only
        # where something does.
        # Gated on the swap interval, NOT on margin. Demand is
        # `routed & ~gpu_experts_mask[topk_ids]` and hits are its complement --
        # both functions of the routed ids and the residency mask alone. The
        # margin only decides how demand splits into kept-on-CPU vs
        # substituted-to-GPU, and that split cancels in the sum the policy
        # reads (kt_expert_swap.py:283). So swapping does not need margin, and
        # exact routing + adaptive placement is now a legal configuration.
        # See SPEC-SWAP-DEMAND.md.
        if self._counters_enabled:
            # Demand the router asked for and we did NOT substitute away. With
            # margin unset nothing is ever substituted, so this holds all of it;
            # under margin routing it is the "insist" half and the counter below
            # carries the rest. Either way the policy sums the two.
            self._margin_insist_count = torch.zeros(
                num_experts, dtype=torch.int32, device=target_device
            )
            self._margin_override_count = torch.zeros(
                num_experts, dtype=torch.int32, device=target_device
            )

        # Full override computes no CPU expert at all, so none of the CPU-side
        # machinery below is ever used: not the stream, not the staging buffer,
        # not the doorbell, and above all not the expert weights. Loading them
        # anyway cost most of startup and ~1.45 TB of host RAM on K3 (cold-only
        # cannot help -- it is refused without a margin, and full override sets
        # only the internal self._margin), and it evicted the page cache so the
        # NEXT run reloaded cold.
        #
        # self.wrapper deliberately stays None rather than being built and left
        # unloaded: every CPU-path call site already guards on `wrapper is None`
        # (that is how tp_rank != 0 behaves), so None keeps all of them
        # fail-safe. A built-but-unloaded wrapper would read as "CPU path
        # available" and die inside kt-kernel on a null `moe` instead.
        _skip_cpu_side = self._skip_cpu_path

        # Initialize dual-stream for CPU-GPU parallelism (rank 0 only)
        if self.tp_rank == 0 and not _skip_cpu_side:
            self._cpu_stream = get_stream("kt_cpu")
            self._sync_done_event = torch.cuda.Event()

            # Get or create shared staging buffer (shared across all MoE layers to save GPU memory)
            self._shared_staging_buffer = get_or_create_shared_staging_buffer(
                max_tokens=self._staging_buffer_max_size,
                hidden_size=hidden_size,
                dtype=params_dtype,
                device=target_device,
            )

        # 2. Initialize KT wrapper for CPU experts
        # CPU experts are identified by gpu_experts_mask=False
        if self.tp_rank == 0 and not _skip_cpu_side:
            # SwiGLU activation params for CPU experts. Source of truth is
            # MoeRunnerConfig, populated by the model file from HF config:
            #   - minimax_m3.py forwards config.swiglu_alpha / swiglu_limit
            #     as gemm1_alpha / gemm1_clamp_limit (swiglu_oai path)
            #   - deepseek_v2.py forwards config.swiglu_limit into the
            #     legacy swiglu_limit slot (DSV4 plain-silu clamp path)
            # kt-kernel C++ accepts a single (alpha, limit) pair and
            # disambiguates by alpha != 0 (swiglu_oai vs plain silu).
            _mrc = getattr(layer, "moe_runner_config", None)
            _cfg_alpha = getattr(_mrc, "gemm1_alpha", None) if _mrc is not None else None
            _cfg_clamp = getattr(_mrc, "gemm1_clamp_limit", None) if _mrc is not None else None
            _cfg_swglim = getattr(_mrc, "swiglu_limit", None) if _mrc is not None else None
            _kt_swiglu_alpha = float(_cfg_alpha) if _cfg_alpha is not None else 0.0
            _kt_swiglu_limit = float(
                _cfg_clamp if _cfg_clamp is not None else (_cfg_swglim or 0.0)
            )
            # kt-kernel guards swiglu_limit to MXFP4/MXFP8 only.
            # Zero it out for other methods (AMXINT4, BF16, etc.)
            # so V4-Flash + non-MXFP runs don't crash at init.
            if (self.kt_config.method or "").upper() not in ("MXFP4", "MXFP8"):
                _kt_swiglu_limit = 0.0
                _kt_swiglu_alpha = 0.0
            # Kimi-K3 SiTU: a dedicated kt-kernel ctor channel
            # (situ_beta/situ_linear_beta), separate from swiglu_alpha/limit.
            # Values arrive via MoeRunnerConfig gemm1_alpha/gemm1_clamp_limit,
            # populated by kimi_k3.py from config.activation_situ_beta /
            # activation_situ_linear_beta -- never hardcoded here.
            _kt_situ_kwargs = {}
            if getattr(_mrc, "activation", None) == "situ":
                _kt_method_uc = (self.kt_config.method or "").upper()
                if _kt_method_uc not in KT_SITU_SUPPORTED_METHODS:
                    raise ValueError(
                        f"activation='situ' is not supported for --kt-method "
                        f"{self.kt_config.method!r}; kt-kernel implements situ "
                        f"only for {sorted(KT_SITU_SUPPORTED_METHODS)}."
                    )
                if not KT_WHEEL_SUPPORTS_SITU:
                    raise RuntimeError(
                        "activation='situ' requires a kt-kernel wheel whose "
                        "KTMoEWrapper ctor accepts situ_beta/situ_linear_beta "
                        "(custom >=0.6.1-k3 build); the installed wheel does not."
                    )
                if _cfg_alpha is None:
                    raise ValueError(
                        "activation='situ' requires gemm1_alpha "
                        "(config.activation_situ_beta) to be set."
                    )
                _kt_situ_kwargs = dict(
                    situ_beta=float(_cfg_alpha),
                    situ_linear_beta=(
                        float(_cfg_clamp) if _cfg_clamp is not None else 0.0
                    ),
                )
                _kt_swiglu_limit = 0.0
                _kt_swiglu_alpha = 0.0
            common_wrapper_kwargs = dict(
                layer_idx=self.kt_config.layer_idx,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                hidden_size=hidden_size,
                moe_intermediate_size=intermediate_size_full,
                # Read at MOEConfig construction, before kt allocates the
                # per-expert weight buffers -- passing it later would silently
                # allocate all 896 and look like the feature did nothing.
                cold_only_cpu_experts=True,
                gpu_experts_mask=self.gpu_experts_mask,
                cpuinfer_threads=self.kt_config.cpuinfer_threads,
                threadpool_count=self.kt_config.threadpool_count,
                numa_nodes=self.kt_config.numa_nodes,
                weight_path=self.kt_config.weight_path,
                chunked_prefill_size=self.kt_config.chunked_prefill_size,
            )
            self.wrapper = KTMoEWrapper(
                **common_wrapper_kwargs,
                swiglu_limit=_kt_swiglu_limit,
                swiglu_alpha=_kt_swiglu_alpha,
                method=self.kt_config.method,
                max_deferred_experts_per_token=layer_max_deferred,
                **_kt_situ_kwargs,
            )
            self._bind_reap_norms_buffer()
            if layer_max_deferred > 0:
                # The wheel's default deferral selector is placement-blind:
                # it defers the token's lowest-score experts, most of which
                # are GPU-resident and cost the CPU nothing — so the sync
                # still waits for nearly all real CPU work.  Install a
                # selector that defers CPU-resident experts specifically.
                self.wrapper.select_deferred_experts = (
                    make_placement_aware_deferred_selector(
                        self.gpu_experts_mask_cuda
                    )
                )

        # The doorbell page must be allocated and the poller running before any
        # capture: cudaHostAlloc is illegal during capture, and the graph bakes
        # the page's device addresses. Slots are bound later, per batch size --
        # binding allocates no device memory, so it is capture-safe.
        if (
            self.kt_config.transport == "doorbell"
            and self.tp_rank == 0
            and not _skip_cpu_side
        ):
            kt_doorbell_init(self.kt_config.transport_pollers)

        # Swap driver registry: keep the layer with the method, since the
        # weight mover writes into the layer's resident parameter rows.
        # No margin term: demand and hits are functions of the routed ids and
        # the residency mask alone, so a server at the default 0.0 budget still
        # feeds the policy. Requiring a margin here used to leave the driver
        # with nothing to iterate on exactly the exact-routing configuration
        # the counter gate above declares legal.
        if self.kt_config.expert_swap_transitions > 0:
            self._swap_layer = layer
            self._swap_policy = None  # built lazily, needs num_experts
            _KT_EP_METHODS.append(self)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Process weights after loading from checkpoint.

        Args:
            layer: The MoE layer module
        """
        # 1. Process GPU weights.  Fully-CPU layers (layer_concentrated
        # placement) keep their zero-size GPU params unprocessed: the pin's
        # Mxfp4MoEMethod post-load indexes expert row 0, which does not exist
        # at num_gpu_experts == 0, and there is nothing to shuffle anyway.
        if self.num_gpu_experts > 0 and hasattr(
            self.gpu_method, "process_weights_after_loading"
        ):
            with _scoped_layer_num_local_experts(layer, self.num_gpu_experts):
                self.gpu_method.process_weights_after_loading(layer)

        # 1b. Split-slice prefill: the cold slice needs its OWN per-expert
        # scalar vectors.  The kernel indexes gemm1_alpha / gemm1_beta by
        # LOCAL expert index, so a 620-length vector against a 276-expert
        # slice reads out of bounds.  Both are constant-filled per layer, so
        # the cold copies are just the resident value repeated.
        if self._split_prefill and self.num_gpu_experts > 0:
            n_cold = self.global_num_experts - self.num_gpu_experts
            self._cold_scalars = {
                "alpha": layer.gemm1_alpha[:1].repeat(n_cold).contiguous(),
                "beta": layer.gemm1_clamp_limit[:1].repeat(n_cold).contiguous(),
            }
            _register_split_prefill_layer(self, layer)

        # 2. Expert location map, on EVERY rank. Swap-plan ids are kt buffer
        # SLOTS (physical); the checkpoint is logical; this map is the bridge,
        # and every rank owns checkpoint reads (promotion fallback), so every
        # rank needs it -- computed locally from process-global metadata, NOT
        # shipped over the arena share channel, because the ranks that fall
        # back to the checkpoint are exactly the ranks that channel failed.
        from sglang.srt.eplb.expert_location_dispatch import (
            get_global_expert_location_metadata,
        )

        metadata = get_global_expert_location_metadata()
        if (
            metadata is not None
            and getattr(metadata, "physical_to_logical_map_cpu", None) is not None
        ):
            physical_to_logical_map_cpu = (
                metadata.physical_to_logical_map_cpu[self.kt_config.layer_idx]
                .contiguous()
            )
        else:
            # Fallback for setups without EPLB metadata: identity mapping.
            physical_to_logical_map_cpu = torch.arange(
                layer.num_experts, dtype=torch.int64, device="cpu"
            )
        self._kt_physical_to_logical = physical_to_logical_map_cpu.tolist()

        # 3. Load CPU weights using KT wrapper
        if self.tp_rank == 0 and self.wrapper is not None:
            torch.cuda.synchronize()
            self.wrapper.load_weights(physical_to_logical_map_cpu)

        # 4. KT_BUFFER_B_MEMFD: hand every rank a read-only mapping of this
        # layer's kt expert buffers (rank 0 exports memfds, peers map them),
        # so swap-window promotions read kt RAM instead of the checkpoint.
        # Deliberately OUTSIDE the rank-0 gate -- the share is a rendezvous
        # every rank participates in, in the same per-layer order.
        from sglang.srt.layers.moe.kt_arena_share import share_layer_arenas

        share_layer_arenas(method=self)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"
    ):
        """Create MoE runner for computation.

        Args:
            layer: The MoE layer module
            moe_runner_config: Configuration for MoE runner
        """
        self.moe_runner_config = moe_runner_config

        # Create a separate config for GPU method without routed_scaling_factor.
        # This is because:
        # 1. GPU method's moe_sum_reduce would apply routed_scaling_factor internally
        # 2. KT CPU kernel does NOT apply routed_scaling_factor
        # 3. The combined output (GPU + CPU) would have inconsistent scaling
        # 4. routed_scaling_factor is applied uniformly in deepseek_v2.py forward_normal
        # So we disable it in GPU method to avoid double scaling on GPU part.
        gpu_runner_config = replace(moe_runner_config, routed_scaling_factor=None)
        if self.override_num_local_experts:
            gpu_runner_config = replace(
                gpu_runner_config, num_local_experts=self.num_gpu_experts
            )

        # Delegate to GPU method to create its runner
        self.gpu_method.create_moe_runner(layer, gpu_runner_config)

    def _submit_cpu_forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        self.wrapper.submit_forward(
            hidden_states,
            topk_ids,
            topk_weights,
            torch.cuda.current_stream(hidden_states.device).cuda_stream,
        )

    def _sync_cpu_forward(self, ref_tensor: torch.Tensor) -> torch.Tensor:
        return self.wrapper.sync_forward(
            ref_tensor,
            torch.cuda.current_stream(ref_tensor.device).cuda_stream,
        )

    def submit(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> None:
        """Submit CPU expert computation asynchronously (non-blocking).

        This method submits the CPU expert computation to AMX/AVX without waiting
        for completion, allowing GPU computation to proceed in parallel.

        Args:
            layer: The MoE layer module
            dispatch_output: Dispatched tokens and routing information
        """
        if self.tp_rank != 0 or self.wrapper is None:
            return

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights, topk_ids, _ = topk_output

        # Submit forward task to CPU (non-blocking)
        self._submit_cpu_forward(x, topk_ids, topk_weights)

    def sync(self, x: torch.Tensor) -> torch.Tensor:
        """Synchronize and retrieve CPU expert computation results.

        This method waits for the CPU computation to complete and returns the results.

        Args:
            x: Reference tensor for shape and device information

        Returns:
            CPU expert computation results
        """
        if self.tp_rank != 0 or self.wrapper is None:
            return torch.zeros_like(x)

        # Wait for CPU computation and retrieve results
        return self._sync_cpu_forward(x)

    def _submit_with_staged_input(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
        staged_hidden_states: torch.Tensor,
    ) -> None:
        """Submit CPU expert computation using staged hidden states.

        Args:
            layer: The MoE layer module
            dispatch_output: Dispatched tokens and routing information
            staged_hidden_states: Pre-copied hidden states in staging buffer
        """
        _activation = self.moe_runner_config.activation
        if _activation not in KT_ALLOWED_ACTIVATIONS:
            raise ValueError(
                f"KT EP wrapper supports activations "
                f"{sorted(KT_ALLOWED_ACTIVATIONS)}, got {_activation!r}."
            )

        if self.tp_rank != 0 or self.wrapper is None:
            return

        topk_output = dispatch_output.topk_output
        topk_weights, topk_ids, _ = topk_output

        # Submit forward task using staged buffer
        self._submit_cpu_forward(staged_hidden_states, topk_ids, topk_weights)

    def _sync_with_staged_input(
        self, staged_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Synchronize CPU computation using staged hidden states reference.

        Args:
            staged_hidden_states: Staged buffer used in submit

        Returns:
            CPU expert computation results
        """
        if self.tp_rank != 0 or self.wrapper is None:
            return torch.zeros_like(staged_hidden_states)

        return self._sync_cpu_forward(staged_hidden_states)

    # -- split-slice full-expert prefill -----------------------------------

    def _split_prefill_apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
        num_tokens: int,
    ) -> "CombineInput":
        """Compute every routed expert on GPU as two disjoint expert slices.

        The resident weights stay exactly as decode leaves them (a dense
        ``[num_gpu, ...]`` stack indexed by gpu_index); the cold slice comes
        from the prefetch pipeline.  Because both slices are addressed in ONE
        slot space -- residents at ``[0, num_gpu)``, cold at
        ``[num_gpu, num_experts)`` -- topk ids need only a bijective gather,
        not the mask-and-remap decode uses.

        No margin routing, no CPU submit/sync: every expert the router picked
        is evaluated, which is the quality claim.
        """
        from sglang.kernels.ops.moe import trtllm_gen_moe as situ_moe
        from sglang.kernels.ops.moe.pack_topk_ids import PackTopkIds
        from sglang.kernels.ops.quantization.per_token_group_quant import (
            per_token_group_quant,
        )
        from sglang.srt.layers.moe import route_quant_handoff
        from sglang.srt.layers.moe.expert_split_moe import split_slice_moe
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        layer_idx = self.kt_config.layer_idx
        # The first MoE layer of a forward re-primes both slots: a prefill pass
        # is not guaranteed to have run to completion (aborts, chunk
        # boundaries), so slot occupancy from a previous pass is not reusable.
        if _KT_SPLIT_PREFILL_LAYERS and (
            _KT_SPLIT_PREFILL_LAYERS[0][0] is self
        ):
            self._cold_pipeline.reset()
            self._cold_pipeline.prime()
        cold_buf = self._cold_pipeline.wait_prefetch(layer_idx)

        x = dispatch_output.hidden_states
        if x.dim() > 2:
            x = x.view(-1, x.shape[-1])

        # The fused route+quant handoff publishes ids packed from LOGICAL
        # expert ids, which would bypass the slot remap below and silently
        # address the wrong rows.  Drain it and quantize ourselves; the
        # handoff is disarmed under KT anyway (KimiK3MoE gates it on
        # isinstance(method, Mxfp4MoEMethod), and ours is the KT wrapper),
        # so this is a guard rather than a hot path.
        prepared = route_quant_handoff.take(x)
        if prepared is not None:
            _, x_quant, x_scale = prepared
            x_scale = x_scale.view(torch.float8_e4m3fn)
        else:
            x_quant, x_scale = per_token_group_quant(
                x, group_size=32, scale_ue8m0=True
            )
            x_scale = x_scale.view(torch.float8_e4m3fn)

        # One slot space for both slices, so this is a bijective gather --
        # no -1, nothing dropped (contrast mask_and_remap_expert_ids).
        topk_output = dispatch_output.topk_output
        slot_ids = self.logical_to_slot_cuda[topk_output.topk_ids.long()].to(
            torch.int32
        )
        packed = PackTopkIds.execute(
            slot_ids, topk_output.topk_weights.to(torch.float32)
        )

        out = split_slice_moe(
            situ_moe=situ_moe,
            packed_topk=packed,
            hidden_states=x_quant,
            hidden_states_scale=x_scale,
            resident={
                "w13": layer.w13_weight,
                "w13_scale": layer.w13_weight_scale,
                "w2": layer.w2_weight,
                "w2_scale": layer.w2_weight_scale,
                "alpha": layer.gemm1_alpha,
                "beta": layer.gemm1_clamp_limit,
            },
            cold={
                "w13": cold_buf["w13_weight"],
                "w13_scale": cold_buf["w13_weight_scale"],
                "w2": cold_buf["w2_weight"],
                "w2_scale": cold_buf["w2_weight_scale"],
                "alpha": self._cold_scalars["alpha"],
                "beta": self._cold_scalars["beta"],
            },
            num_experts=self.global_num_experts,
            num_resident=self.num_gpu_experts,
            top_k=packed.shape[1],
            intermediate_size=self.gpu_method.intermediate_size_per_partition,
            token_tile=self._split_prefill_token_tile,
        )

        self._cold_pipeline.record_compute_and_prefetch_next(layer_idx)
        return StandardCombineInput(hidden_states=out)

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        """Execute hybrid CPU+GPU MoE forward pass with parallelism.

        This is the main computation method that coordinates:
        1. Submit CPU expert computation (non-blocking)
        2. Execute GPU expert computation in parallel
        3. Synchronize CPU results and merge with GPU results

        Args:
            layer: The MoE layer module
            dispatch_output: Dispatched tokens and routing information

        Returns:
            Combined computation results from CPU and GPU experts
        """
        from sglang.srt.eplb.expert_distribution import (
            get_global_expert_distribution_recorder,
        )
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        # Record GPU expert mask for distribution tracking (rank 0 only)
        # Use gpu_experts_mask_cuda which is already on GPU for CUDA graph compatibility
        if self.tp_rank == 0:
            recorder = get_global_expert_distribution_recorder()
            recorder.on_gpu_expert_mask(
                self.kt_config.layer_idx, self.gpu_experts_mask_cuda
            )

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        num_tokens = int(x.shape[0]) if x.dim() > 0 else 0

        # Split-slice full-expert prefill: evaluate the resident and cold
        # expert sets as two disjoint slices on GPU and merge, so every routed
        # expert is actually computed.  Only above a token threshold (decode
        # keeps the margin-routed CPU path) and never under stream capture --
        # this path is eager by construction.  is_extend_in_batch is NOT a
        # usable signal here; plain TP serving never writes it.
        if (
            self._split_prefill_ready
            and num_tokens >= self._split_prefill_threshold
            and not torch.cuda.is_current_stream_capturing()
        ):
            # Count demand before returning. Split prefill computes every
            # expert on GPU, so residency does not change this forward's
            # result -- but the router's behaviour over the PROMPT is the best
            # available forecast of what the decode about to start will ask
            # for, since the two share a domain. Observing it here is what
            # lets a swap at the prefill->decode boundary cut a set for the
            # request's own domain. Free of consequence as well as cheap:
            # nothing here can perturb what it measures. See SPEC-SWAP-DEMAND.
            if self._counters_enabled:
                self._update_demand_counters(dispatch_output.topk_output.topk_ids)
            return self._split_prefill_apply(layer, dispatch_output, num_tokens)

        # Margin routing (SPEC-MARGIN-ROUTING P1): rewrite below-margin
        # CPU-resident picks to resident alternatives BEFORE the Step-1 CPU
        # submit — the CPU side receives raw topk_ids and applies its own
        # pinned membership mask in C++, so a GPU-only rewrite at the Step-2
        # mask would desync the two halves (CPU computing overridden experts,
        # or dropped/double-counted contributions).  Counters accumulate on
        # the ORIGINAL ids so they measure true router preference, and the
        # scatter_add_ runs over all slots with 0/1 addends (static shapes,
        # in-place persistent buffers) so it is capture-safe and keeps
        # counting across decode graph replays.  Split prefill above returns
        # before this on purpose: it computes every routed expert on GPU, so
        # overriding there would cost accuracy for nothing.
        #
        # Margin-unset serving still feeds the swap policy. Demand and hits are
        # functions of the routed ids and the residency mask alone, so nothing
        # about them requires a margin (SPEC-SWAP-DEMAND). This branch is what
        # makes exact routing WITH adaptive placement a legal configuration --
        # bit-exact output, resident set still following the workload.
        # Margin routing runs when the SERVER set a nonzero budget, or when
        # any request in this batch asked for one. Gating on the server value
        # alone silently dropped SamplingParams.kt_routing_margin on a server
        # left at the default: the value reached ForwardBatch and the graph
        # buffer, and then nothing read it.
        _per_req_margin = self._any_request_margin()

        # THE DEFAULT IS 0.0, so this is the path most servers take and it must
        # not pay for the machinery it does not use. At a scalar 0.0 budget the
        # rewrite provably returns the router's own ids and weights, and the
        # decision below already declines to apply them -- but the kernel still
        # RAN, at every one of the 92 layers of every decode step, for a result
        # thrown away. The counters it would have produced reduce to demand and
        # hits, which _update_demand_counters computes directly from the routed
        # ids and the residency mask (SPEC-SWAP-DEMAND). So exact routing keeps
        # its full swap signal and skips the argsort/cumsum/scatter.
        #
        # Full override sits at the default budget too, and must NOT take this
        # path: its whole effect is the rewrite.
        _exact_routing = (
            not self._margin and not _per_req_margin and not self._full_override
        )

        if _exact_routing and self._counters_enabled:
            self._update_demand_counters(dispatch_output.topk_output.topk_ids)
            # Swap windows moved to the scheduler (SPEC-SWAP-DEMAND Phase 3):
            # mid-prompt re-cuts stall the throughput-critical path, and this
            # call site never ran under --kt-expert-split-prefill anyway --
            # split prefill returns before it and decode replays a graph.

        if not _exact_routing:
            from sglang.srt.layers.moe.topk import StandardTopKOutput

            _format_ok = (
                isinstance(topk_output, StandardTopKOutput)
                and topk_output.router_logits is not None
                and topk_output.router_logits.shape[-1] == self.global_num_experts
                # topk_weights IS the gate now: the budget is spent against
                # it and the post-substitution weights are derived from it, so
                # it is checked here rather than trusted.
                and topk_output.topk_weights is not None
                and topk_output.topk_weights.shape == topk_output.topk_ids.shape
            )
            if self._skip_cpu_path and not _format_ok:
                # Fail closed: with the CPU path statically skipped, silently
                # falling back to unmodified routing drops every CPU-resident
                # pick instead of computing it.
                raise RuntimeError(
                    f"--kt-routing-full-override requires the standard topk "
                    f"output with full router logits (layer "
                    f"{self.kt_config.layer_idx} got "
                    f"{type(topk_output).__name__}); the CPU path is skipped, "
                    f"so unoverridden CPU picks would be dropped."
                )
            if _format_ok:
                (
                    new_topk_ids,
                    new_topk_weights,
                    _insist_slots,
                    _override_slots,
                ) = (
                    margin_override_topk_ids(
                        topk_output.topk_ids,
                        topk_output.router_logits,
                        topk_output.topk_weights,
                        self.correction_bias,
                        self.gpu_experts_mask_cuda,
                        self._resolve_margin(topk_output.topk_ids),
                        self._full_override,
                    )
                )
                if self._counters_enabled:
                    self._update_margin_counters(
                        topk_output.topk_ids,
                        new_topk_ids,
                        _insist_slots,
                        _override_slots,
                    )
                # Reachable at a server margin of 0.0 only via a per-request
                # budget (the scalar-0.0 batch took the exact-routing path
                # above). With per-request margins the decision is per token,
                # made inside the kernel by the `margin > 0` term, so the
                # rewrite is applied and tokens at 0.0 come back unchanged.
                if self._full_override or _per_req_margin or self._margin > 0.0:
                    # Both, always: the weights belong to the new expert set,
                    # so applying the ids alone would run it at the old set's
                    # weights -- exactly the mismatch the recompute removes.
                    topk_output = topk_output._replace(
                        topk_ids=new_topk_ids, topk_weights=new_topk_weights
                    )
                    dispatch_output = dispatch_output._replace(
                        topk_output=topk_output
                    )
                self._maybe_log_margin_stats()
                # The first registered layer drives the swap window for the
                # whole model: later layers have not read their membership
                # yet this forward, so one window keeps the batch consistent.
                # Swap windows moved to the scheduler (SPEC-SWAP-DEMAND Phase 3):
                # mid-prompt re-cuts stall the throughput-critical path, and this
                # call site never ran under --kt-expert-split-prefill anyway --
                # split prefill returns before it and decode replays a graph.
            elif not self._margin_format_warned:
                self._margin_format_warned = True
                logger.warning(
                    "[kt-margin] layer=%s: --kt-routing-margin needs the "
                    "standard topk output (router logits); got %s — margin "
                    "routing is OFF for this layer.",
                    self.kt_config.layer_idx,
                    type(topk_output).__name__,
                )

        # Step 1: Copy hidden_states to staging buffer and submit CPU computation
        # Staging buffer allows GPU computation to proceed without waiting for D2H copy
        #
        # _skip_cpu_path (full override) elides Steps 1 and 4 entirely: no
        # token can reach a CPU expert, so the CPU half would compute exact
        # zeros and only cost the per-layer round-trip (staging D2H, submit,
        # sync, event join, merge add).  The flag is fixed in create_weights,
        # never derived from tensor data, so decode graph capture records the
        # same shape it replays — the same static-branch contract the
        # num_gpu_experts == 0 and tp_rank != 0 paths already rely on.
        staging_buffer = None
        # Slot this forward rang, carried from the ring (step 1) to the wait
        # (step 4) so the two cannot drift onto different batch-size tiers.
        _db_slot = None
        # Whether this forward packed its inputs, carried from the dispatch
        # (step 1) to the sync (step 4): the two must agree about which buffer
        # the CPU wrote into, and step 4 cannot re-derive it -- the batch-size
        # rule is evaluated once, before the pack.
        _fused = False
        if self.tp_rank == 0 and self._cpu_stream is not None and not self._skip_cpu_path:
            # Use shared staging buffer (shared across all MoE layers to save GPU memory)
            assert self._shared_staging_buffer is not None, "Shared staging buffer not initialized"
            staging_buffer = self._shared_staging_buffer.get_slice(x.shape[0])

            # Slot resolution and the device-side pack happen on the MAIN
            # stream, BEFORE forking. Per layer the expert GEMM ends ~20 us in
            # while the layer runs ~145 us, so whatever sits on the CPU
            # stream ahead of the ring decides whether the dispatch lands
            # while the GEMM is still running or after it has finished.
            # Measured: the staging copy completed AFTER the GEMM on 52% of
            # layers, and a late dispatch exposes the whole CPU latency
            # because there is no GPU work left to overlap with.
            if self._db_enabled:
                _db_slot = self._kt_doorbell_slot(staging_buffer, dispatch_output)
            _fused = _db_slot is None and self._kt_fused_staging_ok(x)
            if _db_slot is not None or _fused:
                # Pack from x DIRECTLY. staging_buffer is not filled on this
                # path, and packing from it shipped a previous layer's
                # activations to the CPU -- caught by byte identity, invisible
                # to nats and gsm8k.
                #
                # Reading x here is safe precisely because this runs on the
                # MAIN stream before the expert GEMM is issued: the copy is
                # ordered ahead of anything that could modify x, which is the
                # concern staging_buffer existed to solve.
                self._kt_pack_inputs(dispatch_output, x)
            else:
                # Unpacked host-node path stages through the shared buffer so
                # the GPU may modify x freely.
                staging_buffer.copy_(x, non_blocking=True)

            # Fork to cpu_stream (waits for the pack/staging copy)
            self._cpu_stream.wait_stream(torch.cuda.current_stream(x.device))
            with torch.cuda.stream(self._cpu_stream):
                if _db_slot is not None:
                    _db_stream = torch.cuda.current_stream(x.device).cuda_stream
                    # Arm BEFORE the ring: retract the previous replay's
                    # completion so this replay's wait cannot be satisfied by a
                    # stale value. A captured node writes a constant, so
                    # without the arm every replay after the first would sail
                    # through the wait reading the first replay's output.
                    kt_doorbell_arm(_db_slot, _db_stream)
                    # Then FLUSH, then ring. Only the single D2H sits between
                    # the fork and the ring -- the pack already ran on the main
                    # stream. The poller's whole decision reads these ids, so a
                    # ring visible before the flush would have it judge the
                    # PREVIOUS step's batch.
                    self._kt_flush_inputs(staging_buffer)
                    kt_doorbell_ring(_db_slot, _db_stream)
                elif _fused:
                    # One D2H, then the dispatch. The pack already ran on the
                    # main stream, so this is all that stands between the fork
                    # and the poller learning there is work.
                    self._kt_flush_inputs(x)
                    self.wrapper.submit_forward_packed(
                        x, torch.cuda.current_stream(x.device).cuda_stream
                    )
                else:
                    self._submit_with_staged_input(
                        layer, dispatch_output, staging_buffer
                    )

        # Step 2: Prepare GPU computation by masking and remapping expert IDs
        # CPU expert IDs are set to -1; GPU expert IDs are remapped to GPU weight indices
        topk_ids = topk_output.topk_ids
        masked_topk_ids = mask_and_remap_expert_ids(
            topk_ids, self.gpu_experts_mask_cuda, self.logical_to_gpu_index_cuda
        )

        # Create modified dispatch output for GPU computation
        masked_topk_output = topk_output._replace(topk_ids=masked_topk_ids)
        masked_dispatch_output = dispatch_output._replace(
            topk_output=masked_topk_output
        )

        # Step 3: Execute GPU expert computation on main stream
        # No wait needed - staging buffer decouples CPU and GPU data access
        # When num_gpu_experts == 0 the gpu_method's weights have shapes that
        # are incompatible with its own apply() (e.g. on SM_120 with V4 Flash
        # where the only routed-expert quant method available, the FP8 fused
        # MoE Triton path, asserts hidden_states.shape[1] == w1.shape[2] -
        # padded_size, which fails because w1 is the empty 0-expert slice).
        # Skip the GPU GEMM entirely and start from zeros; the CPU path then
        # provides 100% of the routed-expert contribution.
        # Origin: kt-sglang 耦合 (sglang/kt_ep_wrapper.py).
        if not getattr(self, "_diag_logged", False) and logger.isEnabledFor(logging.DEBUG):
            self._diag_logged = True
            try:
                _mask_sum = int(self.gpu_experts_mask.sum().item())
            except Exception as e:  # pragma: no cover
                _mask_sum = f"err:{e}"
            logger.debug(
                "[kt-ep-diag] layer=%s num_gpu_experts=%d mask_sum=%s "
                "mask_shape=%s gpu_method=%s",
                getattr(self.kt_config, 'layer_idx', '?'),
                self.num_gpu_experts,
                _mask_sum,
                tuple(self.gpu_experts_mask.shape),
                type(self.gpu_method).__name__,
            )
        if self.num_gpu_experts == 0:
            output = torch.zeros_like(x)
        else:
            with _scoped_layer_num_local_experts(layer, self.num_gpu_experts):
                output = self._apply_gpu_experts(
                    layer, masked_dispatch_output, topk_output
                )

        # Step 4: Sync CPU results on cpu_stream, then synchronize streams
        if self.tp_rank == 0 and self._cpu_stream is not None and not self._skip_cpu_path:
            with torch.cuda.stream(self._cpu_stream):
                # Use staging_buffer for sync to get correct buffer reference
                if _db_slot is not None:
                    kt_doorbell_wait(
                        _db_slot, torch.cuda.current_stream(x.device).cuda_stream
                    )
                    cpu_output = self._kt_doorbell_output(staging_buffer)
                elif _fused:
                    # x, not staging_buffer: the packed path never fills the
                    # shared buffer. Both name the same [bs, hidden] shape and
                    # sync_forward keys its rings by shape alone, so this is
                    # the same buffer either way -- passing x keeps the packed
                    # path's data flow readable end to end.
                    cpu_output = self._sync_cpu_forward(x)
                else:
                    cpu_output = self._sync_with_staged_input(staging_buffer)
                self._sync_done_event.record(self._cpu_stream)

            # Main stream waits for cpu_stream to complete before merging results
            torch.cuda.current_stream(x.device).wait_event(self._sync_done_event)
            # cpu_output is None only when the merge already happened inside
            # the conditional body, where it had to be in-place.
            if cpu_output is not None:
                output = output + cpu_output
        return StandardCombineInput(hidden_states=output)

    def _bind_reap_norms_buffer(self) -> None:
        """Hand kt this layer's row of the REAP norms buffer, once.

        kt writes into it every forward and never allocates or accumulates:
        the weights and ids that turn a norm into a score live on this side,
        and so does the knowledge of which rows of a padded decode batch are
        real. See SPEC-REAP-PLACEMENT.md.
        """
        stage = reap_stage()
        if stage is None or self.wrapper is None:
            return
        layer_idx = self.kt_config.layer_idx
        if layer_idx >= stage.num_layers:
            return
        row = stage.kt_norms[layer_idx]
        if not hasattr(self.wrapper, "set_reap_norms_buffer"):
            raise RuntimeError(
                "this kt build has no set_reap_norms_buffer; expert swapping "
                "cannot score the experts kt serves, so promotion would never "
                "fire. Rebuild kt-kernel, or run with "
                "--kt-expert-swap-transitions 0"
            )
        self.wrapper.set_reap_norms_buffer(
            row.data_ptr(), row.shape[0], row.shape[1]
        )

    def _apply_gpu_experts(self, layer, masked_dispatch_output, topk_output):
        """Run the GPU experts, parking their per-expert outputs on the way.

        Finalize sums every expert's contribution into one row per token, so
        the deferred seam is the only point at which an expert's output is
        still separable from the mixture it lands in.
        """
        stage = reap_stage()
        if stage is None:
            return self.gpu_method.apply(layer, masked_dispatch_output).hidden_states

        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
            FlashInferTrtllmDeferredFinalizeOutput,
            finalize_flashinfer_trtllm_deferred_output,
            flashinfer_trtllm_deferred_finalize_context,
        )

        with flashinfer_trtllm_deferred_finalize_context():
            deferred = self.gpu_method.apply(
                layer, masked_dispatch_output
            ).hidden_states
        if not isinstance(deferred, FlashInferTrtllmDeferredFinalizeOutput):
            # Only the mxfp4 SiTU path honours the context under the standard
            # (unpacked) routing this wrapper must produce. Everywhere else the
            # scores would stay zero for the life of the process, which reads
            # exactly like "measured, and every expert scored nothing" -- so
            # say so once instead of leaving it to be discovered.
            if not self._reap_seam_warned:
                type(self)._reap_seam_warned = True
                logger.warning(
                    "[kt-reap] %s finalizes internally, so per-expert outputs "
                    "are not observable and placement cannot adapt; expert "
                    "swapping will hold its startup cut",
                    type(self.gpu_method).__name__,
                )
            return deferred

        stage.stage(
            layer_idx=self.kt_config.layer_idx,
            gemm2_out=deferred.gemm2_out,
            permuted_idx=deferred.expanded_idx_to_permuted_idx,
            weights=deferred.expert_weights,
            served_ids=topk_output.topk_ids,
        )
        return finalize_flashinfer_trtllm_deferred_output(deferred, None)

    def _kt_doorbell_slot(self, staging_buffer, dispatch_output) -> Optional[int]:
        """This layer's doorbell slot for the batch size in flight, or None.

        Bound on first sight of each size, which is the first forward of that
        tier -- ahead of the ring recorded a few lines later, so the poller can
        never be rung at a slot it has no closure for.

        None for any batch size kt-kernel does not cache: the poller holds RAW
        POINTERS into that size's rings for the life of the process, and
        KExpertsCPUBuffer only keeps a tuple alive when the size is in
        `capture_bs`. Every other size shares one `temp_buffer` that the next
        differently-sized forward replaces, which would leave the closure
        reading freed memory. Those sizes take the host-node path instead --
        they are prefill shapes, and decode is what this transport is for.
        """
        batch_size = staging_buffer.shape[0]
        slot = self._db_slots.get(batch_size)
        if slot is None:
            if batch_size not in self.wrapper.get_capture_batch_sizes():
                return None
            _, topk_ids, _ = dispatch_output.topk_output
            slot = kt_doorbell_bind_slot(self, staging_buffer, topk_ids)
            self._db_slots[batch_size] = slot
        return slot

    def _any_request_margin(self) -> bool:
        """True when some request in this batch carried a margin override.

        Cheap: the per-token tensor is None unless a request asked, so this is
        an identity check, not a scan.
        """
        from sglang.srt.model_executor.forward_context import (
            get_forward_context,
            has_forward_context,
        )

        if not has_forward_context():
            return False
        return get_forward_context().kt_routing_margin is not None

    def _resolve_margin(self, topk_ids):
        """The margin threshold for this forward: a float, or one per token.

        Returns ``self._margin`` unless some request in the batch carried a
        ``SamplingParams.kt_routing_margin``, in which case a [num_tokens]
        tensor is returned with the server default substituted wherever a
        request did not ask for one (sentinel < 0).

        Falls back to the scalar whenever the per-token tensor is absent or the
        wrong length. That direction is deliberate: getting this wrong by
        length would apply one request's quality setting to another's tokens,
        which is silent and unattributable, whereas falling back merely ignores
        an override and leaves behaviour at the documented server default.
        """
        from sglang.srt.model_executor.forward_context import (
            get_forward_context,
            has_forward_context,
        )

        # What a token gets when its request named no budget. The server
        # default if there is one; otherwise 0.0, which is not a fallback but
        # the exact meaning of an unset server margin -- 0.0 is count-only by
        # the flag's contract: record what WOULD override, route exactly. So a
        # request opting in on a server that did not enable margin routing
        # leaves every other request bit-exact.
        #
        # NEVER None: the caller can enter the routing block on a per-request
        # budget alone, and a None returned here would reach `budget * total`
        # in the greedy and raise TypeError -- killing the scheduler, and at
        # decode-graph CAPTURE rather than on a request, because the capture
        # context always carries the graph-resident margin slot. The server
        # margin is a plain float now, so this is a float too.
        default = float(self._margin)

        # get_forward_context() asserts rather than returning None, and this
        # code also runs from paths that publish no context (unit tests, the
        # standalone probes), so the guard is required.
        if not has_forward_context():
            return default
        per_token = get_forward_context().kt_routing_margin
        if per_token is None:
            # The common path: nobody asked, so there is nothing to resolve.
            return default
        have, want = per_token.shape[0], topk_ids.shape[0]
        if have > want:
            # More budgets than tokens is not a shape this batch can explain,
            # so refuse rather than guess which rows to drop.
            self._warn_per_request_budget_ignored("length-mismatch", have, want)
            return default
        # Sentinel (negative) means "this request did not ask"; SamplingParams
        # rejects negative margins, so the value cannot be a real request's.
        resolved = torch.where(
            per_token < 0.0, torch.full_like(per_token, default), per_token
        )
        if have < want:
            # A SHORT tensor means the token axis was padded after the tensor
            # was built, and the padding is invisible to it: ForwardContext is
            # published from forward_batch.kt_routing_margin at the top of
            # _forward_raw, while _prepare_eager_forward_batch pads INSIDE that
            # scope, and _pad_tensor_to_size returns a new tensor. So padding
            # the field at its source would be dead code; extending here is
            # where the value is actually read.
            #
            # The rows line up because every remaining producer of a short
            # tensor APPENDS: _pad_inputs_to_size is the only one
            # (prepare_attn_tp_scatter_input delegates to it rather than
            # slicing), and the TBO split and the speculative-verify expansion
            # both produce correctly-sized tensors. The appended rows are
            # padding whose output is discarded, so they take the server
            # default.
            resolved = torch.cat(
                [
                    resolved,
                    torch.full(
                        (want - have,),
                        default,
                        dtype=resolved.dtype,
                        device=resolved.device,
                    ),
                ]
            )
        return resolved

    def _warn_per_request_budget_ignored(
        self, reason: str, have: int = 0, want: int = 0
    ) -> None:
        """Say so, once per cause: per-request budgets were dropped.

        The fallback direction is deliberate -- one scalar for every token beats
        applying one request's quality setting to another's -- but it is uniform
        and otherwise invisible, so without this a deployment serves every
        request at the server default while its clients believe they are setting
        a budget. Same reasoning as the counter-fallback warning above: taking
        the safe path costs only a feature, which means it presents as a knob
        that mysteriously does nothing rather than as a failure.

        Keyed by cause rather than by a plain one-shot bool, because a bool is
        burned during decode-graph capture, before a request exists.
        """
        if reason in type(self)._per_request_budget_warned:
            return
        type(self)._per_request_budget_warned.add(reason)
        _default = self._margin
        logger.warning(
            "[kt-margin] per-request kt_routing_margin IGNORED: the per-token "
            "budget tensor has %d rows but the MoE sees %d, so every token in "
            "this batch falls back to the server default (%s). Reachable "
            "wherever the token axis is padded or split (attn-tp gather, "
            "moe_dense_tp_size, speculative verify batches).",
            have,
            want,
            _default,
        )

    def _update_demand_counters(self, topk_ids) -> None:
        """Fold one forward into the demand counters, with no margin involved.

        For the paths that never produce insist/override slots: split prefill
        (which computes every expert on GPU and returns before the margin
        block) and margin-unset serving. Demand is defined directly:

            demand = routed & ~gpu_experts_mask[topk_ids]

        which is the same quantity ``_update_margin_counters`` produces as
        insist + override -- margin only partitions it. Everything lands in the
        insist counter because nothing was substituted.

        Counts on the ids the ROUTER chose. Under split prefill that is also
        the id actually computed (all 896 are), so there is no pre/post
        distinction to get wrong here.
        """
        # Both guards are load-bearing and independent: the swap driver checks
        # gpu_experts_mask_cuda and the counter separately (:6599), so the mask
        # can be absent while the counters exist. Indexing a None mask here
        # would crash the split-prefill path.
        if self._margin_insist_count is None or self.gpu_experts_mask_cuda is None:
            return
        safe_ids = topk_ids.clamp_min(0).reshape(-1).to(torch.int64)
        routed = (topk_ids >= 0).reshape(-1)
        resident = self.gpu_experts_mask_cuda[safe_ids]
        self._margin_insist_count.scatter_add_(
            0, safe_ids, (routed & ~resident).to(torch.int32)
        )

    def _update_margin_counters(
        self, topk_ids, served_ids, insist_slots, override_slots
    ) -> None:
        """Fold one forward into the per-expert demand counters.

        The swap driver cannot work without these -- promotion reads demand for
        non-resident experts, demotion reads traffic served by resident ones --
        so the answer to their cost is a cheaper measurement, not no
        measurement.

        The torch form below runs about a dozen kernels per layer per step
        (clamp, dtype casts, bitwise ops, three scatter_add_), which is a
        material share of decode once multiplied by the layer count. The fused
        kernel does the same arithmetic in one launch; the counters are
        integers accumulated by atomicAdd, so the values are bit-identical,
        not merely equivalent.
        """
        from sglang.kernels.ops.kimi_k3 import kt_margin_counters as ktmc

        if ktmc.covered(topk_ids, insist_slots, override_slots):
            ktmc.kt_margin_counters(
                self._margin_insist_count,
                self._margin_override_count,
                topk_ids,
                insist_slots,
                override_slots,
            )
            return

        # Say so, once. The fallback produces identical numbers, so taking it
        # costs only speed -- which means a mismatch presents as an
        # optimisation that mysteriously did nothing rather than as a failure.
        # That is exactly what happened: the router emits int32 ids, the first
        # covered() demanded int64, and a full measurement round reported "the
        # fused kernel recovered -6%" before the cause was found.
        if not type(self)._kt_counter_fallback_warned:
            type(self)._kt_counter_fallback_warned = True
            logger.warning(
                "[kt-margin] fused demand counters unavailable (%s); using the "
                "torch fallback. Numbers are identical; decode is ~10%% slower.",
                ktmc.why_not_covered(topk_ids, insist_slots, override_slots),
            )
        # Fallback for shapes/dtypes the kernel does not claim. Kept because
        # the counters feed a serving decision: silently not counting would
        # starve the swap policy rather than fail.
        _orig_safe_ids = topk_ids.clamp_min(0).reshape(-1).to(torch.int64)
        # DEMAND on the ORIGINAL id: what the router asked for.
        self._margin_insist_count.scatter_add_(
            0, _orig_safe_ids, insist_slots.reshape(-1).to(torch.int32)
        )
        self._margin_override_count.scatter_add_(
            0, _orig_safe_ids, override_slots.reshape(-1).to(torch.int32)
        )

    def _kt_doorbell_output(self, staging_buffer) -> torch.Tensor:
        """Result tensor for the merge; the wait node already ordered it."""
        return self.wrapper.doorbell_output(staging_buffer)

    def _maybe_log_margin_stats(self) -> None:
        """Rate-limited INFO line with cumulative insist/override counts.

        Runs only on eager (non-captured) applies — graph-mode decode replays
        never execute Python, so decode traffic surfaces here at the next
        eager forward (prefill or eager decode).  The counter reads below are
        host syncs and must never run under capture.
        """
        if self.tp_rank != 0 or torch.cuda.is_current_stream_capturing():
            return
        _cls = type(self)
        if not hasattr(_cls, "_kt_margin_step"):
            _cls._kt_margin_step = {}
        # BEFORE the margin counters, on its own rate limit, and before their
        # early return -- so it is emitted whether or not the swap driver
        # allocated them. Inside the margin limit the only line ever printed is
        # eager step 1, which happens during warmup before capture has bound a
        # single slot: it would report zeros forever and look exactly like a
        # transport that never ran. The counters are cumulative and
        # process-wide, and this runs on EAGER steps, so a line printed while
        # prefilling one request already carries the previous request's DECODE
        # traffic. That is what makes `served` readable with swapping off --
        # the only configuration that can be byte-compared.
        if _KT_DOORBELL["inited"]:
            _cls._kt_db_log_step = getattr(_cls, "_kt_db_log_step", 0) + 1
            if _cls._kt_db_log_step % 256 == 1:
                logger.info("[kt-doorbell] %s", kt_doorbell_stats())
        if self._margin_insist_count is None:
            # Counters are off (no swap driver, no full override): nothing else
            # to report.
            return
        _li = self.kt_config.layer_idx
        _cls._kt_margin_step[_li] = _cls._kt_margin_step.get(_li, 0) + 1
        _step = _cls._kt_margin_step[_li]
        if _step != 1 and _step % 64 != 0:
            return
        insists = self._margin_insist_count
        total_insist = int(insists.sum().item())
        total_override = int(self._margin_override_count.sum().item())
        if self._skip_cpu_path and total_insist:
            # The static analysis said this is impossible (>= top_k residents
            # and finite logits); this is the end-to-end falsification hook.
            # Every insist here is a routed contribution the skipped CPU path
            # never computed — wrong numbers, so stop rather than serve them.
            raise RuntimeError(
                f"[kt-margin] layer={self.kt_config.layer_idx}: "
                f"{total_insist} CPU-resident picks survived full override "
                f"while the CPU path is skipped — their contribution was "
                f"dropped. Routing/mask invariant violated."
            )
        top_vals, top_ids = torch.topk(insists, k=min(8, insists.numel()))
        top = [
            (int(i), int(v))
            for i, v in zip(top_ids.tolist(), top_vals.tolist())
            if v > 0
        ]
        logger.info(
            "[kt-margin] layer=%s eager_step=%d budget=%s "
            "insists=%d overrides=%d top_insisted=%s",
            _li,
            _step,
            self._margin,
            total_insist,
            total_override,
            top,
        )

    def __getattr__(self, name: str):
        """Delegate attribute access to the wrapped GPU method.

        This allows the wrapper to transparently expose attributes and methods
        from the wrapped GPU quantization method.

        Args:
            name: Attribute name

        Returns:
            Attribute value from gpu_method
        """
        # Avoid infinite recursion for internal attributes
        if name in ("gpu_method", "wrapper", "kt_config"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        return getattr(self.gpu_method, name)

def _kt_ep_predicate(layer, server_args):
    return create_kt_config_from_server_args(server_args, layer.layer_id)


def _kt_ep_factory(layer, gpu_method, kt_config):
    return KTEPWrapperMethod(gpu_method, kt_config)


from sglang.srt.layers.moe.quant_method_registry import register_moe_quant_wrapper

# priority=20 → wraps after mxfp4 (matches PR #38 Phase 3 → outer wrapper)
register_moe_quant_wrapper(
    "kt_ep", _kt_ep_predicate, _kt_ep_factory, priority=20
)


def _kt_swap_tables(method) -> "object":
    """Bundle a wrapper's four membership tables for the swap driver."""
    from sglang.srt.layers.moe.kt_expert_swap import SwapTables

    return SwapTables(
        gpu_experts_mask=method.gpu_experts_mask,
        gpu_experts_mask_cuda=method.gpu_experts_mask_cuda,
        logical_to_gpu_index=method.logical_to_gpu_index,
        logical_to_gpu_index_cuda=method.logical_to_gpu_index_cuda,
        gpu_index_to_logical=method.gpu_index_to_logical,
        pinned_mask=(
            method.wrapper.gpu_experts_mask if method.wrapper is not None else None
        ),
        logical_to_slot=method.logical_to_slot,
        logical_to_slot_cuda=method.logical_to_slot_cuda,
    )


# Phase-3 boundary state (SPEC-SWAP-DEMAND). Module-level for the same reason
# _KT_SWAP_STATE is: the driver is a free function over the registered wrappers,
# not a method on any one of them.
# Act on one boundary in every `expert_swap_transitions`.
#
# MUST BE DETERMINISTIC ACROSS TP RANKS. Every rank runs its own scheduler
# process and calls this independently, and all ranks hold the same resident
# expert SET (TP shards the hidden dim, not the expert index), so they must
# reach the SAME swap decision or membership diverges and each rank computes a
# different model -- silently. A wall-clock rate limit does exactly that: two
# ranks straddling the threshold disagree. A counter over prefill->decode
# transitions cannot, because every rank sees the same batches in the same
# order. This is why the original gate counted eager forwards rather than
# seconds, and the reason survives the move to the scheduler.

_KT_BOUNDARY_STATE = {
    "last_was_extend": False,
    "transitions": 0,
}


def maybe_run_expert_swap_at_decode_boundary(
    is_decode: bool,
    is_extend: bool,
) -> None:
    """Run one swap window at a prefill->decode boundary, from the scheduler.

    WHY NOT WHERE IT USED TO BE. ``maybe_run_expert_swap_window`` is called from
    a layer's ``apply()`` and only executes eagerly, so under
    ``--kt-expert-split-prefill`` it never runs at all: split prefill returns
    before the margin block that hosts the call, and decode replays a captured
    graph in which no Python executes. Swapping was inert in exactly the
    configuration this campaign ships.

    WHY HERE. The scheduler loop is Python between batches, so a device sync is
    legal and nothing is mid-chunk. It is also the only place that knows a
    request just left prefill -- the moment when the demand observed over the
    prompt is freshest and the decode about to consume the resident set has not
    started. One window per request-arrival instead of several per prompt.

    RATE LIMIT IS NOT OPTIONAL. Under continuous batching prefill->decode
    transitions arrive constantly, and an unthrottled window costs far more
    than it returns. The limit counts TRANSITIONS
    (--kt-expert-swap-transitions), not wall clock: a window costs a quiesce
    plus weight copies, and what earns that back is the demand observed since
    the last one, which arrives per transition rather than per second.
    """
    if not _KT_EP_METHODS:
        return
    anchor = _KT_EP_METHODS[0]
    cfg = anchor.kt_config
    if cfg.expert_swap_transitions <= 0:
        _KT_BOUNDARY_STATE["last_was_extend"] = is_extend
        return

    prev_extend = _KT_BOUNDARY_STATE["last_was_extend"]
    prev_decode = _KT_BOUNDARY_STATE.get("last_was_decode", False)
    _KT_BOUNDARY_STATE["last_was_extend"] = is_extend
    _KT_BOUNDARY_STATE["last_was_decode"] = is_decode

    crossed = is_decode and prev_extend
    steady = is_decode and prev_decode

    # NEVER ON THE PREFILL BOUNDARY ITSELF. Under split prefill every expert is
    # computed on GPU, so a prefill does not care which experts are offloaded:
    # placement only matters for decode. Running the window at the
    # prefill->decode crossing therefore buys nothing and lands squarely in
    # the request's time-to-first-token.
    #
    # So the crossing only DECIDES; the window runs at the first steady decode
    # step after it, once the first token is already out. The condition is a
    # pure function of the batch-mode sequence, which every TP rank sees
    # identically -- a queue-depth test would not be, and rank divergence here
    # is the M9/M11/M12 failure class. Under DP attention "identically" needs
    # the all-gathered inputs substituted above; the local mode is per-rank.
    if crossed:
        _KT_BOUNDARY_STATE["transitions"] += 1
        n = _KT_BOUNDARY_STATE["transitions"]
        every = max(1, cfg.expert_swap_transitions)
        # Observe on every transition, act on every `every`-th, and never on
        # the first: a cumulative counter's first delta is the whole launch
        # history, so acting on it is acting on a baseline.
        if n > 1 and n % every == 0:
            _KT_BOUNDARY_STATE["act_pending"] = True
    elif not (steady and _KT_BOUNDARY_STATE.get("act_pending")):
        return

    act_now = bool(steady and _KT_BOUNDARY_STATE.get("act_pending"))
    if act_now:
        _KT_BOUNDARY_STATE["act_pending"] = False
    try:
        maybe_run_expert_swap_window(anchor, force=True, act=act_now)
    except Exception:
        # NOT swallowed any more. "Serving continues" was the wrong policy: a
        # window that failed part-way has already moved kt's ownership and
        # staged GPU rows whose tables never flipped, so continuing serves
        # wrong experts silently and forever.
        _fatal_swap_failure("the decode-boundary swap window raised")


def maybe_run_expert_swap_window(
    anchor: "KTEPWrapperMethod",
    force: bool = False,
    act: Optional[bool] = None,
) -> None:
    """Quiesce and re-cut expert membership for the whole model.

    Driven from the scheduler at a prefill->decode boundary
    (``maybe_run_expert_swap_at_decode_boundary``), not from a layer.

    It used to be called from the FIRST registered layer's ``apply()``, and
    that was legal for a specific reason: membership is consumed exclusively
    inside ``apply()`` -- the margin override, the id remap, kt-kernel's own
    mask read -- so at layer 0 no later layer had read its tables yet and the
    batch still saw one consistent membership. That call site is gone. Under
    ``--kt-expert-split-prefill`` it never ran (split prefill returns before
    the margin block that hosted it, and decode replays a captured graph in
    which no Python executes), so swapping was inert in the shipping config.

    From the scheduler the consistency argument is strictly stronger: the call
    lands BETWEEN batches, so no layer of any forward has read membership yet,
    rather than merely no layer after this one.

    ``force`` skips the eager-forward sampling gate; ``act`` overrides whether
    this window actually swaps (the boundary driver decides, and passes False
    on its first call so the EMA has history before anything rests on it).
    The two are separate on purpose -- conflating "skip the gate" with "act"
    is what made the first boundary window swap on a launch-history baseline.

    The device sync is the quiesce: when it returns, every previously issued
    forward has completed, including the host nodes that enqueue CPU expert
    work, so nothing is mid-flight while weights and masks change.
    """
    from sglang.srt.layers.moe.kt_expert_swap import (
        ExpertSwapPolicy,
        run_swap_window,
    )

    cfg = anchor.kt_config
    _KT_SWAP_STATE["eager_forwards"] += 1
    n = _KT_SWAP_STATE["eager_forwards"]
    # Observe more often than we act. The policy's first observation is
    # baseline-only by construction (a cumulative counter's first "delta" is
    # the whole launch history), so welding observation to action wasted an
    # entire interval AND left the first real decision resting on a single
    # sample. Sampling at interval/5 means the EMA already has history when
    # the first window acts -- and a short run still swaps instead of doing
    # nothing at all, which is how three separate runs came back empty.
    sample_every = max(1, 2 * cfg.expert_swap_transitions)
    if not force and n % sample_every:
        return
    # A boundary call has already decided WHETHER to act -- it fires once
    # per prefill->decode transition and is rate-limited on wall clock, so
    # re-gating on the eager-forward counter would drop most windows. It
    # still passes act=False for its first call, because a cumulative
    # counter's first delta is the whole launch history and acting on that
    # baseline is what the interval//5 sampling exists to prevent.
    act = act if act is not None else (n % cfg.expert_swap_transitions) == 0

    # One take for the whole window, before the per-layer loop: the
    # accumulators are shared across layers and taking them per layer would
    # zero the rows the later layers still need.
    stage = reap_stage()
    window = _take_reap_window(stage)

    entries = []
    for method in _KT_EP_METHODS:
        if method.gpu_experts_mask_cuda is None or method._margin_insist_count is None:
            continue
        if method._swap_policy is None:
            method._swap_policy = ExpertSwapPolicy(
                method.global_num_experts,
                hysteresis=cfg.expert_swap_hysteresis,
                max_swaps=cfg.expert_swap_max,
            )
        if window is not None:
            method._swap_policy.observe(
                window[0][method.kt_config.layer_idx],
                window[1][method.kt_config.layer_idx],
            )
        entries.append(
            {
                "policy": method._swap_policy,
                "tables": _kt_swap_tables(method),
                "layer": method._swap_layer,
                "num_gpu_experts": method.num_gpu_experts,
                "layer_idx": method.kt_config.layer_idx,
                "method": method,
            }
        )
    if not entries or not act:
        # Sampling-only pass: counters folded into the EMAs, nothing moved.
        #
        # This used to also kick off a background checkpoint prefetch of what
        # the NEXT window would demote. It fed a read path that no longer
        # exists: rank-write captures each rank's own slice off its own GPU
        # rows, so no demoted expert's bytes come off disk at all.
        return

    # Phase accounting for one window, logged on the way out.
    _timing = {
        "read_s": 0.0,
        "install_s": 0.0,
        # Phase breakdown of what the window's timing line calls "elsewhere",
        # because guessing where that time went was wrong twice. The spans
        # below tile the whole per-layer body, so "unattributed" in the log is
        # a real residue rather than a span nobody thought to measure.
        "select_s": 0.0,      # policy.select over 896 experts, per layer
        "rows_s": 0.0,        # the demoted rows' l2g lookup, per swap
        "begin_s": 0.0,       # begin_layer: rank-write capture + validate
        "move_s": 0.0,        # move_weights (records the move), per swap
        "stage_s": 0.0,       # arena -> device DMA of the promoted rows
        "flush_gpu_s": 0.0,   # swizzle + scatter
        "finish_s": 0.0,      # finish_layer: the staged flush
        "apply_s": 0.0,       # apply_swaps_to_tables (4 tables + 3 H2D)
        "tables_s": 0.0,      # assert_tables_consistent
        "after_s": 0.0,       # after_flip bookkeeping
        # THE RESIDUE, split out. These three were inside the window timer and
        # outside every span, and together they were a third of the window.
        # None is swap work: they are what the window WAITS on.
        "arm_s": 0.0,         # the rank-write arming all_reduce (rank skew)
        "quiesce_s": 0.0,     # torch.cuda.synchronize: drain the in-flight fwd
        "barrier_s": 0.0,     # the end-of-window barrier (rank skew again)
        "d2h_sync_s": 0.0,    # waiting for this rank's demotion D2H to land
    }
    _window_t0 = time.perf_counter()

    # RANK-WRITE ARMING IS DECIDED ONCE PER WINDOW, SYMMETRICALLY. The writer
    # itself is per-rank fallible -- kt_arena_share degrades a rank that could
    # not map the arena, by design -- so gating anything directly on "do I
    # have a writer" splits the ranks, which is precisely the M9/M11/M12
    # failure class. One MIN all_reduce here, on a call every rank reaches
    # (entries and act are plan data), and every later branch keys off the
    # result instead: the barrier, the per-layer capture, and the install.
    _rank_writer = _get_or_create_rank_writer(entries[0])
    _t_arm = time.perf_counter()
    # Still a MIN all_reduce, because the writer is per-rank fallible --
    # kt_arena_share degrades a rank that could not map the arena, by design --
    # and a split rank set is silent corruption rather than a crash. What
    # changed is the verdict: there is nowhere left to degrade TO, so a rank
    # that cannot write is a dead server, not a slower one. Every rank reaches
    # this call (entries and act are plan data), so the terminate is unanimous.
    _KT_SWAP_STATE["rank_write_armed"] = _all_tp_ranks_succeeded(
        _rank_writer is not None
    )
    _timing["arm_s"] += time.perf_counter() - _t_arm
    if not _KT_SWAP_STATE["rank_write_armed"]:
        _fatal_swap_failure(
            "rank-write could not arm on every rank (this rank: "
            f"writer={'yes' if _rank_writer is not None else 'no'}). A demoted "
            "expert owns no CPU buffers under cold-only residency, so without "
            "the arena writer it would become routable with no weights"
        )

    # Batched move state, flushed through run_swap_window's finish_layer hook.
    # Flushing lazily on "the layer changed" instead looks equivalent and is
    # not: the next layer's first move runs AFTER this layer's tables have been
    # flipped, so a failed write left the tables advertising experts whose rows
    # still held the previous occupants, and the error was reported against the
    # following layer. The layer-change flush below is now only a safety net.
    _pending = {
        "layer": None,
        "layer_idx": None,
        "items": [],
    }

    def _flush_moves():
        """Apply one layer's staged swaps as a handful of bulk copies.

        WHY THIS EXISTS. The per-expert version issued a BLOCKING D2H for
        every (swap, tensor) pair -- thousands of synchronising round trips per
        window. The traffic was never the problem; the stalls were.

        This mirrors ColdExpertPipeline, which streams a layer as four bulk
        copies (one per weight name) rather than one per expert. Per layer:
        4 gathers + 4 async D2H + ONE sync + 4 scatters, instead of 32 copies
        and 32 syncs.

        The read-before-write rule is preserved and in fact widened: every
        promoted row was cloned out of the store by stage_row at record time,
        and every demoted row is gathered off the GPU here BEFORE any promoted
        row is written back, so a promote and a demote trading the same places
        cannot destroy each other's source.
        """
        pend = _pending
        items = pend["items"]
        if not items:
            return
        layer, layer_idx = pend["layer"], pend["layer_idx"]
        pend["items"] = []
        # Past this point the recorded rows hold PROMOTED experts. Demotions
        # never read them back: _begin_layer took every demoted row for this
        # layer before its first move ran.

        rows = [it["dst_row"] for it in items]
        dev = getattr(layer, _MXFP4_TRTLLM_RESIDENT_PARAM_NAMES[0]).data.device
        idx = torch.tensor(rows, dtype=torch.long, device=dev)

        # index_select/index_copy_ have no fp8 CUDA kernels and the scale
        # params are Float8_e4m3fn, so the bulk moves run on byte views. The
        # view is exact -- fp8 and uint8 are both one byte and the rows are
        # contiguous -- and costs nothing. Doing this on the raw params instead
        # is what made every layer fail with "index_copy_cuda not implemented
        # for 'Float8_e4m3fn'".
        def _bytes(t):
            return t if t.dtype == torch.uint8 else t.view(torch.uint8)

        # NO READ-BACK. The demoted rows used to be gathered off the GPU into
        # pinned staging and handed to the store, behind a full
        # torch.cuda.synchronize() per layer. Nothing needs them any more: the
        # demoted expert never lost its CPU buffers, and where the arena has to
        # be refreshed the rank-write path writes GPU -> kt directly. The sync
        # went with it.
        _t_gpu = time.perf_counter()

        # WRITE: scatter every promoted row back, again one kernel per name.
        #
        # PREALLOCATED, CONSTANT-SHAPE BATCH. torch.stack allocated fresh
        # device tensors per layer whose shape followed len(items), which
        # varies per layer -- so the caching allocator could not reuse blocks
        # and each miss forced a free/synchronize. That dominated the window;
        # the copies themselves were a rounding error.
        #
        # Split prefill never had this problem because its buffers are
        # allocated once per slot at construction and every layer presents the
        # same shape, so the allocator serves it from cache. Do the same here:
        # one buffer set sized to the swap budget, filled in place, and
        # ALWAYS processed at full width so every layer presents an identical
        # shape. Rows beyond len(items) hold stale bytes and are simply not
        # scattered -- swizzling a few unused rows costs microseconds against
        # milliseconds per allocator miss.
        # The DMA already wrote row i of every buffer, so there is nothing to
        # gather: the staging path allocates batch_bufs before any flush can
        # run, and it is the only producer of `items`. The branch that used to
        # sit here brought host or store tensors together for the deleted
        # store/export transports.
        promoted = _KT_SWAP_STATE["batch_bufs"]
        # Every remaining source hands back CHECKPOINT layout -- kt's arena is
        # checkpoint layout by construction -- so the resident row's trtllm
        # layout is produced HERE, on the GPU, in the same four-gather form
        # split prefill uses, over this layer's swaps instead of its cold set.
        promoted = _swizzle_promoted_rows(promoted)
        for name in _MXFP4_TRTLLM_RESIDENT_PARAM_NAMES:
            dst = _bytes(getattr(layer, name).data)
            # The swizzle emits scales FLAT per expert (the interleave map is
            # flat by construction) while the resident scale params are 3-D;
            # the byte ORDER is already the resident order, so reshaping to
            # the destination row shape is exact -- and reshape raises on any
            # numel mismatch, which is the failure mode this wants. A1's first
            # windows died here 2-D-vs-3-D AFTER w13_weight scattered, leaving
            # rows with new weights and old scales; the reshape must therefore
            # happen for EVERY name before ANY scatter runs.
            promoted[name] = _bytes(promoted[name]).reshape(
                (-1,) + tuple(dst.shape[1:])
            )
        for name in _MXFP4_TRTLLM_RESIDENT_PARAM_NAMES:
            # idx names len(items) rows; the batch is processed at full width,
            # so scatter only the live prefix.
            _bytes(getattr(layer, name).data).index_copy_(
                0, idx, promoted[name][: len(items)]
            )
        _timing["flush_gpu_s"] += time.perf_counter() - _t_gpu

    def _move(layer, dst_row, logical_id, demoted_id):
        """Record expert ``logical_id`` -> resident row ``dst_row``.

        Records rather than applies: the copies are batched per layer by
        _flush_moves. The promoted bytes are read HERE, at record time,
        because the buffers they come from are the ones the demoted expert
        will take.
        """
        layer_idx = layer.layer_id
        # DMA PROMOTION, the mirror of the demotion write: the promoted
        # expert's bytes are already in kt's arena and this rank needs only its
        # own slice, so the copy engine reads them straight to device. It needs
        # the rank writer's offset table and nothing else, and apply_move keeps
        # that table current across swaps -- unlike the load-time read
        # registry, which is empty under cold-only precisely because it would
        # go stale. This used to be decided AFTER a cold-store slot lookup,
        # which made it unreachable once the store was gone: every promotion
        # fell through to a per-expert checkpoint read, up to
        # 92 x --kt-expert-swap-max per window. Nothing in the log said so,
        # because the rank-write line it prints is about demotions.
        _rw = _KT_SWAP_STATE.get("rank_writer")
        _dma_armed = (
            _rw is not None
            and getattr(_rw, "_dma", None) is not None
            and bool(_KT_SWAP_STATE.get("rank_write_armed"))
        )
        _dma_row = (
            _rw._offsets[layer_idx].get(int(logical_id), _rw._g.part)
            if _dma_armed
            else None
        )
        if _dma_row is None:
            # None means the promoted expert holds no CPU buffer, i.e. it is
            # already GPU-resident -- and with THIS policy that cannot happen.
            # ExpertSwapPolicy.select draws promotes from ~mask and demotes
            # from mask via nonzero(), so within one layer's plan the promote
            # ids are distinct and disjoint from the demote ids; the only thing
            # that nulls an entry is apply_move(promote=X), and X appears once.
            # An expert null BEFORE the window is refused earlier still, by
            # can_move inside validate().
            #
            # So this is a policy invariant asserted at the point that depends
            # on it, not a fallback. It used to read the expert off the
            # checkpoint instead, which is exactly the failure recorded above:
            # a silent per-expert disk read while the rank-write log claimed
            # the fast path. If select() ever emits a non-disjoint plan, the
            # server must say so rather than quietly serve at 1/100th the rate.
            _fatal_swap_failure(
                f"promoted expert {int(logical_id)} on layer {layer_idx} holds "
                f"no arena buffer: the swap plan is not disjoint, which "
                f"ExpertSwapPolicy.select is supposed to guarantee"
            )

        if _pending["layer_idx"] != layer_idx:
            _flush_moves()
            _pending["layer"], _pending["layer_idx"] = layer, layer_idx

        if _dma_row is not None:
            # PROMOTION BY DMA, the mirror of the demotion write. The promoted
            # expert's bytes are already in kt's arena and this rank needs only
            # its own slice, so the copy engine reads them straight to device.
            # No stage_row clone (a host read plus a host write of 2.19 MB per
            # expert), no pinned staging, and the read happens BEFORE
            # install_cpu_expert moves the slot, so the promoted expert still
            # owns these buffers -- the same read-before-move rule the writer
            # relies on, inverted.
            #
            # AND THE TWO ALIAS. After the move the demoted expert owns exactly
            # the buffers this read is sourcing, and the rank-write D2H lands
            # in them. Both are issued on the CURRENT stream, so the read is
            # ordered ahead of the write and completes first; putting either on
            # a private stream would silently reintroduce the race.
            _t = time.perf_counter()
            _row = _dma_row
            _raw = _KT_SPLIT_PREFILL_STATE["raw_shapes"]
            _dev = getattr(
                layer, _MXFP4_TRTLLM_RESIDENT_PARAM_NAMES[0]
            ).data.device
            # REUSED LANDING BUFFERS, one set per position in the layer's swap
            # budget. Allocating fresh device tensors per expert misses the
            # caching allocator once it is near full, and every miss forces a
            # free/synchronize -- which dominated the window. The buffers are
            # identical in shape for every expert, so one set per slot serves
            # the whole run.
            # DMA STRAIGHT INTO THE BATCH ROW. This used to land in a separate
            # per-slot buffer that _flush_moves then copied into the batch --
            # two device buffers and a full D2D copy of every promoted expert,
            # for no reason beyond the two pools having been added at different
            # times. The batch row is contiguous and exactly the landing shape,
            # so the copy engine can write it directly.
            #
            # RUN-SCOPED, not per window: _pending is local to this function, so
            # a pool kept there is rebuilt on every window. The shapes depend
            # only on the swap budget and the layer geometry, both fixed for the
            # process, so one set serves the whole run.
            _cap = max(1, cfg.expert_swap_max)
            _bat = _KT_SWAP_STATE.get("batch_bufs")
            if _bat is None or _bat[_MXFP4_TRTLLM_RESIDENT_PARAM_NAMES[0]].shape[0] != _cap:
                _bat = {
                    n: torch.empty(
                        (_cap,) + tuple(_raw[n][0]), dtype=_raw[n][1], device=_dev
                    )
                    for n in _MXFP4_TRTLLM_RESIDENT_PARAM_NAMES
                }
                _KT_SWAP_STATE["batch_bufs"] = _bat
            _k = len(_pending["items"])
            if _k >= _cap:
                raise RuntimeError(
                    f"swap batch overflow: {_k + 1} promotions in one layer "
                    f"against --kt-expert-swap-max {cfg.expert_swap_max}"
                )
            _staged_row = {n: _bat[n][_k] for n in _MXFP4_TRTLLM_RESIDENT_PARAM_NAMES}
            _rw._dma.read(
                layer_idx=layer_idx,
                row=_row,
                out={
                    "w13": _staged_row[_MXFP4_TRTLLM_RESIDENT_PARAM_NAMES[0]],
                    "w13_scale": _staged_row[_MXFP4_TRTLLM_RESIDENT_PARAM_NAMES[1]],
                    "w2": _staged_row[_MXFP4_TRTLLM_RESIDENT_PARAM_NAMES[2]],
                    "w2_scale": _staged_row[_MXFP4_TRTLLM_RESIDENT_PARAM_NAMES[3]],
                },
                stream=torch.cuda.current_stream().cuda_stream,
            )
            _timing["stage_s"] += time.perf_counter() - _t
            _pending["items"].append(
                {
                    "dst_row": dst_row,
                    "demoted_id": demoted_id,
                    "promoted": _staged_row,
                }
            )

    def _install_cpu(entry, promote_id, demote_id):
        """Give the demoted expert the promoted one's CPU weight buffers.

        Only under cold-only residency. Without it every expert already holds
        CPU weights, so a demotion needs nothing -- which is precisely why
        swapping worked before and why cold-only broke it.

        Blocking, and before the tables flip: serving must not resume, nor the
        expert become routable, until its weights are present.
        """
        method = entry.get("method")
        if method is None:
            return

        # RANK-WRITE PATH. Every rank writes its own slice of the demoted
        # expert into kt's shared arena; rank 0 additionally does the
        # bookkeeping move. Deliberately BEFORE the `wrapper is None` return:
        # the peers have no kt wrapper and still must write, which is the
        # whole point.
        #
        # The move and the writes need no ordering between them. kt's move
        # only reassigns which expert id owns a BufferB -- it does not touch
        # the bytes -- so a peer writing at the promoted expert's offset is
        # writing exactly the address the demoted expert will own. The window
        # is quiesced, so nothing reads either expert in between. The one
        # ordering that matters (all writes land before serving resumes) is
        # the window-end barrier.
        # Unconditional: arming is a window-scoped MIN consensus that
        # terminates the server if any rank lacks a writer, and _begin_layer
        # sets rank_write for every layer it returns from. There is no second
        # source for a demoted expert's bytes any more.
        writer = _KT_SWAP_STATE["rank_writer"]
        t0 = time.perf_counter()
        # MOVE, COMMIT, THEN WRITE -- and never raise from here.
        #
        # Order: writing before the move would blit the demoted expert's
        # bytes over the PROMOTED expert's LIVE buffer, and an aborted
        # layer leaves that expert CPU-routable with corrupted weights.
        # The move must therefore come first; everything that could
        # refuse was hoisted into validate() at _begin_layer, before
        # anything moved, so the irreversible step is only taken once the
        # write is known to be possible.
        #
        # THERE IS NO ABORT PATH HERE, DELIBERATELY. Round 2 tried to make
        # a failure recoverable and made it worse: move_slot_only nulls
        # gate/up/down_bb_[promote] irreversibly, so a layer that aborts
        # after it leaves the promoted expert advertised as CPU-served
        # (the tables never flipped) with a null BufferB -- the next token
        # routed there null-derefs inside the AMX GEMM. The mirror image,
        # writing before the move, corrupts that expert's live weights
        # instead. Neither ordering has a safe unwind.
        #
        # And unwinding is not even available: kt's move runs on a
        # CPUInfer worker thread whose loop has no try/catch, so a C++
        # precondition throw terminates the PROCESS -- Python never sees
        # it. Safety therefore has to come from never entering the region
        # unless it will succeed, which is what validate() establishes at
        # _begin_layer (it mirrors kt's own checks on every partition) and
        # what the arming capability probe establishes once per process.
        # Anything that still raises here is a bug, and it propagates.
        if method.wrapper is not None:
            method.wrapper.move_expert_slot(promote_id, demote_id)
        # Every rank mirrors the move kt just made, wrapper or not: the
        # arena is shared, so a rank that misses one is silently a swap
        # behind for the rest of the process's life.
        writer.commit_move(entry["layer_idx"], promote_id, demote_id)
        if not writer.write(entry["layer_idx"], promote_id, demote_id):
            # kt's slot has already moved; there is no state to return to.
            _fatal_swap_failure(
                f"rank-write refused AFTER validation for demote="
                f"{demote_id} on layer {entry.get('layer_idx')} -- a bug "
                "in validate(), and kt's ownership has already moved"
            )
        _timing["install_s"] += time.perf_counter() - t0
        return

    def _begin_layer(entry, swaps, rows):
        """Read this layer's demoted experts off the GPU, before any move.

        Runs once per layer with the whole plan, so the collectives inside are
        keyed to the plan -- identical on every rank -- rather than to per-swap
        conditions. It also guarantees read-before-write unconditionally: no
        move has run yet, so every row still holds its demoted occupant.
        """
        filtered = None
        method = entry.get("method")

        # RANK-WRITE DEMOTIONS. Capture this rank's own slice of every
        # demoted expert BEFORE any move overwrites its GPU row -- purely
        # local, no collective. Whether it worked then rides the SAME single
        # per-layer collective below, so the layer is either rank-write on
        # every rank or on none: a rank that silently skipped its write would
        # leave a hole in the expert, which is wrong bytes rather than a
        # crash.
        _pending["rank_write"] = False
        writer = _KT_SWAP_STATE.get("rank_writer")
        captured = False
        if writer is not None and swaps:
            t0 = time.perf_counter()
            captured = writer.capture(
                entry["layer"],
                entry["layer_idx"],
                rows,
                [s.demote for s in swaps],
            )
            if not captured:
                _fatal_swap_failure(
                    f"rank-write capture failed on layer {entry.get('layer_idx')}"
                )
            # Every refusal, checked while everything is still undone: after
            # this the install performs an IRREVERSIBLE kt move.
            #
            # A refusal is FATAL, not a fallback. can_move mirrors kt's own
            # preconditions exactly, so refusing means this rank's table and
            # kt genuinely disagree -- and the "fallback" would hand that same
            # pair to swap_expert_slot, whose C++ re-checks the identical
            # conditions and throws on a worker thread with no handler, i.e.
            # std::terminate. Proving a pair illegal and then giving it to kt
            # anyway is strictly worse than stopping here.
            why = writer.validate(entry["layer_idx"], swaps)
            if why is not None:
                _fatal_swap_failure(
                    f"layer {entry.get('layer_idx')} not installable ({why}): "
                    "this rank's offset table and kt disagree"
                )
            _timing["read_s"] += time.perf_counter() - t0

        # NO per-layer consensus here, and none is needed: under the fail-fast
        # policy a rank that could not capture or validate has already
        # terminated every rank, so there is no surviving disagreement to
        # reconcile -- and arming itself is a window-scoped MIN that every rank
        # reached, so all eight take this branch or the server is already down.
        _pending["rank_write"] = True
        return filtered

    # after_flip / on_layer_abort existed ONLY to release the direct-dma
    # transport's provisional page pins. Both hooks are optional, so with that
    # transport gone they are not replaced by no-ops -- there is nothing left
    # to reconcile: the arena sources pin nothing per swap.
    def _timed_quiesce():
        # A full device sync: it drains whatever forward was in flight at the
        # decode boundary. Timed because it sits inside the window and is not
        # swap work -- charging it to the swap made the window look worse than
        # it is.
        _t = time.perf_counter()
        torch.cuda.synchronize(anchor.gpu_experts_mask_cuda.device)
        _timing["quiesce_s"] += time.perf_counter() - _t

    try:
        result = run_swap_window(
            entries,
            move_weights=_move,
            install_cpu_expert=_install_cpu,
            begin_layer=_begin_layer,
            finish_layer=_flush_moves,
            phase_timing=_timing,
            quiesce=_timed_quiesce,
        )
    finally:
        # Drain the last layer. In a finally because staged items surviving a
        # failed window would be applied during the NEXT one -- writing a stale
        # layer's rows, which is silent and unattributable.
        _flush_moves()
        # Rank-write demotions: every rank's slices must have LANDED before
        # serving resumes, or kt computes a demoted expert with another
        # rank's hole still in it. TWO orderings are needed and they are not
        # interchangeable:
        #
        #   1. THIS rank's D2H copies must be visible to THIS rank's CPU.
        #      RankShardWriter issues them with cudaMemcpyAsync on the current
        #      stream and never synchronizes, so nothing had made them
        #      host-visible: kt reads the arena from the CPU, and an async D2H
        #      is not complete just because the launching thread moved on.
        #   2. Every OTHER rank's slice must be in before anyone reads the
        #      whole expert -- that is the barrier.
        #
        # The stream sync must come FIRST. Barrier-then-sync lets a rank clear
        # the barrier with copies still in flight, which is the same hole in a
        # different place. This was previously carried by the barrier alone
        # plus the next forward's natural lockstep -- and the comment here
        # already said why that is not enough: "mostly" is not a memory
        # ordering. It is also where the demotion transfer was hiding, ~6.3 GB
        # per window at swap-max 32 that no span could see.
        #
        # Keyed to the WINDOW consensus, never to this rank's writer: a
        # barrier some ranks skip is a hang, and the writer is the one
        # precondition that is per-rank fallible.
        _rw = _KT_SWAP_STATE.get("rank_writer")
        _t_sync = time.perf_counter()
        torch.cuda.current_stream().synchronize()
        _timing["d2h_sync_s"] += time.perf_counter() - _t_sync
        if dist.is_initialized() and get_parallel().tp_size > 1:
            _t_bar = time.perf_counter()
            dist.barrier(group=get_tp_group().cpu_group)
            _timing["barrier_s"] += time.perf_counter() - _t_bar
        if _rw is not None:
            logger.info("%s", _rw.end_window())
    _KT_SWAP_STATE["windows"] += 1
    _KT_SWAP_STATE["swaps"] += result.swaps_applied
    if _KT_DOORBELL["inited"]:
        # Decode replays a graph that runs no Python, so this periodic window
        # is the only place decode-time transport counters can be observed.
        # Without it a doorbell that silently never bound a slot -- and so fell
        # back to host nodes everywhere -- would look exactly like a working
        # one: identical output, identical speed, and no way to tell which.
        logger.info("[kt-doorbell] %s", kt_doorbell_stats())
    if result.swaps_applied or result.skipped_layers:
        logger.info(
            "[kt-swap] window %d: %d swap(s) across %d layer(s), %d skipped "
            "(cumulative %d)",
            _KT_SWAP_STATE["windows"],
            result.swaps_applied,
            result.layers_touched,
            result.skipped_layers,
            _KT_SWAP_STATE["swaps"],
        )
        # Fetch and install are timed separately rather than one being
        # inferred from the other: the install is a memcpy plus a strided
        # restride of down_proj, with no format conversion, so an estimate
        # derived from a disk rate does not describe it.
        logger.info(
            "[kt-swap] window %d timing: total %.2fs = fetch %.2fs + install "
            "%.2fs (+%.2fs elsewhere)",
            _KT_SWAP_STATE["windows"],
            time.perf_counter() - _window_t0,
            _timing["read_s"],
            _timing["install_s"],
            (time.perf_counter() - _window_t0)
            - _timing["read_s"]
            - _timing["install_s"],
        )
        # The "elsewhere" term, attributed. Without this the only way to say
        # where a window's time goes is to guess, and two careful guesses
        # (the per-layer all_reduces; the pinned staging allocations) were
        # both wrong -- the GPU flush measures 0.09 s per window on this node.
        _phases = (
            "select_s", "rows_s", "begin_s", "move_s", "stage_s",
            "flush_gpu_s", "finish_s", "apply_s",
            "tables_s", "after_s", "arm_s", "quiesce_s", "barrier_s",
            "d2h_sync_s",
        )
        _attributed = sum(_timing[k] for k in _phases)
        logger.info(
            "[kt-swap] window %d elsewhere: %s = %.2fs attributed, "
            "%.2fs unattributed",
            _KT_SWAP_STATE["windows"],
            " + ".join(f"{k[:-2]} {_timing[k]:.2f}s" for k in _phases),
            _attributed,
            (time.perf_counter() - _window_t0)
            - _timing["read_s"]
            - _timing["install_s"]
            - _attributed,
        )


class _GpuResidentExpertReader:
    """Read a resident expert back out of GPU memory, in checkpoint layout.

    WHY. Under cold-only residency a demoted expert owns no CPU buffers, so
    every demotion must be given weights before it becomes routable. Reading
    them off the checkpoint costs tens of MB per demotion, which multiplied
    by the swap budget and the layer count was essentially the entire cost of a
    batched swap window. The bytes are already in device memory; only the
    layout differs.

    One thing stands between the resident row and the exported bytes: the
    trtllm-gen shuffle, inverted exactly by ``unswizzle_trtllm_expert``.

    NO COLLECTIVE, AND NO WHOLE EXPERT. Each rank returns its OWN shard and
    writes it into its own slice of kt's arena, which is what
    ``RankShardWriter`` consumes -- so TP never has to be undone. The
    all-gathering variants this class used to carry served the readback route
    that fed ``_read_demoted_expert``; that route was never enabled, and
    rank-write reaches the same bytes without a collective at all.

    Everything here is shape-derived and cached per layer shape, so the cost
    per demotion is the unswizzle alone.
    """

    def __init__(self, param_names):
        self.param_names = param_names
        self._by_shape = {}

    def _prepare(self, layer):
        """Inverse indices and scale shapes for this layer, cached per shape.

        Keyed by shape, not computed once: one reader serves every MoE layer,
        and a layer of a different shape reusing another's indices would
        produce wrong bytes of the right size -- exactly the failure the
        bitwise gate cannot catch once it has already passed on some other
        shape.

        The permutations depend only on shapes, so a sample of the right shape
        is enough and no checkpoint read is needed to build them. Scales carry
        one E8M0 code per 32 values while weights pack two values per byte, so
        a scale tensor is exactly 1/16 the columns of its weight.
        """
        from sglang.srt.layers.moe.kt_mxfp4_export import (
            trtllm_inverse_indices,
            trtllm_permute_indices,
        )

        w13_n, _, w2_n, _ = self.param_names
        w13 = getattr(layer, w13_n).data
        w2 = getattr(layer, w2_n).data
        dev = w13.device
        w13_shape = tuple(w13.shape[1:])
        w2_shape = tuple(w2.shape[1:])
        w13_scale_shape = (w13_shape[0], w13_shape[1] // 16)
        w2_scale_shape = (w2_shape[0], w2_shape[1] // 16)

        key = (w13_shape, w2_shape, str(dev))
        prepared = self._by_shape.get(key)
        if prepared is not None:
            return prepared

        def sample(shape):
            return torch.empty(shape, dtype=torch.uint8, device=dev)

        indices = trtllm_permute_indices(
            w13_sample=sample(w13_shape),
            w13_scale_sample=sample(w13_scale_shape),
            w2_sample=sample(w2_shape),
            w2_scale_sample=sample(w2_scale_shape),
            w13_gate_up_halves=True,
        )
        prepared = (
            trtllm_inverse_indices(
                indices,
                w13_scale_shape=w13_scale_shape,
                w2_scale_shape=w2_scale_shape,
                device=dev,
            ),
            w13_scale_shape,
            w2_scale_shape,
        )
        self._by_shape[key] = prepared
        return prepared

    def read_own_shards(self, layer, dst_rows):
        """THIS RANK's unswizzled shards for ``dst_rows`` -- no collective.

        The rank-write demotion path needs exactly this and nothing more: it
        writes its own slice into kt's shared arena, so it never wants the
        gathered full expert. Keeping it a separate method means the two
        paths share one proven unswizzle and one shape cache, and the
        collective lives only in the caller that actually needs it.

        Every row must still hold its DEMOTED occupant: call before any move.
        """
        from sglang.srt.layers.moe.kt_mxfp4_export import (
            Mxfp4ExpertBytes,
            apply_batched_unswizzle,
        )

        inverse, w13_scale_shape, w2_scale_shape = self._prepare(layer)
        w13_n, w13_s_n, w2_n, w2_s_n = self.param_names
        if not dst_rows:
            return []
        # ONE KERNEL PER TENSOR, not one per expert. This was a Python loop
        # calling the per-expert unswizzle -- the same per-expert-vs-per-batch
        # gap the forward swizzle already closed. The returned rows are VIEWS
        # into the batched result, so the caller's per-expert loop stays free.
        dev = getattr(layer, w13_n).data.device
        idx = torch.tensor(list(dst_rows), dtype=torch.long, device=dev)
        b13, b13s, b2, b2s = apply_batched_unswizzle(
            inverse=inverse,
            w13=torch.index_select(getattr(layer, w13_n).data, 0, idx),
            w13_scale=torch.index_select(getattr(layer, w13_s_n).data, 0, idx),
            w2=torch.index_select(getattr(layer, w2_n).data, 0, idx),
            w2_scale=torch.index_select(getattr(layer, w2_s_n).data, 0, idx),
            w13_scale_shape=w13_scale_shape,
            w2_scale_shape=w2_scale_shape,
        )
        return [
            Mxfp4ExpertBytes(
                w13=b13[i],
                w13_scale_e8m0=b13s[i],
                w2=b2[i],
                w2_scale_e8m0=b2s[i],
            )
            for i in range(len(dst_rows))
        ]


def _register_split_prefill_layer(method, layer) -> None:
    """Record a layer that armed split-slice prefill (post-load, per layer)."""
    layer_idx = method.kt_config.layer_idx
    for existing, _ in _KT_SPLIT_PREFILL_LAYERS:
        if existing.kt_config.layer_idx == layer_idx:
            raise RuntimeError(
                f"split-prefill: layer {layer_idx} registered twice -- call "
                f"reset_split_prefill() between engine constructions"
            )
    _KT_SPLIT_PREFILL_LAYERS.append((method, layer))


def reset_split_prefill() -> None:
    """Drop all split-prefill state (engine teardown / re-construction).

    The sources own process-level resources (the export source's rings and
    worker thread; the direct source's cudaHostRegister'd units and their
    VRAM page tables) with no finalizers -- dropping the reference without
    close() would leak them across an in-process engine reconstruction AND
    make the next direct-DMA arm fail on already-registered pages.
    """
    _KT_SPLIT_PREFILL_LAYERS.clear()
    for key in ():
        src = _KT_SPLIT_PREFILL_STATE.get(key)
        if src is not None:
            try:
                src.close()
            except Exception:
                logger.exception("[split-prefill] %s teardown failed", key)
    _KT_SPLIT_PREFILL_STATE["pipeline"] = None


def finalize_split_prefill(server_args) -> bool:
    """Build the cold-expert source and prefetch pipeline, then arm every layer.

    Called once after ALL layers have loaded -- the store needs the full MoE
    layer list, and the pipeline's slot parity is defined over it.  Returns
    True if the path is armed.

    On any failure the path is disarmed everywhere and serving continues on
    the existing margin-routed CPU path: a partially-built split-prefill would
    compute a subset of experts and silently degrade quality.
    """
    if not _KT_SPLIT_PREFILL_LAYERS:
        return False
    if _KT_SPLIT_PREFILL_STATE["pipeline"] is not None:
        # Already armed. Defence in depth behind the draft-worker gate in
        # ModelRunner: a second call here does not re-arm anything, it builds a
        # WHOLE SECOND cold store, whose only visible symptom is host
        # memory, because the layer list and the
        # arming consensus both look exactly the same the second time.
        logger.info(
            "[split-prefill] already armed on %d layers; ignoring a second "
            "finalize rather than rebuilding the store",
            len(_KT_SPLIT_PREFILL_LAYERS),
        )
        return True

    from sglang.srt.layers.moe.kt_mxfp4_export import WEIGHT_NAMES
    from sglang.srt.layers.moe.expert_pipeline import ColdExpertPipeline

    anchor, anchor_layer = _KT_SPLIT_PREFILL_LAYERS[0]
    layer_indices = [m.kt_config.layer_idx for m, _ in _KT_SPLIT_PREFILL_LAYERS]
    device = anchor_layer.w13_weight.device

    # Bound BEFORE the try so the except handler can tear down whatever was
    # built before the failure -- a raise after a successful direct-DMA build
    # (e.g. the pipeline's device buffers OOMing) must not orphan ~110 GB of
    # registrations and their VRAM page tables for the process lifetime.
    source = pipeline = None
    try:
        per_expert_shapes = {
            name: (tuple(getattr(anchor_layer, name).shape[1:]),
                   getattr(anchor_layer, name).dtype)
            for name in WEIGHT_NAMES
        }
        num_gpu = anchor.num_gpu_experts
        num_cold = anchor.global_num_experts - num_gpu

        # ONE COLD SOURCE. kt's memfd arena already holds every cold expert's
        # checkpoint-layout bytes, and kt_arena_dma has already registered this
        # rank's view of them -- one registration serving both directions, which
        # is why the writer is built just above -- so the copy engine reads them
        # IN PLACE: six pitched copies per layer, one DRAM transit, no prepare
        # stage and no second copy of anything.
        #
        # Four alternatives used to stand beside this and every one of them was
        # another copy of those same bytes: a 51 GiB/rank pinned store, kt's
        # batched raw export into per-rank rings, a Python gather over the
        # arena mapping, and a "direct-dma" transport with its own load-time
        # interval registration. The last two also required FULL kt residency
        # (896 experts, 1.35 TB of arenas) because their address plans go stale
        # when a swap moves a BufferB block -- which is exactly the staleness
        # the writer's offset table handles for us here.
        #
        # So there is no ladder left to fall down, and that is deliberate: a
        # fallback here does not save a boot, it serves a slower transport
        # under the name of the one that was asked for.
        raw_shapes = swizzle_plan = None
        num_gpu = anchor.num_gpu_experts
        num_cold = anchor.global_num_experts - num_gpu

        _dma_writer = _get_or_create_rank_writer({"method": anchor})
        if _dma_writer is None or getattr(_dma_writer, "_dma", None) is None:
            raise RuntimeError(
                "split prefill has no cold source: the arena-DMA writer did "
                "not arm. Read the [kt-rankwrite] and [kt] KT_BUFFER_B_MEMFD "
                "lines above to see why."
            )

        raw_shapes, swizzle_plan = _build_dynamic_swizzle_plan(anchor, device)
        if swizzle_plan is None or raw_shapes is None:
            # The arena is checkpoint layout; without the plan there is no way
            # to produce a resident row from it.
            raise RuntimeError(
                "arena-DMA cold source needs the swizzle plan and it could "
                "not be built"
            )

        from sglang.srt.layers.moe.kt_arena_dma import ArenaDmaColdSource

        source = ArenaDmaColdSource(
            dma=_dma_writer._dma,
            offsets_by_layer=_dma_writer._offsets,
            geometry=_dma_writer._g,
            layers=layer_indices,
            num_cold=num_cold,
            experts=anchor.global_num_experts,
        )
        logger.info(
            "[split-prefill] cold experts stream by DMA out of kt's "
            "registered arena: no pinned store built, %d GiB per rank "
            "saved, six pitched copies per layer",
            51,
        )

        pipeline = ColdExpertPipeline(
            source=source,
            device=device,
            per_expert_shapes=per_expert_shapes,
            moe_layer_indices=layer_indices,
            swizzle_plan=swizzle_plan,
            raw_shapes=raw_shapes,
        )
    except Exception:
        logger.exception(
            "[split-prefill] build failed; falling back to the margin-routed "
            "CPU path for every layer"
        )
        source = pipeline = None


    # Arming is all-or-nothing ACROSS RANKS. The hot gate is rank-local, and a
    # split rank set is silent corruption, not a crash: armed ranks contribute
    # full-expert shards while a disarmed rank contributes its margin-routed
    # resident-only shard, and the row-parallel all-reduce sums them into
    # every output token of every large prefill. A single rank's build is the
    # likeliest thing in this file to fail alone -- kt_arena_share degrades a
    # rank that could not map the arena, BY DESIGN -- so unanimity is decided
    # with the same symmetric consensus every other rank-divergence risk here
    # uses. Reached from the success AND failure paths, so it cannot itself
    # desynchronise. There is no per-rank fallback left to degrade INTO: the
    # whole split-prefill path disarms on every rank, together.
    if not _all_tp_ranks_succeeded(pipeline is not None):
        if pipeline is not None:
            logger.error(
                "[split-prefill] disarmed: another rank failed to build; "
                "every rank keeps the margin-routed CPU path"
            )
        for method, _ in _KT_SPLIT_PREFILL_LAYERS:
            method._split_prefill_ready = False
            _KT_SPLIT_PREFILL_STATE["pipeline"] = None
            return False

    _KT_SPLIT_PREFILL_STATE["pipeline"] = pipeline
    # The swap path needs these too: against a raw store a promotion must
    # swizzle on the way to the GPU and a demotion must unswizzle on the way
    # back, or the two sides silently disagree about layout.
    _KT_SPLIT_PREFILL_STATE["swizzle_plan"] = swizzle_plan
    # The checkpoint-layout shapes, kept because DMA promotion allocates its
    # landing buffers from them: the arena hands back raw bytes and
    # _swizzle_promoted_rows wants exactly these shapes.
    _KT_SPLIT_PREFILL_STATE["raw_shapes"] = raw_shapes
    for method, _ in _KT_SPLIT_PREFILL_LAYERS:
        method._cold_pipeline = pipeline
        method._split_prefill_ready = True

    logger.info(
        "[split-prefill] armed on %d layers: %d resident + %d cold experts, "
        "threshold %d tokens, cold source %s",
        len(_KT_SPLIT_PREFILL_LAYERS),
        anchor.num_gpu_experts,
        source.num_cold,
        anchor._split_prefill_threshold,
        "kt-arena",
    )

    # BUILD THE DEMOTION WRITER NOW, AT BOOT, not on the first window that
    # needs it. Registering the arena is the expensive part, and paying it
    # lazily puts that stall inside the first swap window -- inside serving,
    # where it looks like a pathological window rather than a one-off setup
    # cost. Everything it needs exists by this point: the arenas are mapped
    # (kt_arena_share ran per layer during load) and the layer list is
    # complete.
    #
    # Failure policy lives in the callee and in the window's arming consensus:
    # a writer that cannot be built terminates at the first window, because
    # there is no second source for a demoted expert's weights. Doing it here
    # only moves WHEN that is decided, which is itself worth something -- a
    # boot that cannot arm now fails at boot instead of minutes into serving.
    try:
        _get_or_create_rank_writer({"method": anchor})
    except Exception:
        logger.exception(
            "[kt-rankwrite] boot-time writer construction failed; the first "
            "window will retry"
        )
    return True


def _swizzle_promoted_rows(promoted):
    """Checkpoint-layout promoted rows -> the resident trtllm layout, on GPU.

    Against a raw store this is what replaces asking kt for a GPU-layout
    export. kt's own export costs 57-68 ms per layer's cold set, ~38.7 ms of it
    CPU work reading AMX buffers and expanding E8M0 scales to bf16 -- work this
    path would immediately undo, since the resident layout wants those codes
    back as bytes. Doing the transform on device instead touches no CPU at all
    and reuses the same four-gather form split prefill uses.
    """
    from sglang.srt.layers.moe.kt_mxfp4_export import (
        apply_batched_swizzle,
        swizzle_out_buffers,
    )

    plan = _KT_SPLIT_PREFILL_STATE.get("swizzle_plan")
    if plan is None:
        raise RuntimeError(
            "raw cold store but no swizzle plan: promotions cannot be written "
            "to a resident row without one"
        )
    names = _MXFP4_TRTLLM_RESIDENT_PARAM_NAMES
    # Reuse the gather destinations across layers. Their shape is fixed by the
    # plan and the batch width, and the batch is now processed at a constant
    # width, so one set serves every layer of every window. Without this the
    # four gathers allocate on each call -- the other half of the per-layer
    # allocation cost that the fixed-width batch addressed.
    _swz = _KT_SWAP_STATE.get("swizzle_out")
    if _swz is None or _swz[0].shape[0] != promoted[names[0]].shape[0]:
        _swz = swizzle_out_buffers(
            plan=plan,
            raw_w13=promoted[names[0]],
            raw_w13_scale=promoted[names[1]],
            raw_w2=promoted[names[2]],
            raw_w2_scale=promoted[names[3]],
        )
        _KT_SWAP_STATE["swizzle_out"] = _swz
    out = apply_batched_swizzle(
        out=_swz,
        plan=plan,
        raw_w13=promoted[names[0]],
        raw_w13_scale=promoted[names[1]],
        raw_w2=promoted[names[2]],
        raw_w2_scale=promoted[names[3]],
    )
    return dict(zip(names, out))


def _build_dynamic_swizzle_plan(anchor, device):
    """Raw per-expert shapes and the per-layer swizzle maps, from one sample.

    Both are shape-derived, so a single expert settles them for every layer.
    Returns ``(raw_shapes, plan)`` in ``kt_mxfp4_export.WEIGHT_NAMES`` order, or
    ``(None, None)`` if anything is missing -- in which case the caller
    refuses to arm rather than guessing.

    THE SAMPLE IS A RULER, NOT DATA. ``trtllm_permute_indices`` builds a row
    permutation, and the builder underneath it caches on ``tuple(x.shape)``
    plus the tile size and device -- it never reads a byte of the tensor. So
    the sample only has to have the right SHAPE, and this reads NO expert
    bytes: ``raw_shard_shapes`` states them arithmetically.

    Two earlier versions did read them. The first pulled an expert off the
    CHECKPOINT through a whole expert-mover module; the second materialised a
    real arena shard, a 2.19 MB copy out of mapped memory plus a 2.19 MB
    host-to-device copy, at boot, for bytes nobody looks at. Both were paying
    for data to measure it.

    The shapes come from ``kt_ram_source`` rather than being recomputed here on
    purpose: that module owns the axis arithmetic, and a second copy of it
    beside the plan builder is free to drift from the shard it is supposed to
    describe. The write source is registered per layer during load and the rank
    writer is built just above this call, so it is available by construction.
    """
    from sglang.srt.layers.moe.kt_arena_share import arena_write_source_for
    from sglang.srt.layers.moe.kt_mxfp4_export import (
        WEIGHT_NAMES,
        trtllm_batched_swizzle,
        trtllm_permute_indices,
    )

    try:
        cold = torch.where(~anchor.gpu_experts_mask)[0].tolist()
        if not cold:
            return None, None
        li = anchor.kt_config.layer_idx
        source = arena_write_source_for(li)
        if source is None:
            logger.error(
                "[split-prefill] no arena source for layer %d; cannot derive "
                "the swizzle plan",
                li,
            )
            return None, None
        shapes = source.raw_shard_shapes()
        # uint8 throughout: kt's arena is mapped as uint8 and the shard slicing
        # never changes dtype, which raw_shard_shapes states in its docstring.
        # empty(), not zeros(): nothing reads these.
        on_dev = {
            n: torch.empty(shapes[k], dtype=torch.uint8, device=device)
            for n, k in zip(
                WEIGHT_NAMES, ("w13", "w13_scale", "w2", "w2_scale")
            )
        }
        raw_shapes = {n: (tuple(t.shape), t.dtype) for n, t in on_dev.items()}
        w13, w13_scale, w2, w2_scale = (on_dev[n] for n in WEIGHT_NAMES)
        indices = trtllm_permute_indices(
            w13_sample=w13,
            w13_scale_sample=w13_scale,
            w2_sample=w2,
            w2_scale_sample=w2_scale,
            w13_gate_up_halves=True,
        )
        plan = trtllm_batched_swizzle(
            indices,
            w13_scale_shape=tuple(w13_scale.shape),
            w2_scale_shape=tuple(w2_scale.shape),
            device=device,
        )
        logger.info(
            "[split-prefill] dynamic swizzle armed from kt arena: raw w13 %s, "
            "w2 %s",
            tuple(w13.shape),
            tuple(w2.shape),
        )
        return raw_shapes, plan
    except Exception:
        logger.exception(
            "[split-prefill] could not build the dynamic swizzle plan"
        )
        return None, None


def _fatal_swap_failure(context: str) -> None:
    """A swap window failed: take the whole server down. Never returns.

    POLICY (user decision, 2026-08-17): do not recover, do not verify, do not
    continue serving -- terminate and let the process be restarted.

    It is the only sound response, and four review rounds converged on why.
    A window mutates three things that must agree: kt's BufferB ownership,
    the resident GPU rows, and the routing tables. They are committed at
    different points, by different ranks, and the pieces cannot be unwound --
    ``move_slot_only`` nulls the promoted entry with no inverse, and a kt-side
    throw runs on a CPUInfer worker with no handler, so it is std::terminate
    anyway. Every alternative was tried and each was worse:

      * skip the layer and keep serving (the old behaviour) -- the window-end
        drain then flushes that layer's STAGED promotions into rows whose
        tables never flipped: silent wrong weights, indefinitely;
      * raise on one rank -- seven peers keep issuing per-layer collectives
        while it leaves for the barrier: the M9/M11/M12 hang;
      * agree on an abort first -- the consensus itself became a collective a
        rank could skip, and the state it "recovered" to still had kt's slot
        moved under unflipped tables.

    Killing every rank makes all of that unreachable: a dead server serves no
    tokens, and the next boot rebuilds kt's ownership from the checkpoint,
    which is the only state that is trustworthy by construction. Failures
    here have never been observed (99 windows, ~65k swaps, 0 skipped), so
    this trades an unmeasurable availability cost for the removal of a silent
    corruption class.
    """
    import os
    import signal

    from sglang.srt.utils import kill_process_tree

    logger.critical(
        "[kt-swap] FATAL: %s -- terminating every rank. The swap window's "
        "partial state (kt ownership moved, rows staged, tables unflipped) "
        "cannot be reconciled in-process; restart to rebuild it from the "
        "checkpoint.",
        context,
        exc_info=True,
    )
    for handler in list(logging.getLogger().handlers) + list(logger.handlers):
        try:
            handler.flush()
        except Exception:
            pass
    # ONE SYSCALL, NOT A LOOP. kill_process_tree(os.getppid()) looks right and
    # is not: it enumerates the launcher's children -- a list that CONTAINS
    # this rank -- and SIGKILLs them in /proc order, so when it reaches our
    # own pid we die mid-loop and every later rank plus the parent survive
    # (reproduced: a fatal on rank 3 left ranks 4-7 running and the launcher
    # alive; ranks spawn in ascending pid order, so a fatal on rank 0 kills
    # almost nobody). The survivors then hang on the window-end barrier the
    # dead ranks never join, behind a live HTTP front end.
    #
    # killpg is atomic with respect to ordering: every rank, the launcher and
    # the HTTP server share the process group under launch.sh, and the signal
    # is delivered to all of them regardless of where we are in the group.
    # SIGKILL rather than a graceful teardown on purpose -- we are abandoning
    # in-memory state deliberately, so there is nothing to flush, and a
    # graceful path would try to run the very collectives that are broken.
    # Nothing leaks: the cold rings are unlinked at creation, kt's arenas are
    # memfds freed when the last reference drops, and pinned host memory is
    # ordinary process memory.
    try:
        os.killpg(os.getpgrp(), signal.SIGKILL)
    except Exception:
        logger.exception("[kt-swap] killpg failed; falling back to the tree")
        try:
            # skip_pid matters for the same reason: without it this kills us
            # before it reaches the peers.
            kill_process_tree(os.getppid(), skip_pid=os.getpid())
        except Exception:
            logger.exception("[kt-swap] could not kill the process tree")
    os._exit(70)  # EX_SOFTWARE; only reached if the group kill somehow missed us


def _get_or_create_rank_writer(entry):
    """Process-wide rank-write demotion writer, or None.

    None is fatal at the next swap window: it is the ONLY source of a demoted
    expert's weights, since cold-only residency leaves that expert owning no
    CPU buffers. The window's arming consensus turns it into a terminate.

    Built once, at boot (finalize_split_prefill) so the arena registration
    that direct DMA needs is not paid inside the first serving window. The
    first window still calls this and gets the cached instance; the lazy path
    remains only as the fallback if boot-time construction was skipped.
    Every precondition is uniform across ranks by construction -- the env
    gate, the launch config, and whether kt exposes the slot move -- EXCEPT
    the arena mapping, which `kt_arena_share` degrades per rank by design.
    That last one is why the caller folds the outcome into the per-layer
    consensus instead of trusting this to agree.
    """
    if "rank_writer" in _KT_SWAP_STATE:
        return _KT_SWAP_STATE["rank_writer"]

    writer = None
    try:
        method = entry.get("method")
        if method is not None and method.kt_config.split_prefill:
            from sglang.srt.layers.moe.kt_arena_share import (
                arena_write_source_for,
            )
            from sglang.srt.layers.moe.kt_arena_dma import (
                RankShardWriter,
                SlotOffsets,
            )
            from sglang.srt.layers.moe.kt_arena_geometry import ArenaExpertRanges

            # CAPABILITY PROBE, once, before anything can move. The install
            # region is infallible by construction only if kt actually
            # exposes the bookkeeping-only slot move; a kt built before
            # c68d1ce raises NotImplementedError from inside the region, and
            # a kt-side C++ throw runs on a CPUInfer worker whose loop has no
            # try/catch -- it terminates the process rather than surfacing as
            # an exception. So this is checked while refusing is still free.
            for m in _KT_EP_METHODS:
                if m.wrapper is not None and not hasattr(
                    m.wrapper, "move_expert_slot"
                ):
                    raise RuntimeError(
                        "this kt build has no move_expert_slot (needs kt "
                        "c68d1ce or later); rank-write cannot arm"
                    )

            layers = [m.kt_config.layer_idx for m in _KT_EP_METHODS]
            # The WRITE registry: under cold-only the read one stays empty by
            # design, and borrowing from it would hand the promotion path
            # offsets that go stale at the first swap.
            sources = {li: arena_write_source_for(li) for li in layers}
            missing = [li for li, s in sources.items() if s is None]
            if missing:
                raise RuntimeError(
                    f"no arena mapping for {len(missing)} layer(s) "
                    f"(first {missing[0]}) -- is KT_BUFFER_B_MEMFD=1 set?"
                )
            geom = ArenaExpertRanges(next(iter(sources.values())))
            arenas = {li: s._arenas[geom.part] for li, s in sources.items()}
            dma = None
            if method.kt_config.split_prefill:
                # One registration per (layer, partition) MAPPING -- 92 of
                # ~2275 MiB, not the 150,144 per-expert ranges that made the
                # direct-DMA transport fail with rc=2. Cost is per page, so
                # expect seconds and a few hundred MB of page tables, once.
                from sglang.srt.layers.moe.kt_arena_dma import ArenaDmaWriter
                from sglang.srt.layers.moe.kt_arena_geometry import (
                    CudaCopyLib,
                    cudart_register_fns,
                )

                # NO FALLBACK. Direct DMA is opt-in, so a failure here means
                # the operator asked for a transport the machine will not give
                # -- and the alternatives are both bad in a way that hides it:
                # degrading to the staged host path silently returns the DRAM
                # hop this exists to delete, and the outer handler disarms
                # rank-write entirely and sends every demotion back to the
                # checkpoint read (~28 s per window against ~0.5 s). Neither
                # should be discovered from a throughput graph a day later.
                #
                # THE FAILURE YOU WILL ACTUALLY SEE, and it is not a bug in
                # this code: cudaHostRegister rc=1 partway through the arenas.
                # It pins EXISTING pages, so it is charged against
                # RLIMIT_MEMLOCK -- unlike the cudaHostAlloc backing a pinned
                # store, which is why a much larger store can allocate fine on
                # a box where registering the arena does not. Check `ulimit -l`;
                # an unprivileged container may cap it low enough that nothing
                # can be done from inside, and docker then needs
                # `--ulimit memlock=-1`.
                _t_reg = time.perf_counter()
                try:
                    reg_fn, _ = cudart_register_fns()
                    dma = ArenaDmaWriter(
                        arena_by_layer=arenas,
                        geometry=geom,
                        copy_lib=CudaCopyLib(),
                        register_fn=reg_fn,
                    )
                except Exception:
                    logger.exception(
                        "[kt-rankwrite] direct DMA (--kt-expert-split-prefill) "
                        "could not arm; "
                        "check `ulimit -l` -- cudaHostRegister is charged "
                        "against RLIMIT_MEMLOCK"
                    )
                    _fatal_swap_failure(
                        "direct DMA requested but the arena could not be "
                        "registered"
                    )
                logger.info(
                    "[kt-rankwrite] direct DMA armed: registered %d arena "
                    "mapping(s), %.0f GiB, in %.1fs -- demotions go GPU -> kt "
                    "in one hop, no pinned staging and no host memcpy",
                    len(arenas),
                    dma.registered_bytes / (1 << 30),
                    time.perf_counter() - _t_reg,
                )
            writer = RankShardWriter(
                arena_by_layer=arenas,
                offsets_by_layer={
                    li: SlotOffsets(s._rows, experts=s.experts, numa=s.numa)
                    for li, s in sources.items()
                },
                geometry=geom,
                shard_reader=_GpuResidentExpertReader(
                    _MXFP4_TRTLLM_RESIDENT_PARAM_NAMES
                ),
                dma=dma,
            )
            logger.info(
                "[kt-rankwrite] armed on %d layers: this rank writes its own "
                "slice (partition %d, local rank %d) of every demoted expert "
                "-- no gather, no checkpoint read",
                len(sources),
                geom.part,
                geom.local_rank,
            )
    except Exception:
        logger.exception(
            "[kt-rankwrite] could not arm; the next swap window terminates -- "
            "a demoted expert owns no CPU buffers under cold-only residency "
            "and this writer is the only thing that gives it any"
        )
        writer = None

    _KT_SWAP_STATE["rank_writer"] = writer
    return writer


# ---------------------------------------------------------------------------
# Doorbell transport (--kt-transport doorbell)
# ---------------------------------------------------------------------------
# Replaces the two cudaLaunchHostFunc nodes per layer with device value
# writes plus a wait node, served by a spinning CPU poller in kt-kernel.
# Measured target: 62.4 us/layer, 77% of the residue left after 7004e15.
#
# PROTOCOL. A captured graph node writes a CONSTANT -- whatever value is
# recorded is what every replay writes. A monotonic sequence therefore cannot
# work: it advances once at capture, after which the poller's "changed?" test
# is false forever and the GPU's wait is already satisfied by the stale value,
# so the CPU experts stop computing while the merge keeps reading the first
# replay's output. Silently wrong, and invisible to a smoke test. The protocol
# is instead built from constants that stay correct under unlimited replay:
#
#     arm    write completion[slot] = 0
#     stage  D2H of activations + ids
#     ring   write ring = slot + 1
#     wait   completion[slot] == slot + 1
#
# See kt-kernel cpu_backend/doorbell.h for the poller half and for why the
# ring is a single global word rather than a per-slot flag.

# Slots are (layer, batch size) pairs. 92 MoE layers x ~52 captured decode
# tiers is ~4.8k; the page is 128 B/slot, so sizing generously costs ~1 MB of
# pinned memory and removes a hard failure at the tail of the tier list.
_KT_DOORBELL_MAX_SLOTS = 8192

_KT_DOORBELL = {"inited": False, "next_slot": 0}


def _kt_doorbell_ext():
    # The extension is a submodule of the kt_kernel package, not a top-level
    # module: `import kt_kernel_ext` raises ModuleNotFoundError.
    from kt_kernel import kt_kernel_ext

    return kt_kernel_ext.doorbell


def kt_doorbell_init(num_pollers: int) -> None:
    """Allocate the doorbell page and start the poller. Idempotent.

    Must run before any graph capture: cudaHostAlloc is illegal during
    capture, and the captured nodes bake this page's device addresses.
    """
    if _KT_DOORBELL["inited"]:
        return
    db = _kt_doorbell_ext()
    db.init(_KT_DOORBELL_MAX_SLOTS, num_pollers)
    db.start()
    _KT_DOORBELL["inited"] = True


def kt_doorbell_bind_slot(method, staging_buffer, topk_ids) -> int:
    """Allocate and bind this (layer, batch size) pair's slot.

    Binding is host-only -- it hands kt-kernel the ring pointers for this
    batch size and stores a closure -- so it is safe on a tier's first forward
    even inside that tier's graph capture.
    """
    slot = _KT_DOORBELL["next_slot"]
    if slot >= _KT_DOORBELL_MAX_SLOTS:
        raise RuntimeError(
            f"doorbell: out of slots ({_KT_DOORBELL_MAX_SLOTS}); raise "
            "_KT_DOORBELL_MAX_SLOTS"
        )
    _KT_DOORBELL["next_slot"] = slot + 1
    method.wrapper.register_doorbell_slot(slot, staging_buffer, topk_ids)
    # Binding happens during warmup/capture, where Python still runs, so this
    # is the one place the transport can prove it is live in a run with
    # swapping off (the swap window is the only decode-time Python hook). A
    # doorbell that bound nothing falls back to host nodes everywhere and is
    # otherwise indistinguishable from a working one.
    if slot < 3 or slot % 256 == 0:
        logger.info(
            "[kt-doorbell] bound slot %d (layer batch size %d, %d bound so far)",
            slot,
            staging_buffer.shape[0],
            slot + 1,
        )
    return slot


def kt_doorbell_arm(slot: int, stream) -> None:
    """Retract the previous replay's completion for this slot.

    Without this a replay's wait is satisfied the instant it is reached, by
    the value the poller wrote last time -- the CPU path would appear to work
    while contributing nothing but stale data.
    """
    from cuda.bindings import driver

    db = _kt_doorbell_ext()
    (err,) = driver.cuStreamWriteValue64(
        stream, driver.CUdeviceptr(db.completion_dev_addr(slot)), 0, 0
    )
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"doorbell arm failed: {err}")


def kt_doorbell_ring(slot: int, stream) -> None:
    """Record the device-side ring for this slot.

    MUST be recorded AFTER the ids/activation D2H copies. The poller's whole
    decision reads those ids; if the ring became visible first it would read
    the PREVIOUS step's batch and could declare a batch empty that is not --
    wrong numbers, silently. Ordering here is by stream position.
    """
    from cuda.bindings import driver

    db = _kt_doorbell_ext()
    (err,) = driver.cuStreamWriteValue64(
        stream, driver.CUdeviceptr(db.ring_dev_addr()), slot + 1, 0
    )
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"doorbell ring failed: {err}")


def kt_doorbell_wait(slot: int, stream) -> None:
    """Block the stream until the poller publishes this slot's completion."""
    from cuda.bindings import driver

    db = _kt_doorbell_ext()
    (err,) = driver.cuStreamWaitValue64(
        stream,
        driver.CUdeviceptr(db.completion_dev_addr(slot)),
        slot + 1,
        driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ,
    )
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"doorbell wait failed: {err}")


def kt_doorbell_stats() -> dict:
    """Poller counters, for the transport gates."""
    db = _kt_doorbell_ext()
    served = db.served()
    return {
        "served": served,
        "spins": db.spins(),
        "unbound": db.unbound(),
        "slots_bound": _KT_DOORBELL["next_slot"],
        "work_us_mean": (db.work_ns_total() / served / 1000.0) if served else 0.0,
        "work_us_max": db.work_ns_max() / 1000.0,
    }
