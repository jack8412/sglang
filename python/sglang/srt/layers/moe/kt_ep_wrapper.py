# SPDX-License-Identifier: Apache-2.0
"""
KT Expert Parallelism Wrapper for MoE layers.

This module provides a generic wrapper that enables CPU-GPU expert parallelism
for any MoE quantization method. It coordinates parallel execution of GPU experts
(using any quantization method) and CPU experts (using AMX/AVX instructions).

Diagnostic / escape-hatch environment variables (KT-DEBUG-ONLY; not for prod):

    SGLANG_DEBUG_KT_HYBRID_TIMING=1 (legacy alias: SGLANG_KT_HYBRID_TIMING)
        Per-call wall-time breakdown of submit / mask / gpu / sync / merge
        / cpu_wait stages. Logged at DEBUG for layers (0, 5, 20, 35) on TP0.

    SGLANG_DEBUG_KT_HYBRID_TIMING_DEEP=1 (legacy alias: SGLANG_KT_HYBRID_TIMING_DEEP)
        Insert torch.cuda.synchronize() at each timing stage so DEEP numbers
        reflect real GPU work rather than async-launch return time. Slows
        decode meaningfully; only enable for one-shot triage.

    SGLANG_DISABLE_KT_CPU_STREAM=1 (legacy alias: SGLANG_KT_HYBRID_NO_CPU_STREAM)
        Collapse the CPU-experts CUDA stream onto the main stream. Useful
        when isolating regressions caused by the multi-stream submit path.

    SGLANG_DEBUG_KT_BYPASS_GPU_MOE=1 (legacy alias: SGLANG_KT_BYPASS_GPU_MOE)
        Force GPU-experts apply() to a zero return; routed expert output
        comes purely from the CPU side. "Plan-C" fallback for diagnosing
        whether a regression sits in the GPU MoE path or the merge math.
"""

import bisect
import copy
import ctypes
import gc
import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, replace
from multiprocessing import shared_memory
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import msgspec
import torch
import torch.distributed as dist

from sglang.srt.distributed import get_tp_group
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_buffer, get_parallel, get_stream
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from sglang.srt.layers.quantization.marlin_utils import marlin_permute_scales
from sglang.srt.utils import get_compiler_backend, is_cuda

if is_cuda():
    from sglang.kernels.ops.quantization.gptq_marlin_repack import gptq_marlin_repack

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
        gpu_prefill_token_threshold: token threshold for enabling full GPU fallback
        kt_enable_dynamic_expert_update: Enable dynamic GPU expert updates based on runtime statistics
        routing_margin: Router-logit margin for GPU-preferred routing overrides
            (None = feature off, bit-exact routing; 0.0 = count-only)
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
    gpu_prefill_token_threshold: Optional[int] = None
    kt_enable_dynamic_expert_update: bool = False
    routing_margin: Optional[float] = None
    routing_full_override: bool = False
    transport: str = "hostnode"
    transport_pollers: int = 2
    conditional_cpu_branch: bool = False
    cold_only_cpu_experts: bool = False
    expert_swap_interval: int = 0
    expert_swap_max: int = 4
    expert_swap_hysteresis: float = 2.0
    split_prefill: bool = False
    split_prefill_token_tile: int = 0
    cold_transport: str = "ring-export"


# Process-level registries for the MXFP4 layerwise-prefill slot machinery
# (mutated in place only — no module rebinding; carried from ktfork #72/#73).
# Every wrapped MoE layer, in construction order, so the swap driver can walk
# them. Registered unconditionally: the layerwise-prefill registry below is
# populated only when gpu_prefill_token_threshold > 0, and the validated
# recipe runs 0, which would leave a swap driver with nothing to iterate.
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
    "store": None,
    "pipeline": None,
    "export_source": None,
    "direct_source": None,
}

# Smallest chunk worth paying the cold-expert stream for. The stream is a
# fixed per-forward cost (~2.0 s measured: every cold expert lands once
# regardless of token count), so the split path beats the ~1,400 tok/s
# CPU-expert path above ~2,800 tokens. Rounded up for margin.
_SPLIT_PREFILL_MIN_TOKENS = 4096

_MXFP4_PREFILL_LAYER_REGISTRY = {}
_MXFP4_LAYERWISE_MANAGERS = {}
_MXFP4_LAYERWISE_DISABLED_REASONS = {}

# Prepared-slot target layouts for the MXFP4 layerwise-prefill pipeline.
# DSV4's DeepSeekMxfp4MoEMethod consumes prepared Marlin weights; K3's native
# Mxfp4MoEMethod consumes the trtllm-gen shuffled layout (the SiTU-capable
# B200 kernel — Marlin's epilogue lacks SiTU, so it is never a K3 target).
_MXFP4_LAYOUT_MARLIN = "marlin"
_MXFP4_LAYOUT_TRTLLM = "trtllm"

# Host SHM staging ring for MXFP4 transports.  The layerwise manager consumes
# the ring as two half-ring chunks so one TP consensus + one kt sync covers
# ring/2 experts instead of one; serial transports keep using slots 0/1.
_KT_MXFP4_HOST_RING_DEPTH = 64

# Raw (pre-swizzle) slot tensor names per prepared layout.  They mirror the
# attribute names each GPU method's create_weights registers:
#   marlin: DeepSeekMxfp4MoEMethod.create_weights (mxfp4_deepseek.py L151-180)
#   trtllm: Mxfp4MoEMethod.create_weights (mxfp4.py L475-533)
_MXFP4_RAW_NAMES_BY_LAYOUT = {
    _MXFP4_LAYOUT_MARLIN: (
        "w13_weight",
        "w13_weight_scale_inv",
        "w2_weight",
        "w2_weight_scale_inv",
    ),
    _MXFP4_LAYOUT_TRTLLM: (
        "w13_weight",
        "w13_weight_scale",
        "w2_weight",
        "w2_weight_scale",
    ),
}

# Resident trtllm-gen parameter names (F2 dynamic expert update).
# ``Mxfp4MoEMethod.process_weights_after_loading`` rebinds exactly these four
# attributes to the shuffled weight stacks and interleaved fp8-viewed scale
# stacks (mxfp4.py L827-830), so the raw names double as the prepared names.
_MXFP4_TRTLLM_RESIDENT_PARAM_NAMES = _MXFP4_RAW_NAMES_BY_LAYOUT[
    _MXFP4_LAYOUT_TRTLLM
]


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


class SharedFullContext:
    def __init__(
        self,
        layer: torch.nn.Module,
        init_args: tuple,
        global_num_experts: int,
        moe_runner_config: "MoeRunnerConfig",
        defer_cpu_buffers: bool = False,
    ):
        self._build_layers(layer, init_args, global_num_experts, moe_runner_config)

        # Capture original tensors to support restoration before loading
        self.original_params = {
            name: param for name, param in self.gpu_layer.named_parameters()
        }
        self.original_buffers = {
            name: buf for name, buf in self.gpu_layer.named_buffers()
        }

        # The MXFP4 layerwise manager defers SHM creation until every TP rank
        # has successfully allocated both GPU layer slots.  This keeps an OOM
        # on one rank from stranding peers inside an SHM collective.
        self._cpu_buffers_initialized = False
        if not defer_cpu_buffers:
            self.initialize_cpu_buffers()

        # For M3 MXFP8 layerwise prefill, cache the canonical uint8 ue8m0
        # scale Parameter objects so `_prepare_weight_mxfp8` can rebind to
        # them before each byte-copy. Native MXFP8 path keeps the scale in
        # this layout throughout (no convert to block-fp8).
        if getattr(self, "_is_mxfp8_quant", False):
            self._init_mxfp8_aux()

    def initialize_cpu_buffers(self) -> None:
        if self._cpu_buffers_initialized:
            return
        self._create_cpu_buffers()
        self._cpu_buffers_initialized = True

    def _init_mxfp8_aux(self) -> None:
        """Cache the canonical uint8 ue8m0 [E, N, K//32] scale Parameters.

        Fp8MoEMethod.create_weights (use_mxfp8=True) allocated them; we just
        save references so the shadow gpu_layer can rebind back to the same
        objects across rounds of layerwise prefill.
        """
        layer = self.gpu_layer
        self._w13_scale_mxfp8_param = layer.w13_weight_scale_inv
        self._w2_scale_mxfp8_param = layer.w2_weight_scale_inv

    def _build_layers(self, layer, init_args, global_num_experts, moe_runner_config):
        from sglang.srt.layers.moe.fused_moe_triton.layer import (
            UnquantizedFusedMoEMethod,
        )

        hidden_size, intermediate_size_per_partition, params_dtype = init_args
        target_device = next(layer.parameters()).device

        # Create gpu_layer as a shallow copy, then override specific attributes
        self.gpu_layer = copy.copy(layer)
        # Clear module state that shouldn't be shared
        self.gpu_layer._parameters = {}
        self.gpu_layer._buffers = {}
        self.gpu_layer._modules = {}

        # Override expert counts for full GPU execution
        self.gpu_layer.num_experts = global_num_experts
        self.gpu_layer.num_local_experts = global_num_experts
        self.gpu_layer.num_gpu_experts = global_num_experts

        # Create quant_method for gpu_layer
        if self.gpu_layer.quant_config is not None:
            self.gpu_method = self.gpu_layer.quant_config.get_quant_method(
                self.gpu_layer, prefix=""
            )
        else:
            self.gpu_method = UnquantizedFusedMoEMethod(
                self.gpu_layer.use_triton_kernels
            )
        # V4-Flash routed experts are MXFP4-packed (FP4 e2m1 in int8 + ue8m0
        # scales) but quant_config.get_quant_method picks Fp8MoEMethod
        # (V4-Flash uses FP8 for attn / shared experts). The default
        # mxfp4_deepseek pipeline wraps Fp8MoEMethod with DeepSeekMxfp4MoEMethod
        # for V4 routed experts; replicate that wrap here so kt_ep_wrapper's
        # gpu_method correctly handles MXFP4 (shape K = hidden, not hidden/2
        # as the FP8 path assumes), and so the capability-driven dispatch in
        # mxfp4_deepseek.apply (trtllm vs triton_kernels) fires on the
        # wrapped path. Gate by `--kt-method MXFP4` (an explicit user choice
        # that means "this run uses MXFP4 routed experts on the kt path"),
        # NOT by SGLANG_V4_USE_TRITON_KERNELS (which is now a diagnostic
        # override only, see v4_triton_kernels_moe.use_v4_triton_kernels
        # docstring). Origin: kt-sglang 耦合 (V4-Flash routed experts MXFP4
        # detection in kt_ep_wrapper).
        try:
            from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
            from sglang.srt.layers.quantization.mxfp4_deepseek import (
                DeepSeekMxfp4MoEMethod,
            )
            from sglang.srt.runtime_context import get_exec
            _v4_env = envs.SGLANG_V4_USE_TRITON_KERNELS.get()
            if _v4_env == "1":
                _do_v4_wrap = True
            elif _v4_env == "0":
                _do_v4_wrap = False
            else:
                _do_v4_wrap = (
                    (get_exec().moe.kt_method or "").upper() == "MXFP4"
                )
            if _do_v4_wrap and isinstance(self.gpu_method, Fp8MoEMethod):
                self.gpu_method = DeepSeekMxfp4MoEMethod(self.gpu_method, prefix="")
        except Exception as _v4_tk_wrap_exc:
            logger.warning(
                f"[kt-ep-wrapper] V4-Flash MXFP4 wrap skipped: {_v4_tk_wrap_exc}"
            )
        self.gpu_layer.quant_method = self.gpu_method

        self.gpu_method.create_weights(
            layer=self.gpu_layer,
            num_experts=global_num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
        )

        # Detect quantization type for weight loading based on actually created weights.
        # This is more robust than class-based detection when quant methods are wrapped
        # (e.g., KT wrapper -> compressed-tensors scheme), especially in layerwise prefill.
        # Run once at init.  Freeze the result into _is_* so downstream
        # always sees the original quant type even if a Marlin repack
        # later renames attributes (e.g. _inv → _weight_scale).
        self._detect_quant_type_from_created_weights()
        for _attr in ("is_mxfp4_quant", "is_mxfp8_quant", "is_fp8_quant",
                       "is_fp8_channel_quant", "is_bf16_quant"):
            if hasattr(self, _attr):
                setattr(self, f"_{_attr}", getattr(self, _attr))

        # Move all parameters to target device
        for param in self.gpu_layer.parameters():
            if param.device != target_device:
                param.data = param.data.to(target_device)

        # Save weight/scale shapes after device move so _restore_raw_attrs
        # can re-create tensors on the correct device later.
        self._raw_weight_shapes = {}
        for _sn in self.weight_names:
            if hasattr(self.gpu_layer, _sn):
                _t = getattr(self.gpu_layer, _sn)
                self._raw_weight_shapes[_sn] = (tuple(_t.shape), _t.dtype, _t.device)

        # Create runner config - update both num_experts and num_local_experts for full GPU fallback
        # Set routed_scaling_factor=None to avoid double scaling:
        # - moe_sum_reduce would apply routed_scaling_factor internally
        # - deepseek_v2.py forward_normal also applies routed_scaling_factor for KTEPWrapperMethod
        # By setting it to None here, we ensure it's only applied once in forward_normal
        runner_config = replace(
            moe_runner_config,
            num_experts=global_num_experts,
            num_local_experts=global_num_experts,
            routed_scaling_factor=None,
        )
        self.gpu_layer.moe_runner_config = runner_config
        self.gpu_method.create_moe_runner(self.gpu_layer, runner_config)

    def _get_base_quant_method(self):
        """Unwrap nested quant methods to get the underlying base method.

        Some paths may wrap the real quant method with KT wrappers/schemes.
        """
        method = self.gpu_method
        visited = set()

        while method is not None and id(method) not in visited:
            visited.add(id(method))

            # KT wrapper pattern: method.gpu_method
            nested = getattr(method, "gpu_method", None)
            if nested is not None and nested is not method:
                method = nested
                continue

            # Compressed-tensors scheme pattern: method.scheme
            nested = getattr(method, "scheme", None)
            if nested is not None and nested is not method:
                method = nested
                continue

            break

        return method

    def _detect_quant_type_from_created_weights(self) -> None:
        """Detect quant type from weight attributes created on gpu_layer."""
        layer = self.gpu_layer

        # Prepared-slot target layout for the MXFP4 layerwise-prefill
        # pipeline; stays None for every non-MXFP4 layout.
        self.mxfp4_prepared_layout = None

        # V4-Flash MXFP4 (must come before FP8 block — both register
        # `w13_weight_scale_inv`, but MXFP4 is FP4 nibble-packed weights with
        # ue8m0 scales rather than FP8 e4m3 weights with FP8 scales). Use the
        # quant method's class name as discriminator to avoid a circular import
        # of DeepSeekMxfp4MoEMethod. Origin: sglang 本身 (V4-Flash full-GPU
        # prefill fallback compat).
        if self.gpu_method.__class__.__name__ == "DeepSeekMxfp4MoEMethod":
            self.is_mxfp4_quant = True
            self.is_mxfp8_quant = False
            self.is_fp8_quant = False
            self.is_fp8_channel_quant = False
            self.is_bf16_quant = False
            self.mxfp4_prepared_layout = _MXFP4_LAYOUT_MARLIN
            return

        # K3 native MXFP4 (must come before FP8 per-channel — Mxfp4MoEMethod
        # registers `w13_weight_scale`/`w2_weight_scale` too, but they are
        # uint8 E8M0 group scales over FP4 nibble-packed weights). Class-name
        # check for the same circular-import reason as above.
        if self.gpu_method.__class__.__name__ == "Mxfp4MoEMethod":
            self.is_mxfp4_quant = True
            self.is_mxfp8_quant = False
            self.is_fp8_quant = False
            self.is_fp8_channel_quant = False
            self.is_bf16_quant = False
            self.mxfp4_prepared_layout = _MXFP4_LAYOUT_TRTLLM
            return

        # INT4 Marlin
        if hasattr(layer, "w13_weight_packed") and hasattr(layer, "w2_weight_packed"):
            self.is_mxfp4_quant = False
            self.is_mxfp8_quant = False
            self.is_fp8_quant = False
            self.is_fp8_channel_quant = False
            self.is_bf16_quant = False
            return

        # M3 MXFP8 block (must come before FP8 block — both register
        # `w13_weight_scale_inv`, but MXFP8 stores uint8 ue8m0 scales with
        # block_size=[1,32] while FP8 block uses fp32 scales with [128,128]).
        # The `format_ue8m0` attribute set by Fp8MoEMethod.create_weights
        # when use_mxfp8=True (fp8.py:914) is the canonical discriminator.
        # Origin: kt-sglang 耦合 (M3 MXFP8 layerwise prefill).
        if (
            hasattr(layer, "w13_weight_scale_inv")
            and hasattr(layer, "w2_weight_scale_inv")
            and getattr(layer.w13_weight_scale_inv, "format_ue8m0", False)
        ):
            self.is_mxfp4_quant = False
            self.is_mxfp8_quant = True
            self.is_fp8_quant = False
            self.is_fp8_channel_quant = False
            self.is_bf16_quant = False
            return

        # FP8 block
        if hasattr(layer, "w13_weight_scale_inv") and hasattr(layer, "w2_weight_scale_inv"):
            self.is_mxfp4_quant = False
            self.is_mxfp8_quant = False
            self.is_fp8_quant = True
            self.is_fp8_channel_quant = False
            self.is_bf16_quant = False
            return

        # FP8 per-channel
        if hasattr(layer, "w13_weight_scale") and hasattr(layer, "w2_weight_scale"):
            self.is_mxfp4_quant = False
            self.is_mxfp8_quant = False
            self.is_fp8_quant = False
            self.is_fp8_channel_quant = True
            self.is_bf16_quant = False
            return

        # BF16 / unquantized
        if hasattr(layer, "w13_weight") and hasattr(layer, "w2_weight"):
            self.is_mxfp4_quant = False
            self.is_mxfp8_quant = False
            self.is_fp8_quant = False
            self.is_fp8_channel_quant = False
            self.is_bf16_quant = True
            return

        # Fallback to class-based detection for unknown layouts.
        self.is_mxfp4_quant = False
        self.is_mxfp8_quant = False
        self.is_fp8_quant = self._detect_fp8_quant()
        self.is_fp8_channel_quant = self._detect_fp8_channel_quant()
        self.is_bf16_quant = self._detect_bf16_quant()

    def _detect_fp8_quant(self) -> bool:
        """Detect if the quantization method is FP8 block quant.

        Returns:
            True if FP8 block quant, False otherwise (INT4 Marlin, BF16, etc.)
        """
        from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod

        method = self._get_base_quant_method()
        # Check for Fp8MoEMethod with block_quant
        if isinstance(method, Fp8MoEMethod) and getattr(method, "block_quant", False):
            return True

        # Check for CompressedTensorsW8A8Fp8MoEMethod with block_quant
        method_name = method.__class__.__name__
        if "W8A8Fp8" in method_name and getattr(method, "block_quant", False):
            return True

        return False

    def _detect_fp8_channel_quant(self) -> bool:
        """Detect if the quantization method is FP8 per-channel quant.

        Per-channel FP8 differs from block FP8:
        - Per-channel: scale shape is (num_experts, output_dim, 1), weight_scale name
        - Block FP8: scale shape is (num_experts, blocks_n, blocks_k), weight_scale_inv name

        Returns:
            True if FP8 per-channel quant, False otherwise
        """
        try:
            from compressed_tensors.quantization import QuantizationStrategy
        except ImportError:
            return False

        method = self._get_base_quant_method()
        method_name = method.__class__.__name__

        # Check for CompressedTensorsW8A8Fp8MoEMethod with channel strategy
        if "W8A8Fp8" in method_name:
            weight_quant = getattr(method, "weight_quant", None)
            if weight_quant is not None:
                if weight_quant.strategy == QuantizationStrategy.CHANNEL:
                    return True

        return False

    def _detect_bf16_quant(self) -> bool:
        """Detect if the quantization method is BF16/unquantized.

        Returns:
            True if BF16/unquantized, False otherwise (INT4 Marlin, FP8, etc.)
        """
        from sglang.srt.layers.moe.fused_moe_triton.layer import (
            UnquantizedFusedMoEMethod,
        )

        method = self._get_base_quant_method()
        # Check for UnquantizedFusedMoEMethod
        if isinstance(method, UnquantizedFusedMoEMethod):
            return True

        return False

    def _resolve_int4_quant_params(self):
        """Resolve INT4 quant params from potentially wrapped quant methods.

        Some quantization paths (e.g., compressed-tensors) expose INT4 metadata on
        the underlying scheme instead of the outer fused method wrapper.
        """
        candidates = []
        seen = set()

        def add_candidate(obj):
            if obj is None:
                return
            obj_id = id(obj)
            if obj_id in seen:
                return
            seen.add(obj_id)
            candidates.append(obj)

        base_method = self._get_base_quant_method()
        add_candidate(self.gpu_method)
        add_candidate(getattr(self.gpu_method, "gpu_method", None))
        add_candidate(getattr(self.gpu_method, "scheme", None))
        add_candidate(base_method)
        add_candidate(getattr(base_method, "scheme", None))
        add_candidate(getattr(self.gpu_layer, "scheme", None))

        required = ("num_bits", "packed_factor", "group_size")
        for candidate in candidates:
            if all(hasattr(candidate, attr) for attr in required):
                return (
                    getattr(candidate, "num_bits"),
                    getattr(candidate, "packed_factor"),
                    getattr(candidate, "group_size"),
                    getattr(candidate, "actorder", None),
                )

        raise AttributeError(
            "Unable to resolve INT4 quantization params: expected attributes "
            "num_bits/packed_factor/group_size on quant method or scheme"
        )

    @property
    def weight_names(self) -> list:
        """Get weight names based on quantization type."""
        if getattr(self, "_is_mxfp4_quant", False):
            if self.mxfp4_prepared_layout == _MXFP4_LAYOUT_TRTLLM:
                # K3 native MXFP4: Mxfp4MoEMethod.create_weights registers
                # w13_weight_scale / w2_weight_scale (no `_inv` suffix).
                return self.WEIGHT_NAMES_MXFP4_TRTLLM
            # V4-Flash MXFP4 uses the same flat names as FP8 block (w13_weight,
            # w13_weight_scale_inv, w2_weight, w2_weight_scale_inv); the
            # underlying byte payload differs (FP4 nibble + ue8m0 scale) but
            # the staging buffers don't care about content.
            return self.WEIGHT_NAMES_FP8
        if getattr(self, "_is_mxfp8_quant", False):
            # M3 MXFP8 reuses the FP8 block flat names. Byte payload is
            # MXFP8 (fp8 + uint8 ue8m0 [1,32]) — staging buffer dtype/shape
            # follow gpu_layer.w13_weight_scale_inv (uint8) so byte-copy
            # transports the canonical layout. fused_experts_mxfp8 consumes
            # it directly; no convert step.
            return self.WEIGHT_NAMES_FP8
        if self._is_fp8_quant:
            return self.WEIGHT_NAMES_FP8
        elif self._is_fp8_channel_quant:
            return self.WEIGHT_NAMES_FP8_CHANNEL
        elif self._is_bf16_quant:
            return self.WEIGHT_NAMES_BF16
        else:
            return self.WEIGHT_NAMES_INT4

    # Weight names for shared memory buffers (INT4 Marlin format)
    WEIGHT_NAMES_INT4 = [
        "w13_weight_packed",
        "w13_weight_scale",
        "w2_weight_packed",
        "w2_weight_scale",
    ]

    # Weight names for FP8 block quant format
    WEIGHT_NAMES_FP8 = [
        "w13_weight",
        "w13_weight_scale_inv",
        "w2_weight",
        "w2_weight_scale_inv",
    ]

    # Weight names for FP8 per-channel quant format
    # Per-channel differs from block quant:
    # - Scale shape: (num_experts, output_dim, 1) vs (num_experts, blocks_n, blocks_k)
    # - Weight name: w13_weight_scale vs w13_weight_scale_inv
    WEIGHT_NAMES_FP8_CHANNEL = [
        "w13_weight",
        "w13_weight_scale",
        "w2_weight",
        "w2_weight_scale",
    ]

    # Weight names for K3 native MXFP4 (Mxfp4MoEMethod): FP4 nibble-packed
    # weights + E8M0 group scales, registered without the `_inv` suffix.
    WEIGHT_NAMES_MXFP4_TRTLLM = [
        "w13_weight",
        "w13_weight_scale",
        "w2_weight",
        "w2_weight_scale",
    ]

    # Weight names for BF16/unquantized format (no scales)
    WEIGHT_NAMES_BF16 = [
        "w13_weight",
        "w2_weight",
    ]

    def _cleanup_cpu_buffers_after_failure(self) -> None:
        """Best-effort symmetric cleanup for a failed SHM setup phase."""

        if torch.cuda.is_available():
            for tensor in getattr(self, "_registered_host_buffers", []):
                try:
                    torch.cuda.cudart().cudaHostUnregister(tensor.data_ptr())
                except Exception:
                    pass
        self._registered_host_buffers = []

        # Drop exported torch buffers before closing their SharedMemory maps.
        self.cpu_buffers = {}
        gc.collect()

        for shm in getattr(self, "_opened_shm_refs", {}).values():
            try:
                shm.close()
            except Exception:
                pass
        self._opened_shm_refs = {}

        for shm in getattr(self, "shm_handles", {}).values():
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass
            try:
                shm.close()
            except Exception:
                pass
        self.shm_handles = {}

    def _commit_cpu_buffer_phase(
        self, local_error: Optional[Exception], phase: str
    ) -> None:
        """Make every TP rank commit or abort an SHM initialization phase."""

        if _all_tp_ranks_succeeded(local_error is None):
            return

        self._cleanup_cpu_buffers_after_failure()
        message = f"KT shared-memory {phase} failed on at least one TP rank"
        if local_error is not None:
            raise RuntimeError(message) from local_error
        raise RuntimeError(message)

    def _create_cpu_buffers(self):
        """Create CPU buffers in POSIX shared memory and register as pinned memory.

        Uses double buffering (2 experts) to reduce memory usage while maintaining
        pipeline efficiency: write(e+1) || copy(e) only needs 2 buffers.
        """
        self.cpu_buffers = {}
        self.shm_handles: Dict[str, shared_memory.SharedMemory] = {}
        self._opened_shm_refs: Dict[str, shared_memory.SharedMemory] = {}
        self._registered_host_buffers: List[torch.Tensor] = []
        tp_rank = get_parallel().tp_rank
        num_experts = self.gpu_layer.num_experts

        # No rank may enter a later SHM phase until every peer has completed
        # the current one.  In particular, a local libnuma/SHM/register error
        # must not leave peers waiting forever in a barrier.
        numa_error = None
        try:
            libnuma = ctypes.CDLL("libnuma.so.1")
            if libnuma.numa_available() < 0:
                raise RuntimeError("NUMA is not available on this system")
            libnuma.numa_set_localalloc()
        except Exception as exc:
            numa_error = exc
        self._commit_cpu_buffer_phase(numa_error, "NUMA setup")

        # Generate unique ID on rank 0 and broadcast to all ranks
        if tp_rank == 0:
            self.shm_unique_id = uuid.uuid4().hex[:8]
        else:
            self.shm_unique_id = None
        if dist.is_initialized():
            unique_id_list = [self.shm_unique_id]
            dist.broadcast_object_list(
                unique_id_list,
                src=get_tp_group().first_rank,
                group=get_tp_group().cpu_group,
            )
            self.shm_unique_id = unique_id_list[0]

        allocation_error = None
        try:
            # Serial transports ping-pong slots 0/1; the MXFP4 layerwise
            # manager batches its per-expert control plane over half-ring
            # chunks, so it stages a deeper ring (~2.3 MiB/slot/rank).
            self.host_ring_depth = (
                _KT_MXFP4_HOST_RING_DEPTH
                if getattr(self, "_is_mxfp4_quant", False)
                else 2
            )
            self.host_expert_nbytes = {}
            for name in self.weight_names:
                gpu_tensor = getattr(self.gpu_layer, name)
                expert_shape = gpu_tensor.shape[1:]  # Shape per expert
                if (
                    getattr(self, "_is_mxfp4_quant", False)
                    and name in (
                        "w13_weight_scale_inv",
                        "w2_weight_scale_inv",
                        "w13_weight_scale",
                        "w2_weight_scale",
                    )
                ):
                    # kt-kernel's write_weight_scale_to_buffer keeps its bf16
                    # scale contract regardless of the resident scale layout;
                    # the gpu_layer attr dtype (fp32 for DSV4, uint8 E8M0 for
                    # K3 trtllm) does not describe the export payload.
                    buf_dtype = torch.bfloat16
                else:
                    buf_dtype = gpu_tensor.dtype
                element_size = torch.empty((), dtype=buf_dtype).element_size()
                expert_nbytes = gpu_tensor.numel() // num_experts * element_size
                self.host_expert_nbytes[name] = expert_nbytes
                ring_nbytes = expert_nbytes * self.host_ring_depth

                shm_name = f"kt_buf_{name}_r{tp_rank}_{self.shm_unique_id}"
                shm = shared_memory.SharedMemory(
                    name=shm_name, create=True, size=ring_nbytes
                )
                self.shm_handles[name] = shm

                # Shape: [host_ring_depth, ...expert_shape...]
                cpu_buffer = torch.frombuffer(shm.buf, dtype=buf_dtype).reshape(
                    (self.host_ring_depth,) + expert_shape
                )

                # Register as pinned memory for fast DMA
                if torch.cuda.is_available():
                    register_result = torch.cuda.cudart().cudaHostRegister(
                        cpu_buffer.data_ptr(), ring_nbytes, 0
                    )
                    if int(register_result) != 0:
                        raise RuntimeError(
                            "cudaHostRegister failed for "
                            f"{name} with error code {int(register_result)}"
                        )
                    self._registered_host_buffers.append(cpu_buffer)

                self.cpu_buffers[name] = cpu_buffer
        except Exception as exc:
            allocation_error = exc
        self._commit_cpu_buffer_phase(allocation_error, "allocation")

        pointer_error = None
        try:
            self.all_rank_buffer_ptrs = self._collect_all_rank_buffer_pointers()
            if tp_rank == 0:
                tp_world_size = get_parallel().tp_size
                valid = all(
                    len(ptrs) == tp_world_size and all(ptr > 0 for ptr in ptrs)
                    for ptrs in self.all_rank_buffer_ptrs.values()
                )
                if not valid:
                    raise RuntimeError(
                        "TP0 could not map every rank's shared-memory buffer"
                    )
        except Exception as exc:
            pointer_error = exc
        self._commit_cpu_buffer_phase(pointer_error, "pointer collection")

        # Unlink shared memory after all ranks have collected pointers.
        # The memory remains accessible as long as we hold references via mmap.
        for shm in self.shm_handles.values():
            try:
                shm.unlink()
            except FileNotFoundError:
                pass

    def _collect_all_rank_buffer_pointers(self) -> Dict[str, List[int]]:
        """Collect CPU buffer pointers from all ranks."""
        tp_rank = get_parallel().tp_rank
        tp_world_size = get_parallel().tp_size
        buffer_names = list(self.cpu_buffers.keys())
        all_rank_ptrs: Dict[str, List[int]] = {name: [] for name in buffer_names}
        self._opened_shm_refs: Dict[str, shared_memory.SharedMemory] = {}

        for rank in range(tp_world_size):
            for name in buffer_names:
                if rank == tp_rank:
                    ptr = self.cpu_buffers[name].data_ptr()
                elif tp_rank == 0:
                    shm_name = f"kt_buf_{name}_r{rank}_{self.shm_unique_id}"
                    try:
                        shm = shared_memory.SharedMemory(name=shm_name)
                        self._opened_shm_refs[f"{name}_r{rank}"] = shm
                        ptr = ctypes.addressof(ctypes.c_char.from_buffer(shm.buf))
                    except Exception:
                        logger.error(
                            "Rank %d: Failed to open shared memory '%s'",
                            tp_rank,
                            shm_name,
                        )
                        ptr = 0
                else:
                    ptr = 0
                all_rank_ptrs[name].append(ptr)

        return all_rank_ptrs

    def _prepare_weight_int4(self, wrapper):
        """Prepare INT4 Marlin weights by writing from KT, copying to GPU, and postprocessing.

        Pipeline: write(e+1) || copy(e) || postprocess(e-1)

        Postprocessing extracted from CompressedTensorsWNA16MoEMethod.process_weights_after_loading
        in python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py
        """
        # Bind Python thread to specific CPU core (last cores for each rank)
        tp_rank = get_parallel().tp_rank
        num_cpus = os.cpu_count()
        target_cpu = num_cpus - 1 - tp_rank
        os.sched_setaffinity(0, {target_cpu})

        layer = self.gpu_layer
        num_bits, packed_factor, group_size, actorder = (
            self._resolve_int4_quant_params()
        )
        num_experts = layer.num_experts
        device = layer.w13_weight_packed.device

        # Create empty g_idx tensors for non-grouped actorder
        if actorder != "group":
            for name in [
                "w13_weight_g_idx",
                "w2_weight_g_idx",
                "w13_g_idx_sort_indices",
                "w2_g_idx_sort_indices",
            ]:
                setattr(
                    layer,
                    name,
                    torch.nn.Parameter(
                        torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                        requires_grad=False,
                    ),
                )

        # Prepare weight tensors (cpu_buf is double-buffered with shape [2, ...])
        weight_infos = []
        for name in self.WEIGHT_NAMES_INT4:
            cpu_buf = self.cpu_buffers[name]  # Shape: [2, ...expert_shape...]
            gpu_t = getattr(layer, name)  # Shape: [num_experts, ...expert_shape...]
            # Reshape gpu_t to match expert shape for per-expert copy
            expert_shape = cpu_buf.shape[1:]
            gpu_t.set_(gpu_t.view((num_experts,) + expert_shape))
            weight_infos.append((cpu_buf, gpu_t))

        w13_p, w13_s = layer.w13_weight_packed, layer.w13_weight_scale
        w2_p, w2_s = layer.w2_weight_packed, layer.w2_weight_scale
        w13_k, w13_n = w13_p.shape[1] * packed_factor, w13_p.shape[2]
        w2_k, w2_n = w2_p.shape[1] * packed_factor, w2_p.shape[2]
        w2_sk = w2_s.shape[1] * (group_size if group_size != -1 else packed_factor)
        perm = torch.empty(0, dtype=torch.int32, device=device)

        # Tmp buffers for transpose
        tmp_bufs = [
            torch.empty(t.size(1), t.size(2), dtype=t.dtype, device=device)
            for _, t in weight_infos
        ]

        def postprocess_expert(e):
            # Transpose
            for (_, gpu_t), tmp in zip(weight_infos, tmp_bufs):
                d1, d2 = gpu_t.size(1), gpu_t.size(2)
                tmp.copy_(gpu_t[e].reshape(d2, d1).T, non_blocking=True)
                gpu_t[e].copy_(tmp, non_blocking=True)
            # Repack weights
            w13_p[e].copy_(
                gptq_marlin_repack(w13_p[e], perm, w13_k, w13_n, num_bits).view(
                    w13_p[e].shape
                )
            )
            w2_p[e].copy_(
                gptq_marlin_repack(w2_p[e], perm, w2_k, w2_n, num_bits).view(
                    w2_p[e].shape
                )
            )
            # Permute scales
            w13_s[e].copy_(
                marlin_permute_scales(w13_s[e], w13_n, w13_s.shape[2], group_size).view(
                    w13_s[e].shape
                )
            )
            w2_s[e].copy_(
                marlin_permute_scales(w2_s[e], w2_sk, w2_s.shape[2], group_size).view(
                    w2_s[e].shape
                )
            )

        # Pipeline: write(e+1) || copy(e) || postprocess(e-1)
        copy_stream = torch.cuda.Stream(device=device)
        post_stream = torch.cuda.Stream(device=device)
        events = [torch.cuda.Event() for _ in range(num_experts)]

        # Prepare write pipeline (rank 0 only)
        tp_world_size = get_parallel().tp_size
        do_write = tp_rank == 0 and wrapper is not None

        if do_write:
            # Calculate per-expert byte sizes (buffer is double-buffered: [2, ...])
            w13_packed_buf = self.cpu_buffers["w13_weight_packed"]
            w13_scale_buf = self.cpu_buffers["w13_weight_scale"]
            w2_packed_buf = self.cpu_buffers["w2_weight_packed"]
            w2_scale_buf = self.cpu_buffers["w2_weight_scale"]

            # Per-expert sizes are fixed at ring allocation time.
            w13_packed_expert_nbytes = self.host_expert_nbytes["w13_weight_packed"]
            w13_scale_expert_nbytes = self.host_expert_nbytes["w13_weight_scale"]
            w2_packed_expert_nbytes = self.host_expert_nbytes["w2_weight_packed"]
            w2_scale_expert_nbytes = self.host_expert_nbytes["w2_weight_scale"]

            def submit_write_expert(expert_id):
                # Use expert_id % 2 for double buffering slot selection
                slot = expert_id % 2
                w13_packed_ptrs = [
                    ptr + slot * w13_packed_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w13_weight_packed"]
                ]
                w13_scale_ptrs = [
                    ptr + slot * w13_scale_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w13_weight_scale"]
                ]
                w2_packed_ptrs = [
                    ptr + slot * w2_packed_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w2_weight_packed"]
                ]
                w2_scale_ptrs = [
                    ptr + slot * w2_scale_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w2_weight_scale"]
                ]
                wrapper.submit_write_weight_scale_to_buffer(
                    tp_world_size,
                    expert_id,
                    w13_packed_ptrs,
                    w13_scale_ptrs,
                    w2_packed_ptrs,
                    w2_scale_ptrs,
                )

            # Submit expert 0 ahead of time
            submit_write_expert(0)

        for e in range(num_experts):
            # Sync write for expert e, submit write for expert e+1
            if do_write:
                wrapper.sync_write_weight_scale_to_buffer()
                if e + 1 < num_experts:
                    # Before writing to slot (e+1)%2, make sure the previous
                    # copy from that slot has completed to avoid overwriting
                    # pinned host memory while DMA is in-flight.
                    if e > 0:
                        events[e - 1].synchronize()
                    submit_write_expert(e + 1)

            # Barrier to ensure all ranks see the written data
            if dist.is_initialized():
                dist.barrier(group=get_tp_group().device_group)

            with torch.cuda.stream(copy_stream):
                slot = e % 2  # Double buffering
                for cpu_buf, gpu_t in weight_infos:
                    gpu_t[e].copy_(cpu_buf[slot], non_blocking=True)
                events[e].record(copy_stream)

            if e > 0:
                with torch.cuda.stream(post_stream):
                    post_stream.wait_event(events[e - 1])
                    postprocess_expert(e - 1)

        with torch.cuda.stream(post_stream):
            post_stream.wait_event(events[-1])
            postprocess_expert(num_experts - 1)

        torch.cuda.current_stream(device).wait_stream(post_stream)

        # Reshape to final shape
        w13_p.set_(w13_p.view(num_experts, w13_k // 16, w13_n * (num_bits // 2)))
        w2_p.set_(w2_p.view(num_experts, w2_k // 16, w2_n * (num_bits // 2)))

    def _prepare_weight_fp8(self, wrapper, original_layer=None, gpu_experts_mask=None,
                            logical_to_gpu_index=None):
        """Prepare FP8 block quant weights by writing from KT and copying to GPU.

        Pipeline: write(e+1) || copy(e) || postprocess(e-1)

        FP8 block quant is simpler than INT4 Marlin:
        - No transpose needed (weight layout is already correct)
        - No marlin_repack needed (only INT4 Marlin needs this)
        - No permute_scales needed (only Marlin format needs this)

        The postprocess stage is a no-op for FP8 but provides pipeline synchronization
        to ensure copy(e-2) completes before write(e) overwrites the same slot.

        Optional DeepGemm ue8m0 conversion is handled after all experts are loaded.

        Optimization: If original_layer and gpu_experts_mask are provided, experts
        already on GPU are copied directly (fast GPU-to-GPU), while CPU experts
        use the KT wrapper pipeline.
        """
        # Bind Python thread to specific CPU core (last cores for each rank)
        tp_rank = get_parallel().tp_rank
        num_cpus = os.cpu_count()
        target_cpu = num_cpus - 1 - tp_rank
        os.sched_setaffinity(0, {target_cpu})

        layer = self.gpu_layer
        num_experts = layer.num_experts
        device = layer.w13_weight.device

        # Prepare weight tensors (cpu_buf is double-buffered with shape [2, ...])
        weight_infos = []
        for name in self.WEIGHT_NAMES_FP8:
            cpu_buf = self.cpu_buffers[name]  # Shape: [2, ...expert_shape...]
            gpu_t = getattr(layer, name)  # Shape: [num_experts, ...expert_shape...]
            weight_infos.append((name, cpu_buf, gpu_t))

        # Separate GPU experts (direct copy) from CPU experts (KT transfer)
        gpu_expert_ids = []
        cpu_expert_ids = []
        if gpu_experts_mask is not None and original_layer is not None and logical_to_gpu_index is not None:
            for e in range(num_experts):
                if gpu_experts_mask[e].item():
                    gpu_expert_ids.append(e)
                else:
                    cpu_expert_ids.append(e)
        else:
            # Fallback: all experts from CPU
            cpu_expert_ids = list(range(num_experts))

        # --- Phase 1: Copy GPU experts directly (fast GPU-to-GPU) ---
        if gpu_expert_ids:
            for e in gpu_expert_ids:
                gpu_idx = logical_to_gpu_index[e].item()
                for name, _, dst in weight_infos:
                    src = getattr(original_layer, name)  # [num_gpu_experts, ...]
                    dst[e].copy_(src[gpu_idx], non_blocking=True)

        # --- Phase 2: Transfer CPU experts via KT pipeline ---
        if not cpu_expert_ids:
            # All experts are on GPU, nothing more to do
            return

        # Pipeline: write(e+1) || copy(e) || postprocess(e-1)
        copy_stream = torch.cuda.Stream(device=device)
        post_stream = torch.cuda.Stream(device=device)
        # Events indexed by position in cpu_expert_ids
        events = [torch.cuda.Event() for _ in range(len(cpu_expert_ids))]

        def postprocess_expert(idx):
            # FP8 doesn't need actual postprocessing (no repack/permute).
            # This function provides a pipeline synchronization point and
            # can be extended for future FP8-specific processing if needed.
            pass

        # Prepare write pipeline (rank 0 only)
        tp_world_size = get_parallel().tp_size
        do_write = tp_rank == 0 and wrapper is not None

        if do_write:
            # Calculate per-expert byte sizes (buffer is double-buffered: [2, ...])
            w13_weight_buf = self.cpu_buffers["w13_weight"]
            w13_scale_buf = self.cpu_buffers["w13_weight_scale_inv"]
            w2_weight_buf = self.cpu_buffers["w2_weight"]
            w2_scale_buf = self.cpu_buffers["w2_weight_scale_inv"]

            # Per-expert sizes are fixed at ring allocation time.
            w13_weight_expert_nbytes = self.host_expert_nbytes["w13_weight"]
            w13_scale_expert_nbytes = self.host_expert_nbytes["w13_weight_scale_inv"]
            w2_weight_expert_nbytes = self.host_expert_nbytes["w2_weight"]
            w2_scale_expert_nbytes = self.host_expert_nbytes["w2_weight_scale_inv"]

            def submit_write_expert(expert_id, slot):
                # Use provided slot for double buffering
                w13_weight_ptrs = [
                    ptr + slot * w13_weight_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w13_weight"]
                ]
                w13_scale_ptrs = [
                    ptr + slot * w13_scale_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w13_weight_scale_inv"]
                ]
                w2_weight_ptrs = [
                    ptr + slot * w2_weight_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w2_weight"]
                ]
                w2_scale_ptrs = [
                    ptr + slot * w2_scale_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w2_weight_scale_inv"]
                ]
                wrapper.submit_write_weight_scale_to_buffer(
                    tp_world_size,
                    expert_id,
                    w13_weight_ptrs,
                    w13_scale_ptrs,
                    w2_weight_ptrs,
                    w2_scale_ptrs,
                )

            # Submit first CPU expert ahead of time
            submit_write_expert(cpu_expert_ids[0], 0)

        for idx, e in enumerate(cpu_expert_ids):
            slot = idx % 2  # Double buffering based on iteration index

            # Sync write for expert e, submit write for next CPU expert
            if do_write:
                wrapper.sync_write_weight_scale_to_buffer()
                if idx + 1 < len(cpu_expert_ids):
                    next_slot = (idx + 1) % 2
                    # Before writing to next_slot, ensure copy from that slot is complete.
                    if idx > 0:
                        events[idx - 1].synchronize()
                    submit_write_expert(cpu_expert_ids[idx + 1], next_slot)

            # Barrier to ensure all ranks see the written data
            if dist.is_initialized():
                dist.barrier(group=get_tp_group().device_group)

            with torch.cuda.stream(copy_stream):
                for _, cpu_buf, gpu_t in weight_infos:
                    gpu_t[e].copy_(cpu_buf[slot], non_blocking=True)
                events[idx].record(copy_stream)

            # Postprocess expert idx-1: provides pipeline structure for future extensions
            if idx > 0:
                with torch.cuda.stream(post_stream):
                    post_stream.wait_event(events[idx - 1])
                    postprocess_expert(idx - 1)

        # Process last CPU expert
        if cpu_expert_ids:
            with torch.cuda.stream(post_stream):
                post_stream.wait_event(events[-1])
                postprocess_expert(len(cpu_expert_ids) - 1)

        torch.cuda.current_stream(device).wait_stream(post_stream)

    # NOTE: DeepGemm ue8m0 conversion is not used in KT fallback path.
    # The conversion is handled separately in the normal weight loading path.

    def _prepare_weight_mxfp8(self, wrapper, original_layer=None, gpu_experts_mask=None,
                              logical_to_gpu_index=None):
        """Byte-copy M3 MXFP8 weights from CPU staging buffer to GPU for the
        full-GPU layerwise prefill fallback.

        Shadow ``gpu_method`` stays in the MXFP8 view (``use_mxfp8=True``,
        ``weight_block_size=[1, 32]``). ``Fp8MoEMethod.apply`` then routes
        through ``get_triton_quant_info`` -> ``fused_experts_mxfp8``, which
        consumes the uint8 ue8m0 scale directly via ``tl.dot_scaled`` — no
        block-FP8 conversion, no precision loss.

        Origin: kt-sglang 耦合 (M3 MXFP8 layerwise prefill, native MXFP8).
        """
        # Reset shadow to MXFP8 view (idempotent; ensures the Parameter slot
        # points at the canonical uint8 ue8m0 tensor before byte-copy).
        self.gpu_method.use_mxfp8 = True
        self.gpu_method.weight_block_size = [1, 32]
        self.gpu_layer.w13_weight_scale_inv = self._w13_scale_mxfp8_param
        self.gpu_layer.w2_weight_scale_inv = self._w2_scale_mxfp8_param

        # Byte-copy via the FP8 pipeline (uint8 ue8m0 scale + fp8 weight
        # both copied bytewise from kt-kernel CPU staging buffer).
        # original_layer=None disables the GPU shortcut: the real layer's
        # scale slot may have been mutated by Fp8MoEMethod's post-load step
        # on other paths; force the CPU staging route for canonical bytes.
        self._prepare_weight_fp8(
            wrapper,
            original_layer=None,
            gpu_experts_mask=None,
            logical_to_gpu_index=None,
        )

    def _prepare_weight_mxfp4(self, wrapper, original_layer=None, gpu_experts_mask=None,
                              logical_to_gpu_index=None):
        """Prepare V4-Flash MXFP4 weights for the full-GPU prefill fallback.

        V4-Flash MXFP4 routed-experts share flat attribute names with FP8 block
        (`w13_weight` / `w13_weight_scale_inv` / `w2_weight` / `w2_weight_scale_inv`)
        but with different payload semantics: FP4 e2m1 nibble-packed weights +
        ue8m0 per-kgroup scales instead of FP8 e4m3 weights + FP8 scales. The
        staging-buffer byte-copy machinery in `_prepare_weight_fp8` does not
        care about content semantics, so we reuse it as-is for the 144 GPU +
        112 CPU expert load.

        After all 256 experts are filled into `gpu_layer.w13_weight` etc., we
        re-run `gpu_method.process_weights_after_loading(gpu_layer)`, which
        invokes `convert_v4_weights_to_triton_kernels` and stores the swizzled
        result in `gpu_layer._v4_tk_w13` / `_v4_tk_w13_pcg` / `_v4_tk_w2` /
        `_v4_tk_w2_pcg` — exactly what the downstream `gpu_method.apply` →
        `apply_v4_triton_kernels_moe` path expects to read. This requires the
        outer model loader to have skipped the post-swizzle deletes (gated on
        `kt_gpu_prefill_token_threshold > 0` in `mxfp4_deepseek.py`).

        **No caching**: SharedFullContext is a global singleton whose single
        `gpu_layer` holds one layer's swizzled weights at a time. After layer N
        loads, layer N-1's data is overwritten. A boolean or per-layer-set
        cache would be stale when a different layer has since loaded into the
        same gpu_layer. Every load() call must therefore run the full pipeline.

        Origin: sglang 本身 (V4-Flash full-GPU prefill fallback compat).
        """
        # Phase 1+2: byte-copy via the FP8 path (works for FP4-packed bytes).
        self._prepare_weight_fp8(
            wrapper,
            original_layer=original_layer,
            gpu_experts_mask=gpu_experts_mask,
            logical_to_gpu_index=logical_to_gpu_index,
        )

        # Phase 3: re-swizzle the now-256-expert flat tensors into the
        # triton_kernels form `gpu_method.apply` will consume. Ensure all
        # in-flight CPU→GPU copies from Phase 2 are visible first.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.gpu_method.process_weights_after_loading(self.gpu_layer)

    def _prepare_weight_fp8_channel(self, wrapper, original_layer=None, gpu_experts_mask=None,
                                     logical_to_gpu_index=None):
        """Prepare FP8 per-channel quant weights by writing from KT and copying to GPU.

        Pipeline: write(e+1) || copy(e) || postprocess(e-1)

        FP8 per-channel quant differs from FP8 block quant:
        - Per-channel scale shape: (num_experts, output_dim, 1) vs (num_experts, blocks_n, blocks_k)
        - Weight name: w13_weight_scale vs w13_weight_scale_inv
        - Both use float8_e4m3fn weights

        Similar to block FP8:
        - No transpose needed (weight layout is already correct)
        - No marlin_repack needed (only INT4 Marlin needs this)
        - No permute_scales needed (only Marlin format needs this)

        The postprocess stage is a no-op for FP8 but provides pipeline synchronization
        to ensure copy(e-2) completes before write(e) overwrites the same slot.

        Optimization: If original_layer and gpu_experts_mask are provided, experts
        already on GPU are copied directly (fast GPU-to-GPU), while CPU experts
        use the KT wrapper pipeline.
        """
        # Bind Python thread to specific CPU core (last cores for each rank)
        tp_rank = get_parallel().tp_rank
        num_cpus = os.cpu_count()
        target_cpu = num_cpus - 1 - tp_rank
        os.sched_setaffinity(0, {target_cpu})

        layer = self.gpu_layer
        num_experts = layer.num_experts
        device = layer.w13_weight.device

        # Prepare weight tensors (cpu_buf is double-buffered with shape [2, ...])
        weight_infos = []
        for name in self.WEIGHT_NAMES_FP8_CHANNEL:
            cpu_buf = self.cpu_buffers[name]  # Shape: [2, ...expert_shape...]
            gpu_t = getattr(layer, name)  # Shape: [num_experts, ...expert_shape...]
            weight_infos.append((name, cpu_buf, gpu_t))

        # Separate GPU experts (direct copy) from CPU experts (KT transfer)
        gpu_expert_ids = []
        cpu_expert_ids = []
        if gpu_experts_mask is not None and original_layer is not None and logical_to_gpu_index is not None:
            for e in range(num_experts):
                if gpu_experts_mask[e].item():
                    gpu_expert_ids.append(e)
                else:
                    cpu_expert_ids.append(e)
        else:
            # Fallback: all experts from CPU
            cpu_expert_ids = list(range(num_experts))

        # --- Phase 1: Copy GPU experts directly (fast GPU-to-GPU) ---
        if gpu_expert_ids:
            for e in gpu_expert_ids:
                gpu_idx = logical_to_gpu_index[e].item()
                for name, _, dst in weight_infos:
                    src = getattr(original_layer, name)  # [num_gpu_experts, ...]
                    dst[e].copy_(src[gpu_idx], non_blocking=True)

        # --- Phase 2: Transfer CPU experts via KT pipeline ---
        if not cpu_expert_ids:
            # All experts are on GPU, nothing more to do
            return

        # Pipeline: write(e+1) || copy(e) || postprocess(e-1)
        copy_stream = torch.cuda.Stream(device=device)
        post_stream = torch.cuda.Stream(device=device)
        # Events indexed by position in cpu_expert_ids
        events = [torch.cuda.Event() for _ in range(len(cpu_expert_ids))]

        def postprocess_expert(idx):
            # FP8 per-channel doesn't need actual postprocessing (no repack/permute).
            # This function provides a pipeline synchronization point and
            # can be extended for future FP8-specific processing if needed.
            pass

        # Prepare write pipeline (rank 0 only)
        tp_world_size = get_parallel().tp_size
        do_write = tp_rank == 0 and wrapper is not None

        if do_write:
            # Calculate per-expert byte sizes (buffer is double-buffered: [2, ...])
            w13_weight_buf = self.cpu_buffers["w13_weight"]
            w13_scale_buf = self.cpu_buffers["w13_weight_scale"]
            w2_weight_buf = self.cpu_buffers["w2_weight"]
            w2_scale_buf = self.cpu_buffers["w2_weight_scale"]

            # Per-expert sizes are fixed at ring allocation time.
            w13_weight_expert_nbytes = self.host_expert_nbytes["w13_weight"]
            w13_scale_expert_nbytes = self.host_expert_nbytes["w13_weight_scale"]
            w2_weight_expert_nbytes = self.host_expert_nbytes["w2_weight"]
            w2_scale_expert_nbytes = self.host_expert_nbytes["w2_weight_scale"]

            def submit_write_expert(expert_id, slot):
                # Use provided slot for double buffering
                w13_weight_ptrs = [
                    ptr + slot * w13_weight_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w13_weight"]
                ]
                w13_scale_ptrs = [
                    ptr + slot * w13_scale_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w13_weight_scale"]
                ]
                w2_weight_ptrs = [
                    ptr + slot * w2_weight_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w2_weight"]
                ]
                w2_scale_ptrs = [
                    ptr + slot * w2_scale_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w2_weight_scale"]
                ]
                wrapper.submit_write_weight_scale_to_buffer(
                    tp_world_size,
                    expert_id,
                    w13_weight_ptrs,
                    w13_scale_ptrs,
                    w2_weight_ptrs,
                    w2_scale_ptrs,
                )

            # Submit first CPU expert ahead of time
            submit_write_expert(cpu_expert_ids[0], 0)

        for idx, e in enumerate(cpu_expert_ids):
            slot = idx % 2  # Double buffering based on iteration index

            # Sync write for expert e, submit write for next CPU expert
            if do_write:
                wrapper.sync_write_weight_scale_to_buffer()
                if idx + 1 < len(cpu_expert_ids):
                    next_slot = (idx + 1) % 2
                    # Before writing to next_slot, ensure copy from that slot is complete.
                    if idx > 0:
                        events[idx - 1].synchronize()
                    submit_write_expert(cpu_expert_ids[idx + 1], next_slot)

            # Barrier to ensure all ranks see the written data
            if dist.is_initialized():
                dist.barrier(group=get_tp_group().device_group)

            with torch.cuda.stream(copy_stream):
                for _, cpu_buf, gpu_t in weight_infos:
                    gpu_t[e].copy_(cpu_buf[slot], non_blocking=True)
                events[idx].record(copy_stream)

            # Postprocess expert idx-1: provides pipeline structure for future extensions
            if idx > 0:
                with torch.cuda.stream(post_stream):
                    post_stream.wait_event(events[idx - 1])
                    postprocess_expert(idx - 1)

        # Process last CPU expert
        if cpu_expert_ids:
            with torch.cuda.stream(post_stream):
                post_stream.wait_event(events[-1])
                postprocess_expert(len(cpu_expert_ids) - 1)

        torch.cuda.current_stream(device).wait_stream(post_stream)

    def _prepare_weight_bf16(self, wrapper, original_layer=None, gpu_experts_mask=None,
                             logical_to_gpu_index=None):
        """Prepare BF16/unquantized weights by writing from KT and copying to GPU.

        Pipeline: write(e+1) || copy(e) || postprocess(e-1)

        BF16/unquantized is similar to FP8 block quant:
        - No transpose needed (weight layout is already correct)
        - No marlin_repack needed (only INT4 Marlin needs this)
        - No permute_scales needed (only Marlin format needs this)
        - No scales at all (unlike FP8 which has scale_inv)

        The postprocess stage is a no-op for BF16 but provides pipeline synchronization
        to ensure copy(e-2) completes before write(e) overwrites the same slot.

        Optimization: If original_layer and gpu_experts_mask are provided, experts
        already on GPU are copied directly (fast GPU-to-GPU), while CPU experts
        use the KT wrapper pipeline.
        """
        # Bind Python thread to specific CPU core (last cores for each rank)
        tp_rank = get_parallel().tp_rank
        num_cpus = os.cpu_count()
        target_cpu = num_cpus - 1 - tp_rank
        os.sched_setaffinity(0, {target_cpu})

        layer = self.gpu_layer
        num_experts = layer.num_experts
        device = layer.w13_weight.device

        # Prepare weight tensors (cpu_buf is double-buffered with shape [2, ...])
        weight_infos = []
        for name in self.WEIGHT_NAMES_BF16:
            cpu_buf = self.cpu_buffers[name]  # Shape: [2, ...expert_shape...]
            gpu_t = getattr(layer, name)  # Shape: [num_experts, ...expert_shape...]
            weight_infos.append((name, cpu_buf, gpu_t))

        # Separate GPU experts (direct copy) from CPU experts (KT transfer)
        gpu_expert_ids = []
        cpu_expert_ids = []
        if gpu_experts_mask is not None and original_layer is not None and logical_to_gpu_index is not None:
            for e in range(num_experts):
                if gpu_experts_mask[e].item():
                    gpu_expert_ids.append(e)
                else:
                    cpu_expert_ids.append(e)
        else:
            # Fallback: all experts from CPU
            cpu_expert_ids = list(range(num_experts))

        # --- Phase 1: Copy GPU experts directly (fast GPU-to-GPU) ---
        if gpu_expert_ids:
            for e in gpu_expert_ids:
                gpu_idx = logical_to_gpu_index[e].item()
                for name, _, dst in weight_infos:
                    src = getattr(original_layer, name)  # [num_gpu_experts, ...]
                    dst[e].copy_(src[gpu_idx], non_blocking=True)

        # --- Phase 2: Transfer CPU experts via KT pipeline ---
        if not cpu_expert_ids:
            # All experts are on GPU, nothing more to do
            return

        # Pipeline: write(e+1) || copy(e) || postprocess(e-1)
        copy_stream = torch.cuda.Stream(device=device)
        post_stream = torch.cuda.Stream(device=device)
        # Events indexed by position in cpu_expert_ids
        events = [torch.cuda.Event() for _ in range(len(cpu_expert_ids))]

        def postprocess_expert(idx):
            # BF16 doesn't need actual postprocessing (no repack/permute/transpose).
            # This function provides a pipeline synchronization point and
            # can be extended for future BF16-specific processing if needed.
            pass

        # Prepare write pipeline (rank 0 only)
        tp_world_size = get_parallel().tp_size
        do_write = tp_rank == 0 and wrapper is not None

        if do_write:
            # Calculate per-expert byte sizes (buffer is double-buffered: [2, ...])
            w13_weight_buf = self.cpu_buffers["w13_weight"]
            w2_weight_buf = self.cpu_buffers["w2_weight"]

            # Per-expert sizes are fixed at ring allocation time.
            w13_weight_expert_nbytes = self.host_expert_nbytes["w13_weight"]
            w2_weight_expert_nbytes = self.host_expert_nbytes["w2_weight"]

            def submit_write_expert(expert_id, slot):
                # Use provided slot for double buffering
                w13_weight_ptrs = [
                    ptr + slot * w13_weight_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w13_weight"]
                ]
                w2_weight_ptrs = [
                    ptr + slot * w2_weight_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w2_weight"]
                ]
                # For BF16, we pass empty scale pointer lists (no scales)
                w13_scale_ptrs = [0] * tp_world_size
                w2_scale_ptrs = [0] * tp_world_size
                wrapper.submit_write_weight_scale_to_buffer(
                    tp_world_size,
                    expert_id,
                    w13_weight_ptrs,
                    w13_scale_ptrs,
                    w2_weight_ptrs,
                    w2_scale_ptrs,
                )

            # Submit first CPU expert ahead of time
            submit_write_expert(cpu_expert_ids[0], 0)

        for idx, e in enumerate(cpu_expert_ids):
            slot = idx % 2  # Double buffering based on iteration index

            # Sync write for expert e, submit write for next CPU expert
            if do_write:
                wrapper.sync_write_weight_scale_to_buffer()
                if idx + 1 < len(cpu_expert_ids):
                    next_slot = (idx + 1) % 2
                    # Before writing to next_slot, ensure copy from that slot is complete.
                    if idx > 0:
                        events[idx - 1].synchronize()
                    submit_write_expert(cpu_expert_ids[idx + 1], next_slot)

            # Barrier to ensure all ranks see the written data
            if dist.is_initialized():
                dist.barrier(group=get_tp_group().device_group)

            with torch.cuda.stream(copy_stream):
                for _, cpu_buf, gpu_t in weight_infos:
                    gpu_t[e].copy_(cpu_buf[slot], non_blocking=True)
                events[idx].record(copy_stream)

            # Postprocess expert idx-1: provides pipeline structure for future extensions
            if idx > 0:
                with torch.cuda.stream(post_stream):
                    post_stream.wait_event(events[idx - 1])
                    postprocess_expert(idx - 1)

        # Process last CPU expert
        if cpu_expert_ids:
            with torch.cuda.stream(post_stream):
                post_stream.wait_event(events[-1])
                postprocess_expert(len(cpu_expert_ids) - 1)

        torch.cuda.current_stream(device).wait_stream(post_stream)

    def _restore_raw_attrs(self):
        """Restore ctx.gpu_layer weight/scale attributes to raw (pre-repack) format.

        After Marlin repack, attribute shapes and dtypes change (fp8→int32)
        and _weight_scale_inv is renamed to _weight_scale.  This helper
        reverts those changes so downstream code always sees raw format.
        """
        for _stale in ("w13_weight_scale", "w2_weight_scale"):
            if hasattr(self.gpu_layer, _stale):
                delattr(self.gpu_layer, _stale)
        for _sn, (_shape, _dtype, _device) in self._raw_weight_shapes.items():
            _cur = getattr(self.gpu_layer, _sn, None)
            if _cur is None or _cur.shape != _shape or _cur.dtype != _dtype:
                setattr(self.gpu_layer, _sn,
                        torch.nn.Parameter(
                            torch.empty(_shape, dtype=_dtype, device=_device)))
    def load(self, layer_idx, wrapper, original_layer=None, gpu_experts_mask=None,
             logical_to_gpu_index=None):
        """Load weights from disk to GPU via shared memory.

        Args:
            layer_idx: Layer index in the model
            wrapper: KT wrapper for CPU expert weight loading
            original_layer: Original MoE layer with GPU experts (optional)
            gpu_experts_mask: bool tensor [num_experts], True = on GPU (optional)
            logical_to_gpu_index: int tensor [num_experts], maps logical ID to GPU index (optional)
        """
        for name, param in self.original_params.items():
            setattr(self.gpu_layer, name, param)
        for name, buf in self.original_buffers.items():
            self.gpu_layer.register_buffer(name, buf)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        tp_rank = get_parallel().tp_rank
        t0 = time.perf_counter()

        # Restore raw format before loading new layer's weights
        self._restore_raw_attrs()

        # Select appropriate prepare_weight method based on quantization type
        # FP8/BF16 methods support GPU expert optimization; INT4 uses full CPU pipeline
        if getattr(self, "_is_mxfp4_quant", False):
            # V4-Flash MXFP4: byte-copy via FP8 path + re-swizzle into
            # triton_kernels form. Origin: sglang 本身.
            self._prepare_weight_mxfp4(wrapper, original_layer, gpu_experts_mask,
                                       logical_to_gpu_index)
        elif getattr(self, "_is_mxfp8_quant", False):
            # M3 MXFP8: byte-copy via FP8 path with original_layer=None
            # (Phase 1 shortcut disabled) + Triton MXFP8->block-FP8 convert
            # on shadow gpu_layer so apply() runs the standard block-FP8
            # deep_gemm path. Origin: kt-sglang 耦合 (v2 bridge).
            self._prepare_weight_mxfp8(wrapper, original_layer, gpu_experts_mask,
                                       logical_to_gpu_index)
        elif self._is_fp8_quant:
            # When the inference layer is Marlin-repacked (int32), the
            # raw fp8 context layer can't share GPU→GPU copies.
            # Disable Phase 1 shortcut by passing original_layer=None.
            self._prepare_weight_fp8(wrapper, None, gpu_experts_mask,
                                     logical_to_gpu_index)
        elif self._is_fp8_channel_quant:
            self._prepare_weight_fp8_channel(wrapper, None, gpu_experts_mask,
                                             logical_to_gpu_index)
        elif self._is_bf16_quant:
            self._prepare_weight_bf16(wrapper, original_layer, gpu_experts_mask,
                                      logical_to_gpu_index)
        else:
            # INT4 Marlin format: write(e+1) || copy(e) || postprocess(e-1)
            self._prepare_weight_int4(wrapper)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        total_time = (time.perf_counter() - t0) * 1000.0

        if tp_rank == 0:
            logger.info(
                "KT layerwise prefill: layer %d prepare weight = %.2f ms",
                layer_idx,
                total_time,
            )


class _Mxfp4PrefillSlot:
    """One complete MXFP4 layer image used by the layerwise prefill pipeline.

    ``raw_names`` tags the raw tensor layout and ``prepared`` holds the
    layout's prepared image (``V4MarlinPreparedWeights`` for DSV4's marlin
    target, ``TrtllmPreparedWeights`` for K3's trtllm-gen target); see
    ``_MXFP4_RAW_NAMES_BY_LAYOUT``.
    """

    def __init__(
        self,
        index: int,
        raw_tensors: Dict[str, torch.Tensor],
        prepared,
        *,
        raw_names: tuple,
    ):
        self.index = index
        self.raw_names = tuple(raw_names)
        for name in self.raw_names:
            setattr(self, name, raw_tensors[name])
        self.prepared = prepared
        # Stable nn.Parameter views over `prepared` for layouts whose apply
        # reads prepared tensors straight off layer attributes (trtllm);
        # filled by the manager, None for the marlin layout.
        self.prepared_params = None

        self.state = "EMPTY"
        self.layer_idx: Optional[int] = None
        self.epoch = -1
        self.has_consumed_event = False
        self.reuse_guard = None

        self.raw_ready_event = torch.cuda.Event()
        self.ready_event = torch.cuda.Event()
        self.consumed_event = torch.cuda.Event()

        self.intermediate_size = self.w2_weight.shape[2] * 2
        self.num_experts = self.w13_weight.shape[0]

    def invalidate(self) -> None:
        # Keep the prepared tensors alive.  Their backing may still be in use on
        # the compute stream; the consumed event protects the next overwrite.
        self.state = "EMPTY"
        self.layer_idx = None
        self.epoch = -1


class _Mxfp4LayerwisePrefillManager:
    """Persistent two-slot MXFP4 layer-to-layer prefill scheduler.

    GPU work is ordered exclusively with stream events.  Python launches the
    current layer first, then performs the successor's KT host write and GPU
    enqueue work while the current layer is already running on the main stream.
    """

    def __init__(
        self,
        context: SharedFullContext,
        signature: tuple,
        slot0_raw_tensors: Dict[str, torch.Tensor],
        slot1_raw_tensors: Dict[str, torch.Tensor],
        slot_prepared: tuple,
    ):
        self.context = context
        self.signature = signature
        self.prepared_layout = context.mxfp4_prepared_layout
        self.raw_names = _MXFP4_RAW_NAMES_BY_LAYOUT[self.prepared_layout]
        self.device = context.gpu_layer.w13_weight.device
        self.slots = (
            _Mxfp4PrefillSlot(
                0, slot0_raw_tensors, slot_prepared[0], raw_names=self.raw_names
            ),
            _Mxfp4PrefillSlot(
                1, slot1_raw_tensors, slot_prepared[1], raw_names=self.raw_names
            ),
        )
        if self.prepared_layout == _MXFP4_LAYOUT_TRTLLM:
            for slot in self.slots:
                slot.prepared_params = {
                    "w13_weight": torch.nn.Parameter(
                        slot.prepared.w13, requires_grad=False
                    ),
                    "w13_weight_scale": torch.nn.Parameter(
                        slot.prepared.w13_scale, requires_grad=False
                    ),
                    "w2_weight": torch.nn.Parameter(
                        slot.prepared.w2, requires_grad=False
                    ),
                    "w2_weight_scale": torch.nn.Parameter(
                        slot.prepared.w2_scale, requires_grad=False
                    ),
                }
            self._initialize_trtllm_static_layer_attrs()
        self.transfer_stream = torch.cuda.Stream(device=self.device)
        self.postprocess_stream = torch.cuda.Stream(device=self.device)
        # Runtime transport control must not use the main stream: it is
        # launched after layer N's compute so layer N+1 transport can overlap
        # that compute.  A one-element NCCL reduction on this dedicated stream
        # is substantially cheaper than a Gloo round-trip per transport phase.
        self.control_stream = torch.cuda.Stream(device=self.device)
        self.host_slot_free_events = (torch.cuda.Event(), torch.cuda.Event())
        self.host_slot_was_used = [False, False]
        self.host_write_status = torch.ones(
            (1,), dtype=torch.int32, device="cpu"
        )
        self.device_phase_status = torch.ones(
            (1,), dtype=torch.int32, device=self.device
        )

        self.epoch = -1
        self.last_layer_position: Optional[int] = None
        self.current_slot_index: Optional[int] = None
        self.round_active = False

    def _initialize_trtllm_static_layer_attrs(self) -> None:
        """Create the per-layer-invariant attributes Mxfp4MoEMethod.apply
        reads that its process_weights_after_loading would have set.

        Mirrors mxfp4.py L629-644 (gemm1_alpha / gemm1_beta /
        gemm1_clamp_limit from the runner config) and L831-838 (float32
        shuffled bias stacks).  The kt export carries no bias channel and the
        shadow layer's biases are zero-initialized, so the shuffled biases
        are exact zeros of the post-shuffle shape.
        """
        layer = self.context.gpu_layer
        slot = self.slots[0]
        num_experts = slot.num_experts
        w13_rows = slot.w13_weight.shape[1]
        hidden_size = slot.w2_weight.shape[1]
        alpha = layer.moe_runner_config.gemm1_alpha or 1.702
        limit = layer.moe_runner_config.gemm1_clamp_limit or 7.0
        layer.gemm1_alpha = torch.nn.Parameter(
            torch.full(
                (num_experts,),
                float(alpha),
                dtype=torch.float32,
                device=self.device,
            ),
            requires_grad=False,
        )
        layer.gemm1_beta = torch.nn.Parameter(
            torch.ones(num_experts, dtype=torch.float32, device=self.device),
            requires_grad=False,
        )
        layer.gemm1_clamp_limit = torch.nn.Parameter(
            torch.full(
                (num_experts,),
                float(limit),
                dtype=torch.float32,
                device=self.device,
            ),
            requires_grad=False,
        )
        layer.w13_weight_bias = torch.nn.Parameter(
            torch.zeros(
                (num_experts, w13_rows), dtype=torch.float32, device=self.device
            ),
            requires_grad=False,
        )
        layer.w2_weight_bias = torch.nn.Parameter(
            torch.zeros(
                (num_experts, hidden_size),
                dtype=torch.float32,
                device=self.device,
            ),
            requires_grad=False,
        )

    @property
    def registry(self):
        return _MXFP4_PREFILL_LAYER_REGISTRY.get(self.signature, {})

    @property
    def layer_order(self) -> List[int]:
        return sorted(self.registry)

    def successor_layer_idx(self, layer_idx: int) -> Optional[int]:
        order = self.layer_order
        pos = bisect.bisect_right(order, layer_idx)
        return order[pos] if pos < len(order) else None

    def abort_round(self) -> None:
        if not self.round_active:
            return
        self.epoch += 1
        self.last_layer_position = None
        self.current_slot_index = None
        self.round_active = False
        for slot in self.slots:
            slot.invalidate()

    def _advance_round(self, layer_idx: int) -> None:
        order = self.layer_order
        pos = bisect.bisect_left(order, layer_idx)
        if pos >= len(order) or order[pos] != layer_idx:
            raise RuntimeError(
                f"MXFP4 layerwise prefill layer {layer_idx} is not registered; "
                f"registered layers are {order}"
            )

        if (
            not self.round_active
            or self.last_layer_position is None
            or pos <= self.last_layer_position
        ):
            self.epoch += 1
            self.round_active = True
            for slot in self.slots:
                if slot.epoch != self.epoch:
                    slot.invalidate()
        self.last_layer_position = pos

    def _find_ready_slot(self, layer_idx: int) -> Optional[_Mxfp4PrefillSlot]:
        for slot in self.slots:
            if (
                slot.state == "READY"
                and slot.layer_idx == layer_idx
                and slot.epoch == self.epoch
            ):
                return slot
        return None

    @staticmethod
    def _record_prepared_backing_on_stream(
        slot: _Mxfp4PrefillSlot, stream: torch.cuda.Stream
    ) -> None:
        """Tell the caching allocator which prepared tensors compute consumes."""

        prepared = slot.prepared
        for tensor in (
            prepared.w13,
            prepared.w13_scale,
            prepared.w2,
            prepared.w2_scale,
        ):
            tensor.record_stream(stream)

    def _bind_slot(self, slot: _Mxfp4PrefillSlot) -> None:
        layer = self.context.gpu_layer
        if self.prepared_layout == _MXFP4_LAYOUT_MARLIN:
            layer._v4_marlin_weights = slot.prepared
            layer._v4_marlin_path = True
            layer._v4_tk_path = False
            return
        # trtllm: Mxfp4MoEMethod.apply reads the shuffled stacks straight off
        # the layer attributes its process_weights_after_loading rebinds
        # (mxfp4.py L827-830; consumed at L1575-1582 situ / L1655-1662
        # trtllm).  Install the slot's stable Parameter views over exactly
        # those names and delegate to the resident method's apply.
        for name, param in slot.prepared_params.items():
            setattr(layer, name, param)

    def _tp_phase_succeeded(self, local_success: bool) -> bool:
        if (
            not dist.is_initialized()
            or get_parallel().tp_size == 1
        ):
            return local_success
        status = getattr(self, "host_write_status", None)
        if status is None:
            status = torch.ones((1,), dtype=torch.int32, device="cpu")
            self.host_write_status = status
        status.fill_(int(local_success))
        dist.all_reduce(
            status,
            op=dist.ReduceOp.MIN,
            group=get_tp_group().cpu_group,
        )
        return bool(status.item())

    def _commit_tp_runtime_phase(
        self, local_error: Optional[Exception], phase: str
    ) -> None:
        if self._tp_phase_succeeded(local_error is None):
            return
        message = f"MXFP4 {phase} failed on at least one TP rank"
        if local_error is not None:
            raise RuntimeError(message) from local_error
        raise RuntimeError(message)

    def _tp_device_phase_succeeded(self, local_success: bool) -> bool:
        """Fast TP consensus for the per-expert transport control plane.

        Gloo remains the initialization/layer-boundary error plane.  The hot
        loop instead reduces a persistent CUDA scalar on a dedicated control
        stream, so it neither waits for the main compute stream nor allocates.
        Recoverable launch/control errors are propagated here; a poisoned CUDA
        context is process-fatal and is left to the process-group watchdog.
        """

        if (
            not dist.is_initialized()
            or get_parallel().tp_size == 1
        ):
            return local_success
        with torch.cuda.stream(self.control_stream):
            self.device_phase_status.fill_(int(local_success))
            dist.all_reduce(
                self.device_phase_status,
                op=dist.ReduceOp.MIN,
                group=get_tp_group().device_group,
            )
            # Keep the blocking scalar read on the stream where Work.wait()
            # inserted NCCL's completion dependency.  Reading after leaving
            # this context would race that dependency on the default stream.
            return bool(self.device_phase_status.item())

    def _commit_tp_device_runtime_phase(
        self, local_error: Optional[Exception], phase: str
    ) -> None:
        if self._tp_device_phase_succeeded(local_error is None):
            return
        message = f"MXFP4 {phase} failed on at least one TP rank"
        if local_error is not None:
            raise RuntimeError(message) from local_error
        raise RuntimeError(message)

    def _submit_host_write(self, method, expert_id: int, host_slot: int) -> None:
        """Queue one expert's SHM write on the kt task queue (no sync).

        The caller batches submissions and issues a single
        ``sync_write_weight_scale_to_buffer`` per chunk — the kt task queue
        preserves submission order and ``sync`` drains every pending task.
        """
        pointers = self.context.all_rank_buffer_ptrs
        expert_nbytes = self.context.host_expert_nbytes

        offsets = {
            name: host_slot * expert_nbytes[name] for name in self.raw_names
        }

        def rank_pointers(name: str) -> List[int]:
            return [ptr + offsets[name] for ptr in pointers[name]]

        # raw_names order is (w13, w13_scale, w2, w2_scale) in every layout —
        # the positional contract of write_weight_scale_to_buffer.
        w13_name, w13_scale_name, w2_name, w2_scale_name = self.raw_names
        method.wrapper.submit_write_weight_scale_to_buffer(
            get_parallel().tp_size,
            expert_id,
            rank_pointers(w13_name),
            rank_pointers(w13_scale_name),
            rank_pointers(w2_name),
            rank_pointers(w2_scale_name),
        )

    def _copy_resident_trtllm_experts(
        self,
        *,
        slot: _Mxfp4PrefillSlot,
        method,
        original_layer: torch.nn.Module,
        gpu_expert_ids: List[int],
    ) -> None:
        """Copy GPU-resident experts straight into the prepared slot image.

        After Mxfp4MoEMethod.process_weights_after_loading the resident layer
        holds only trtllm-gen shuffled stacks (mxfp4.py L827-830; the raw
        checkpoint layout is not preserved).  Both the weight shuffle and the
        scale interleave permute within one expert, so the resident expert's
        shuffled image is byte-identical to what the export+swizzle path
        would produce — copy it into ``slot.prepared`` and skip its raw rows
        entirely.  Caller runs on the transfer stream after the slot reuse
        fences.
        """
        prepared = slot.prepared
        resident_pairs = (
            (original_layer.w13_weight, prepared.w13),
            (original_layer.w13_weight_scale, prepared.w13_scale),
            (original_layer.w2_weight, prepared.w2),
            (original_layer.w2_weight_scale, prepared.w2_scale),
        )
        if not gpu_expert_ids:
            return
        device = prepared.w13.device
        dst_index = torch.tensor(gpu_expert_ids, dtype=torch.long, device=device)
        src_index = (
            method.logical_to_gpu_index[gpu_expert_ids]
            .to(device=device, dtype=torch.long)
        )
        # Gather-scatter in bounded chunks: advanced indexing materializes the
        # gathered rows, so cap the transient at ~1/4 of a layer's residents.
        chunk = max(1, len(gpu_expert_ids) // 4)
        for start in range(0, len(gpu_expert_ids), chunk):
            dst_part = dst_index[start : start + chunk]
            src_part = src_index[start : start + chunk]
            for source, destination in resident_pairs:
                destination[dst_part] = source[src_part]

    def _postprocess_slot(
        self, slot: _Mxfp4PrefillSlot, *, cpu_expert_ids: List[int]
    ) -> None:
        with torch.cuda.stream(self.postprocess_stream):
            self.postprocess_stream.wait_event(slot.raw_ready_event)
            try:
                # Repack and scale-swizzle into stable caller-owned storage.  The
                # kernels are current-stream ordered and publish no events; the
                # layerwise scheduler owns the raw/ready/consumed lifecycle.
                if self.prepared_layout == _MXFP4_LAYOUT_MARLIN:
                    from sglang.srt.layers.quantization.v4_marlin_moe import (
                        prepare_v4_mxfp4_marlin,
                    )

                    prepare_v4_mxfp4_marlin(
                        slot.w13_weight,
                        slot.w13_weight_scale_inv,
                        slot.w2_weight,
                        slot.w2_weight_scale_inv,
                        out=slot.prepared,
                    )
                else:
                    # trtllm: only CPU-resident experts hold export bytes in
                    # the raw slot; GPU-resident experts were copied into the
                    # prepared image directly (already shuffled) and their
                    # prepared rows must not be overwritten.
                    from sglang.srt.layers.moe.kt_mxfp4_export import (
                        prepare_trtllm_mxfp4,
                    )

                    prepare_trtllm_mxfp4(
                        slot.w13_weight,
                        slot.w13_weight_scale,
                        slot.w2_weight,
                        slot.w2_weight_scale,
                        out=slot.prepared,
                        expert_ids=cpu_expert_ids,
                    )
            finally:
                # Even an exceptional conversion attempt must publish a fence
                # before this slot can be overwritten on a retry.
                try:
                    slot.ready_event.record(self.postprocess_stream)
                    slot.reuse_guard = "ready"
                except Exception:
                    self.postprocess_stream.synchronize()
                    slot.reuse_guard = "synchronized"
                    raise

    def _load_slot(
        self,
        slot: _Mxfp4PrefillSlot,
        layer_idx: int,
        method,
        original_layer: torch.nn.Module,
    ) -> None:
        if getattr(slot, "reuse_guard", None) == "poisoned":
            raise RuntimeError(
                f"MXFP4 slot {slot.index} cannot be reused after its CUDA "
                "transfer stream failed to synchronize"
            )
        if slot.state == "LOADING":
            raise RuntimeError(
                f"MXFP4 slot {slot.index} is already loading layer {slot.layer_idx}"
            )

        # A prefetched slot may be invalidated before it is ever consumed.
        # In that case there is no consumed event for this generation, so its
        # ready event (postprocess completion) is the overwrite fence.
        reuse_guard = getattr(slot, "reuse_guard", None)
        if reuse_guard is None and slot.has_consumed_event:
            reuse_guard = "consumed"

        # reuse_guard describes the *current* load generation.  Retain the
        # prior generation only in the local variable above so an exception
        # after partially enqueueing this load cannot mistake an old ready
        # fence for protection of the new DMA.
        slot.reuse_guard = "loading"
        slot.state = "LOADING"
        slot.layer_idx = layer_idx
        slot.epoch = self.epoch

        saved_affinity = None
        try:
            setup_error = None
            try:
                weight_infos = [
                    (
                        name,
                        self.context.cpu_buffers[name],
                        getattr(slot, name),
                    )
                    for name in self.raw_names
                ]
                gpu_expert_ids = []
                cpu_expert_ids = []
                for expert_id in range(slot.num_experts):
                    if method.gpu_experts_mask[expert_id].item():
                        gpu_expert_ids.append(expert_id)
                    else:
                        cpu_expert_ids.append(expert_id)

                if hasattr(os, "sched_getaffinity"):
                    saved_affinity = os.sched_getaffinity(0)
                    available_cpus = sorted(saved_affinity)
                    if available_cpus:
                        # A per-rank band (not a single core) keeps the
                        # transport thread away from the kt threadpool's
                        # cores without serializing it behind whatever else
                        # the scheduler parks on one CPU.
                        band_width = 4
                        band_end = len(available_cpus) - method.tp_rank * band_width
                        band_start = band_end - band_width
                        band = set(
                            available_cpus[max(0, band_start) : max(0, band_end)]
                        )
                        os.sched_setaffinity(0, band or set(available_cpus))
            except Exception as exc:
                setup_error = exc
            self._commit_tp_runtime_phase(
                setup_error, f"transport setup for layer {layer_idx}"
            )

            gpu_copy_error = None
            try:
                with torch.cuda.stream(self.transfer_stream):
                    if reuse_guard == "consumed":
                        self.transfer_stream.wait_event(slot.consumed_event)
                    elif reuse_guard == "ready":
                        self.transfer_stream.wait_event(slot.ready_event)
                    elif reuse_guard == "raw":
                        self.transfer_stream.wait_event(slot.raw_ready_event)
                    if self.prepared_layout == _MXFP4_LAYOUT_MARLIN:
                        for expert_id in gpu_expert_ids:
                            gpu_index = method.logical_to_gpu_index[
                                expert_id
                            ].item()
                            for name, _, destination in weight_infos:
                                source = getattr(original_layer, name)
                                destination[expert_id].copy_(
                                    source[gpu_index], non_blocking=True
                                )
                    else:
                        self._copy_resident_trtllm_experts(
                            slot=slot,
                            method=method,
                            original_layer=original_layer,
                            gpu_expert_ids=gpu_expert_ids,
                        )
            except Exception as exc:
                gpu_copy_error = exc
            self._commit_tp_runtime_phase(
                gpu_copy_error, f"GPU expert copy for layer {layer_idx}"
            )

            # The SHM ring is consumed as two half-ring chunks so the control
            # plane runs per chunk, not per expert: one host reuse fence, one
            # batched kt submit + single sync, one producer-ready consensus,
            # then the whole chunk's H2D enqueues.  Halves ping-pong so rank
            # 0's SHM writes for chunk N+1 overlap the DMA drain of chunk N.
            chunk_capacity = max(1, self.context.host_ring_depth // 2)
            pending_h2d_error = None
            chunks = [
                cpu_expert_ids[start : start + chunk_capacity]
                for start in range(0, len(cpu_expert_ids), chunk_capacity)
            ]
            for chunk_idx, chunk in enumerate(chunks):
                half = chunk_idx % 2
                base_slot = half * chunk_capacity

                # Rank 0 writes every rank's SHM.  Every rank must therefore
                # finish its own DMA before this half-ring can be overwritten.
                # A prior H2D enqueue failure is sticky until this common
                # control point, which lets every rank leave the hot loop in
                # the same collective order instead of stranding a peer.
                host_free_error = pending_h2d_error
                pending_h2d_error = None
                try:
                    if self.host_slot_was_used[half]:
                        self.host_slot_free_events[half].synchronize()
                except Exception as exc:
                    if host_free_error is None:
                        host_free_error = exc
                self._commit_tp_device_runtime_phase(
                    host_free_error,
                    f"host half-ring {half} reuse for chunk {chunk_idx}",
                )

                write_error = None
                if method.tp_rank == 0:
                    try:
                        if method.wrapper is None:
                            raise RuntimeError(
                                "MXFP4 TP0 has no KT wrapper for host weight "
                                "transport"
                            )
                        for position, expert_id in enumerate(chunk):
                            self._submit_host_write(
                                method, expert_id, base_slot + position
                            )
                        method.wrapper.sync_write_weight_scale_to_buffer()
                    except Exception as exc:
                        write_error = exc

                # The device reduction is both the producer-ready fence and
                # an error broadcast.  TP0 therefore cannot strand peer ranks
                # in a later phase if its KT writer fails.
                self._commit_tp_device_runtime_phase(
                    write_error,
                    f"host write for chunk {chunk_idx} "
                    f"({len(chunk)} experts)",
                )

                host_free_recorded = False
                try:
                    with torch.cuda.stream(self.transfer_stream):
                        try:
                            for position, expert_id in enumerate(chunk):
                                cpu_slot = base_slot + position
                                for _, cpu_buffer, destination in weight_infos:
                                    destination[expert_id].copy_(
                                        cpu_buffer[cpu_slot], non_blocking=True
                                    )
                        finally:
                            # Once any DMA may have been enqueued, a peer's
                            # failure must not make this rank forget the local
                            # half-ring fence before the consensus raises.
                            self.host_slot_was_used[half] = True
                            self.host_slot_free_events[half].record(
                                self.transfer_stream
                            )
                            host_free_recorded = True
                except Exception as exc:
                    pending_h2d_error = exc
                    if self.host_slot_was_used[half] and not host_free_recorded:
                        try:
                            # Event publication itself failed.  A local-stream
                            # sync is the exception-only safe fallback before
                            # this rank reports failure to its peers.
                            self.transfer_stream.synchronize()
                            self.host_slot_was_used[half] = False
                        except Exception:
                            pass

            # H2D launch errors are reported at the next pre-write consensus.
            # The final expert has no successor, so combine its sticky status
            # with raw-fence publication and commit it once per layer.
            raw_ready_error = pending_h2d_error
            try:
                with torch.cuda.stream(self.transfer_stream):
                    slot.raw_ready_event.record(self.transfer_stream)
                slot.reuse_guard = "raw"
            except Exception as exc:
                if raw_ready_error is None:
                    raw_ready_error = exc
                try:
                    self.transfer_stream.synchronize()
                    slot.reuse_guard = "synchronized"
                except Exception:
                    pass
            self._commit_tp_runtime_phase(
                raw_ready_error, f"raw-ready fence for layer {layer_idx}"
            )

            postprocess_error = None
            try:
                self._postprocess_slot(slot, cpu_expert_ids=cpu_expert_ids)
            except Exception as exc:
                postprocess_error = exc
            self._commit_tp_runtime_phase(
                postprocess_error, f"postprocess for layer {layer_idx}"
            )
            slot.state = "READY"
        except Exception:
            # Fence any partial transfer generation before a caller can retry
            # and overwrite this slot.  _postprocess_slot upgrades the guard
            # to "ready" when it enqueues any postprocess work.
            fence_error = None
            if slot.reuse_guard not in ("ready", "synchronized"):
                try:
                    with torch.cuda.stream(self.transfer_stream):
                        # A setup failure can happen before the normal
                        # transfer block has consumed the prior generation's
                        # guard.  Preserve that dependency before publishing
                        # a replacement raw fence.
                        if reuse_guard == "consumed":
                            self.transfer_stream.wait_event(slot.consumed_event)
                        elif reuse_guard == "ready":
                            self.transfer_stream.wait_event(slot.ready_event)
                        elif reuse_guard == "raw":
                            self.transfer_stream.wait_event(slot.raw_ready_event)
                        slot.raw_ready_event.record(self.transfer_stream)
                    slot.reuse_guard = "raw"
                except Exception:
                    try:
                        # Event publication failed, so establish the fence
                        # synchronously.  This is exception-only and never
                        # enters the healthy layerwise pipeline.
                        self.transfer_stream.synchronize()
                        slot.reuse_guard = "synchronized"
                    except Exception as exc:
                        # The CUDA context can no longer provide an overwrite
                        # fence.  Keep the slot permanently non-reusable; the
                        # process-group watchdog handles this fatal condition.
                        slot.reuse_guard = "poisoned"
                        fence_error = exc
            slot.invalidate()
            if fence_error is not None:
                raise RuntimeError(
                    f"MXFP4 failed to fence slot {slot.index} after a "
                    "transport error"
                ) from fence_error
            raise
        finally:
            if saved_affinity is not None:
                try:
                    os.sched_setaffinity(0, saved_affinity)
                except Exception:
                    logger.warning(
                        "Failed to restore CPU affinity after MXFP4 transport",
                        exc_info=True,
                    )

    def _acquire(self, layer_idx: int, method, layer: torch.nn.Module):
        self._advance_round(layer_idx)
        ready = self._find_ready_slot(layer_idx)
        if ready is not None:
            return ready, True

        if self.current_slot_index is None:
            slot = self.slots[0]
        else:
            slot = self.slots[1 - self.current_slot_index]
        self._load_slot(slot, layer_idx, method, layer)
        return slot, False

    def _prefetch_successor(self, current: _Mxfp4PrefillSlot) -> None:
        successor_idx = self.successor_layer_idx(current.layer_idx)
        if successor_idx is None:
            return
        entry = self.registry.get(successor_idx)
        if entry is None:
            raise RuntimeError(
                f"MXFP4 successor layer {successor_idx} disappeared from registry"
            )
        next_method, next_layer = entry
        target = self.slots[1 - current.index]
        if (
            target.state == "READY"
            and target.layer_idx == successor_idx
            and target.epoch == self.epoch
        ):
            return
        self._load_slot(target, successor_idx, next_method, next_layer)

    def apply(self, method, layer, dispatch_output):
        layer_idx = method.kt_config.layer_idx
        slot, prefetch_hit = self._acquire(layer_idx, method, layer)
        if slot.layer_idx != layer_idx or slot.epoch != self.epoch:
            raise RuntimeError(
                "MXFP4 layerwise prefill acquired stale weights: "
                f"wanted layer={layer_idx}/epoch={self.epoch}, got "
                f"layer={slot.layer_idx}/epoch={slot.epoch}"
            )

        main_stream = None
        result = None
        compute_error = None
        try:
            main_stream = torch.cuda.current_stream(self.device)
            main_stream.wait_event(slot.ready_event)
            self._bind_slot(slot)
            self._record_prepared_backing_on_stream(slot, main_stream)
            result = self.context.gpu_method.apply(
                self.context.gpu_layer, dispatch_output
            )
        except Exception as exc:
            compute_error = exc
        finally:
            if main_stream is not None:
                try:
                    # Fence even when apply enqueues partial weight-reading
                    # work and then raises.
                    slot.consumed_event.record(main_stream)
                    slot.has_consumed_event = True
                    slot.reuse_guard = "consumed"
                    slot.state = "IN_USE"
                    self.current_slot_index = slot.index
                except Exception as exc:
                    if compute_error is None:
                        compute_error = exc
                    try:
                        main_stream.synchronize()
                        slot.reuse_guard = "synchronized"
                        slot.state = "IN_USE"
                        self.current_slot_index = slot.index
                    except Exception:
                        pass

        # A rank-local Python launch error must be observed by every peer
        # before any successful rank enters successor transport collectives.
        self._commit_tp_runtime_phase(
            compute_error, f"compute launch for layer {layer_idx}"
        )

        # Dynamic expert update (F2), manager path. Placed AFTER the compute
        # consensus so promotion's collectives can never pair with a diverged
        # rank's phase commit; its own phase commit keeps rank-local
        # promotion errors symmetric. The gate (kt_config) is replicated, so
        # every rank enters or skips this block together.
        if method.kt_config.kt_enable_dynamic_expert_update:
            promo_error = None
            promoted = False
            try:
                promoted = method._maybe_promote_experts_from_slot(
                    layer=layer, slot=slot, dispatch_output=dispatch_output
                )
            except Exception as exc:  # noqa: BLE001 — fed into the consensus
                promo_error = exc
            self._commit_tp_runtime_phase(
                promo_error, f"dynamic expert promotion for layer {layer_idx}"
            )
            if promoted and main_stream is not None:
                # Push the reuse fence past the promotion copies: no waiter
                # has observed the earlier record yet — this slot's next
                # load is scheduled later on this same host thread.
                slot.consumed_event.record(main_stream)

        # GPU compute is now enqueued.  Host KT writes and successor transfer
        # scheduling can overlap it without requiring an async kt-kernel API.
        self._prefetch_successor(slot)
        if method.tp_rank == 0:
            logger.info(
                "KT MXFP4 layerwise prefill: layer=%d epoch=%d slot=%d %s",
                layer_idx,
                self.epoch,
                slot.index,
                "prefetch-hit" if prefetch_hit else "prime",
            )
        return result


def _mxfp4_pipeline_signature(method, layer: torch.nn.Module) -> tuple:
    device = next(layer.parameters()).device
    return (
        str(device),
        method.kt_config.weight_path,
        method.kt_config.num_layers,
        method.global_num_experts,
        method._full_init_args,
    )


def _mxfp4_pipeline_requested(method) -> bool:
    return (
        method.gpu_prefill_token_threshold > 0
        and (method.kt_config.method or "").upper() == "MXFP4"
    )


def _mxfp4_pipeline_layout_or_reason(method) -> Tuple[Optional[str], Optional[str]]:
    """Classify the resident GPU method into a prepared-slot layout.

    Returns ``(layout, None)`` for a supported configuration,
    ``(None, reason)`` for a *recognized-but-unsupported* MXFP4 mode (the
    caller records the reason so threshold-qualified prefills stay on the
    hybrid path instead of the incompatible serial fallback), and
    ``(None, None)`` for unknown layouts — which keep today's plain
    unsupported semantics.
    """
    gpu_method = method.gpu_method
    if gpu_method.__class__.__name__ == "DeepSeekMxfp4MoEMethod":
        return _MXFP4_LAYOUT_MARLIN, None

    from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod

    # Precise class check: the DSV4 wrap chain (DeepSeekMxfp4MoEMethod over
    # Fp8MoEMethod) is handled above and subclasses are not vetted.
    if type(gpu_method) is not Mxfp4MoEMethod:
        return None, None
    # Mode checks follow Mxfp4MoEMethod.apply's dispatch priority
    # (mxfp4.py L1368 deep_gemm, L1426 marlin, L1430-1433 cutlass,
    # L1434 flashinfer trtllm).
    if gpu_method.use_deep_gemm:
        return None, (
            "resident Mxfp4MoEMethod is in deep_gemm mode; the layerwise "
            "slot pipeline supports only the flashinfer trtllm-gen (SM100) "
            "backend"
        )
    if gpu_method.use_marlin:
        return None, (
            "resident Mxfp4MoEMethod is in marlin mode, whose epilogue "
            "lacks SiTU; the layerwise slot pipeline supports only the "
            "flashinfer trtllm-gen (SM100) backend"
        )
    if gpu_method._fi_kernel != "trtllm_sm100":
        return None, (
            "resident Mxfp4MoEMethod is not in flashinfer trtllm-gen mode "
            f"(fi_kernel={gpu_method._fi_kernel!r}, "
            f"use_flashinfer={gpu_method.use_flashinfer}); the layerwise "
            "slot pipeline supports only the SM100 trtllm-gen backend"
        )
    if method.moe_runner_config.activation != "situ":
        return None, (
            "resident Mxfp4MoEMethod trtllm-gen path needs "
            "activation='situ' (the KT dispatch provides precomputed "
            "standard routing, which the non-situ trtllm branch does not "
            f"accept); got activation="
            f"{method.moe_runner_config.activation!r}"
        )
    return _MXFP4_LAYOUT_TRTLLM, None


def _mxfp4_pipeline_backend_supported(method, layer: torch.nn.Module) -> bool:
    if not _mxfp4_pipeline_requested(method) or not torch.cuda.is_available():
        return False
    layout, unsupported_reason = _mxfp4_pipeline_layout_or_reason(method)
    if layout is None:
        if unsupported_reason is not None:
            # Recognized MXFP4 mode the pipeline cannot serve: record a
            # TP-consistent disabled reason (the classification is derived
            # from process-global config, identical on every rank) so these
            # layers use hybrid CPU/GPU MoE.  Unknown layouts fall through
            # with no reason, keeping today's semantics.
            signature = _mxfp4_pipeline_signature(method, layer)
            if signature not in _MXFP4_LAYERWISE_DISABLED_REASONS:
                _disable_mxfp4_layerwise_pipeline(
                    signature, unsupported_reason
                )
        return False
    if layout == _MXFP4_LAYOUT_MARLIN:
        # Respect both diagnostic overrides.  The default capability-driven
        # path uses the prepared Marlin backend on Ada and Blackwell
        # consumer GPUs.
        if envs.SGLANG_V4_USE_TRITON_KERNELS.get() in ("0", "1"):
            return False
        device = next(layer.parameters()).device
        return torch.cuda.get_device_capability(device) in ((8, 9), (12, 0))
    # trtllm: _fi_kernel == "trtllm_sm100" already encodes the SM100
    # capability check performed at Mxfp4MoEMethod construction.
    return True


def _mxfp4_pipeline_runtime_supported(method, layer: torch.nn.Module) -> bool:
    if not _mxfp4_pipeline_backend_supported(method, layer):
        return False
    layout, _ = _mxfp4_pipeline_layout_or_reason(method)
    return all(
        hasattr(layer, name) for name in _MXFP4_RAW_NAMES_BY_LAYOUT[layout]
    )


def _mxfp4_raw_slot_storage_nbytes(
    *, num_experts: int, hidden_size: int, intermediate_size: int
) -> int:
    """Return the bytes in one full-expert raw MXFP4 slot.

    This mirrors ``DeepSeekMxfp4MoEMethod.create_weights`` without creating
    any tensors.  The resulting value is used only to limit KV-cache sizing;
    the actual full-expert slot remains lazy.
    """
    int8_size = torch.tensor([], dtype=torch.int8).element_size()
    float32_size = torch.tensor([], dtype=torch.float32).element_size()
    return (
        num_experts * (2 * intermediate_size) * (hidden_size // 2) * int8_size
        + num_experts * hidden_size * (intermediate_size // 2) * int8_size
        + num_experts
        * (2 * intermediate_size)
        * (hidden_size // 32)
        * float32_size
        + num_experts
        * hidden_size
        * (intermediate_size // 32)
        * float32_size
    )


def _mxfp4_trtllm_raw_slot_storage_nbytes(
    *, num_experts: int, hidden_size: int, intermediate_size: int
) -> int:
    """One full-expert raw slot for the trtllm layout, without allocating.

    Uint8 FP4 nibble weights shaped as ``Mxfp4MoEMethod.create_weights``
    registers them, plus **bf16** export scales (the SHM/H2D payload dtype;
    the created uint8 E8M0 scale params cannot serve as raw slot storage —
    see ``_allocate_mxfp4_slot_storage``).
    """
    uint8_size = torch.tensor([], dtype=torch.uint8).element_size()
    bf16_size = torch.tensor([], dtype=torch.bfloat16).element_size()
    weight_bytes = (
        num_experts * (2 * intermediate_size) * (hidden_size // 2) * uint8_size
        + num_experts * hidden_size * (intermediate_size // 2) * uint8_size
    )
    scale_elems = num_experts * (2 * intermediate_size) * (
        hidden_size // 32
    ) + num_experts * hidden_size * (intermediate_size // 32)
    return weight_bytes + scale_elems * bf16_size


def _mxfp4_trtllm_shadow_static_nbytes(
    *, num_experts: int, hidden_size: int, intermediate_size: int
) -> int:
    """Shadow-layer allocations the trtllm pipeline retains beyond the two
    raw and two prepared slots: create_weights' uint8 E8M0 scale params and
    bf16 biases (kept alive via ``SharedFullContext.original_params``) plus
    the float32 static attrs ``_initialize_trtllm_static_layer_attrs`` adds.
    """
    scale_elems = num_experts * (2 * intermediate_size) * (
        hidden_size // 32
    ) + num_experts * hidden_size * (intermediate_size // 32)
    bias_elems = num_experts * (2 * intermediate_size) + num_experts * hidden_size
    scalar_elems = 3 * num_experts
    # uint8 scales + bf16 raw biases + float32 shuffled biases + float32
    # gemm1_alpha/beta/clamp_limit.
    return scale_elems + bias_elems * (2 + 4) + scalar_elems * 4


def mxfp4_layerwise_prefill_reservation_gib(server_args) -> float:
    """KV-budget reservation for the lazily-allocated MXFP4 prefill slots.

    Zero unless the MXFP4 layerwise-prefill path is enabled; called by
    KVCacheConfigurator._profile_available_bytes after model load (the layer
    registry is populated during weight loading)."""
    if (server_args.kt_method or "").upper() != "MXFP4":
        return 0.0
    if (server_args.kt_gpu_prefill_token_threshold or 0) <= 0:
        return 0.0
    return get_mxfp4_layerwise_prefill_reservation_bytes() / (1 << 30)


def get_mxfp4_layerwise_prefill_reservation_bytes() -> int:
    """Return the unallocated MXFP4 slot capacity needed after a long request.

    KV-cache profiling normally consumes all currently free VRAM.  With
    layerwise slots allocated lazily, that would leave no capacity when the
    first threshold-qualified request arrives.  Account for two raw and two
    prepared slots here, but do not materialize them until that request.
    """
    total_bytes = 0
    for signature, registry in _MXFP4_PREFILL_LAYER_REGISTRY.items():
        if (
            not registry
            or signature in _MXFP4_LAYERWISE_MANAGERS
            or signature in _MXFP4_LAYERWISE_DISABLED_REASONS
        ):
            continue

        first_layer_idx = min(registry)
        method, layer = registry[first_layer_idx]
        if not _mxfp4_pipeline_runtime_supported(method, layer):
            continue

        init_args = getattr(method, "_full_init_args", None)
        num_experts = getattr(method, "global_num_experts", None)
        if init_args is None or num_experts is None:
            continue
        hidden_size, intermediate_size, _ = init_args

        layout, _ = _mxfp4_pipeline_layout_or_reason(method)
        if layout == _MXFP4_LAYOUT_MARLIN:
            from sglang.srt.layers.quantization.v4_marlin_moe import (
                get_v4_mxfp4_marlin_storage_nbytes,
            )

            raw_slot_bytes = _mxfp4_raw_slot_storage_nbytes(
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
            )
            prepared_slot_bytes = get_v4_mxfp4_marlin_storage_nbytes(
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
            )
            total_bytes += 2 * (raw_slot_bytes + prepared_slot_bytes)
        elif layout == _MXFP4_LAYOUT_TRTLLM:
            if hidden_size % 128 or intermediate_size % 128:
                # Mxfp4MoEMethod.create_weights would pad these dims, so the
                # export-shaped slot layout cannot apply; initialization
                # disables the pipeline with a reason and no slot capacity
                # is ever allocated.
                continue
            from sglang.srt.layers.moe.kt_mxfp4_export import (
                get_trtllm_mxfp4_storage_nbytes,
            )

            raw_slot_bytes = _mxfp4_trtllm_raw_slot_storage_nbytes(
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
            )
            prepared_slot_bytes = get_trtllm_mxfp4_storage_nbytes(
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
            )
            total_bytes += 2 * (
                raw_slot_bytes + prepared_slot_bytes
            ) + _mxfp4_trtllm_shadow_static_nbytes(
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
            )

    return total_bytes


def _register_mxfp4_prefill_layer(method, layer: torch.nn.Module) -> None:
    if not _mxfp4_pipeline_requested(method):
        return
    signature = _mxfp4_pipeline_signature(method, layer)
    method._mxfp4_pipeline_signature = signature
    registry = _MXFP4_PREFILL_LAYER_REGISTRY.setdefault(signature, {})
    layer_idx = method.kt_config.layer_idx
    existing = registry.get(layer_idx)
    if existing is not None and (
        existing[0] is not method or existing[1] is not layer
    ):
        raise RuntimeError(
            f"duplicate MXFP4 layerwise prefill registration for layer {layer_idx}"
        )
    registry[layer_idx] = (method, layer)


def _all_tp_ranks_succeeded(local_success: bool) -> bool:
    if not dist.is_initialized() or get_parallel().tp_size == 1:
        return local_success
    status = torch.tensor([int(local_success)], dtype=torch.int32, device="cpu")
    dist.all_reduce(status, op=dist.ReduceOp.MIN, group=get_tp_group().cpu_group)
    return bool(status.item())


def _any_tp_rank_true(local_value: bool) -> bool:
    if not dist.is_initialized() or get_parallel().tp_size == 1:
        return local_value
    status = torch.tensor([int(local_value)], dtype=torch.int32, device="cpu")
    dist.all_reduce(status, op=dist.ReduceOp.MAX, group=get_tp_group().cpu_group)
    return bool(status.item())


def _all_tp_ranks_succeeded_vec(local_ok: List[bool]) -> List[bool]:
    """Element-wise unanimity across ranks; ONE fixed-shape collective.

    The shape is len(local_ok), which every caller derives from a
    plan-deterministic input (identical on all ranks), so the collective is
    symmetric by construction -- the same discipline as
    _all_tp_ranks_succeeded, vectorized for per-pair decisions.
    """
    if not dist.is_initialized() or get_parallel().tp_size == 1:
        return list(local_ok)
    status = torch.tensor([int(v) for v in local_ok], dtype=torch.int32, device="cpu")
    dist.all_reduce(status, op=dist.ReduceOp.MIN, group=get_tp_group().cpu_group)
    return [bool(v) for v in status.tolist()]


def _disable_mxfp4_layerwise_pipeline(signature: tuple, reason: str) -> None:
    _MXFP4_LAYERWISE_DISABLED_REASONS[signature] = reason
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if get_parallel().tp_rank == 0:
        logger.warning(
            "KT MXFP4 layerwise prefill disabled; using hybrid CPU/GPU MoE: %s",
            reason,
        )


def _try_mxfp4_initialization(factory):
    """Run an allocation without leaking exception tracebacks to the caller."""

    try:
        return factory(), None
    except torch.cuda.OutOfMemoryError as exc:
        return None, ("oom", str(exc))
    except Exception as exc:
        return None, ("fatal", f"{type(exc).__name__}: {exc}")


def _allocate_mxfp4_slot_storage(context: SharedFullContext):
    """Build (slot0_raw, slot1_raw, slot_prepared) for the context's layout.

    Marlin (DSV4): slot 0 raw storage reuses all four created params; the
    prepared Marlin images live in separate storage.

    trtllm (K3): slot 0 reuses the created uint8 weight params, but the raw
    scale storage must be fresh **bf16** tensors — the export payload is
    bf16 while ``Mxfp4MoEMethod.create_weights`` registers uint8 E8M0 scale
    params, and a bf16→uint8 ``copy_`` would numerically cast instead of
    transporting bytes.
    """
    layer = context.gpu_layer
    if context.mxfp4_prepared_layout == _MXFP4_LAYOUT_MARLIN:
        from sglang.srt.layers.quantization.v4_marlin_moe import (
            allocate_v4_mxfp4_marlin,
        )

        raw_names = _MXFP4_RAW_NAMES_BY_LAYOUT[_MXFP4_LAYOUT_MARLIN]
        slot0_raw = {name: getattr(layer, name).data for name in raw_names}
        slot1_raw = {
            name: torch.empty_like(getattr(layer, name).data)
            for name in raw_names
        }
        num_experts = slot0_raw["w13_weight"].shape[0]
        hidden_size = slot0_raw["w13_weight"].shape[2] * 2
        intermediate_size = slot0_raw["w2_weight"].shape[2] * 2
        device = slot0_raw["w13_weight"].device
        slot_prepared = tuple(
            allocate_v4_mxfp4_marlin(
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                device=device,
            )
            for _ in range(2)
        )
        return slot0_raw, slot1_raw, slot_prepared

    from sglang.srt.layers.moe.kt_mxfp4_export import allocate_trtllm_mxfp4

    device = layer.w13_weight.device
    slot0_raw = {
        "w13_weight": layer.w13_weight.data,
        "w13_weight_scale": torch.empty(
            tuple(layer.w13_weight_scale.shape),
            dtype=torch.bfloat16,
            device=device,
        ),
        "w2_weight": layer.w2_weight.data,
        "w2_weight_scale": torch.empty(
            tuple(layer.w2_weight_scale.shape),
            dtype=torch.bfloat16,
            device=device,
        ),
    }
    slot1_raw = {
        name: torch.empty_like(tensor) for name, tensor in slot0_raw.items()
    }
    num_experts = slot0_raw["w13_weight"].shape[0]
    hidden_size = slot0_raw["w13_weight"].shape[2] * 2
    intermediate_size = slot0_raw["w2_weight"].shape[2] * 2
    slot_prepared = tuple(
        allocate_trtllm_mxfp4(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            device=device,
        )
        for _ in range(2)
    )
    return slot0_raw, slot1_raw, slot_prepared


def _trtllm_export_shape_mismatch(
    *, context: SharedFullContext, method
) -> Optional[str]:
    """Return a disable reason if the shadow layer cannot hold kt-export
    bytes, else None.

    ``Mxfp4MoEMethod.create_weights`` pads hidden/intermediate to multiples
    of 128 on the SM100 flashinfer branch (mxfp4.py L414-417).  The kt
    export writes unpadded ``[2I, H/2]`` / ``[H, I/2]`` shards, so any
    padding breaks the raw byte layout (row-stride mismatch) and the
    pipeline must fall back to hybrid CPU/GPU MoE.
    """
    hidden_size, intermediate_size, _ = method._full_init_args
    layer = context.gpu_layer
    expected_w13 = (
        method.global_num_experts,
        2 * intermediate_size,
        hidden_size // 2,
    )
    expected_w2 = (
        method.global_num_experts,
        hidden_size,
        intermediate_size // 2,
    )
    actual_w13 = tuple(layer.w13_weight.shape)
    actual_w2 = tuple(layer.w2_weight.shape)
    if actual_w13 != expected_w13 or actual_w2 != expected_w2:
        return (
            "trtllm-gen layerwise slots need kt-export-shaped raw weights; "
            f"Mxfp4MoEMethod.create_weights padded them (w13 {actual_w13} "
            f"vs expected {expected_w13}, w2 {actual_w2} vs expected "
            f"{expected_w2})"
        )
    if hidden_size % 128 or intermediate_size % 128:
        return (
            "trtllm-gen shuffled layout requires hidden/intermediate "
            f"multiples of 128, got {hidden_size}/{intermediate_size}"
        )
    return None


def _initialize_mxfp4_layerwise_pipeline(method, layer: torch.nn.Module) -> None:
    if not _mxfp4_pipeline_backend_supported(method, layer):
        return
    signature = getattr(method, "_mxfp4_pipeline_signature", None)
    if signature is None:
        signature = _mxfp4_pipeline_signature(method, layer)
        method._mxfp4_pipeline_signature = signature
    if signature in _MXFP4_LAYERWISE_MANAGERS:
        return
    if signature in _MXFP4_LAYERWISE_DISABLED_REASONS:
        return

    context, context_failure = _try_mxfp4_initialization(
        lambda: SharedFullContext(
            layer=layer,
            init_args=method._full_init_args,
            global_num_experts=method.global_num_experts,
            moe_runner_config=method.moe_runner_config,
            defer_cpu_buffers=True,
        )
    )

    if not _all_tp_ranks_succeeded(context_failure is None):
        context = None
        fatal_error = _any_tp_rank_true(
            context_failure is not None and context_failure[0] == "fatal"
        )
        if fatal_error:
            message = "MXFP4 slot 0 initialization failed on at least one TP rank"
            if context_failure is not None:
                message = f"{message}: {context_failure[1]}"
            raise RuntimeError(message)
        context_failure = None
        _disable_mxfp4_layerwise_pipeline(
            signature, "full-layer slot 0 allocation failed on at least one TP rank"
        )
        return
    if context is None or not getattr(context, "_is_mxfp4_quant", False):
        raise RuntimeError("MXFP4 layerwise prefill built a non-MXFP4 full context")
    expected_layout, _ = _mxfp4_pipeline_layout_or_reason(method)
    if context.mxfp4_prepared_layout != expected_layout:
        raise RuntimeError(
            "MXFP4 layerwise prefill context layout "
            f"{context.mxfp4_prepared_layout!r} does not match the resident "
            f"method's prepared layout {expected_layout!r}"
        )
    if expected_layout == _MXFP4_LAYOUT_TRTLLM:
        # Deterministic on every rank (pure shape math on shared config), so
        # no TP consensus round is needed before disabling.
        shape_mismatch = _trtllm_export_shape_mismatch(
            context=context, method=method
        )
        if shape_mismatch is not None:
            context = None
            _disable_mxfp4_layerwise_pipeline(signature, shape_mismatch)
            return

    slot_storage, slot_failure = _try_mxfp4_initialization(
        lambda: _allocate_mxfp4_slot_storage(context)
    )

    if not _all_tp_ranks_succeeded(slot_failure is None):
        slot_storage = None
        context = None
        fatal_error = _any_tp_rank_true(
            slot_failure is not None and slot_failure[0] == "fatal"
        )
        if fatal_error:
            message = "MXFP4 raw/prepared slot allocation failed on at least one TP rank"
            if slot_failure is not None:
                message = f"{message}: {slot_failure[1]}"
            raise RuntimeError(message)
        slot_failure = None
        _disable_mxfp4_layerwise_pipeline(
            signature,
            "full-layer raw/prepared allocation failed on at least one TP rank",
        )
        return
    slot0_raw, slot1_raw, slot_prepared = slot_storage

    manager, manager_failure = _try_mxfp4_initialization(
        lambda: _Mxfp4LayerwisePrefillManager(
            context=context,
            signature=signature,
            slot0_raw_tensors=slot0_raw,
            slot1_raw_tensors=slot1_raw,
            slot_prepared=slot_prepared,
        )
    )
    if not _all_tp_ranks_succeeded(manager_failure is None):
        manager = None
        slot_storage = None
        slot0_raw.clear()
        slot1_raw.clear()
        slot_prepared = None
        context = None
        fatal_error = _any_tp_rank_true(
            manager_failure is not None and manager_failure[0] == "fatal"
        )
        if fatal_error:
            message = "MXFP4 prepared slot setup failed on at least one TP rank"
            if manager_failure is not None:
                message = f"{message}: {manager_failure[1]}"
            raise RuntimeError(message)
        manager_failure = None
        _disable_mxfp4_layerwise_pipeline(
            signature,
            "persistent MXFP4 prepared slot setup failed on at least one TP rank",
        )
        return

    context.initialize_cpu_buffers()
    _MXFP4_LAYERWISE_MANAGERS[signature] = manager
    get_buffer("kt_full_context", dict).setdefault("ctx", context)
    if method.tp_rank == 0:
        raw_bytes = sum(
            getattr(slot, name).numel() * getattr(slot, name).element_size()
            for slot in manager.slots
            for name in manager.raw_names
        )
        prepared_bytes = sum(
            tensor.numel() * tensor.element_size()
            for slot in manager.slots
            for tensor in (
                slot.prepared.w13,
                slot.prepared.w13_scale,
                slot.prepared.w2,
                slot.prepared.w2_scale,
            )
        )
        logger.info(
            "KT MXFP4 layerwise prefill lazily initialized two raw + two "
            "%s-prepared full-layer slots on %s (raw=%.2f GiB, "
            "prepared=%.2f GiB, total=%.2f GiB)",
            manager.prepared_layout,
            manager.device,
            raw_bytes / 1024**3,
            prepared_bytes / 1024**3,
            (raw_bytes + prepared_bytes) / 1024**3,
        )


def _get_or_initialize_mxfp4_layerwise_manager(
    method, layer: torch.nn.Module
) -> Optional[_Mxfp4LayerwisePrefillManager]:
    """Return the persistent manager, allocating its slots on first use.

    This function is called only from a threshold-qualified prefill forward.
    Each TP rank executes it before the manager enters its transport
    collectives, and `_initialize_mxfp4_layerwise_pipeline` keeps allocation
    failures consistent across ranks.  An OOM records a disabled reason and
    returns ``None`` so the triggering request can use hybrid CPU/GPU MoE.
    """

    signature = getattr(method, "_mxfp4_pipeline_signature", None)
    if signature is None:
        signature = _mxfp4_pipeline_signature(method, layer)
        method._mxfp4_pipeline_signature = signature

    manager = _MXFP4_LAYERWISE_MANAGERS.get(signature)
    if manager is not None or signature in _MXFP4_LAYERWISE_DISABLED_REASONS:
        return manager

    _initialize_mxfp4_layerwise_pipeline(method, layer)
    return _MXFP4_LAYERWISE_MANAGERS.get(signature)


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

    # NOTE: hash-layer skip experiment was tried here (return None when
    # layer_idx < num_hash_layers); it didn't help because the underlying
    # fused_moe shape-mismatch in V4 hash MoE happens with or without KT wrap.
    # Reverted; root cause is in V4 MoE weight layout vs sglang fused_moe.

    # Get mask for this specific layer
    gpu_experts_mask = masks[layer_idx]

    if bool(gpu_experts_mask.all()):
        # Fully GPU-resident layer: leave the plain quant method unwrapped —
        # no CPU submit, no staging, no per-layer round-trip.  The loader and
        # the layerwise-prefill registry tolerate per-layer absence.
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
        gpu_prefill_token_threshold=server_args.kt_gpu_prefill_token_threshold,
        kt_enable_dynamic_expert_update=server_args.kt_enable_dynamic_expert_update,
        routing_margin=server_args.kt_routing_margin,
        routing_full_override=server_args.kt_routing_full_override,
        transport=server_args.kt_transport,
        transport_pollers=server_args.kt_transport_pollers,
        conditional_cpu_branch=server_args.kt_conditional_cpu_branch,
        cold_only_cpu_experts=server_args.kt_cold_only_cpu_experts,
        expert_swap_interval=server_args.kt_expert_swap_interval,
        expert_swap_max=server_args.kt_expert_swap_max,
        expert_swap_hysteresis=server_args.kt_expert_swap_hysteresis,
        split_prefill=server_args.kt_expert_split_prefill,
        split_prefill_token_tile=server_args.kt_expert_split_prefill_token_tile,
        cold_transport=server_args.kt_cold_transport,
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


def _margin_override_topk_ids_impl(
    topk_ids: torch.Tensor,
    router_logits: torch.Tensor,
    gpu_experts_mask: torch.Tensor,
    margin: float,
    full_override: bool = False,
) -> tuple:
    """Margin routing (SPEC-MARGIN-ROUTING P1): GPU-preferred top-k rewrite.

    For each routed slot holding a CPU-resident expert, compare its router
    logit against the token's best GPU-resident expert that is not already
    selected.  If the lead is below ``margin`` the slot is rewritten to a
    resident alternative (an "override"); otherwise the CPU expert is kept
    (an "insist").  The i-th overridden slot of a token takes the token's
    i-th best unselected resident, so multiple overrides in one token land
    on distinct experts.  Slot weights are NOT touched: the substitute
    inherits the overridden slot's weight (least-perturbation stand-in).

    Margins are compared in router-logit space: per token, logit order
    equals score order for both sigmoid and softmax routers, so the rule is
    activation-independent (the noaux_tc selection bias is not visible here;
    the margin sweep absorbs that systematic offset).

    Pure tensor ops over the LIVE ``gpu_experts_mask`` — CUDA-graph
    capturable, and replays follow in-place mask updates (same contract as
    ``make_placement_aware_deferred_selector``).

    Returns:
        (new_topk_ids, insist_slots, override_slots) — masks are per-slot
        bools aligned with the ORIGINAL topk_ids (true router preference).
    """
    safe_ids = topk_ids.clamp_min(0).to(torch.int64)
    routed = topk_ids >= 0
    cpu_routed = ~gpu_experts_mask[safe_ids] & routed

    logits = router_logits.float()
    neg_inf = float("-inf")
    resident_scores = logits.masked_fill(~gpu_experts_mask.unsqueeze(0), neg_inf)
    # An already-selected expert (resident picks included) is not an
    # alternative: substituting a duplicate would only re-weight it.
    resident_scores = resident_scores.scatter(-1, safe_ids, neg_inf)

    k = topk_ids.shape[-1]
    alt_scores, alt_ids = torch.topk(resident_scores, k=k, dim=-1)
    best_alt = alt_scores[:, :1]
    finite_alt = torch.isfinite(best_alt)
    if full_override:
        # Static Python bool: Dynamo specializes it into its own graph, so the
        # lead comparison is compiled out rather than run against a sentinel
        # (a float("inf") margin would also break under non-finite logits).
        # The isfinite rail stays: without a real resident alternative there
        # is nothing to override to, and dropping it would let topk hand back
        # an expert from the -inf pool — i.e. a CPU expert again.
        override_slots = cpu_routed & finite_alt
    else:
        slot_logit = torch.gather(logits, -1, safe_ids)
        lead = slot_logit - best_alt
        # ``margin`` is either a float (server default for every token) or a
        # [num_tokens] tensor of per-request overrides. The tensor form
        # broadcasts against lead's [num_tokens, top_k] once unsqueezed, so the
        # rule is unchanged per slot -- only the threshold varies by row.
        if isinstance(margin, torch.Tensor):
            # Per-token thresholds. The extra ``margin > 0`` term carries the
            # count-only contract PER TOKEN: with a scalar margin the caller
            # enforces "0.0 means route exactly" by skipping the rewrite for
            # the whole batch, which cannot express one request at 0.0 beside
            # another at 2.0. Without this term such a request would be
            # overridden whenever its lead were negative.
            margin = margin.reshape(-1, 1)
            override_slots = (
                cpu_routed & (lead < margin) & (margin > 0.0) & finite_alt
            )
        else:
            override_slots = cpu_routed & (lead < margin) & finite_alt

    # Rank overridden slots within each token, then drop any slot whose
    # assigned alternative is -inf (fewer unselected residents than
    # overrides — degenerate layers only; production keeps residents >> k).
    alt_rank = (torch.cumsum(override_slots.to(torch.int64), dim=-1) - 1).clamp_min(0)
    override_slots = override_slots & torch.isfinite(
        torch.gather(alt_scores, -1, alt_rank)
    )
    insist_slots = cpu_routed & ~override_slots

    substitute = torch.gather(alt_ids, -1, alt_rank).to(topk_ids.dtype)
    new_topk_ids = torch.where(override_slots, substitute, topk_ids)
    return new_topk_ids, insist_slots, override_slots


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


def select_top_experts_from_batch(
    topk_ids: torch.Tensor,
    num_experts: int,
    num_gpu_experts: int,
) -> torch.Tensor:
    """Select top N most frequently activated experts from batch routing results.

    Args:
        topk_ids: Tensor of shape [num_tokens, top_k] containing logical expert IDs
        num_experts: Total number of experts in the layer
        num_gpu_experts: Number of experts to select for GPU

    Returns:
        Tensor of shape [num_gpu_experts] containing selected expert IDs (sorted)

    Edge cases:
        - If batch has fewer unique experts than num_gpu_experts, fills remaining
          slots with least-activated experts (maintaining determinism)
        - Handles ties by preferring lower expert IDs (deterministic)
    """
    # Count activation frequency for each expert in this batch
    expert_counts = torch.zeros(num_experts, dtype=torch.int64, device=topk_ids.device)

    # Flatten topk_ids and count occurrences
    flat_ids = topk_ids.flatten()
    # Filter out invalid IDs (< 0 or >= num_experts)
    valid_mask = (flat_ids >= 0) & (flat_ids < num_experts)
    valid_ids = flat_ids[valid_mask]

    if valid_ids.numel() > 0:
        expert_counts.index_add_(0, valid_ids, torch.ones_like(valid_ids, dtype=torch.int64))

    # Select top num_gpu_experts by frequency
    # For ties, torch.topk with sorted=True will prefer earlier indices (deterministic)
    _, selected_indices = torch.topk(
        expert_counts,
        k=min(num_gpu_experts, num_experts),
        largest=True,
        sorted=True  # Ensures deterministic tie-breaking
    )

    # Sort selected indices for easier debugging and consistent ordering
    selected_experts = selected_indices.sort()[0]

    return selected_experts


def copy_experts_weights_int4(
    src_layer: torch.nn.Module,
    dst_layer: torch.nn.Module,
    selected_experts: torch.Tensor,
) -> None:
    """Copy INT4 Marlin expert weights from source to destination layer.

    Args:
        src_layer: Source layer (temporary full GPU layer) with all experts
        dst_layer: Destination layer (original layer) with subset of experts
        selected_experts: Tensor of logical expert IDs to copy (shape: [num_gpu_experts])

    This copies:
        - w13_weight_packed: Packed INT4 weights for gate+up projection
        - w13_weight_scale: FP16 scales for w13
        - w2_weight_packed: Packed INT4 weights for down projection
        - w2_weight_scale: FP16 scales for w2
    """
    weight_names = ["w13_weight_packed", "w13_weight_scale", "w2_weight_packed", "w2_weight_scale"]

    # Build mapping: selected logical ID -> dst GPU index
    logical_to_dst_index = {
        int(selected_experts[i].item()): i
        for i in range(len(selected_experts))
    }

    for weight_name in weight_names:
        src_weight = getattr(src_layer, weight_name)  # [global_num_experts, ...]
        dst_weight = getattr(dst_layer, weight_name)  # [num_gpu_experts, ...]

        # Copy each selected expert
        for logical_id, dst_idx in logical_to_dst_index.items():
            # In src_layer, expert at logical_id is at index logical_id
            # In dst_layer, we write to gpu_index dst_idx
            dst_weight[dst_idx].copy_(src_weight[logical_id], non_blocking=False)


def copy_experts_weights_fp8(
    src_layer: torch.nn.Module,
    dst_layer: torch.nn.Module,
    selected_experts: torch.Tensor,
) -> None:
    """Copy FP8 block quant expert weights from source to destination layer.

    Args:
        src_layer: Source layer (temporary full GPU layer) with all experts
        dst_layer: Destination layer (original layer) with subset of experts
        selected_experts: Tensor of logical expert IDs to copy (shape: [num_gpu_experts])

    This copies:
        - w13_weight: FP8 weights for gate+up projection
        - w13_weight_scale_inv: FP32 inverse scales for w13
        - w2_weight: FP8 weights for down projection
        - w2_weight_scale_inv: FP32 inverse scales for w2
    """
    weight_names = ["w13_weight", "w13_weight_scale_inv", "w2_weight", "w2_weight_scale_inv"]

    # Build mapping: selected logical ID -> dst GPU index
    logical_to_dst_index = {
        int(selected_experts[i].item()): i
        for i in range(len(selected_experts))
    }

    for weight_name in weight_names:
        src_weight = getattr(src_layer, weight_name)  # [global_num_experts, ...]
        dst_weight = getattr(dst_layer, weight_name)  # [num_gpu_experts, ...]

        # Copy each selected expert
        for logical_id, dst_idx in logical_to_dst_index.items():
            dst_weight[dst_idx].copy_(src_weight[logical_id], non_blocking=False)


def copy_experts_weights_fp8_channel(
    src_layer: torch.nn.Module,
    dst_layer: torch.nn.Module,
    selected_experts: torch.Tensor,
) -> None:
    """Copy FP8 per-channel quant expert weights from source to destination layer.

    Args:
        src_layer: Source layer (temporary full GPU layer) with all experts
        dst_layer: Destination layer (original layer) with subset of experts
        selected_experts: Tensor of logical expert IDs to copy (shape: [num_gpu_experts])

    This copies:
        - w13_weight: FP8 weights for gate+up projection
        - w13_weight_scale: FP32 per-channel scales for w13
        - w2_weight: FP8 weights for down projection
        - w2_weight_scale: FP32 per-channel scales for w2
    """
    weight_names = ["w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"]

    # Build mapping: selected logical ID -> dst GPU index
    logical_to_dst_index = {
        int(selected_experts[i].item()): i
        for i in range(len(selected_experts))
    }

    for weight_name in weight_names:
        src_weight = getattr(src_layer, weight_name)  # [global_num_experts, ...]
        dst_weight = getattr(dst_layer, weight_name)  # [num_gpu_experts, ...]

        # Copy each selected expert
        for logical_id, dst_idx in logical_to_dst_index.items():
            dst_weight[dst_idx].copy_(src_weight[logical_id], non_blocking=False)


def copy_experts_weights_bf16(
    src_layer: torch.nn.Module,
    dst_layer: torch.nn.Module,
    selected_experts: torch.Tensor,
) -> None:
    """Copy BF16/unquantized expert weights from source to destination layer.

    Args:
        src_layer: Source layer (temporary full GPU layer) with all experts
        dst_layer: Destination layer (original layer) with subset of experts
        selected_experts: Tensor of logical expert IDs to copy (shape: [num_gpu_experts])

    This copies:
        - w13_weight: BF16 weights for gate+up projection
        - w2_weight: BF16 weights for down projection
    """
    weight_names = ["w13_weight", "w2_weight"]

    # Build mapping: selected logical ID -> dst GPU index
    logical_to_dst_index = {
        int(selected_experts[i].item()): i
        for i in range(len(selected_experts))
    }

    for weight_name in weight_names:
        src_weight = getattr(src_layer, weight_name)  # [global_num_experts, ...]
        dst_weight = getattr(dst_layer, weight_name)  # [num_gpu_experts, ...]

        # Copy each selected expert
        for logical_id, dst_idx in logical_to_dst_index.items():
            dst_weight[dst_idx].copy_(src_weight[logical_id], non_blocking=False)


class Mxfp4RawExpertSource(msgspec.Struct, frozen=True):
    """Full-expert raw kt-export stacks for the re-swizzle copy mode.

    ``w13`` holds FP4 nibble bytes as ``[gate | up]`` halves (the kt export
    orientation) and ``w2`` the down projection; scales are the export's bf16
    expansion of the resident E8M0 codes.  fp32 scales are accepted when they
    are an exact staging cast of that bf16 payload (the marlin-layout raw
    slots stage the bf16 export through fp32 params) — the cast back to bf16
    is exact, and ``bf16_scales_to_e8m0`` still asserts E8M0 exactness.
    """

    w13: torch.Tensor
    w13_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor


class Mxfp4DynUpdatePlan(msgspec.Struct, frozen=True):
    """Cached per-layer decision for MXFP4 dynamic expert updates.

    ``disabled_reason is None`` means the resident layer holds the trtllm-gen
    shuffled image and ``param_names`` are the four resident attributes to
    write; otherwise the reason explains the TP-consistent disable.
    """

    param_names: Tuple[str, ...] = ()
    disabled_reason: Optional[str] = None


def _mxfp4_shadow_source_mismatch(
    *,
    ctx: "SharedFullContext",
    resident_layer: torch.nn.Module,
    param_names: Tuple[str, ...],
) -> Optional[str]:
    """Reason the post-fallback shadow cannot source direct row copies, or None.

    At the dynamic-update call site the shadow ``ctx.gpu_layer`` has already
    re-run ``Mxfp4MoEMethod.process_weights_after_loading`` (phase 3 of
    ``_prepare_weight_mxfp4``), so its flat attributes must hold the prepared
    trtllm-gen stacks: interleaved scales viewed as float8_e4m3fn
    (mxfp4.py L805-830) with per-expert rows shaped like the resident's
    (identical create_weights geometry; only the expert count differs).
    """
    if ctx.mxfp4_prepared_layout != _MXFP4_LAYOUT_TRTLLM:
        return (
            "full-expert shadow context targets the "
            f"{ctx.mxfp4_prepared_layout!r} prepared layout, not trtllm"
        )
    for lyr, role in (
        (ctx.gpu_layer, "full-expert shadow"),
        (resident_layer, "resident"),
    ):
        for name in param_names:
            if not hasattr(lyr, name):
                return f"{role} layer has no `{name}` attribute"
        for name in (param_names[1], param_names[3]):
            if getattr(lyr, name).dtype != torch.float8_e4m3fn:
                return (
                    f"{role} `{name}` dtype {getattr(lyr, name).dtype} is not "
                    "the interleaved float8_e4m3fn prepared-scale form"
                )
    for name in param_names:
        src = getattr(ctx.gpu_layer, name)
        dst = getattr(resident_layer, name)
        if src.shape[1:] != dst.shape[1:] or src.dtype != dst.dtype:
            return (
                f"shadow/resident `{name}` per-expert layouts differ: "
                f"{tuple(src.shape[1:])}/{src.dtype} vs "
                f"{tuple(dst.shape[1:])}/{dst.dtype}"
            )
    return None


def resolve_mxfp4_dyn_update_plan(
    *,
    gpu_method,
    resident_layer: torch.nn.Module,
    ctx: "SharedFullContext",
) -> Mxfp4DynUpdatePlan:
    """Classify the resident MXFP4 layer for dynamic expert updates.

    Supported: the ``Mxfp4MoEMethod`` flashinfer trtllm-gen resident — after
    its process_weights_after_loading, per-expert rows of the four resident
    params are independent shuffled/interleaved images (permute indices are
    computed once from a single-expert sample and applied per expert,
    mxfp4.py L740-803), so selected rows can be overwritten in place.

    Disabled with a reason (each derives from process-global config, the
    device capability, or deterministic shapes, so the local result is
    TP-consistent; the caller still commits it through a TP consensus):
    - the V4 triton_kernels resident (``_v4_tk_path``) — a diagnostic path;
      the fork never implemented tk expert copies and neither do we;
    - the V4 prepared-Marlin resident (``_v4_marlin_path``);
    - ``DeepSeekMxfp4MoEMethod``'s trtllm resident — its stacks live under
      ``w13_weight_scale_inv``/``w2_weight_scale_inv`` (mxfp4_deepseek.py
      L398-411) and its row shuffle is composed inside flashinfer's
      ``_maybe_get_cached_w3_w1_permute_indices`` (mxfp4_deepseek.py
      L344-386), which is not verified against the kt export swizzle;
    - non-trtllm ``Mxfp4MoEMethod`` modes (deep_gemm / marlin / cutlass).
    """
    if getattr(resident_layer, "_v4_tk_path", False):
        return Mxfp4DynUpdatePlan(
            disabled_reason=(
                "resident layer is on the V4 triton_kernels (tk) diagnostic "
                "path; dynamic expert updates implement no tk swizzled copies"
            )
        )
    if getattr(resident_layer, "_v4_marlin_path", False):
        return Mxfp4DynUpdatePlan(
            disabled_reason=(
                "resident layer holds prepared V4 Marlin weights; dynamic "
                "expert updates support only the trtllm-gen shuffled resident"
            )
        )

    # Class-name checks avoid circular imports of the quant methods, matching
    # SharedFullContext._detect_quant_type_from_created_weights.
    method_class = gpu_method.__class__.__name__
    if method_class == "DeepSeekMxfp4MoEMethod":
        return Mxfp4DynUpdatePlan(
            disabled_reason=(
                "resident DeepSeekMxfp4MoEMethod trtllm image does not match "
                "the Mxfp4MoEMethod contract (scales under "
                "w13_weight_scale_inv/w2_weight_scale_inv; row shuffle "
                "composed inside flashinfer's cached w3_w1 permute helper, "
                "unverified against the kt export swizzle); only the "
                "Mxfp4MoEMethod resident is supported"
            )
        )
    if method_class != "Mxfp4MoEMethod":
        return Mxfp4DynUpdatePlan(
            disabled_reason=(
                f"unrecognized MXFP4 resident method {method_class}; dynamic "
                "expert updates support only the Mxfp4MoEMethod trtllm-gen "
                "resident"
            )
        )
    if gpu_method.use_deep_gemm:
        return Mxfp4DynUpdatePlan(
            disabled_reason=(
                "resident Mxfp4MoEMethod is in deep_gemm mode; dynamic expert "
                "updates support only the flashinfer trtllm-gen (SM100) "
                "resident image"
            )
        )
    if gpu_method.use_marlin:
        return Mxfp4DynUpdatePlan(
            disabled_reason=(
                "resident Mxfp4MoEMethod is in marlin mode; dynamic expert "
                "updates support only the flashinfer trtllm-gen (SM100) "
                "resident image"
            )
        )
    if gpu_method._fi_kernel != "trtllm_sm100":
        return Mxfp4DynUpdatePlan(
            disabled_reason=(
                "resident Mxfp4MoEMethod is not in flashinfer trtllm-gen mode "
                f"(fi_kernel={gpu_method._fi_kernel!r}, "
                f"use_flashinfer={gpu_method.use_flashinfer}); dynamic expert "
                "updates support only the SM100 trtllm-gen resident image"
            )
        )

    param_names = _MXFP4_TRTLLM_RESIDENT_PARAM_NAMES
    source_mismatch = _mxfp4_shadow_source_mismatch(
        ctx=ctx, resident_layer=resident_layer, param_names=param_names
    )
    if source_mismatch is not None:
        return Mxfp4DynUpdatePlan(disabled_reason=source_mismatch)
    return Mxfp4DynUpdatePlan(param_names=param_names)


def _as_bf16_export_scales(scales: torch.Tensor) -> torch.Tensor:
    """Return the export's bf16 scale payload, undoing an fp32 staging cast.

    bf16 -> fp32 staging is value-exact, so the cast back is too; any value a
    genuine fp32-scale wheel produced still trips ``bf16_scales_to_e8m0``'s
    E8M0-exactness assertion downstream.
    """
    if scales.dtype == torch.bfloat16:
        return scales
    if scales.dtype == torch.float32:
        return scales.to(torch.bfloat16)
    raise TypeError(
        f"MXFP4 export scales must be bf16 (or fp32-staged bf16), got "
        f"{scales.dtype}"
    )


def _copy_mxfp4_experts_from_raw_source(
    *,
    raw_source: Mxfp4RawExpertSource,
    dst_layer: torch.nn.Module,
    selected_ids: List[int],
    param_names: Tuple[str, ...],
) -> None:
    """Re-swizzle raw kt-export experts straight into resident trtllm rows.

    Valid per expert because both the trtllm weight shuffle and the block
    scale interleave permute within one expert (kt_mxfp4_export.py
    ``TrtllmPermuteIndices`` docstring).  All resident writes go through
    ``swizzle_trtllm_expert``'s ``out_*`` path, which is ``copy_``-only into
    the given row views; the swizzle's gather/interleave temporaries are
    source-side scratch and never alias resident storage.
    """
    from sglang.srt.layers.moe.kt_mxfp4_export import (
        Mxfp4ExpertBytes,
        bf16_scales_to_e8m0,
        swizzle_trtllm_expert,
        trtllm_permute_indices,
    )

    w13_name, w13_scale_name, w2_name, w2_scale_name = param_names
    dst_w13 = getattr(dst_layer, w13_name)
    dst_w13_scale = getattr(dst_layer, w13_scale_name)
    dst_w2 = getattr(dst_layer, w2_name)
    dst_w2_scale = getattr(dst_layer, w2_scale_name)
    for name, tensor in (
        (w13_scale_name, dst_w13_scale),
        (w2_scale_name, dst_w2_scale),
    ):
        if tensor.dtype != torch.float8_e4m3fn:
            raise ValueError(
                f"resident `{name}` dtype {tensor.dtype} is not the "
                "interleaved float8_e4m3fn prepared-scale form"
            )

    # Gather selected rows before code recovery: unselected raw rows may
    # never have been written by the export and must not reach the E8M0
    # exactness assertion (mirrors prepare_trtllm_mxfp4).
    index = torch.tensor(
        selected_ids, dtype=torch.long, device=raw_source.w13.device
    )
    codes13 = bf16_scales_to_e8m0(
        _as_bf16_export_scales(raw_source.w13_scale.index_select(0, index))
    )
    codes2 = bf16_scales_to_e8m0(
        _as_bf16_export_scales(raw_source.w2_scale.index_select(0, index))
    )
    indices = trtllm_permute_indices(
        w13_sample=raw_source.w13[selected_ids[0]],
        w13_scale_sample=codes13[0],
        w2_sample=raw_source.w2[selected_ids[0]],
        w2_scale_sample=codes2[0],
        w13_gate_up_halves=True,
    )
    for dst_idx, logical_id in enumerate(selected_ids):
        swizzle_trtllm_expert(
            Mxfp4ExpertBytes(
                w13=raw_source.w13[logical_id],
                w13_scale_e8m0=codes13[dst_idx],
                w2=raw_source.w2[logical_id],
                w2_scale_e8m0=codes2[dst_idx],
            ),
            indices,
            out_w13=dst_w13[dst_idx],
            out_w13_scale=dst_w13_scale[dst_idx],
            out_w2=dst_w2[dst_idx],
            out_w2_scale=dst_w2_scale[dst_idx],
        )


def copy_experts_weights_mxfp4(
    src_layer: Optional[torch.nn.Module],
    dst_layer: torch.nn.Module,
    selected_experts: torch.Tensor,
    *,
    param_names: Tuple[str, ...] = _MXFP4_TRTLLM_RESIDENT_PARAM_NAMES,
    raw_source: Optional[Mxfp4RawExpertSource] = None,
) -> None:
    """Copy MXFP4 expert weights into a trtllm-gen resident layer.

    Args:
        src_layer: Source layer whose four ``param_names`` attributes hold
            the full-expert trtllm-gen prepared image (e.g. the post-fallback
            shadow ``ctx.gpu_layer``, or a layerwise slot's prepared params).
            May be None when ``raw_source`` is given.
        dst_layer: Destination resident layer (subset of GPU experts).
        selected_experts: Logical expert IDs to copy ([num_gpu_experts]);
            position i lands in resident row i (the mapping
            ``update_gpu_expert_mappings`` installs).
        param_names: The four resident attribute names, ordered
            (w13, w13_scale, w2, w2_scale).
        raw_source: Raw kt-export stacks to re-swizzle from when no prepared
            trtllm image is available (marlin-layout slots, tests).

    Direct mode copies the shuffled weight rows and interleaved scale rows
    verbatim — exact because the trtllm shuffle and scale interleave permute
    within one expert (mxfp4.py L740-803) and source/destination share the
    per-expert geometry.  Every write is an in-place ``copy_`` into existing
    resident rows (CUDA-graph safety: no parameter rebinding, no resident
    allocation).
    """
    if (src_layer is None) == (raw_source is None):
        raise ValueError(
            "copy_experts_weights_mxfp4 needs exactly one source: a prepared "
            "src_layer or a raw_source"
        )
    if selected_experts.numel() == 0:
        return

    if raw_source is not None:
        selected_ids = [int(expert_id) for expert_id in selected_experts.tolist()]
        _copy_mxfp4_experts_from_raw_source(
            raw_source=raw_source,
            dst_layer=dst_layer,
            selected_ids=selected_ids,
            param_names=param_names,
        )
        return

    for name in (param_names[1], param_names[3]):
        for lyr, role in ((src_layer, "source"), (dst_layer, "destination")):
            if getattr(lyr, name).dtype != torch.float8_e4m3fn:
                raise ValueError(
                    f"{role} `{name}` dtype {getattr(lyr, name).dtype} is not "
                    "the interleaved float8_e4m3fn prepared-scale form; "
                    "refusing a raw-vs-prepared MXFP4 row copy"
                )
    num_selected = selected_experts.numel()
    gather_index = None
    for name in param_names:
        src_weight = getattr(src_layer, name)  # [global_num_experts, ...]
        dst_weight = getattr(dst_layer, name)  # [num_gpu_experts, ...]
        if src_weight.shape[1:] != dst_weight.shape[1:]:
            raise ValueError(
                f"`{name}` per-expert shapes differ between source "
                f"{tuple(src_weight.shape[1:])} and destination "
                f"{tuple(dst_weight.shape[1:])}"
            )
        if gather_index is None:
            gather_index = selected_experts.to(
                device=src_weight.device, dtype=torch.long
            )
        # Position i lands in resident row i; index_select(out=) writes the
        # gathered rows straight into the existing resident storage (no
        # temporary, no parameter rebinding — CUDA-graph safe).
        torch.index_select(
            src_weight, 0, gather_index, out=dst_weight[:num_selected]
        )


def update_gpu_expert_mappings(
    selected_experts: torch.Tensor,
    num_experts: int,
    device: torch.device,
):
    """Update GPU expert mapping tables based on newly selected experts.

    Args:
        selected_experts: Tensor of logical expert IDs now on GPU (shape: [num_gpu_experts])
        num_experts: Total number of experts in layer
        device: Target CUDA device for mapping tensors

    Returns:
        Tuple of (gpu_experts_mask, logical_to_gpu_index, gpu_index_to_logical):
            - gpu_experts_mask: CPU bool tensor [num_experts], True = on GPU
            - logical_to_gpu_index: CUDA int32 tensor [num_experts], maps logical -> GPU index
            - gpu_index_to_logical: CPU int32 tensor [num_gpu_experts], reverse mapping
    """
    num_gpu_experts = len(selected_experts)

    # Create new mask (CPU tensor)
    gpu_experts_mask_cpu = torch.zeros(num_experts, dtype=torch.bool, device='cpu')
    gpu_experts_mask_cpu[selected_experts.cpu()] = True

    # Create logical_to_gpu_index (CUDA tensor) with one scatter.
    logical_to_gpu_index = torch.full(
        (num_experts,), -1, dtype=torch.int32, device=device
    )
    logical_to_gpu_index[selected_experts.to(device=device, dtype=torch.long)] = (
        torch.arange(num_gpu_experts, dtype=torch.int32, device=device)
    )

    # Create gpu_index_to_logical (CPU tensor for weight loading)
    gpu_index_to_logical_cpu = selected_experts.cpu().to(torch.int32)

    return gpu_experts_mask_cpu, logical_to_gpu_index, gpu_index_to_logical_cpu


def update_kt_wrapper_masks(
    wrapper: Optional["KTMoEWrapper"],
    gpu_experts_mask_cpu: torch.Tensor,
) -> None:
    """Update KT wrapper's internal GPU experts mask (rank 0 only).

    Args:
        wrapper: KTMoEWrapper instance (None if not rank 0)
        gpu_experts_mask_cpu: New GPU experts mask to apply

    The wrapper needs updated masks to correctly route tokens to CPU vs GPU experts.
    This is called on rank 0 only since only rank 0 has the wrapper instance.

    CRITICAL: wrapper.gpu_experts_mask is a pinned memory tensor whose pointer is shared
    with C++ code. We MUST use .copy_() to update in-place, not replace the reference.
    """
    if wrapper is None:
        return

    # Update wrapper's internal mask IN-PLACE
    # CRITICAL: The C++ code holds a pointer to this tensor's memory.
    # Replacing the reference would leave C++ pointing to old/freed memory.
    wrapper.gpu_experts_mask.copy_(gpu_experts_mask_cpu)


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

        # F2 (MXFP4 dynamic expert update) wheel precondition, checked at the
        # earliest point kt_config is available.  The update path recovers the
        # exact resident E8M0 codes from the export's bf16 scale buffers, a
        # contract only E8M0-resident kt-kernel wheels honor (fp32-scale
        # wheels export arbitrary bf16 values); fail at construction instead
        # of tripping the exactness assertion mid-serving.  Feature-check, not
        # a version check.
        if (
            kt_config.kt_enable_dynamic_expert_update
            and (kt_config.method or "").upper() == "MXFP4"
        ):
            from sglang.srt.layers.moe.kt_mxfp4_export import (
                kt_wheel_has_e8m0_resident_scales,
            )

            if not kt_wheel_has_e8m0_resident_scales():
                raise ValueError(
                    "--kt-enable-dynamic-expert-update with --kt-method MXFP4 "
                    "requires a kt-kernel wheel with E8M0-resident MXFP4 "
                    "scales (feat/mxfp4-kimi-k3 line): the dynamic update "
                    "path recovers exact E8M0 codes from the export's bf16 "
                    "scale buffers, which fp32-scale wheels cannot provide"
                )
        # Lazily resolved on the first qualifying fallback fire; None means
        # "not yet resolved", a plan with disabled_reason means the update is
        # TP-consistently off for this layer.
        self._mxfp4_dyn_update_plan: Optional[Mxfp4DynUpdatePlan] = None

        self.gpu_experts_mask = kt_config.gpu_experts_mask  # bool tensor [num_experts], on CPU
        self.num_gpu_experts = int(self.gpu_experts_mask.sum().item())
        self.override_num_local_experts = True
        self.gpu_method.num_gpu_experts = self.num_gpu_experts
        self.tp_rank = get_parallel().tp_rank
        # Debug/kill-switch env knobs, snapshotted once (read on the hot path).
        self._kt_debug_timing = envs.SGLANG_DEBUG_KT_HYBRID_TIMING.get()
        self._kt_debug_timing_deep = envs.SGLANG_DEBUG_KT_HYBRID_TIMING_DEEP.get()
        self._kt_no_cpu_stream = envs.SGLANG_DISABLE_KT_CPU_STREAM.get()
        self._kt_bypass_gpu_moe = envs.SGLANG_DEBUG_KT_BYPASS_GPU_MOE.get()
        self._kt_ablate_hostnodes = envs.SGLANG_KT_ABLATE_HOSTNODES.get()
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
        # poller only once they landed -- measured as the staging completing
        # after the GPU expert GEMM on 52% of layers, which exposes the whole
        # CPU latency because no GPU work is left to hide behind.
        self._fused_enabled = kt_config.transport == "hostnode"
        # CPU-branch elision: a CUDA conditional node skips the whole branch
        # when nothing in the batch routes to a CPU-resident expert. The flag
        # and the body stream are created in create_weights -- both addresses
        # are baked into the captured graph, so neither may be allocated
        # during capture or move afterwards.
        # Per-expert demand counters are maintained only where something reads
        # them: the swap policy, and the full-override falsification check
        # (which needs to see an insist survive a routing that claims none can).
        # Profiling put them at ~5.2% of decode GPU time, so "always on" is a
        # real price, not bookkeeping noise.
        self._counters_enabled = bool(
            kt_config.expert_swap_interval > 0 or kt_config.routing_full_override
        )
        self._cond_enabled = kt_config.conditional_cpu_branch
        self._cond_flag: Optional[torch.Tensor] = None
        self._cond_body_stream: Optional[torch.cuda.Stream] = None
        self._kt_ablate_zero: Optional[torch.Tensor] = None
        # Margin routing (SPEC-MARGIN-ROUTING P1). None = off, bit-exact.
        self._margin = kt_config.routing_margin
        self._full_override = kt_config.routing_full_override
        self._resident_hit_count: Optional[torch.Tensor] = None
        if self._full_override and self._margin is None:
            # Full override subsumes the margin: counters still record what
            # the router wanted, so the mode reports its own quality cost.
            self._margin = 0.0
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
        # Break-even against the CPU-expert path, NOT the chunk size.  The
        # split path's cost is dominated by a FIXED per-forward stream -- every
        # cold expert lands once however many tokens the chunk holds -- so it
        # wins above roughly (stream seconds x CPU tokens/s): measured 1.99 s
        # and ~1,400 tok/s give ~2,800 tokens.
        #
        # Gating on chunked_prefill_size instead meant only an exactly-full
        # chunk qualified, so the scheduler's remainder always fell back: a
        # 65,498-token prompt became 32768 (split, 2.0 s) + 32730 (CPU, 23 s),
        # 38 tokens short of the threshold and 6.4x slower overall.
        self._split_prefill_threshold = _SPLIT_PREFILL_MIN_TOKENS
        self._split_prefill_validate = envs.SGLANG_KT_VERIFY_SPLIT_PREFILL.get()
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

        self.gpu_prefill_token_threshold = kt_config.gpu_prefill_token_threshold or 0
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
        # They are NOT free. Profiling decode (runs/meta/phaseP.sh) put this
        # bookkeeping at ~5.2% of GPU time: three scatter_add_ and four
        # bitwise kernels per layer per step, 11040 and 7360 launches over 40
        # steps -- exactly 92 layers x 3 and 92 x 2. Baked into the captured
        # graph, so they run every step forever whether or not anything reads
        # them. Maintained only where something does.
        # Gated on the swap interval, NOT on margin. Demand is
        # `routed & ~gpu_experts_mask[topk_ids]` and hits are its complement --
        # both functions of the routed ids and the residency mask alone. The
        # margin only decides how demand splits into kept-on-CPU vs
        # substituted-to-GPU, and that split cancels in the sum the policy
        # reads (kt_expert_swap.py:283). So swapping does not need margin, and
        # exact routing + adaptive placement is now a legal configuration.
        # See SPEC-SWAP-DEMAND.md.
        if self._counters_enabled and (
            self._margin is not None or self.kt_config.expert_swap_interval > 0
        ):
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
            self._resident_hit_count = torch.zeros(
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
                **(
                    {"cold_only_cpu_experts": True}
                    if self.kt_config.cold_only_cpu_experts
                    else {}
                ),
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
            if self._cond_enabled:
                # Allocated here, before any capture: the predicate kernel
                # bakes this flag's address into the graph, so it must outlive
                # every replay and never be reallocated. The body stream is
                # shared by every layer -- bodies are captured one at a time
                # and replayed serially, the same reason one doorbell ring
                # word suffices.
                self._cond_flag = torch.zeros(
                    1, dtype=torch.int32, device=self.gpu_experts_mask_cuda.device
                )
                self._cond_body_stream = get_stream("kt_cond_body")

        # Swap driver registry: keep the layer with the method, since the
        # weight mover writes into the layer's resident parameter rows.
        if self.kt_config.expert_swap_interval > 0 and self._margin is not None:
            self._swap_layer = layer
            self._swap_policy = None  # built lazily, needs num_experts
            _KT_EP_METHODS.append(self)

        # Registration happens during model construction, not on the first
        # request, so layer N can identify and prepare N+1 immediately.
        _register_mxfp4_prefill_layer(self, layer)

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
            self._maybe_verify_kt_ram_source()

        # 4. KT_BUFFER_B_MEMFD: hand every rank a read-only mapping of this
        # layer's kt expert buffers (rank 0 exports memfds, peers map them),
        # so swap-window promotions read kt RAM instead of the checkpoint.
        # Deliberately OUTSIDE the rank-0 gate -- the share is a rendezvous
        # every rank participates in, in the same per-layer order.
        from sglang.srt.layers.moe.kt_arena_share import share_layer_arenas

        share_layer_arenas(method=self)

    def _maybe_verify_kt_ram_source(self):
        """SGLANG_KT_VERIFY_RAM_SOURCE=1: prove kt's buffers match the checkpoint.

        Runs here -- immediately after this layer's kt weights load, on the
        rank that owns the wrapper -- because the buffers this checks exist
        nowhere else and at no earlier time. Read-only, a few experts, once per
        process (the mapping is layer-independent, so one layer's evidence
        covers the rest).
        """
        if not envs.SGLANG_KT_VERIFY_RAM_SOURCE.get():
            return
        if _KT_SWAP_STATE.get("ram_source_verified"):
            return
        _KT_SWAP_STATE["ram_source_verified"] = True
        from sglang.srt.layers.moe.kt_ram_source import (
            build_kt_ram_source,
            verify_against_checkpoint,
        )

        source = build_kt_ram_source(
            self, tp_rank=self.tp_rank, tp_size=get_parallel().tp_size
        )
        if source is None:
            logger.error(
                "[kt-ram] verification requested but no source could be built "
                "(no expert_buffer_pointers on this wrapper?)"
            )
            return
        n = source.experts
        # SLOT ids: raw_shard is slot-indexed and verify translates only the
        # checkpoint side. Pre-mapping the ids here compared raw_shard(p2l[s])
        # against checkpoint p2l[s] -- wrong on both sides of a non-identity
        # map, and invisible under the identity maps it was written against.
        ids = sorted({0, n // 3, (2 * n) // 3, n - 1})
        verify_against_checkpoint(
            source,
            weight_path=self.kt_config.weight_path,
            layer_idx=self.kt_config.layer_idx,
            expert_ids=ids,
            physical_to_logical=self._kt_physical_to_logical,
            tp_rank=self.tp_rank,
            tp_size=get_parallel().tp_size,
        )
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
            validate=self._split_prefill_validate,
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

        # No layer filter: placement strategies (layer_concentrated) put
        # wrappers on arbitrary layer indices; the per-layer step rate-limit
        # at the emission site keeps volume bounded.  Never instrument under
        # stream capture — DEEP mode's device synchronize invalidates the
        # graph being captured.
        _kt_timing = (
            self._kt_debug_timing
            and self.tp_rank == 0
            and not torch.cuda.is_current_stream_capturing()
        )
        _kt_t_apply_start = time.perf_counter() if _kt_timing else None
        _kt_t_after_submit = None
        _kt_t_after_mask = None
        _kt_t_after_gpu = None
        _kt_t_after_sync = None
        _kt_t_after_merge = None
        _kt_t_cpu_wait_ms = 0.0

        # Check for full GPU fallback. The full-GPU path's _build_full_context →
        # _prepare_weight_{mxfp4,fp8,fp8_channel,bf16,int4} helpers read flat
        # `w13_weight` / `w13_weight_packed` attributes off `layer`. V4-Flash
        # MXFP4 (triton_kernels path) optionally preserves these when
        # `kt_gpu_prefill_token_threshold > 0` is set (see
        # `mxfp4_deepseek.process_weights_after_loading`); accept either the
        # flat attr or the v4 triton-kernels marker as a hint that the loader
        # can populate the layer. Layouts without either are still skipped to
        # avoid crashing the scheduler. Origin: sglang 本身 (V4-Flash
        # full-GPU prefill fallback compat).
        _full_gpu_fallback_supported = (
            hasattr(layer, "w13_weight")
            or hasattr(layer, "w13_weight_packed")
            or getattr(layer, "_v4_tk_path", False)
        )
        # Never open the layerwise gate under stream capture: hostfunc-heavy
        # layerwise work must not bake into a decode/verify graph.  (The
        # is_extend_in_batch contextvar is NOT a usable signal here — plain
        # TP serving never writes it, which silently killed layerwise for
        # every prefill.  Eager TARGET_VERIFY passes clearing the threshold
        # remain a theoretical hazard only: bs x draft_tokens tops out at
        # 232 under the mamba-capped tiers vs threshold 1024.)
        _full_gpu_gate = (
            self.gpu_prefill_token_threshold > 0
            and num_tokens >= self.gpu_prefill_token_threshold
            and _full_gpu_fallback_supported
            and not (
                torch.cuda.is_available()
                and torch.cuda.is_current_stream_capturing()
            )
        )
        _mxfp4_requested = _mxfp4_pipeline_requested(self)
        _mxfp4_signature = getattr(self, "_mxfp4_pipeline_signature", None)
        _mxfp4_manager = (
            _MXFP4_LAYERWISE_MANAGERS.get(_mxfp4_signature)
            if _mxfp4_signature is not None
            else None
        )
        _mxfp4_disabled_after_oom = (
            _mxfp4_signature in _MXFP4_LAYERWISE_DISABLED_REASONS
            if _mxfp4_signature is not None
            else False
        )

        if _mxfp4_manager is not None and not _full_gpu_gate:
            _mxfp4_manager.abort_round()

        # Allocate the persistent full-layer slots only when a request actually
        # enters the MXFP4 layerwise path.  Startup reserves only their
        # KV-cache budget, not the tensors themselves; an allocation OOM still
        # records a disabled reason and this same request falls through to the
        # existing hybrid CPU/GPU path below.
        if (
            _full_gpu_gate
            and _mxfp4_requested
            and not _mxfp4_disabled_after_oom
            and _mxfp4_manager is None
            and _mxfp4_pipeline_runtime_supported(self, layer)
        ):
            _mxfp4_manager = _get_or_initialize_mxfp4_layerwise_manager(
                self, layer
            )
            _mxfp4_signature = getattr(self, "_mxfp4_pipeline_signature", None)
            _mxfp4_disabled_after_oom = (
                _mxfp4_signature in _MXFP4_LAYERWISE_DISABLED_REASONS
                if _mxfp4_signature is not None
                else False
            )

        if _full_gpu_gate and not (
            _mxfp4_requested and _mxfp4_disabled_after_oom
        ):
            if _mxfp4_pipeline_runtime_supported(self, layer):
                if _mxfp4_manager is None:
                    raise RuntimeError(
                        "MXFP4 layerwise prefill is supported but was not "
                        "initialized during lazy slot allocation"
                    )
                return _mxfp4_manager.apply(self, layer, dispatch_output)

            if _mxfp4_manager is not None:
                raise RuntimeError(
                    "MXFP4 layerwise prefill was initialized, but the "
                    "runtime layer has no compatible MXFP4 weights"
                )

            # Non-MXFP4 and unsupported MXFP4 backends retain the existing
            # serialized full-GPU fallback.
            ctx = self._build_full_context(layer)

            # Re-run quant post-processing on the full-expert gpu_layer
            # if supported (e.g. Marlin repack for Fp8MarlinMoEMethod).
            # The ctx.load() call writes raw fp8 weights; downstream apply()
            # expects repacked format.
            # MXFP4's serial load helper already performs its required
            # postprocess.  Running it again double-swizzles the same layer.
            _needs_repack = (
                hasattr(ctx.gpu_method, "process_weights_after_loading")
                and not getattr(ctx, "_is_mxfp4_quant", False)
            )
            if _needs_repack:
                ctx.gpu_method.process_weights_after_loading(ctx.gpu_layer)

            t_compute = time.perf_counter()
            result = ctx.gpu_method.apply(ctx.gpu_layer, dispatch_output)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            compute_time = (time.perf_counter() - t_compute) * 1000.0

            # Dynamic expert update: analyze batch and update GPU experts.
            # MUST run BEFORE _restore_raw_attrs() because on Ampere FP8 the
            # Marlin repack changes weight shapes/dtypes, and
            # _restore_raw_attrs() creates empty tensors (torch.empty) to
            # restore the raw fp8 format, destroying the weight data.
            # MXFP4 (F2): after the fire the shadow gpu_layer holds the
            # full-expert trtllm-gen prepared image (its PWAL re-ran in
            # `_prepare_weight_mxfp4` phase 3), so selected rows copy straight
            # into a trtllm-gen resident via `copy_experts_weights_mxfp4`.
            # Unsupported MXFP4 residents (tk / marlin / DeepSeek trtllm)
            # resolve to a TP-consistent, once-logged disable — every rank
            # takes the same skip, keeping the broadcast flow aligned.
            _dyn_update_enabled = self.kt_config.kt_enable_dynamic_expert_update
            if _dyn_update_enabled and ctx._is_mxfp4_quant:
                _dyn_update_enabled = (
                    self._mxfp4_dyn_update_plan_for(
                        ctx=ctx, layer=layer
                    ).disabled_reason
                    is None
                )
            if _dyn_update_enabled:
                t_update = time.perf_counter()
                self._update_gpu_experts_from_batch(
                    layer=layer,
                    ctx=ctx,
                    dispatch_output=dispatch_output,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                update_time = (time.perf_counter() - t_update) * 1000.0

                if self.tp_rank == 0:
                    logger.info(
                        "KT layerwise prefill: layer %d compute = %.2f ms, expert update = %.2f ms",
                        self.kt_config.layer_idx,
                        compute_time,
                        update_time,
                    )
            else:
                if self.tp_rank == 0:
                    logger.info(
                        "KT layerwise prefill: layer %d compute = %.2f ms",
                        self.kt_config.layer_idx,
                        compute_time,
                    )

            # Restore raw format AFTER dynamic update so the context is
            # clean for the next layer's load() call.
            if _needs_repack:
                ctx._restore_raw_attrs()

            return result

        # Margin routing (SPEC-MARGIN-ROUTING P1): rewrite below-margin
        # CPU-resident picks to resident alternatives BEFORE the Step-1 CPU
        # submit — the CPU side receives raw topk_ids and applies its own
        # pinned membership mask in C++, so a GPU-only rewrite at the Step-2
        # mask would desync the two halves (CPU computing overridden experts,
        # or dropped/double-counted contributions).  Counters accumulate on
        # the ORIGINAL ids so they measure true router preference, and the
        # scatter_add_ runs over all slots with 0/1 addends (static shapes,
        # in-place persistent buffers) so it is capture-safe and keeps
        # counting across decode graph replays.  The full-GPU prefill paths
        # above bypass this on purpose: they compute every routed expert on
        # GPU, so overriding there would cost accuracy for nothing.
        #
        # Margin-unset serving still feeds the swap policy. Demand and hits are
        # functions of the routed ids and the residency mask alone, so nothing
        # about them requires a margin (SPEC-SWAP-DEMAND). This branch is what
        # makes exact routing WITH adaptive placement a legal configuration --
        # bit-exact output, resident set still following the workload.
        # Margin routing runs when the SERVER set a default, or when any
        # request in this batch asked for one. Gating on the server default
        # alone silently dropped SamplingParams.kt_routing_margin on a server
        # started without --kt-routing-margin: the value reached ForwardBatch
        # and the graph buffer, and then nothing read it.
        _per_req_margin = self._any_request_margin()

        if self._margin is None and not _per_req_margin and self._counters_enabled:
            self._update_demand_counters(dispatch_output.topk_output.topk_ids)
            # Swap windows moved to the scheduler (SPEC-SWAP-DEMAND Phase 3):
            # mid-prompt re-cuts stall the throughput-critical path, and this
            # call site never ran under --kt-expert-split-prefill anyway --
            # split prefill returns before it and decode replays a graph.

        if self._margin is not None or _per_req_margin:
            from sglang.srt.layers.moe.topk import StandardTopKOutput

            _format_ok = (
                isinstance(topk_output, StandardTopKOutput)
                and topk_output.router_logits is not None
                and topk_output.router_logits.shape[-1] == self.global_num_experts
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
                new_topk_ids, _insist_slots, _override_slots = (
                    margin_override_topk_ids(
                        topk_output.topk_ids,
                        topk_output.router_logits,
                        self.gpu_experts_mask_cuda,
                        self._resolve_margin(topk_output.topk_ids),
                        self._full_override,
                    )
                )
                if self._counters_enabled:
                    self._update_margin_counters(
                        topk_output.topk_ids, _insist_slots, _override_slots
                    )
                # margin == 0.0 is count-only (documented flag contract):
                # counters record what WOULD override, routing stays exact.
                # With per-request margins the decision is per token, made
                # inside the kernel by the `margin > 0` term, so the rewrite is
                # applied and tokens at 0.0 come back unchanged. The
                # `is not None` guard matters: a server with no default but a
                # request that asked reaches here with self._margin None, and
                # `None > 0.0` raises.
                if (
                    self._full_override
                    or _per_req_margin
                    or (self._margin is not None and self._margin > 0.0)
                ):
                    topk_output = topk_output._replace(topk_ids=new_topk_ids)
                    dispatch_output = dispatch_output._replace(
                        topk_output=topk_output
                    )
                self._maybe_log_margin_stats()
                self._maybe_verify_expert_mover(layer)
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
        # _db_elide travels with it: both of a layer's conditional regions must
        # agree, and step 4 must know whether step 1 opened one at all.
        _db_slot = None
        _db_elide = False
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

            # SGLANG_DISABLE_KT_CPU_STREAM=1 collapses cpu_stream onto main stream.
            _no_cpu_stream = self._kt_no_cpu_stream
            if not _no_cpu_stream:
                # Fork to cpu_stream (waits for the pack/staging copy)
                self._cpu_stream.wait_stream(torch.cuda.current_stream(x.device))
            from contextlib import nullcontext as _ctx_null
            _stream_ctx = _ctx_null() if _no_cpu_stream else torch.cuda.stream(self._cpu_stream)
            with _stream_ctx:
                # Elide the whole branch when no slot routes off-GPU. Only
                # under capture: a conditional node has to be spliced into a
                # graph, and eager forwards have none -- they simply run the
                # branch, which is what they already did.
                _db_elide = (
                    _db_slot is not None
                    and self._cond_enabled
                    and torch.cuda.is_current_stream_capturing()
                )
                if _db_elide:
                    self._kt_cond_predicate(dispatch_output)
                if _db_slot is not None:
                    with self._kt_cond_region(_db_elide):
                        # Inside the IF body the recording stream is the body
                        # stream, so the memops must be read off the CURRENT
                        # stream rather than captured before the region.
                        _db_stream = torch.cuda.current_stream(x.device).cuda_stream
                        # Arm BEFORE the ring: retract the previous replay's
                        # completion so this replay's wait cannot be satisfied
                        # by a stale value. A captured node writes a constant,
                        # so without the arm every replay after the first would
                        # sail through the wait reading the first replay's
                        # output.
                        kt_doorbell_arm(_db_slot, _db_stream)
                        # Then FLUSH, then ring. Only the single D2H sits
                        # between the fork and the ring -- the pack already
                        # ran on the main stream. The poller's whole decision
                        # reads these ids, so a ring visible before the flush
                        # would have it judge the PREVIOUS step's batch.
                        self._kt_flush_inputs(staging_buffer)
                        kt_doorbell_ring(_db_slot, _db_stream)
                elif _fused and not self._kt_ablate_hostnodes:
                    # One D2H, then the dispatch. The pack already ran on the
                    # main stream, so this is all that stands between the fork
                    # and the poller learning there is work.
                    self._kt_flush_inputs(x)
                    self.wrapper.submit_forward_packed(
                        x, torch.cuda.current_stream(x.device).cuda_stream
                    )
                elif not self._kt_ablate_hostnodes:
                    self._submit_with_staged_input(
                        layer, dispatch_output, staging_buffer
                    )
        if _kt_timing:
            if self._kt_debug_timing_deep:
                torch.cuda.synchronize(x.device)
            _kt_t_after_submit = time.perf_counter()

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
        if _kt_timing:
            if self._kt_debug_timing_deep:
                torch.cuda.synchronize(x.device)
            _kt_t_after_mask = time.perf_counter()

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
        # SGLANG_DEBUG_KT_BYPASS_GPU_MOE=1 also short-circuits to zeros, because
        # the kt mask generator returns an all-True (num_gpu_experts ==
        # num_total_experts) per-layer mask in some configurations (e.g. V4
        # Flash + --kt-num-gpu-experts=0), which defeats the
        # num_gpu_experts==0 short-circuit. The env var lets the operator
        # force the bypass without untangling the mask generator.
        if self.num_gpu_experts == 0 or self._kt_bypass_gpu_moe:
            gpu_combine_input = None
            output = torch.zeros_like(x)
        else:
            with _scoped_layer_num_local_experts(layer, self.num_gpu_experts):
                gpu_combine_input = self.gpu_method.apply(
                    layer, masked_dispatch_output
                )
            output = gpu_combine_input.hidden_states
        if _kt_timing:
            if self._kt_debug_timing_deep:
                torch.cuda.synchronize(x.device)
            _kt_t_after_gpu = time.perf_counter()

        # Step 4: Sync CPU results on cpu_stream, then synchronize streams
        if self.tp_rank == 0 and self._cpu_stream is not None and not self._skip_cpu_path:
            _no_cpu_stream = self._kt_no_cpu_stream
            from contextlib import nullcontext as _ctx_null
            if _db_elide and not _no_cpu_stream:
                # The merge moves INSIDE the IF body (a skipped body must
                # leave `output` untouched), so the CPU stream now has to see
                # the finished GPU result. It did not before, because the
                # merge ran on the main stream. This does not undo the
                # overlap: the ring went out in step 1, so the poller has been
                # working throughout the GPU compute -- only the WAIT is
                # ordered after it, which is exactly where it belongs.
                self._cpu_stream.wait_stream(torch.cuda.current_stream(x.device))
            _stream_ctx = _ctx_null() if _no_cpu_stream else torch.cuda.stream(self._cpu_stream)
            with _stream_ctx:
                # Use staging_buffer for sync to get correct buffer reference
                _kt_t_sync_pre = time.perf_counter() if _kt_t_apply_start is not None else None
                if _db_elide:
                    with self._kt_cond_region(True):
                        kt_doorbell_wait(
                            _db_slot,
                            torch.cuda.current_stream(x.device).cuda_stream,
                        )
                        # Merged in place, inside the body. `output + cpu` is
                        # the same arithmetic, but it would produce a NEW
                        # tensor the skipped path never writes, leaving the
                        # caller holding whichever one the branch happened to
                        # take.
                        output.add_(self._kt_doorbell_output(staging_buffer))
                    cpu_output = None
                elif _db_slot is not None:
                    kt_doorbell_wait(
                        _db_slot, torch.cuda.current_stream(x.device).cuda_stream
                    )
                    cpu_output = self._kt_doorbell_output(staging_buffer)
                elif _fused and not self._kt_ablate_hostnodes:
                    # x, not staging_buffer: the packed path never fills the
                    # shared buffer. Both name the same [bs, hidden] shape and
                    # sync_forward keys its rings by shape alone, so this is
                    # the same buffer either way -- passing x keeps the packed
                    # path's data flow readable end to end.
                    cpu_output = self._sync_cpu_forward(x)
                elif self._kt_ablate_hostnodes:
                    # Same shape and same merge-add, without the sync host
                    # node: isolates dispatch cost from the copies/merge.
                    # Grow-on-demand: the staging slice is batch-sized, so a
                    # buffer cached from the first (small) batch cannot serve a
                    # later larger one. Reallocating only on growth keeps the
                    # steady-state cost at zero so the measurement stays clean.
                    if (
                        self._kt_ablate_zero is None
                        or self._kt_ablate_zero.shape[0] < staging_buffer.shape[0]
                    ):
                        self._kt_ablate_zero = torch.zeros_like(staging_buffer)
                    cpu_output = self._kt_ablate_zero[: staging_buffer.shape[0]]
                else:
                    cpu_output = self._sync_with_staged_input(staging_buffer)
                if _kt_t_sync_pre is not None:
                    _kt_t_cpu_wait_ms = (time.perf_counter() - _kt_t_sync_pre) * 1000.0
                if not _no_cpu_stream:
                    self._sync_done_event.record(self._cpu_stream)
            if _kt_timing:
                _kt_t_after_sync = time.perf_counter()

            # Main stream waits for cpu_stream to complete before merging results
            if not _no_cpu_stream:
                torch.cuda.current_stream(x.device).wait_event(self._sync_done_event)
            # cpu_output is None only when the merge already happened inside
            # the conditional body, where it had to be in-place.
            if cpu_output is not None:
                output = output + cpu_output
        if _kt_timing:
            _kt_t_after_merge = time.perf_counter()
            # Optional: synchronize GPU at end of apply() to capture true GPU
            # work latency (otherwise gpu_apply Python time only captures
            # kernel-launch CPU overhead, not actual GPU compute). DEEP mode
            # serialises streams so per-stage numbers reflect GPU work, not
            # async launch return.
            if self._kt_debug_timing_deep:
                torch.cuda.synchronize(x.device)
                _kt_t_after_merge = time.perf_counter()

        if _kt_t_apply_start is not None:
            _kt_total_ms = (_kt_t_after_merge - _kt_t_apply_start) * 1000.0
            _stage_submit_ms = (_kt_t_after_submit - _kt_t_apply_start) * 1000.0
            _stage_mask_ms = (_kt_t_after_mask - _kt_t_after_submit) * 1000.0
            _stage_gpu_ms = (_kt_t_after_gpu - _kt_t_after_mask) * 1000.0
            _stage_sync_ms = (
                (_kt_t_after_sync - _kt_t_after_gpu) * 1000.0
                if _kt_t_after_sync is not None else 0.0
            )
            _stage_merge_ms = (
                (_kt_t_after_merge - _kt_t_after_sync) * 1000.0
                if _kt_t_after_sync is not None
                else (_kt_t_after_merge - _kt_t_after_gpu) * 1000.0
            )
            _cls = type(self)
            if not hasattr(_cls, '_kt_layer_step'):
                _cls._kt_layer_step = {}
            _li = getattr(self.kt_config, 'layer_idx', -1)
            _cls._kt_layer_step[_li] = _cls._kt_layer_step.get(_li, 0) + 1
            _step = _cls._kt_layer_step[_li]
            if _step <= 16 or _step % 16 == 0:
                # INFO on purpose: the env flag is the opt-in; requiring
                # --log-level debug on top buried the numbers under the
                # whole server's debug firehose.
                logger.info(
                    "[kt-time] layer=%s step=%d total=%.2fms submit=%.2f "
                    "mask=%.2f gpu=%.2f sync=%.2f merge=%.2f "
                    "cpu_wait=%.2fms num_tokens=%d",
                    _li, _step, _kt_total_ms, _stage_submit_ms,
                    _stage_mask_ms, _stage_gpu_ms, _stage_sync_ms,
                    _stage_merge_ms, _kt_t_cpu_wait_ms, num_tokens,
                )
        return StandardCombineInput(hidden_states=output)

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

        # get_forward_context() asserts rather than returning None, and this
        # code also runs from paths that publish no context (unit tests, the
        # standalone probes in runs/meta), so the guard is required.
        if not has_forward_context():
            return self._margin
        per_token = get_forward_context().kt_routing_margin
        if per_token is None or per_token.shape[0] != topk_ids.shape[0]:
            return self._margin
        # What a token gets when its request named no margin. The server
        # default if there is one; otherwise 0.0, which is not a fallback but
        # the exact meaning of an unset server margin -- margin 0.0 is
        # count-only by the flag's contract: record what WOULD override,
        # route exactly. So a request opting in to margin routing on a server
        # that did not enable it leaves every other request bit-exact.
        default = 0.0 if self._margin is None else float(self._margin)
        # Sentinel (negative) means "this request did not ask"; SamplingParams
        # rejects negative margins, so the value cannot be a real request's.
        return torch.where(
            per_token < 0.0, torch.full_like(per_token, default), per_token
        )

    def _update_demand_counters(self, topk_ids) -> None:
        """Fold one forward into the demand counters, with no margin involved.

        For the paths that never produce insist/override slots: split prefill
        (which computes every expert on GPU and returns before the margin
        block) and margin-unset serving. Demand is defined directly:

            demand = routed & ~gpu_experts_mask[topk_ids]
            hits   = routed &  gpu_experts_mask[topk_ids]

        which is the same quantity ``_update_margin_counters`` produces as
        insist + override -- margin only partitions it. Everything lands in the
        insist counter because nothing was substituted; ``snapshot_counters``
        sums the pair, so the policy sees an identical figure either way.

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
        self._resident_hit_count.scatter_add_(
            0, safe_ids, (routed & resident).to(torch.int32)
        )

    def _update_margin_counters(self, topk_ids, insist_slots, override_slots) -> None:
        """Fold one forward into the per-expert demand counters.

        The swap driver cannot work without these -- promotion reads demand for
        non-resident experts, demotion reads traffic served by resident ones --
        so the answer to their cost is a cheaper measurement, not no
        measurement.

        The torch form below ran ~11 kernels per layer per step (clamp, two
        dtype casts, three more casts, four bitwise ops, three scatter_add_),
        about 920 launches per decode step across 92 layers. Profiling put that
        at ~10% of the step. The fused kernel does the same arithmetic in one
        launch; the counters are integers accumulated by atomicAdd, so the
        values are bit-identical, not merely equivalent.
        """
        from sglang.kernels.ops.kimi_k3 import kt_margin_counters as ktmc

        if ktmc.covered(topk_ids, insist_slots, override_slots):
            ktmc.kt_margin_counters(
                self._margin_insist_count,
                self._margin_override_count,
                self._resident_hit_count,
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
        self._margin_insist_count.scatter_add_(
            0, _orig_safe_ids, insist_slots.reshape(-1).to(torch.int32)
        )
        self._margin_override_count.scatter_add_(
            0, _orig_safe_ids, override_slots.reshape(-1).to(torch.int32)
        )
        _resident_slots = (topk_ids >= 0) & ~(insist_slots | override_slots)
        self._resident_hit_count.scatter_add_(
            0, _orig_safe_ids, _resident_slots.reshape(-1).to(torch.int32)
        )

    def _kt_cond_predicate(self, dispatch_output) -> None:
        """Set this layer's branch flag from the routed ids, on device.

        Must read the FINAL ids: margin routing rewrites topk_ids before this
        point, and a predicate over the pre-override ids would elide a branch
        the rewritten routing still needs (or keep one it does not).

        Device memory, not a Python bool: a node captured in a graph writes
        whatever was recorded, so a host-side predicate would freeze the
        branch at capture time and every replay would take the same path.
        """
        from sglang.kernels.ops.kimi_k3 import kt_cpu_branch

        _, topk_ids, _ = dispatch_output.topk_output
        kt_cpu_branch.kt_cpu_branch_flag(
            self._cond_flag, topk_ids, self.gpu_experts_mask_cuda
        )

    def _kt_cond_region(self, elide: bool):
        """The CPU branch, gated on this layer's flag -- or run unconditionally.

        Both of a layer's regions read the same flag, and nothing rewrites it
        between them, so they cannot disagree about whether this step has CPU
        work. If the branch is skipped nothing rings, so the poller is never
        invoked and the skipped wait has nothing to wait for.
        """
        if not elide:
            from contextlib import nullcontext

            return nullcontext()
        from sglang.kernels.ops.kimi_k3 import kt_cpu_branch

        return kt_cpu_branch.kt_conditional(self._cond_flag, self._cond_body_stream)

    def _kt_fused_staging_ok(self, hidden_states) -> bool:
        """May this forward use packed staging? Both transports share the rule.

        Only for batch sizes kt caches: `get_packed` keys its buffers by size
        and caches only sizes in `capture_bs`, so any other size would allocate
        a fresh pinned block per layer per step. Those are prefill shapes;
        decode is what the packing is for, and they fall back to the three
        separate copies.

        Deferral is excluded at config time, not here -- it needs a second task
        over a second ids ring that one packed block cannot carry.
        """
        if not self._fused_enabled or self.wrapper is None:
            return False
        return hidden_states.shape[0] in self.wrapper.get_capture_batch_sizes()

    def _kt_pack_inputs(self, dispatch_output, hidden_states) -> None:
        """Device-side pack of THIS step's activations, on the MAIN stream.

        Takes the live hidden states, not the shared staging buffer: on the
        packed path nothing fills that buffer, so packing from it feeds the CPU
        a previous layer's activations. Ordered before the expert GEMM on the
        same stream, so x cannot be modified underneath it.
        """
        topk_weights, topk_ids, _ = dispatch_output.topk_output
        self.wrapper.pack_forward_inputs(hidden_states, topk_ids, topk_weights)

    def _kt_flush_inputs(self, hidden_states) -> None:
        """The single D2H; the only thing between the fork and the dispatch."""
        self.wrapper.flush_forward_inputs(hidden_states)

    def _kt_doorbell_stage(self, layer, dispatch_output, staging_buffer) -> None:
        """Copy this step's ids/weights into the kt ring the poller reads.

        The host-node path did this inside submit_forward; with the doorbell
        the copies must still happen (the poller reads the same rings) but
        without the enqueue, so this mirrors submit_forward's staging half
        and stops there.
        """
        topk_weights, topk_ids, _ = dispatch_output.topk_output
        self.wrapper.stage_forward_inputs(staging_buffer, topk_ids, topk_weights)

    def _kt_doorbell_output(self, staging_buffer) -> torch.Tensor:
        """Result tensor for the merge; the wait node already ordered it."""
        return self.wrapper.doorbell_output(staging_buffer)

    def _maybe_verify_expert_mover(self, layer) -> None:
        """SGLANG_KT_VERIFY_EXPERT_MOVER=1: prove the swap mover, once.

        Rebuilds an expert that is ALREADY resident straight from the
        checkpoint and compares byte-for-byte with the row the production
        loader filled. This is the gate the weight mover has to pass before it
        is allowed to rewrite anything: a wrong TP slice or gate/up assembly
        produces a correctly-shaped tensor full of the wrong numbers, which
        degrades output without ever raising.

        Read-only and one-shot; never runs under capture.
        """
        if getattr(type(self), "_kt_mover_verified", False):
            return
        if self.tp_rank != 0 or torch.cuda.is_current_stream_capturing():
            return
        if not envs.SGLANG_KT_VERIFY_EXPERT_MOVER.get():
            return
        type(self)._kt_mover_verified = True
        try:
            from sglang.srt.layers.moe.kt_expert_mover import CheckpointExpertMover

            resident_rows = torch.nonzero(self.gpu_experts_mask).flatten()
            if resident_rows.numel() == 0:
                return
            logical_id = int(resident_rows[0].item())
            row = int(self.logical_to_gpu_index[logical_id].item())
            layer_idx = self.kt_config.layer_idx
            mover = CheckpointExpertMover(
                self.kt_config.weight_path,
                expert_prefix_for_layer=lambda _l: (
                    f"language_model.model.layers.{layer_idx}"
                    f".block_sparse_moe.experts"
                ),
                tp_rank=get_parallel().tp_rank,
                tp_size=get_parallel().tp_size,
                param_names=_MXFP4_TRTLLM_RESIDENT_PARAM_NAMES,
            )
            ok = mover.verify_row(layer, row, logical_id)
            logger.info(
                "[kt-swap-verify] layer=%d expert=%d row=%d -> %s",
                layer_idx,
                logical_id,
                row,
                "PASS" if ok else "FAIL",
            )
            mover.reader.close()
        except Exception:
            logger.exception("[kt-swap-verify] mover verification errored")

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
        if self._margin_insist_count is None:
            # Counters are off (no swap driver, no full override), so there is
            # nothing to report. The doorbell line below still matters and is
            # emitted before the return.
            if _KT_DOORBELL["inited"]:
                _cls._kt_db_log_step = getattr(_cls, "_kt_db_log_step", 0) + 1
                if _cls._kt_db_log_step % 256 == 1:
                    logger.info("[kt-doorbell] %s", kt_doorbell_stats())
            return
        _li = self.kt_config.layer_idx
        _cls._kt_margin_step[_li] = _cls._kt_margin_step.get(_li, 0) + 1
        _step = _cls._kt_margin_step[_li]
        if _KT_DOORBELL["inited"]:
            # Emitted BEFORE the margin rate limit, on its own counter. Inside
            # it, the only line ever printed is eager step 1 -- which happens
            # during warmup, before capture has bound a single slot, so it
            # reports zeros forever and looks exactly like a transport that
            # never ran. The counters are cumulative and process-wide, and
            # this runs on EAGER steps, so a line printed while prefilling one
            # request already carries the previous request's DECODE traffic.
            # That is what makes `served` readable with swapping off -- the
            # only configuration that can be byte-compared.
            _cls._kt_db_log_step = getattr(_cls, "_kt_db_log_step", 0) + 1
            if _cls._kt_db_log_step % 256 == 1:
                logger.info("[kt-doorbell] %s", kt_doorbell_stats())
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
            "[kt-margin] layer=%s eager_step=%d margin=%.4g "
            "insists=%d overrides=%d top_insisted=%s",
            _li,
            _step,
            self._margin,
            total_insist,
            total_override,
            top,
        )

    def _mxfp4_dyn_update_plan_for(
        self, *, ctx: "SharedFullContext", layer: torch.nn.Module
    ) -> Mxfp4DynUpdatePlan:
        """Resolve (once) whether MXFP4 dynamic expert updates can run here.

        The first qualifying fallback fire classifies the resident layout and
        commits the local verdict through a TP consensus so every rank takes
        the same update-or-skip path; a disable is logged once (rank 0) and
        cached, so later fires cost one attribute read and no collective.
        """
        if self._mxfp4_dyn_update_plan is not None:
            return self._mxfp4_dyn_update_plan
        plan = resolve_mxfp4_dyn_update_plan(
            gpu_method=self.gpu_method, resident_layer=layer, ctx=ctx
        )
        if not _all_tp_ranks_succeeded(plan.disabled_reason is None):
            if plan.disabled_reason is None:
                plan = Mxfp4DynUpdatePlan(
                    disabled_reason=(
                        "MXFP4 dynamic expert update is unsupported on at "
                        "least one TP rank"
                    )
                )
        if plan.disabled_reason is not None and self.tp_rank == 0:
            logger.warning(
                "KT MXFP4 dynamic expert update disabled for layer %d; GPU "
                "expert placement stays static: %s",
                self.kt_config.layer_idx,
                plan.disabled_reason,
            )
        self._mxfp4_dyn_update_plan = plan
        return plan

    def _maybe_promote_experts_from_slot(
        self, *, layer: torch.nn.Module, slot, dispatch_output
    ) -> bool:
        """Manager-path dynamic expert update (F2).

        On the layerwise-prefill manager path the serial ctx fallback never
        runs, so promotion sources from the fired slot instead: its prepared
        parameters ARE the full-expert trtllm-gen image for this layer, and
        promotion reduces to per-expert row copies into the resident layer.
        Runs after the caller's compute-launch TP consensus; the caller
        re-records the slot's consumed fence afterwards so the next
        postprocess cannot overwrite prepared storage mid-copy."""
        if not self.kt_config.kt_enable_dynamic_expert_update:
            return False
        if slot.prepared_params is None:
            # Marlin-prepared slots (DSV4) have no direct-copy source here;
            # their (disabled) plan resolution belongs to the serial ctx path.
            return False
        from types import SimpleNamespace

        source_ctx = SimpleNamespace(
            _is_mxfp4_quant=True,
            mxfp4_prepared_layout=_MXFP4_LAYOUT_TRTLLM,
            gpu_layer=SimpleNamespace(**slot.prepared_params),
        )
        plan = self._mxfp4_dyn_update_plan_for(ctx=source_ctx, layer=layer)
        if plan.disabled_reason is not None:
            return False
        self._update_gpu_experts_from_batch(
            layer=layer, ctx=source_ctx, dispatch_output=dispatch_output
        )
        return True

    def _update_gpu_experts_from_batch(
        self,
        layer: torch.nn.Module,
        ctx: "SharedFullContext",
        dispatch_output: "StandardDispatchOutput",
    ) -> None:
        """Update original layer's GPU experts based on current batch statistics.

        This method:
        1. Analyzes topk_ids to find most frequently activated experts
        2. Copies selected expert weights from ctx.gpu_layer to layer
        3. Updates all mapping tables (gpu_experts_mask, logical_to_gpu_index, etc.)
        4. Broadcasts changes across TP ranks for consistency

        Args:
            layer: Original MoE layer with subset of GPU experts
            ctx: SharedFullContext containing temporary full GPU layer
            dispatch_output: Current batch dispatch output with routing information
        """
        # Step 1: Select top experts (rank 0 computes, broadcasts to all ranks)
        topk_ids = dispatch_output.topk_output.topk_ids
        device = topk_ids.device

        if self.tp_rank == 0:
            selected_experts = select_top_experts_from_batch(
                topk_ids=topk_ids,
                num_experts=self.global_num_experts,
                num_gpu_experts=self.num_gpu_experts,
            )
        else:
            # Create placeholder on other ranks
            selected_experts = torch.zeros(
                self.num_gpu_experts, dtype=torch.int64, device=device
            )

        # Broadcast selected experts to all ranks for consistent weight updates
        if dist.is_initialized():
            dist.broadcast(selected_experts, src=0, group=get_tp_group().device_group)

        # Step 2: Copy selected expert weights from ctx.gpu_layer to layer.
        # Both are already in inference format: apply() already called
        # process_weights_after_loading() which handles Marlin repack.
        if ctx._is_mxfp4_quant:
            plan = self._mxfp4_dyn_update_plan_for(ctx=ctx, layer=layer)
            if plan.disabled_reason is not None:
                raise ValueError(
                    "MXFP4 dynamic expert update invoked while disabled for "
                    f"layer {self.kt_config.layer_idx}: {plan.disabled_reason}"
                )
            copy_experts_weights_mxfp4(
                src_layer=ctx.gpu_layer,
                dst_layer=layer,
                selected_experts=selected_experts,
                param_names=plan.param_names,
            )
        elif ctx._is_fp8_quant:
            # Ampere Marlin vs native FP8 block quant: Marlin repack renames
            # w13_weight_scale_inv → w13_weight_scale and changes w13_weight
            # dtype fp8→int32.  Use gpu_method class name (invariant) to pick
            # the right attribute-name list.
            if ctx.gpu_method.__class__.__name__.endswith("MarlinMoEMethod"):
                copy_experts_weights_fp8_channel(
                    src_layer=ctx.gpu_layer, dst_layer=layer,
                    selected_experts=selected_experts,
                )
            else:
                copy_experts_weights_fp8(
                    src_layer=ctx.gpu_layer, dst_layer=layer,
                    selected_experts=selected_experts,
                )
        elif ctx._is_fp8_channel_quant:
            copy_experts_weights_fp8_channel(
                src_layer=ctx.gpu_layer,
                dst_layer=layer,
                selected_experts=selected_experts,
            )
        elif ctx._is_bf16_quant:
            copy_experts_weights_bf16(
                src_layer=ctx.gpu_layer,
                dst_layer=layer,
                selected_experts=selected_experts,
            )
        else:
            copy_experts_weights_int4(
                src_layer=ctx.gpu_layer,
                dst_layer=layer,
                selected_experts=selected_experts,
            )

        # Step 3: Update mapping tables
        gpu_experts_mask_cpu, logical_to_gpu_index_cuda, gpu_index_to_logical_cpu = (
            update_gpu_expert_mappings(
                selected_experts=selected_experts,
                num_experts=self.global_num_experts,
                device=device,
            )
        )

        # Update instance variables (both CPU and CUDA versions)
        # CRITICAL: Use .copy_() for CUDA tensors to maintain same buffer for CUDA graph compatibility
        # CUDA graph captures tensor memory addresses during decode phase, so we must update
        # in-place rather than replacing the tensor reference
        self.gpu_experts_mask = gpu_experts_mask_cpu  # CPU tensor, safe to replace
        self.gpu_experts_mask_cuda.copy_(gpu_experts_mask_cpu)  # In-place update for CUDA graph
        self.logical_to_gpu_index = logical_to_gpu_index_cuda.cpu()  # CPU version for weight loading
        self.logical_to_gpu_index_cuda.copy_(logical_to_gpu_index_cuda)  # In-place update for CUDA graph
        self.gpu_index_to_logical = gpu_index_to_logical_cpu  # CPU tensor, safe to replace

        # Step 4: Update KT wrapper (rank 0 only)
        if self.tp_rank == 0:
            update_kt_wrapper_masks(self.wrapper, gpu_experts_mask_cpu)

        # Log expert changes (rank 0 only).  The argument gate matters: the
        # .cpu().tolist() is a host sync that would otherwise serialize the
        # successor transport behind this layer's compute at every level.
        if self.tp_rank == 0 and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "KT dynamic update: layer %d updated GPU experts to: %s",
                self.kt_config.layer_idx,
                selected_experts.cpu().tolist(),
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

    def _build_full_context(self, layer: torch.nn.Module) -> "SharedFullContext":
        holder = get_buffer("kt_full_context", dict)
        if "ctx" not in holder:
            holder["ctx"] = SharedFullContext(
                layer=layer,
                init_args=self._full_init_args,
                global_num_experts=self.global_num_experts,
                moe_runner_config=self.moe_runner_config,
            )
        ctx = holder["ctx"]

        ctx.load(
            layer_idx=self.kt_config.layer_idx,
            wrapper=self.wrapper,
            original_layer=layer,
            gpu_experts_mask=self.gpu_experts_mask,
            logical_to_gpu_index=self.logical_to_gpu_index,
        )
        return ctx


# ---------------------------------------------------------------------------
# Plugin registration: makes KTEPWrapperMethod available to FusedMoE without
# any base-file import. Activated by importing this module, which happens via
# sglang.srt.models.deepseek_v4 -> auto-discovered by ModelRegistry.
# ---------------------------------------------------------------------------

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
# Act on one boundary in every interval/_KT_BOUNDARY_DIVISOR.
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
_KT_BOUNDARY_DIVISOR = 10

# How long a demotion will wait on the background disk prefetch before giving
# up and reading the expert itself. Generous on purpose: the read is already in
# flight, so waiting costs at most what is LEFT of it, while giving up costs
# the whole read again. Bounded only so a stuck reader cannot wedge a window.
_KT_PREFETCH_WAIT_S = 60.0

_KT_BOUNDARY_STATE = {
    "last_was_extend": False,
    "transitions": 0,
}


def maybe_run_expert_swap_at_decode_boundary(is_decode: bool, is_extend: bool) -> None:
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

    RATE LIMIT IS NOT OPTIONAL. With continuous batching at 8 concurrent
    requests a prefill->decode transition lands every ~2.5 s; an unthrottled
    window there costs far more than it returns. The interval is expressed in
    seconds of wall clock rather than forwards because the cost being bounded
    (a quiesce plus weight copies) is wall-clock cost.
    """
    if not _KT_EP_METHODS:
        return
    anchor = _KT_EP_METHODS[0]
    cfg = anchor.kt_config
    if cfg.expert_swap_interval <= 0:
        _KT_BOUNDARY_STATE["last_was_extend"] = is_extend
        return

    crossed = is_decode and _KT_BOUNDARY_STATE["last_was_extend"]
    _KT_BOUNDARY_STATE["last_was_extend"] = is_extend
    if not crossed:
        return

    _KT_BOUNDARY_STATE["transitions"] += 1
    n = _KT_BOUNDARY_STATE["transitions"]
    every = max(1, cfg.expert_swap_interval // _KT_BOUNDARY_DIVISOR)
    # Observe on every transition, act on every `every`-th, and never on the
    # first: a cumulative counter's first delta is the whole launch history,
    # so acting on it is acting on a baseline.
    try:
        maybe_run_expert_swap_window(anchor, force=True, act=(n > 1 and n % every == 0))
    except Exception:
        # NOT swallowed any more. "Serving continues" was the wrong policy: a
        # window that failed part-way has already moved kt's ownership and
        # staged GPU rows whose tables never flipped, so continuing serves
        # wrong experts silently and forever.
        _fatal_swap_failure("the prefill->decode boundary window raised")


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
    from sglang.srt.layers.moe.kt_arena_share import arena_source_for
    from sglang.srt.layers.moe.kt_expert_swap import (
        ExpertSwapPolicy,
        SwapInstallError,
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
    sample_every = max(1, cfg.expert_swap_interval // 5)
    if not force and n % sample_every:
        return
    # A boundary call has already decided WHETHER to act -- it fires once
    # per prefill->decode transition and is rate-limited on wall clock, so
    # re-gating on the eager-forward counter would drop most windows. It
    # still passes act=False for its first call, because a cumulative
    # counter's first delta is the whole launch history and acting on that
    # baseline is what the interval//5 sampling exists to prevent.
    act = act if act is not None else (n % cfg.expert_swap_interval) == 0

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
        method._swap_policy.snapshot_counters(
            method._margin_insist_count,
            method._margin_override_count,
            method._resident_hit_count,
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
        # Sampling-only pass: counters folded into the EMAs, nothing moved --
        # but it is also the last boundary before an acting one, so it is where
        # the demotion reads get started off the critical path.
        if entries:
            _start_demotion_prefetch(anchor, entries)
        return

    mover = _get_or_create_expert_mover(anchor)
    if mover is None:
        return

    # Phase accounting for one window, logged on the way out.
    _timing = {
        "read_s": 0.0,
        "install_s": 0.0,
        "prefetch_hits": 0,
        "prefetch_misses": 0,
    }
    _window_t0 = time.perf_counter()

    store = _KT_SPLIT_PREFILL_STATE["store"]

    # Per-layer physical-to-logical maps for _checkpoint_id: _move only has
    # the layer module in hand, everything else here has an entry.
    p2l_by_layer = {
        e["layer_idx"]: e["method"]._kt_physical_to_logical for e in entries
    }

    # RANK-WRITE ARMING IS DECIDED ONCE PER WINDOW, SYMMETRICALLY. The writer
    # itself is per-rank fallible -- kt_arena_share degrades a rank that could
    # not map the arena, by design -- so gating anything directly on "do I
    # have a writer" splits the ranks, which is precisely the M9/M11/M12
    # failure class. One MIN all_reduce here, on a call every rank reaches
    # (entries and act are plan data), and every later branch keys off the
    # result instead: the barrier, the per-layer capture, and the install.
    _rank_writer = _get_or_create_rank_writer(entries[0])
    _KT_SWAP_STATE["rank_write_armed"] = _all_tp_ranks_succeeded(
        _rank_writer is not None
    )
    if _rank_writer is not None and not _KT_SWAP_STATE["rank_write_armed"]:
        logger.error(
            "[kt-rankwrite] disarmed for this window: another rank has no "
            "arena mapping; every rank takes the checkpoint path"
        )

    # Arena promotion (full-kt + KT_BUFFER_B_MEMFD): promoted bytes come from
    # this rank's read-only mapping of kt's own buffers instead of the
    # checkpoint. The batched swizzle plan it feeds is armed lazily here
    # because without split prefill nothing else builds one.
    _maybe_arm_arena_swizzle_plan(entries)

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
        "prefetched": {},
        "export_rows": {},
        # (layer_idx, demote_id) acquires taken by _begin_layer, provisional
        # until that layer's tables flip; _on_layer_abort releases them.
        "dma_acquired": [],
        # True while the staged items' "promoted" bytes came from the arena
        # source (checkpoint layout, no cold-store slot behind them). A batch
        # is never mixed: arena staging happens only when there is no store.
        "arena": False,
    }
    gpu_reader = _get_or_create_gpu_reader()

    def _flush_moves():
        """Apply one layer's staged swaps as a handful of bulk copies.

        WHY THIS EXISTS. The per-expert version issued, for every swap, four
        `.to("cpu", non_blocking=False)` reads -- a BLOCKING D2H each. At 8
        swaps x 4 tensors x 92 layers that is 2,944 synchronising round trips,
        and it is what made a measured swap window cost 36-78 s against a
        bandwidth model predicting ~116 ms. The traffic was never the problem:
        3.2 GB at 27.9 GB/s is a tenth of a second. The stalls were.

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
            pend["arena"] = False
            return
        layer, layer_idx = pend["layer"], pend["layer_idx"]
        pend["items"] = []
        arena_batch = pend["arena"]
        pend["arena"] = False
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

        dtypes = {
            n: getattr(layer, n).data.dtype
            for n in _MXFP4_TRTLLM_RESIDENT_PARAM_NAMES
        }

        # READ: gather every demoted row, one kernel per weight name, and pull
        # them down asynchronously into pinned staging. Only a store needs the
        # demoted bytes back -- an arena batch runs under full kt residency,
        # where the demoted expert never lost its CPU buffers, so there is
        # nothing to read and nothing to synchronize before the writes.
        staged = {}
        if store is not None:
            for name in _MXFP4_TRTLLM_RESIDENT_PARAM_NAMES:
                gathered = _bytes(getattr(layer, name).data).index_select(0, idx)
                host = torch.empty(
                    gathered.shape, dtype=torch.uint8, device="cpu", pin_memory=True
                )
                host.copy_(gathered, non_blocking=True)
                staged[name] = host
            # The one sync for the whole layer. Everything above must land
            # before the writes below overwrite the rows it read.
            torch.cuda.synchronize()

        # WRITE: scatter every promoted row back, again one kernel per name.
        promoted = {
            name: torch.stack([it["promoted"][name] for it in items]).to(
                dev, non_blocking=True
            )
            for name in _MXFP4_TRTLLM_RESIDENT_PARAM_NAMES
        }
        if arena_batch or (store is not None and getattr(store, "raw_layout", False)):
            # The source holds checkpoint-layout bytes (raw store or arena
            # mapping alike), so the resident row's trtllm layout is produced
            # HERE, on the GPU, rather than by asking the CPU for it. Same
            # four-gather form split prefill uses, just over this layer's
            # swaps instead of its whole cold set.
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
                (len(items),) + tuple(dst.shape[1:])
            )
        for name in _MXFP4_TRTLLM_RESIDENT_PARAM_NAMES:
            _bytes(getattr(layer, name).data).index_copy_(0, idx, promoted[name])

        # The store is authoritative for the cold set, so the demoted rows go
        # back into the slots the promoted experts vacated. Slices of the
        # pinned staging are already on CPU, so write_row's .to("cpu") is free;
        # they are handed back in the param's own dtype, which write_row checks.
        # An arena batch has no store: full kt residency means the demoted
        # expert's bytes never left the CPU, so nothing is written anywhere.
        if store is not None:
            if getattr(store, "raw_layout", False):
                # Mirror of the promotion side: the rows gathered off the GPU
                # are in trtllm layout, and a raw store must not be given those.
                staged = _unswizzle_demoted_rows(staged, dtypes)
            for i, it in enumerate(items):
                store.write_row(
                    layer_idx,
                    it["slot"],
                    {
                        n: staged[n][i].view(dtypes[n])
                        for n in _MXFP4_TRTLLM_RESIDENT_PARAM_NAMES
                    },
                    logical_id=it["demoted_id"],
                )

    def _move(layer, dst_row, logical_id, demoted_id):
        """Record expert ``logical_id`` -> resident row ``dst_row``.

        Records rather than applies: the copies are batched per layer by
        _flush_moves. The promoted shard is cloned out of the cold store HERE,
        at record time, because the slot it comes from is the slot the demoted
        expert will later take.
        """
        layer_idx = layer.layer_id
        slot = None if store is None else store.slot_of(layer_idx, logical_id)
        if slot is None:
            # Storeless promotion, preferred sources in order: the layer's
            # batched ring export (staged by _begin_layer), then this rank's
            # arena mapping -- either way ~2.2 MB of RAM instead of a
            # checkpoint read, in checkpoint layout for the batched swizzle.
            # Kept out of any batch with store-backed items by construction:
            # this branch only runs when there is no store at all.
            if store is None and _KT_SPLIT_PREFILL_STATE.get("swizzle_plan") is not None:
                staged_row = _pending["export_rows"].pop(logical_id, None)
                if staged_row is None:
                    src = arena_source_for(layer_idx)
                    if src is not None:
                        t0 = time.perf_counter()
                        staged_row = _arena_stage_row(src, logical_id)
                        _timing["read_s"] += time.perf_counter() - t0
                if staged_row is not None:
                    if _pending["layer_idx"] != layer_idx:
                        _flush_moves()
                        _pending["layer"] = layer
                        _pending["layer_idx"] = layer_idx
                    _pending["items"].append(
                        {
                            "dst_row": dst_row,
                            "slot": None,
                            "demoted_id": demoted_id,
                            "promoted": staged_row,
                        }
                    )
                    _pending["arena"] = True
                    return
            # No store, or the expert left the cold set (a re-promotion inside
            # one window). The checkpoint path writes the GPU row immediately,
            # so drain anything staged for this layer first -- otherwise a
            # queued scatter could land on top of it.
            _flush_moves()
            mover.move(
                layer,
                dst_row,
                _checkpoint_id(p2l_by_layer.get(layer_idx), logical_id),
            )
            return

        if _pending["layer_idx"] != layer_idx:
            _flush_moves()
            _pending["layer"], _pending["layer_idx"] = layer, layer_idx
        _pending["items"].append(
            {
                "dst_row": dst_row,
                "slot": slot,
                "demoted_id": demoted_id,
                "promoted": store.stage_row(layer_idx, slot),
            }
        )

    def _verify_install_once(entry):
        """SGLANG_KT_VERIFY_CPU_INSTALL=1: prove the demotion install bitwise.

        Re-runs the install's own fill against an expert that is ALREADY
        CPU-resident and compares the AMX buffers byte for byte with what the
        bulk load produced. Shares fill_expert_buffers with the real install,
        deliberately -- a check that reimplements the thing it checks verifies
        nothing.

        This is the gate the demotion path actually needs. End-to-end quality
        can only say "something is worse"; a wrong NUMA slice writes a
        valid-looking expert and fails nothing downstream.
        """
        import os

        if os.environ.get("SGLANG_KT_VERIFY_CPU_INSTALL") != "1":
            return
        if _KT_SWAP_STATE.get("install_verified"):
            return
        _KT_SWAP_STATE["install_verified"] = True
        method = entry.get("method")
        if method is None or method.wrapper is None:
            return
        mask = method.gpu_experts_mask
        resident = [i for i in range(mask.numel()) if not bool(mask[i])]
        if not resident:
            logger.warning("[kt-install-verify] no CPU-resident expert to check")
            return
        eid = resident[len(resident) // 2]
        try:
            # Checkpoint read translated; the kt-side slot id stays physical.
            tensors = mover.read_full_expert(
                entry["layer"], _checkpoint_id(method._kt_physical_to_logical, eid)
            )
            ok = method.wrapper.verify_install_against_loaded(
                eid, *[t.data_ptr() for t in tensors]
            )
        except Exception:
            logger.exception("[kt-install-verify] check itself failed")
            return
        if ok:
            logger.info(
                "[kt-install-verify] expert %d: install BITWISE-MATCHES the "
                "bulk load on every NUMA partition",
                eid,
            )
        else:
            logger.error(
                "[kt-install-verify] expert %d: install DIFFERS from the bulk "
                "load -- demoted experts are being given wrong weights",
                eid,
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
        if method is None or not method.kt_config.cold_only_cpu_experts:
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
        writer = _KT_SWAP_STATE.get("rank_writer")
        if writer is not None and _pending.get("rank_write"):
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

        # CHECKPOINT PATH. It also moves kt's slot (swap_expert_slot routes
        # through the same move_slot_only), so a writer that exists but is
        # not driving this layer STILL has to mirror the move.
        if method.wrapper is None:
            if writer is not None:
                writer.commit_move(entry["layer_idx"], promote_id, demote_id)
            return
        _verify_install_once(entry)
        t0 = time.perf_counter()
        tensors = _read_demoted_expert(entry, demote_id)
        t1 = time.perf_counter()
        method.wrapper.swap_expert_slot(
            promote_id, demote_id, *[t.data_ptr() for t in tensors]
        )
        if writer is not None:
            writer.commit_move(entry["layer_idx"], promote_id, demote_id)
        # Timed separately on purpose. The split between "fetching the bytes"
        # and "handing them to kt" was previously inferred by subtracting an
        # ESTIMATED disk rate from the measured window, which is guesswork
        # dressed as a number -- and the loader shows the install is a memcpy
        # plus a strided restride of down_proj, with no format conversion, so
        # the estimate was probably wrong. Measure both.
        _timing["read_s"] += t1 - t0
        _timing["install_s"] += time.perf_counter() - t1

    def _begin_layer(entry, swaps, rows):
        """Read this layer's demoted experts off the GPU, before any move.

        Runs once per layer with the whole plan, so the collectives inside are
        keyed to the plan -- identical on every rank -- rather than to per-swap
        conditions. It also guarantees read-before-write unconditionally: no
        move has run yet, so every row still holds its demoted occupant even on
        the checkpoint-fallback path that writes its row immediately.

        Under the direct-DMA transport it additionally RETURNS a filtered
        (swaps, rows) plan: a demoted expert must have its arena pages
        registered on EVERY rank before the tables make it routable
        (register-before-routable, SPEC-DIRECT-DMA invariant 1), so pairs
        any rank could not register are dropped everywhere. The filter is
        derived from ONE fixed-shape collective over the plan, so it is
        identical on all ranks by construction.
        """
        _pending["prefetched"] = {}
        _pending["export_rows"] = {}
        _pending["dma_acquired"] = []
        filtered = None
        dma_src = _KT_SPLIT_PREFILL_STATE["direct_source"]
        if dma_src is not None and swaps:
            local_ok = dma_src.window_acquire_demotions(
                entry["layer_idx"], [s.demote for s in swaps]
            )
            ok = _all_tp_ranks_succeeded_vec(local_ok)
            if not all(ok):
                # A locally-successful acquire for a globally-dropped pair
                # must be released: its expert stays resident, so nothing
                # would ever release it later.
                for s, o, local in zip(swaps, ok, local_ok):
                    if local and not o:
                        dma_src.window_release(entry["layer_idx"], s.demote)
                kept = [(s, r) for s, r, o in zip(swaps, rows, ok) if o]
                logger.error(
                    "[kt-dma] layer %s: dropped %d/%d swap pair(s) -- a rank "
                    "could not register the demoted expert's pages",
                    entry.get("layer_idx"),
                    len(swaps) - len(kept),
                    len(swaps),
                )
                swaps = [s for s, _ in kept]
                rows = [r for _, r in kept]
                filtered = (swaps, rows)
            # The kept pairs' acquires are provisional until the tables flip:
            # a layer that aborts after this point never flips, its demoted
            # experts stay RESIDENT, and no future window would ever release
            # them -- on_layer_abort reconciles via this stash, and
            # _after_flip clears it once the flip commits the acquires.
            _pending["dma_acquired"] = [
                (entry["layer_idx"], int(s.demote)) for s in swaps
            ]
        # Export-design promotions: ONE batched raw export of this layer's
        # promoted experts into stage 0 of every rank's cold ring. Every rank
        # calls this with the identical plan (rank 0 exports, peers wait the
        # ring flag), the window is quiesced so stage 0 is free, and the rows
        # are consumed (stack-copied) by _flush_moves before the next layer's
        # begin can overwrite them. ~1 ms per layer at swap-max 8.
        exp_src = _KT_SPLIT_PREFILL_STATE["export_source"]
        if (
            exp_src is not None
            and store is None
            and _KT_SPLIT_PREFILL_STATE.get("swizzle_plan") is not None
        ):
            t0 = time.perf_counter()
            try:
                staged = exp_src.export_experts_sync(
                    entry["layer_idx"], [s.promote for s in swaps]
                )
            except Exception:
                logger.exception(
                    "[kt-export] promotion export failed for layer %s; this "
                    "layer falls back to the checkpoint",
                    entry.get("layer_idx"),
                )
            else:
                from sglang.srt.layers.moe.expert_cold_store import WEIGHT_NAMES

                # CLONES, not views: consumption must complete before the
                # per-layer all_reduce below, which is what stops rank 0's
                # NEXT layer's export from overwriting stage 0 while a
                # lagging peer still reads it (the review's promotion-WAR
                # finding). ~18 MB per layer at swap-max 8.
                _pending["export_rows"] = {
                    int(s.promote): {
                        res: staged[raw][i].clone()
                        for res, raw in zip(
                            _MXFP4_TRTLLM_RESIDENT_PARAM_NAMES, WEIGHT_NAMES
                        )
                    }
                    for i, s in enumerate(swaps)
                }
            _timing["read_s"] += time.perf_counter() - t0
        method = entry.get("method")

        # RANK-WRITE DEMOTIONS. Capture this rank's own slice of every
        # demoted expert BEFORE any move overwrites its GPU row -- purely
        # local, no collective. Whether it worked then rides the SAME single
        # per-layer collective below, so the layer is either rank-write on
        # every rank or on none: a rank that silently skipped its write would
        # leave a hole in the expert, which is wrong bytes rather than a
        # crash.
        _pending["rank_write"] = False
        # The WINDOW-scoped consensus, not this rank's writer: every rank
        # takes the same branch here, so the collective counts below match
        # even when one rank could not map the arena.
        armed = bool(_KT_SWAP_STATE.get("rank_write_armed"))
        writer = _KT_SWAP_STATE.get("rank_writer") if armed else None
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

        want = (
            gpu_reader is not None
            and not _KT_SWAP_STATE.get("gpu_readback_off")
            and method is not None
            and method.kt_config.cold_only_cpu_experts
        )
        if armed:
            # NO per-layer consensus here any more, and none is needed: under
            # the fail-fast policy a rank that could not capture or validate
            # has already terminated every rank, so there is no surviving
            # disagreement to reconcile. Removing it also removes a
            # collective, which is the resource these rounds kept
            # desynchronising. `armed` is the window-scoped consensus, so all
            # eight ranks take this branch or none do.
            _pending["rank_write"] = True
            return filtered
        # NOT part of `want`: whether THIS rank has a kt wrapper to install
        # into. Only some ranks do, and gating the gather on it is what
        # deadlocked M9 and M11 -- rank 0 entered the collective alone and the
        # other seven, having nothing to install, never called it and sailed on
        # through all 92 layers. A rank that will not consume the result still
        # has to contribute its shard.
        #
        # Agreed across ranks rather than assumed: this all_reduce runs on every
        # layer whether or not the gather does, so it is symmetric by
        # construction, and any residual disagreement degrades to "no rank uses
        # the GPU route here" instead of hanging.
        if not _all_tp_ranks_succeeded(want):
            return filtered
        # Not wrapped: absorbing here on one rank would drop it out of the
        # collectives the others are running, which is the failure this hook
        # exists to prevent.
        got = gpu_reader.read_full_experts(entry["layer"], rows)
        _pending["prefetched"] = {
            s.demote: [t.to("cpu") for t in tensors]
            for s, tensors in zip(swaps, got)
        }
        return filtered

    def _read_demoted_expert(entry, demote_id):
        """The demoted expert's full bytes, from the GPU if that is proven.

        Device memory already holds them; the checkpoint read they replace is
        ~17.5 MB per demotion and dominated the swap window. The GPU route is
        used only after it has been shown bitwise-equal to the checkpoint route
        on this process's first demotion, and any failure falls back rather
        than installing bytes nobody has checked.
        """
        # 1. The background disk prefetch started at the previous boundary.
        #    Waiting on it is the point: the read is already in flight, so
        #    waiting costs at most what remains of it, while re-reading
        #    synchronously costs the whole thing again.
        pf = _KT_SWAP_STATE.get("prefetch")
        if pf is not None:
            pf["done"].wait(timeout=_KT_PREFETCH_WAIT_S)
            got = pf["data"].get((entry.get("layer_idx"), demote_id))
            if got is not None:
                _timing["prefetch_hits"] += 1
                return got
            _timing["prefetch_misses"] += 1

        # 2. Prefetched by _begin_layer off the GPU, before any of this layer's
        #    moves ran, in one collective per tensor. Nothing here is per-rank
        #    data: either the whole layer was prefetched on every rank or none.
        got = _pending["prefetched"].get(demote_id)
        if got is None:
            return mover.read_full_expert(
                entry["layer"],
                _checkpoint_id(entry["method"]._kt_physical_to_logical, demote_id),
            )

        if not _KT_SWAP_STATE.get("gpu_readback_verified"):
            # NO COLLECTIVE HERE. The comment that used to sit at this line
            # claimed every rank reaches it; that is false, and it was M11
            # rebuilt: _read_demoted_expert's only caller sits AFTER
            # `if method.wrapper is None: return`, and the wrapper exists on
            # rank 0 alone, so an all_reduce here is issued by one rank while
            # seven march into the next layer -- a permanent one-collective
            # skew on the gloo group that ends in the watchdog.
            #
            # The verdict does not need a collective anyway: turning the
            # route off is rank-local, and the per-layer
            # `_all_tp_ranks_succeeded(want)` in _begin_layer is a MIN, so
            # rank 0's False propagates to every rank on the very next layer
            # through a consensus that IS symmetric.
            _KT_SWAP_STATE["gpu_readback_verified"] = True
            want = mover.read_full_expert(
                entry["layer"],
                _checkpoint_id(entry["method"]._kt_physical_to_logical, demote_id),
            )
            bad = [
                i
                for i, (a, b) in enumerate(zip(got, want))
                if a.shape != b.shape or not torch.equal(a, b)
            ]
            if bad:
                logger.error(
                    "[kt-swap] GPU read-back DIFFERS from the checkpoint on "
                    "tensor(s) %s (expert %d, layer %s)",
                    bad,
                    demote_id,
                    entry.get("layer_idx"),
                )
            if bad:
                # Local flag only; _begin_layer's MIN consensus carries it to
                # every rank at the next layer.
                _KT_SWAP_STATE["gpu_readback_off"] = True
                logger.error(
                    "[kt-swap] GPU read-back disabled; the next layer's "
                    "consensus drops it on every rank and demotions read the "
                    "checkpoint instead"
                )
                return want
            logger.info(
                "[kt-swap] GPU read-back verified bitwise against the "
                "checkpoint; demotions no longer touch disk"
            )
        return got

    def _after_flip(entry, swaps):
        # Direct-DMA bookkeeping keyed to the FLIPPED table: each applied
        # pair's PROMOTED expert just left the cold set, so its page pins go
        # from live to trim-eligible. Releasing on the proposed plan instead
        # would drop pins a skipped pair's still-cold expert needs. The flip
        # also COMMITS this layer's demotion acquires: clear the abort stash.
        _pending["dma_acquired"] = []
        dma_src = _KT_SPLIT_PREFILL_STATE["direct_source"]
        if dma_src is None:
            return
        for s in swaps:
            dma_src.window_release(entry["layer_idx"], s.promote)

    def _on_layer_abort(entry):
        # The layer failed after _begin_layer: tables never flipped, its
        # demoted experts stay resident, and nothing else would ever release
        # the acquires _begin_layer took for them.
        dma_src = _KT_SPLIT_PREFILL_STATE["direct_source"]
        stash = _pending.get("dma_acquired") or []
        _pending["dma_acquired"] = []
        if dma_src is None:
            return
        for layer_idx, demote in stash:
            dma_src.window_release(layer_idx, demote)
        if stash:
            logger.info(
                "[kt-dma] layer %s abort: released %d provisional demotion "
                "acquire(s)",
                entry.get("layer_idx"),
                len(stash),
            )

    try:
        result = run_swap_window(
            entries,
            move_weights=_move,
            install_cpu_expert=_install_cpu,
            begin_layer=_begin_layer,
            finish_layer=_flush_moves,
            after_flip=_after_flip,
            on_layer_abort=_on_layer_abort,
            quiesce=lambda: torch.cuda.synchronize(anchor.gpu_experts_mask_cuda.device),
        )
    finally:
        # Drain the last layer. In a finally because staged items surviving a
        # failed window would be applied during the NEXT one -- writing a stale
        # layer's rows, which is silent and unattributable.
        _flush_moves()
        # Rank-write demotions: every rank's slices must have LANDED before
        # serving resumes, or kt computes a demoted expert with another
        # rank's hole still in it. The natural TP lockstep of the next
        # forward would mostly cover it, but "mostly" is not a memory
        # ordering -- one plan-independent barrier here costs ~1 ms and is
        # symmetric on every path out of the window, including the raising
        # one.
        # Keyed to the WINDOW consensus, never to this rank's writer: a
        # barrier some ranks skip is a hang, and the writer is the one
        # precondition that is per-rank fallible.
        _rw = _KT_SWAP_STATE.get("rank_writer")
        if _KT_SWAP_STATE.get("rank_write_armed"):
            if dist.is_initialized() and get_parallel().tp_size > 1:
                dist.barrier(group=get_tp_group().cpu_group)
        if _rw is not None:
            # After the barrier, so every rank's slice is in: this is the only
            # point where an expert this path built is complete and readable.
            _verify_rank_write_once(entries, _rw, mover)
            logger.info("%s", _rw.end_window())
        # Direct-DMA epilogue, still inside the quiesced window (the finally
        # covers a raising window too -- some layers may have flipped before
        # the failure, and serving must not resume on their stale plans):
        # fold the new logical_to_slot into the copy plans, then trim dead
        # registration units to budget. Trim is legal ONLY here: nothing is
        # mid-DMA while the window holds the pipeline quiesced.
        _dma_src = _KT_SPLIT_PREFILL_STATE["direct_source"]
        if _dma_src is not None:
            _dma_src.invalidate_plans()
            budget = int(1.25 * _dma_src.live_bytes())
            freed = _dma_src.window_trim(budget_bytes=budget)
            if freed:
                logger.info(
                    "[kt-dma] window trim freed %.2f GiB (registered %.1f GiB, "
                    "budget %.1f GiB)",
                    freed / (1 << 30),
                    _dma_src.registered_bytes() / (1 << 30),
                    budget / (1 << 30),
                )
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
        # The phase split, MEASURED. It was previously inferred by subtracting
        # an estimated disk rate from the window total, and the loader says
        # that estimate was probably wrong: the install is a memcpy plus a
        # strided restride of down_proj, with no format conversion in it.
        logger.info(
            "[kt-swap] window %d timing: total %.2fs = fetch %.2fs + install "
            "%.2fs (+%.2fs elsewhere); prefetch %d hit / %d miss",
            _KT_SWAP_STATE["windows"],
            time.perf_counter() - _window_t0,
            _timing["read_s"],
            _timing["install_s"],
            (time.perf_counter() - _window_t0)
            - _timing["read_s"]
            - _timing["install_s"],
            _timing["prefetch_hits"],
            _timing["prefetch_misses"],
        )


class _GpuResidentExpertReader:
    """Read a resident expert back out of GPU memory, in checkpoint layout.

    WHY. Under cold-only residency a demoted expert owns no CPU buffers, so
    every demotion must be given weights before it becomes routable. Reading
    them off the checkpoint costs ~17.5 MB per demotion -- 8 swaps x 92 layers
    = ~12.9 GB per window, which at this node's ~1.1 GB/s is ~11.7 s and is
    essentially the entire 12.5 s a batched swap window still cost. The bytes
    are already in device memory; only the layout differs.

    Two things stand between the resident row and the exported bytes:

      1. the trtllm-gen shuffle, inverted exactly by ``unswizzle_trtllm_expert``
         (bitwise on every tensor -- runs/meta/verify_unswizzle.py); and
      2. TP. A rank holds one eighth of the expert, while kt slices across its
         own NUMA partitions internally and therefore wants the whole thing.
         So the shards are all-gathered -- ~17.5 MB per expert over NVLink,
         which is microseconds against the seconds of disk it replaces.

    Everything here is shape-derived and cached per layer shape, so the cost
    per demotion is the unswizzle plus one collective.
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

    def read_full_experts(self, layer, dst_rows):
        """Every expert in ``dst_rows``, in one collective per tensor.

        This is the shape the swap window must use. Reading rows one at a time
        issues six all-gathers per expert and, worse, does so from inside the
        per-swap install where the decision to read at all is per-rank data --
        ranks then disagree on how many collectives to run and the window
        deadlocks in NCCL instead of falling back. Called once per layer with
        the layer's whole plan, the collective count is a pure function of that
        plan, which is identical on every rank by construction.

        Returns one ``(gate, up, down, gate_s, up_s, down_s)`` tuple per row, in
        the same order and layout ``CheckpointExpertMover.read_full_expert``
        returns, so the two are interchangeable.

        Every row must still hold its DEMOTED occupant: call before any move.
        """
        shards = self.read_own_shards(layer, dst_rows)
        return self._gather_shards(shards, len(dst_rows))

    def read_own_shards(self, layer, dst_rows):
        """THIS RANK's unswizzled shards for ``dst_rows`` -- no collective.

        The rank-write demotion path needs exactly this and nothing more: it
        writes its own slice into kt's shared arena, so it never wants the
        gathered full expert. Keeping it a separate method means the two
        paths share one proven unswizzle and one shape cache, and the
        collective lives only in the caller that actually needs it.

        Every row must still hold its DEMOTED occupant: call before any move.
        """
        from sglang.srt.layers.moe.kt_mxfp4_export import unswizzle_trtllm_expert

        inverse, w13_scale_shape, w2_scale_shape = self._prepare(layer)
        w13_n, w13_s_n, w2_n, w2_s_n = self.param_names
        return [
            unswizzle_trtllm_expert(
                w13=getattr(layer, w13_n).data[r],
                w13_scale=getattr(layer, w13_s_n).data[r],
                w2=getattr(layer, w2_n).data[r],
                w2_scale=getattr(layer, w2_s_n).data[r],
                inverse=inverse,
                w13_scale_shape=w13_scale_shape,
                w2_scale_shape=w2_scale_shape,
            )
            for r in dst_rows
        ]

    def _gather_shards(self, shards, n_rows: int):
        per = shards[0].w13.shape[0] // 2
        per_s = shards[0].w13_scale_e8m0.shape[0] // 2
        # (name, per-expert concat dim). Stacking prepends an expert axis, so
        # the concat dim shifts by one inside _all_gather_batched.
        plan = [
            ([s.w13[:per] for s in shards], 0),
            ([s.w13[per:] for s in shards], 0),
            ([s.w2 for s in shards], 1),
            ([s.w13_scale_e8m0[:per_s] for s in shards], 0),
            ([s.w13_scale_e8m0[per_s:] for s in shards], 0),
            ([s.w2_scale_e8m0 for s in shards], 1),
        ]
        gathered = [
            self._all_gather_batched(torch.stack(parts), dim) for parts, dim in plan
        ]
        return [tuple(g[i].contiguous() for g in gathered) for i in range(n_rows)]

    def _all_gather_batched(self, stacked: torch.Tensor, dim: int) -> torch.Tensor:
        """Gather ``[experts, ...]`` shards from every rank; concat along ``dim``.

        One collective for the whole layer instead of one per expert.
        """
        tp_size = get_parallel().tp_size
        if tp_size == 1:
            return stacked.contiguous()
        stacked = stacked.contiguous()
        out = torch.empty(
            (tp_size,) + tuple(stacked.shape), dtype=stacked.dtype, device=stacked.device
        )
        dist.all_gather_into_tensor(out, stacked, group=get_tp_group().device_group)
        # out is [rank, expert, ...]; the per-expert concat axis is dim + 1.
        return torch.cat(list(out.unbind(0)), dim=dim + 1)

    def read_full_expert(self, layer, dst_row: int):
        """Single-row convenience wrapper. Prefer :meth:`read_full_experts`.

        Kept for the offline verifier and for tp_size == 1, where there is no
        collective and therefore no symmetry requirement.
        """
        from sglang.srt.layers.moe.kt_mxfp4_export import unswizzle_trtllm_expert

        inverse, w13_scale_shape, w2_scale_shape = self._prepare(layer)
        w13_n, w13_s_n, w2_n, w2_s_n = self.param_names

        shard = unswizzle_trtllm_expert(
            w13=getattr(layer, w13_n).data[dst_row],
            w13_scale=getattr(layer, w13_s_n).data[dst_row],
            w2=getattr(layer, w2_n).data[dst_row],
            w2_scale=getattr(layer, w2_s_n).data[dst_row],
            inverse=inverse,
            w13_scale_shape=w13_scale_shape,
            w2_scale_shape=w2_scale_shape,
        )

        # build_expert_bytes packs w13 as [gate | up] ROW halves of this rank's
        # slice, and shards down by COLUMN; undo both, in rank order.
        per = shard.w13.shape[0] // 2
        per_s = shard.w13_scale_e8m0.shape[0] // 2
        parts = [
            (shard.w13[:per], 0),
            (shard.w13[per:], 0),
            (shard.w2, 1),
            (shard.w13_scale_e8m0[:per_s], 0),
            (shard.w13_scale_e8m0[per_s:], 0),
            (shard.w2_scale_e8m0, 1),
        ]
        return tuple(self._all_gather(t, dim) for t, dim in parts)

    def _all_gather(self, shard: torch.Tensor, dim: int) -> torch.Tensor:
        """Concatenate this tensor's TP shards, in rank order, along ``dim``.

        Every rank swaps the same experts (the policy is deterministic for
        exactly this reason), so all ranks reach this collective the same
        number of times and in the same order.
        """
        tp_size = get_parallel().tp_size
        if tp_size == 1:
            return shard.contiguous()
        shard = shard.contiguous()
        out = torch.empty(
            (tp_size,) + tuple(shard.shape), dtype=shard.dtype, device=shard.device
        )
        dist.all_gather_into_tensor(
            out, shard, group=get_tp_group().device_group
        )
        return torch.cat(list(out.unbind(0)), dim=dim).contiguous()


class _PerLayerMover:
    """One CheckpointExpertMover per layer (permute indices are per-shape)."""

    def __init__(self, weight_path, tp_rank, tp_size):
        self._by_layer = {}
        self._args = (weight_path, tp_rank, tp_size)

    def move(self, layer, dst_row, logical_id, demoted_id=None):
        # demoted_id is part of the MoveWeightsFn contract for movers that
        # maintain a cold-side store; the checkpoint mover reads the promoted
        # expert straight from disk and does not need it.
        self._for(layer)(layer, dst_row, logical_id)

    def read_full_expert(self, layer, logical_id):
        """Full unsliced expert bytes, for the cold-only CPU install.

        Delegates to the same per-layer mover move() uses. Defining it only on
        CheckpointExpertMover left this wrapper without it, and the swap
        window swallowed the AttributeError as a failed layer -- so every
        install silently did nothing while the run looked healthy.
        """
        return self._for(layer).read_full_expert(layer, logical_id)

    def _for(self, layer):
        from sglang.srt.layers.moe.kt_expert_mover import CheckpointExpertMover

        layer_idx = getattr(layer, "layer_id", None)
        if layer_idx is None:
            raise RuntimeError("layer has no layer_id; cannot resolve its experts")
        mover = self._by_layer.get(layer_idx)
        if mover is None:
            weight_path, tp_rank, tp_size = self._args
            prefix = (
                f"language_model.model.layers.{layer_idx}"
                f".block_sparse_moe.experts"
            )
            mover = CheckpointExpertMover(
                weight_path,
                expert_prefix_for_layer=lambda _l, _p=prefix: _p,
                tp_rank=tp_rank,
                tp_size=tp_size,
                param_names=_MXFP4_TRTLLM_RESIDENT_PARAM_NAMES,
            )
            self._by_layer[layer_idx] = mover
        return mover


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
    for key in ("export_source", "direct_source"):
        src = _KT_SPLIT_PREFILL_STATE.get(key)
        if src is not None:
            try:
                src.close()
            except Exception:
                logger.exception("[split-prefill] %s teardown failed", key)
    _KT_SPLIT_PREFILL_STATE["store"] = None
    _KT_SPLIT_PREFILL_STATE["pipeline"] = None
    _KT_SPLIT_PREFILL_STATE["export_source"] = None
    _KT_SPLIT_PREFILL_STATE["direct_source"] = None


def _invert_cold_slot_table(
    l2s: torch.Tensor, num_gpu: int, num_cold: int
) -> torch.Tensor:
    """Cold half of ``logical_to_slot``, inverted: row ``j`` -> the expert
    whose slot is ``num_gpu + j``.

    Runs at gather time, i.e. inside a forward where torch's DEFAULT DEVICE is
    cuda -- A3's first request died on exactly that (a device-less
    ``torch.empty`` landed on cuda:0 against the CPU table). Every tensor here
    is therefore pinned to the table's own device, and the result comes back
    on CPU, which is what the gather indexes with.
    """
    is_cold = l2s >= num_gpu
    cold = torch.empty(num_cold, dtype=torch.int64, device=l2s.device)
    cold[(l2s[is_cold] - num_gpu).long()] = torch.nonzero(
        is_cold, as_tuple=False
    ).flatten()
    return cold.cpu()


def _try_build_direct_dma(
    *,
    layer_indices,
    anchor,
    device,
    num_cold,
    cold_slot_expert_ids,
    arena_source_for,
):
    """Build the direct-DMA source, or (None, None, None). Rank-local.

    Everything fallible about this transport happens HERE, with no
    collectives, so the caller can fold the outcome into one consensus. On
    any failure the registrar is torn down (unregistering whatever booted)
    before returning None.
    """
    from sglang.srt.layers.moe.kt_direct_dma import (
        ArenaExpertRanges,
        CudaCopyLib,
        DirectDmaSource,
        IntervalRegistrar,
        boot_acquire_cold_set,
        cudart_register_fns,
    )

    registrar = None
    try:
        arena_sources = {li: arena_source_for(li) for li in layer_indices}
        missing = [li for li, s in arena_sources.items() if s is None]
        if missing:
            logger.error(
                "[kt-dma] no arena mapping for %d layer(s) (first: %s) -- is "
                "KT_BUFFER_B_MEMFD=1 set?",
                len(missing),
                missing[0],
            )
            return None, None, None

        raw_shapes, swizzle_plan = _build_dynamic_swizzle_plan(anchor, device)
        if raw_shapes is None or swizzle_plan is None:
            return None, None, None

        free_b, _total = torch.cuda.mem_get_info(device)
        floor_b = int(envs.SGLANG_KT_DMA_FREE_VRAM_FLOOR_GB.get() * (1 << 30))
        # The registration itself costs ~8 B per 4K page of VRAM (measured
        # exact); insist the floor still holds AFTER the projected cost.
        ranges_by_layer = {
            li: ArenaExpertRanges(src) for li, src in arena_sources.items()
        }
        any_r = next(iter(ranges_by_layer.values()))
        per_expert = (
            2 * any_r.gu_w
            + 2 * any_r.gu_s
            + (any_r.hidden - 1) * any_r.w2_pitch
            + any_r.w2_width
            + any_r.hidden * any_r.w2s_pitch
        )
        projected_pte = (
            len(layer_indices) * num_cold * per_expert // 4096 * 8
        )
        if free_b - projected_pte < floor_b:
            logger.error(
                "[kt-dma] refusing to arm: free VRAM %.2f GiB minus projected "
                "page tables %.2f GiB is under the %.2f GiB floor",
                free_b / (1 << 30),
                projected_pte / (1 << 30),
                floor_b / (1 << 30),
            )
            return None, None, None

        # Landing-row geometry must agree with the swizzle plan's raw shapes
        # BEFORE any plan writes dst offsets with it.
        def _nbytes(name):
            shape, dtype = raw_shapes[name]
            n = 1
            for s in shape:
                n *= int(s)
            return n * torch.empty((), dtype=dtype).element_size()

        checks = (
            ("w13_weight", 2 * any_r.gu_w),
            ("w13_weight_scale", 2 * any_r.gu_s),
            ("w2_weight", any_r.hidden * any_r.w2_width),
            ("w2_weight_scale", any_r.hidden * any_r.w2s_width),
        )
        for name, want in checks:
            got = _nbytes(name)
            if got != want:
                logger.error(
                    "[kt-dma] landing-row mismatch for %s: raw shape says %d "
                    "bytes, arena geometry says %d -- refusing to arm",
                    name,
                    got,
                    want,
                )
                return None, None, None

        # The registration storm charges kernel memory to the container's
        # cgroup; give it reclaimable headroom FIRST or the charges fail with
        # rc=2 while every host-wide metric looks healthy (D2: 22,901
        # memory.max hits during boot).
        from sglang.srt.layers.moe.kt_direct_dma import (
            reclaim_checkpoint_cache,
        )

        reclaim_checkpoint_cache(
            weight_path=anchor.kt_config.weight_path,
            floor_bytes=250 << 30,
        )
        reg_fn, unreg_fn = cudart_register_fns()
        registrar = IntervalRegistrar(
            register_fn=reg_fn, unregister_fn=unreg_fn
        )
        source = DirectDmaSource(
            ranges_by_layer=ranges_by_layer,
            registrar=registrar,
            copy_lib=CudaCopyLib(),
            cold_slot_expert_ids=cold_slot_expert_ids,
            raw_shapes=raw_shapes,
            moe_layer_indices=layer_indices,
            num_cold=num_cold,
            device=device,
        )
        if not boot_acquire_cold_set(
            source=source,
            registrar=registrar,
            ranges_by_layer=ranges_by_layer,
            cold_slot_expert_ids=cold_slot_expert_ids,
        ):
            registrar.close()
            return None, None, None
        return source, raw_shapes, swizzle_plan
    except Exception:
        logger.exception("[kt-dma] build failed on this rank")
        if registrar is not None:
            try:
                registrar.close()
            except Exception:
                pass
        return None, None, None


def finalize_split_prefill(server_args) -> bool:
    """Build the cold-expert store and prefetch pipeline, then arm every layer.

    Called once after ALL layers have loaded -- the store needs the full MoE
    layer list, and the pipeline's slot parity is defined over it.  Returns
    True if the path is armed.

    On any failure the path is disarmed everywhere and serving continues on
    the existing margin-routed CPU path: a partially-built split-prefill would
    compute a subset of experts and silently degrade quality.
    """
    if not _KT_SPLIT_PREFILL_LAYERS:
        return False

    from sglang.srt.layers.moe.expert_cold_store import (
        WEIGHT_NAMES,
        build_cold_store,
    )
    from sglang.srt.layers.moe.expert_pipeline import ColdExpertPipeline

    anchor, anchor_layer = _KT_SPLIT_PREFILL_LAYERS[0]
    layer_indices = [m.kt_config.layer_idx for m, _ in _KT_SPLIT_PREFILL_LAYERS]
    device = anchor_layer.w13_weight.device

    # Bound BEFORE the try so the except handler can tear down whatever was
    # built before the failure -- a raise after a successful direct-DMA build
    # (e.g. the pipeline's device buffers OOMing) must not orphan ~110 GB of
    # registrations and their VRAM page tables for the process lifetime.
    store = source = pipeline = export_source = direct_source = None
    try:
        per_expert_shapes = {
            name: (tuple(getattr(anchor_layer, name).shape[1:]),
                   getattr(anchor_layer, name).dtype)
            for name in WEIGHT_NAMES
        }
        # EXPORT MODE first (the design of record): kt keeps anonymous weight
        # memory and its own pool fills small per-rank pinned rings with a
        # batched raw export -- measured 30.7 ms/layer with NUMA-local rings,
        # bitwise-verified. Capability lives on rank 0's wrapper only, so it
        # is decided ONCE and broadcast before any collective ring work; a
        # per-rank guess here is exactly the class of divergence the arming
        # consensus below exists to catch, but a boot-time broadcast is
        # cheaper than burning the whole build on it.
        methods_by_layer = {
            m.kt_config.layer_idx: m for m, _ in _KT_SPLIT_PREFILL_LAYERS
        }
        num_gpu = anchor.num_gpu_experts
        num_cold = anchor.global_num_experts - num_gpu

        def cold_slot_expert_ids(layer_idx):
            return _invert_cold_slot_table(
                methods_by_layer[layer_idx].logical_to_slot, num_gpu, num_cold
            )

        can_export = False
        if get_parallel().tp_rank == 0:
            wrappers = {
                li: m.wrapper for li, m in methods_by_layer.items()
            }
            can_export = all(
                w is not None and hasattr(w, "submit_write_raw_experts_to_buffer")
                and hasattr(w.moe, "write_raw_experts_to_buffer_task")
                for w in wrappers.values()
            )
        holder = [can_export]
        if dist.is_initialized() and get_parallel().tp_size > 1:
            dist.broadcast_object_list(
                holder,
                src=get_tp_group().first_rank,
                group=get_tp_group().cpu_group,
            )
        # PINNED-STORE mode is the explicit "give me the fastest per-layer
        # path and the fastest promotions, I will pay the RAM" choice: every
        # streaming transport below is skipped so the build falls through to
        # build_cold_store in RESIDENT layout. It composes with FULL kt
        # residency (no --kt-cold-only-cpu-experts), which is what makes a
        # demotion a nop -- the demoted expert never lost its CPU buffers --
        # while promotions read pre-swizzled rows straight out of the cache.
        force_store = anchor.kt_config.cold_transport == "pinned-store"
        use_export = (
            holder[0]
            and not anchor.kt_config.cold_only_cpu_experts
            and not force_store
        )

        # DIRECT-DMA first when the launch asks for it (SPEC-DIRECT-DMA.md):
        # every rank registers its cold read-set of kt's memfd arenas and its
        # copy engine reads the weights in place -- no prepare stage, one
        # DRAM transit. Everything here is per-rank fallible feeding ONE
        # consensus, the same discipline as the branches below: every rank
        # reaches _all_tp_ranks_succeeded no matter where it failed locally,
        # and a non-unanimous outcome tears down cleanly and falls through to
        # the ring-export transport.
        raw_shapes = swizzle_plan = None
        direct_source = None
        use_direct = (
            anchor.kt_config.cold_transport == "direct-dma"
            and not anchor.kt_config.cold_only_cpu_experts
            and not force_store
        )
        if use_direct:
            from sglang.srt.layers.moe.kt_arena_share import arena_source_for

            direct_source, raw_shapes, swizzle_plan = _try_build_direct_dma(
                layer_indices=layer_indices,
                anchor=anchor,
                device=device,
                num_cold=num_cold,
                cold_slot_expert_ids=cold_slot_expert_ids,
                arena_source_for=arena_source_for,
            )
            use_direct = _all_tp_ranks_succeeded(direct_source is not None)
            if not use_direct and direct_source is not None:
                logger.error(
                    "[kt-dma] disarmed: another rank failed to build; every "
                    "rank falls back to the ring-export transport"
                )
            if not use_direct:
                if direct_source is not None:
                    try:
                        direct_source.close()
                    except Exception:
                        logger.exception("[kt-dma] teardown on disarm failed")
                direct_source = None
                raw_shapes = swizzle_plan = None
            else:
                source = direct_source
                dynamic = True
                use_export = False
                logger.info(
                    "[split-prefill] cold experts stream by DIRECT DMA from "
                    "kt's registered arenas; no export, no rings"
                )

        if use_export:
            # Per-rank fallible (checkpoint read, device allocs) feeding a
            # COLLECTIVE ring build: the outcome must be unanimous or one
            # failed rank skips the rings' broadcast/barriers while seven
            # enter them. Every rank reaches this consensus because
            # use_export is uniform (broadcast capability AND launch config).
            raw_shapes, swizzle_plan = _build_dynamic_swizzle_plan(anchor, device)
            use_export = _all_tp_ranks_succeeded(
                swizzle_plan is not None and raw_shapes is not None
            )
        store = None
        export_source = None
        if use_export:
            from sglang.srt.layers.moe.kt_export_source import (
                ExportColdSource,
                build_cold_rings,
            )

            my_ring, peer_rings = build_cold_rings(
                tp_rank=get_parallel().tp_rank,
                tp_size=get_parallel().tp_size,
                num_cold=num_cold,
                raw_shapes=raw_shapes,
            )
            source = ExportColdSource(
                tp_rank=get_parallel().tp_rank,
                tp_size=get_parallel().tp_size,
                my_ring=my_ring,
                peer_rings=peer_rings,
                cold_slot_expert_ids=cold_slot_expert_ids,
                raw_shapes=raw_shapes,
                moe_layer_indices=layer_indices,
                num_cold=num_cold,
                wrappers_by_layer=(
                    wrappers if get_parallel().tp_rank == 0 else None
                ),
            )
            export_source = source
            dynamic = True
            logger.info(
                "[split-prefill] cold experts stream via kt's batched raw "
                "export into per-rank rings; no mapping, no pinned store"
            )

        # ARENA MODE second: under full-kt + KT_BUFFER_B_MEMFD every rank
        # already maps every expert's checkpoint-layout bytes. Kept as the
        # bridge configuration while the export design proves out.
        from sglang.srt.layers.moe.kt_arena_share import arena_source_for

        arena_sources = {li: arena_source_for(li) for li in layer_indices}
        use_arena = (
            not use_direct
            and not use_export
            and not force_store
            and not anchor.kt_config.cold_only_cpu_experts
            and all(s is not None for s in arena_sources.values())
        )
        if use_arena:
            raw_shapes, swizzle_plan = _build_dynamic_swizzle_plan(anchor, device)
            use_arena = swizzle_plan is not None and raw_shapes is not None
        if not use_export and not use_direct:
            dynamic = use_arena
        if use_arena:
            from sglang.srt.layers.moe.expert_pipeline import ArenaColdSource

            source = ArenaColdSource(
                sources_by_layer=arena_sources,
                cold_slot_expert_ids=cold_slot_expert_ids,
                raw_shapes=raw_shapes,
                moe_layer_indices=layer_indices,
                num_cold=num_cold,
            )
            logger.info(
                "[split-prefill] cold experts stream from the kt arena "
                "mapping; no pinned store built"
            )
        elif not use_export and not use_direct:
            dynamic = envs.SGLANG_KT_SPLIT_PREFILL_DYNAMIC_SWIZZLE.get()
            raw_shapes, swizzle_plan = (
                _build_dynamic_swizzle_plan(anchor, device)
                if dynamic
                else (None, None)
            )
            # The plan builder falls back rather than guessing, so honour that
            # here too: without it, `dynamic` would size the store by a None
            # shape map.
            dynamic = dynamic and swizzle_plan is not None and raw_shapes is not None
            # Make room BEFORE allocating, and do not let eight ranks race.
            # MEASURED (S1): kt's full residency (1,454 GiB) plus eight
            # 51.1 GiB stores is 1,863 GiB against a 1,916 GiB cgroup -- it
            # fits in steady state, but allocating 409 GiB of unreclaimable
            # pinned memory from eight processes at once, with ~530 GiB of
            # checkpoint cache still to reclaim, got rank 0 OOM-killed
            # (memory.events oom_kill 6). Dropping the cache first turns the
            # reclaim race into plain free memory; the stagger keeps the
            # remaining allocations from arriving as one 409 GiB burst.
            from sglang.srt.layers.moe.kt_direct_dma import (
                cgroup_headroom_bytes,
                reclaim_checkpoint_cache,
            )

            reclaim_checkpoint_cache(
                weight_path=anchor.kt_config.weight_path,
                floor_bytes=600 << 30,
            )
            head = cgroup_headroom_bytes()
            if head is not None:
                logger.info(
                    "[cold-store] cgroup headroom before allocation: %.0f GB",
                    head / 1e9,
                )
            time.sleep(3.0 * get_parallel().tp_rank)
            # Swapping used to be refused here: the swap path reads and writes
            # these same rows, and against a raw store that mixed layouts
            # silently. It no longer does -- _flush_moves swizzles a promotion
            # on its way to the resident row and unswizzles a demotion on its
            # way back -- so the two can now run together.
            store = build_cold_store(
                layer_indices=layer_indices,
                gpu_experts_mask=anchor.gpu_experts_mask,
                weight_path=anchor.kt_config.weight_path,
                tp_rank=get_parallel().tp_rank,
                tp_size=get_parallel().tp_size,
                expert_prefix_for_layer=lambda li: (
                    f"language_model.model.layers.{li}.block_sparse_moe.experts"
                ),
                # In dynamic mode the store holds checkpoint-layout bytes, so
                # it is sized by the RAW shapes, not the resident-buffer ones.
                per_expert_shapes=raw_shapes if dynamic else per_expert_shapes,
                device=device,
                raw_layout=dynamic,
            )
            source = store
        pipeline = ColdExpertPipeline(
            store=source,
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
        # Release what the failed build already owns -- most importantly the
        # direct source's registrar (pinned pages + VRAM page tables), which
        # has no finalizer and would otherwise leak until process exit.
        for _owned in (direct_source, export_source):
            if _owned is not None:
                try:
                    _owned.close()
                except Exception:
                    logger.exception("[split-prefill] teardown after failed build")
        store = source = pipeline = export_source = direct_source = None

    # Arming is all-or-nothing ACROSS RANKS. The hot gate is rank-local, and a
    # split rank set is silent corruption, not a crash: armed ranks contribute
    # full-expert shards while a disarmed rank contributes its margin-routed
    # resident-only shard, and the row-parallel all-reduce sums them into
    # every output token of every large prefill. A single rank's build is the
    # likeliest thing in this file to fail alone (its arena share can degrade
    # per rank BY DESIGN, sending only it into the 52 GiB pinned-store
    # fallback), so unanimity is decided with the same symmetric consensus
    # every other rank-divergence risk here uses. Reached from the success
    # AND failure paths, so it cannot itself desynchronise.
    if not _all_tp_ranks_succeeded(pipeline is not None):
        if pipeline is not None:
            logger.error(
                "[split-prefill] disarmed: another rank failed to build; "
                "every rank keeps the margin-routed CPU path"
            )
        if export_source is not None:
            try:
                export_source.close()
            except Exception:
                logger.exception("[kt-export] teardown on disarm failed")
        if direct_source is not None:
            try:
                direct_source.close()
            except Exception:
                logger.exception("[kt-dma] teardown on disarm failed")
        for method, _ in _KT_SPLIT_PREFILL_LAYERS:
            method._split_prefill_ready = False
        _KT_SPLIT_PREFILL_STATE["store"] = None
        _KT_SPLIT_PREFILL_STATE["pipeline"] = None
        _KT_SPLIT_PREFILL_STATE["export_source"] = None
        _KT_SPLIT_PREFILL_STATE["direct_source"] = None
        return False

    _KT_SPLIT_PREFILL_STATE["store"] = store
    _KT_SPLIT_PREFILL_STATE["pipeline"] = pipeline
    _KT_SPLIT_PREFILL_STATE["export_source"] = export_source
    _KT_SPLIT_PREFILL_STATE["direct_source"] = direct_source
    # The swap path needs these too: against a raw store a promotion must
    # swizzle on the way to the GPU and a demotion must unswizzle on the way
    # back, or the two sides silently disagree about layout.
    _KT_SPLIT_PREFILL_STATE["swizzle_plan"] = swizzle_plan
    _KT_SPLIT_PREFILL_STATE["swizzle_inverse"] = None
    _KT_SPLIT_PREFILL_STATE["raw_scale_shapes"] = (
        None
        if not dynamic
        else (
            tuple(raw_shapes["w13_weight_scale"][0]),
            tuple(raw_shapes["w2_weight_scale"][0]),
        )
    )
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
        (
            "direct-dma"
            if direct_source is not None
            else "ring-export"
            if export_source is not None
            else "kt-arena"
            if store is None
            else "pinned-store"
        ),
    )
    return True


def _start_demotion_prefetch(anchor, entries):
    """Read what the NEXT window will demote, on a background thread.

    Under cold-only residency a demoted expert owns no CPU buffers, so the
    window must give it some before it becomes routable, and those bytes come
    off the checkpoint: ~17.5 MB per demotion, 8 swaps x 92 layers, against a
    measured window of ~31 s over a 7.1 s baseline. The plan is knowable one
    boundary early -- `select` is pure, it reads the EMAs and mutates nothing --
    so the reading need not sit on the critical path at all.

    Best-effort, and deliberately so. A plan that changes between here and the
    acting window simply misses the cache and is read synchronously. Nothing
    blocks on this thread, and crucially NO COLLECTIVE is involved: a rank that
    skips the prefetch, or finishes late, cannot desynchronise the group. That
    is the whole reason this is a safer shape than gathering the same bytes off
    the GPU, which deadlocked three times for exactly that reason.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    state = _KT_SWAP_STATE.get("prefetch")
    if state is not None and not state["done"].is_set():
        return  # one in flight already
    if not entries:
        return
    method0 = entries[0].get("method")
    if method0 is None or not method0.kt_config.cold_only_cpu_experts:
        return  # nothing to install: the demoted expert keeps its CPU buffers
    # Only the rank that installs needs the bytes. Safe to vary by rank here
    # precisely because there is no collective below.
    if method0.wrapper is None:
        return
    mover = _get_or_create_expert_mover(anchor)
    if mover is None:
        return

    plan = []
    for entry in entries:
        try:
            swaps = entry["policy"].select(entry["tables"].gpu_experts_mask)
        except Exception:
            continue
        for s in swaps:
            plan.append(
                (
                    entry["layer"],
                    entry["layer_idx"],
                    entry["method"]._kt_physical_to_logical,
                    s.demote,
                )
            )
    if not plan:
        return

    done = threading.Event()
    data: dict = {}
    _KT_SWAP_STATE["prefetch"] = {"done": done, "data": data, "planned": len(plan)}

    def _read_one(item):
        layer, layer_idx, p2l, demote_id = item
        try:
            # Cache key stays the PLAN id (that is what the window looks
            # up); only the checkpoint read arg translates.
            data[(layer_idx, demote_id)] = mover.read_full_expert(
                layer, _checkpoint_id(p2l, demote_id)
            )
        except Exception:
            pass  # a miss just costs a synchronous read later

    def _work():
        t0 = time.perf_counter()
        try:
            # PARALLEL, and the parallelism is the point. A window demotes
            # 8 experts x 92 layers = 736 FULL experts (kt needs every NUMA
            # partition, not this rank's slice) = ~12.9 GB, and it arrives
            # as ~4.4k scattered 2-3 MB tensor ranges out of mmap'd
            # safetensors. Read serially that is page-fault-bound at ~0.6
            # GB/s -- the ~22 s of fetch measured in a ~31 s window -- on an
            # NVMe that streams several GB/s. The reads release the GIL and
            # are independent, so overlapping them is what actually lets the
            # device be the limit. (The GPU read-back route avoids the disk
            # entirely and is faster still; this is the floor when it is
            # off or misses.)
            with ThreadPoolExecutor(
                max_workers=8, thread_name_prefix="kt-swap-prefetch"
            ) as pool:
                list(pool.map(_read_one, plan))
        finally:
            _KT_SWAP_STATE["prefetch"]["seconds"] = time.perf_counter() - t0
            done.set()

    threading.Thread(
        target=_work, name="kt-swap-prefetch", daemon=True
    ).start()
    logger.info(
        "[kt-swap] prefetching %d demoted experts off the critical path "
        "(8 reader threads)",
        len(plan),
    )


def _checkpoint_id(p2l, plan_id):
    """Swap-plan ids are kt buffer SLOTS (physical); the checkpoint is logical.

    Every checkpoint read keyed by a plan id goes through this, so the arena
    source (slot-indexed by construction) and the checkpoint fallback name the
    SAME expert for one plan id. Identity map -> no-op, which is why the gap
    was invisible until now; a frequency-placement seed makes it real.
    """
    return int(plan_id) if p2l is None else int(p2l[plan_id])


def _arena_stage_row(source, expert_id):
    """One promoted expert's checkpoint-layout shard, keyed for _flush_moves.

    The keys are the resident param names because that is how _flush_moves
    stacks a batch; the VALUES are raw_shard's checkpoint-layout tensors, the
    exact shapes apply_batched_swizzle takes. Returns None on failure so the
    caller can fall back to the checkpoint path for this expert alone.
    """
    try:
        shard = source.raw_shard(expert_id)
    except Exception:
        logger.exception(
            "[kt-arena] raw_shard(%d) failed; this promotion falls back to "
            "the checkpoint",
            expert_id,
        )
        return None
    names = _MXFP4_TRTLLM_RESIDENT_PARAM_NAMES
    return {
        names[0]: shard["w13"],
        names[1]: shard["w13_scale"],
        names[2]: shard["w2"],
        names[3]: shard["w2_scale"],
    }


def _maybe_arm_arena_swizzle_plan(entries):
    """Build the batched swizzle plan when arena promotion will need it.

    Split prefill arms _KT_SPLIT_PREFILL_STATE at store-build time; the
    full-kt config has no store, so the first acting window pays for the plan
    here instead -- one 2.2 MB checkpoint read, shape-derived, serves every
    layer for the process lifetime. Purely local: no collective, and a failure
    only means promotions keep the checkpoint path.
    """
    from sglang.srt.layers.moe.kt_arena_share import arena_source_for

    if _KT_SPLIT_PREFILL_STATE.get("swizzle_plan") is not None:
        return
    if _KT_SPLIT_PREFILL_STATE.get("arena_swizzle_failed"):
        return
    if not any(
        arena_source_for(e["layer_idx"]) is not None for e in entries
    ):
        return
    anchor = entries[0]["method"]
    device = anchor.gpu_experts_mask_cuda.device
    raw_shapes, plan = _build_dynamic_swizzle_plan(anchor, device)
    if plan is None or raw_shapes is None:
        _KT_SPLIT_PREFILL_STATE["arena_swizzle_failed"] = True
        logger.error(
            "[kt-arena] could not build a swizzle plan; promotions keep the "
            "checkpoint path"
        )
        return
    _KT_SPLIT_PREFILL_STATE["swizzle_plan"] = plan
    _KT_SPLIT_PREFILL_STATE["swizzle_inverse"] = None
    _KT_SPLIT_PREFILL_STATE["raw_scale_shapes"] = (
        tuple(raw_shapes["w13_weight_scale"][0]),
        tuple(raw_shapes["w2_weight_scale"][0]),
    )
    from sglang.srt.layers.moe.kt_arena_share import arena_sources_summary

    logger.info(
        "[kt-arena] promotion from kt RAM armed on this rank (%s)",
        arena_sources_summary(),
    )


def _swizzle_promoted_rows(promoted):
    """Checkpoint-layout promoted rows -> the resident trtllm layout, on GPU.

    Against a raw store this is what replaces asking kt for a GPU-layout
    export. kt's own export costs 57-68 ms per layer's cold set, ~38.7 ms of it
    CPU work reading AMX buffers and expanding E8M0 scales to bf16 -- work this
    path would immediately undo, since the resident layout wants those codes
    back as bytes. Doing the transform on device instead touches no CPU at all
    and reuses the same four-gather form split prefill uses.
    """
    from sglang.srt.layers.moe.kt_mxfp4_export import apply_batched_swizzle

    plan = _KT_SPLIT_PREFILL_STATE.get("swizzle_plan")
    if plan is None:
        raise RuntimeError(
            "raw cold store but no swizzle plan: promotions cannot be written "
            "to a resident row without one"
        )
    names = _MXFP4_TRTLLM_RESIDENT_PARAM_NAMES
    out = apply_batched_swizzle(
        plan=plan,
        raw_w13=promoted[names[0]],
        raw_w13_scale=promoted[names[1]],
        raw_w2=promoted[names[2]],
        raw_w2_scale=promoted[names[3]],
    )
    return dict(zip(names, out))


def _unswizzle_demoted_rows(staged, dtypes):
    """Resident trtllm rows -> checkpoint layout, for writing back to a raw store.

    Per expert rather than batched: a window demotes at most a handful per
    layer (8 at the default swap-max), so ~52 us each is nothing, and the
    per-expert inverse is the form already proved bitwise.
    """
    from sglang.srt.layers.moe.kt_mxfp4_export import (
        trtllm_inverse_indices,
        trtllm_permute_indices,
        unswizzle_trtllm_expert,
    )

    names = _MXFP4_TRTLLM_RESIDENT_PARAM_NAMES
    shapes = _KT_SPLIT_PREFILL_STATE.get("raw_scale_shapes")
    if shapes is None:
        raise RuntimeError("raw cold store but no raw scale shapes recorded")
    w13_scale_shape, w2_scale_shape = shapes

    # staged is PINNED HOST memory, but the permutations and the interleave
    # are CUDA-only, so the transform runs on the plan's device and the result
    # comes back to host for write_row.
    plan = _KT_SPLIT_PREFILL_STATE.get("swizzle_plan")
    if plan is None:
        raise RuntimeError("raw cold store but no swizzle plan for the inverse")
    dev = plan.w13_rows.device
    staged = {n: staged[n].to(dev, non_blocking=True) for n in names}

    inverse = _KT_SPLIT_PREFILL_STATE.get("swizzle_inverse")
    if inverse is None:
        # Built lazily and cached: the first demotion of the process pays for
        # it, and it is shape-derived so it serves every layer thereafter.
        indices = trtllm_permute_indices(
            w13_sample=torch.empty(
                (w13_scale_shape[0], w13_scale_shape[1] * 16),
                dtype=torch.uint8,
                device=dev,
            ),
            w13_scale_sample=torch.empty(
                w13_scale_shape, dtype=torch.uint8, device=dev
            ),
            w2_sample=torch.empty(
                (w2_scale_shape[0], w2_scale_shape[1] * 16),
                dtype=torch.uint8,
                device=dev,
            ),
            w2_scale_sample=torch.empty(
                w2_scale_shape, dtype=torch.uint8, device=dev
            ),
            w13_gate_up_halves=True,
        )
        inverse = trtllm_inverse_indices(
            indices,
            w13_scale_shape=w13_scale_shape,
            w2_scale_shape=w2_scale_shape,
            device=dev,
        )
        _KT_SPLIT_PREFILL_STATE["swizzle_inverse"] = inverse

    n_rows = staged[names[0]].shape[0]
    out = {n: [] for n in names}
    for i in range(n_rows):
        raw = unswizzle_trtllm_expert(
            w13=staged[names[0]][i],
            w13_scale=staged[names[1]][i],
            w2=staged[names[2]][i],
            w2_scale=staged[names[3]][i],
            inverse=inverse,
            w13_scale_shape=w13_scale_shape,
            w2_scale_shape=w2_scale_shape,
        )
        for n, t in zip(
            names, (raw.w13, raw.w13_scale_e8m0, raw.w2, raw.w2_scale_e8m0)
        ):
            out[n].append(t.view(dtypes[n]))
    return {n: torch.stack(v).to("cpu") for n, v in out.items()}


def _build_dynamic_swizzle_plan(anchor, device):
    """Raw per-expert shapes and the per-layer swizzle maps, from one sample.

    Both are shape-derived, so a single expert read settles them for every
    layer. Returns ``(raw_shapes, plan)`` in ColdExpertStore's WEIGHT_NAMES
    order, or ``(None, None)`` if anything is missing -- in which case the
    caller falls back to the pre-swizzled store rather than guessing.
    """
    from sglang.srt.layers.moe.expert_cold_store import WEIGHT_NAMES
    from sglang.srt.layers.moe.kt_expert_mover import (
        CheckpointExpertReader,
        build_expert_bytes,
    )
    from sglang.srt.layers.moe.kt_mxfp4_export import (
        trtllm_batched_swizzle,
        trtllm_permute_indices,
    )

    try:
        cold = torch.where(~anchor.gpu_experts_mask)[0].tolist()
        if not cold:
            return None, None
        reader = CheckpointExpertReader(anchor.kt_config.weight_path)
        try:
            li = anchor.kt_config.layer_idx
            sample = build_expert_bytes(
                reader,
                f"language_model.model.layers.{li}.block_sparse_moe.experts",
                cold[0],
                tp_rank=get_parallel().tp_rank,
                tp_size=get_parallel().tp_size,
            )
        finally:
            reader.close()
        on_dev = type(sample)(
            w13=sample.w13.to(device),
            w13_scale_e8m0=sample.w13_scale_e8m0.to(device),
            w2=sample.w2.to(device),
            w2_scale_e8m0=sample.w2_scale_e8m0.to(device),
        )
        raw_shapes = {
            n: (tuple(t.shape), t.dtype)
            for n, t in zip(
                WEIGHT_NAMES,
                (
                    on_dev.w13,
                    on_dev.w13_scale_e8m0,
                    on_dev.w2,
                    on_dev.w2_scale_e8m0,
                ),
            )
        }
        indices = trtllm_permute_indices(
            w13_sample=on_dev.w13,
            w13_scale_sample=on_dev.w13_scale_e8m0,
            w2_sample=on_dev.w2,
            w2_scale_sample=on_dev.w2_scale_e8m0,
            w13_gate_up_halves=True,
        )
        plan = trtllm_batched_swizzle(
            indices,
            w13_scale_shape=tuple(on_dev.w13_scale_e8m0.shape),
            w2_scale_shape=tuple(on_dev.w2_scale_e8m0.shape),
            device=device,
        )
        logger.info(
            "[split-prefill] dynamic swizzle armed: raw w13 %s, w2 %s",
            tuple(on_dev.w13.shape),
            tuple(on_dev.w2.shape),
        )
        return raw_shapes, plan
    except Exception:
        logger.exception(
            "[split-prefill] could not build the dynamic swizzle plan; using "
            "the pre-swizzled cold store"
        )
        return None, None


def _get_or_create_gpu_reader():
    """Process-wide reader for demoted experts; None if disabled or unbuildable.

    WHY IT IS WORTH USING. A window demotes 8 experts x 92 layers = 736 FULL
    experts, ~12.9 GB, and the disk route reads that as ~4.4k scattered
    2-3 MB ranges out of mmap'd safetensors (measured ~22 s of fetch in a
    ~31 s window). The bytes are already in VRAM -- they are the resident
    rows about to be overwritten -- so this route unswizzles them and
    all-gathers the full expert over NVLink instead, which is microseconds
    of transfer against seconds of disk.

    WHY IT WAS OFF, AND WHY IT IS ON NOW. The read-back is proved bitwise,
    but it reaches the full expert through a TP all-gather, and the install
    path it lived in had data-dependent early-outs (no cold-store slot, a
    row already written, a disabled route). Ranks that disagreed on how many
    collectives to run deadlocked the window rather than falling back --
    observed as an _ALLGATHER_BASE timing out after 600 s with the NCCL
    watchdog killing the process group. The condition for re-enabling was
    that the collective count become a pure function of the per-layer swap
    plan, and it now is: `_begin_layer` runs ONCE per layer with the whole
    plan, and its decision goes through `_all_tp_ranks_succeeded(want)`,
    which every rank executes whether or not it consumes the gather. A
    residual disagreement therefore degrades to "no rank uses the GPU route
    on this layer" instead of hanging. SGLANG_KT_SWAP_GPU_READBACK=0 still
    forces the disk route if a window ever misbehaves.
    """
    if not envs.SGLANG_KT_SWAP_GPU_READBACK.get():
        return None
    reader = _KT_SWAP_STATE.get("gpu_reader")
    if reader is None:
        try:
            reader = _GpuResidentExpertReader(_MXFP4_TRTLLM_RESIDENT_PARAM_NAMES)
            _KT_SWAP_STATE["gpu_reader"] = reader
        except Exception:
            logger.exception("[kt-swap] could not build the GPU expert reader")
            return None
    return reader


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


def _verify_rank_write_once(entries, writer, mover) -> None:
    """SGLANG_KT_VERIFY_CPU_INSTALL=1: prove a rank-written expert bitwise.

    THE gate for this path. Eight ranks each wrote a disjoint slice of an
    expert directly into kt's buffers; every way that can be wrong -- a
    partition off by one, a rank's rows landing at another rank's offset, a
    strip written at the wrong pitch -- produces a valid-looking expert
    holding wrong weights, which nothing downstream fails on. kt's own
    verifier rebuilds the expert from the CHECKPOINT through the same
    fill_expert_buffers the old install used and compares every NUMA
    partition byte for byte, so it answers exactly the question the unit
    tests answer in simulation, on the real thing.

    Once per process, one expert, after the window's barrier. Rank 0 only --
    it owns kt -- and read-only with respect to serving state.
    """
    import os

    if os.environ.get("SGLANG_KT_VERIFY_CPU_INSTALL") != "1":
        return
    if _KT_SWAP_STATE.get("rank_write_verified"):
        return
    if writer.last_installed is None or mover is None:
        return
    layer_idx, expert_id = writer.last_installed
    entry = next(
        (e for e in entries if e.get("layer_idx") == layer_idx), None
    )
    if entry is None:
        return
    method = entry.get("method")
    if method is None or method.wrapper is None:
        return  # peers hold no kt engine to verify against
    _KT_SWAP_STATE["rank_write_verified"] = True
    try:
        tensors = mover.read_full_expert(
            entry["layer"], _checkpoint_id(method._kt_physical_to_logical, expert_id)
        )
        ok = method.wrapper.verify_install_against_loaded(
            expert_id, *[t.data_ptr() for t in tensors]
        )
    except Exception:
        logger.exception("[kt-rankwrite] the bitwise check itself failed")
        return
    if ok:
        logger.info(
            "[kt-rankwrite] expert %d (layer %d): the eight ranks' writes "
            "BITWISE-MATCH the checkpoint on every NUMA partition",
            expert_id,
            layer_idx,
        )
    else:
        logger.error(
            "[kt-rankwrite] expert %d (layer %d): DIFFERS from the checkpoint "
            "-- demoted experts are being given wrong weights; set "
            "SGLANG_KT_DEMOTION_RANK_WRITE=0 and re-check the offset math",
            expert_id,
            layer_idx,
        )


def _get_or_create_rank_writer(entry):
    """Process-wide rank-write demotion writer, or None (checkpoint path).

    Built once, on the first layer of the first window that could use it.
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
        if (
            envs.SGLANG_KT_DEMOTION_RANK_WRITE.get()
            and method is not None
            and method.kt_config.cold_only_cpu_experts
        ):
            from sglang.srt.layers.moe.kt_arena_share import (
                arena_write_source_for,
            )
            from sglang.srt.layers.moe.kt_demotion_writer import (
                RankShardWriter,
                SlotOffsets,
            )
            from sglang.srt.layers.moe.kt_direct_dma import ArenaExpertRanges

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
            writer = RankShardWriter(
                arena_by_layer={
                    li: s._arenas[geom.part] for li, s in sources.items()
                },
                offsets_by_layer={
                    li: SlotOffsets(s._rows, experts=s.experts, numa=s.numa)
                    for li, s in sources.items()
                },
                geometry=geom,
                shard_reader=_GpuResidentExpertReader(
                    _MXFP4_TRTLLM_RESIDENT_PARAM_NAMES
                ),
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
            "[kt-rankwrite] could not arm; demotions keep the checkpoint path"
        )
        writer = None

    _KT_SWAP_STATE["rank_writer"] = writer
    return writer


def _get_or_create_expert_mover(anchor):
    mover = _KT_SWAP_STATE.get("mover")
    if mover is None:
        try:
            mover = _PerLayerMover(
                anchor.kt_config.weight_path,
                get_parallel().tp_rank,
                get_parallel().tp_size,
            )
            _KT_SWAP_STATE["mover"] = mover
        except Exception:
            logger.exception("[kt-swap] could not build the expert mover")
            return None
    return mover


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
