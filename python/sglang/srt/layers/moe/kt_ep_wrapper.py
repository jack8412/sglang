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
    expert_swap_interval: int = 0
    expert_swap_max: int = 4
    expert_swap_hysteresis: float = 2.0


# Process-level registries for the MXFP4 layerwise-prefill slot machinery
# (mutated in place only — no module rebinding; carried from ktfork #72/#73).
# Every wrapped MoE layer, in construction order, so the swap driver can walk
# them. Registered unconditionally: the layerwise-prefill registry below is
# populated only when gpu_prefill_token_threshold > 0, and the validated
# recipe runs 0, which would leave a swap driver with nothing to iterate.
_KT_EP_METHODS = []
_KT_SWAP_STATE = {"eager_forwards": 0, "windows": 0, "swaps": 0}

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
        expert_swap_interval=server_args.kt_expert_swap_interval,
        expert_swap_max=server_args.kt_expert_swap_max,
        expert_swap_hysteresis=server_args.kt_expert_swap_hysteresis,
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

        # CUDA tensors for inference (will be set in create_weights)
        self.gpu_experts_mask_cuda = None
        self.logical_to_gpu_index_cuda = None

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
        if self._margin is not None:
            self._margin_insist_count = torch.zeros(
                num_experts, dtype=torch.int32, device=target_device
            )
            self._margin_override_count = torch.zeros(
                num_experts, dtype=torch.int32, device=target_device
            )
            self._resident_hit_count = torch.zeros(
                num_experts, dtype=torch.int32, device=target_device
            )

        # Initialize dual-stream for CPU-GPU parallelism (rank 0 only)
        if self.tp_rank == 0:
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
        if self.tp_rank == 0:
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

        # 2. Load CPU weights using KT wrapper
        if self.tp_rank == 0 and self.wrapper is not None:
            torch.cuda.synchronize()

            # Get expert location metadata for CPU expert mapping
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
            self.wrapper.load_weights(physical_to_logical_map_cpu)
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
        if self._margin is not None:
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
                        self._margin,
                        self._full_override,
                    )
                )
                _orig_safe_ids = (
                    topk_output.topk_ids.clamp_min(0).reshape(-1).to(torch.int64)
                )
                self._margin_insist_count.scatter_add_(
                    0, _orig_safe_ids, _insist_slots.reshape(-1).to(torch.int32)
                )
                self._margin_override_count.scatter_add_(
                    0, _orig_safe_ids, _override_slots.reshape(-1).to(torch.int32)
                )
                # Resident hits: slots the router picked that were ALREADY on
                # GPU (neither insisted nor overridden).  Counted on the
                # original ids for symmetry with the demand counters, so
                # "demand for a non-resident expert" and "traffic served by a
                # resident expert" are on the same scale.
                _routed = topk_output.topk_ids >= 0
                _resident_slots = (
                    _routed & ~_insist_slots & ~_override_slots
                )
                self._resident_hit_count.scatter_add_(
                    0, _orig_safe_ids, _resident_slots.reshape(-1).to(torch.int32)
                )
                # margin == 0.0 is count-only (documented flag contract):
                # counters record what WOULD override, routing stays exact.
                if self._full_override or self._margin > 0.0:
                    topk_output = topk_output._replace(topk_ids=new_topk_ids)
                    dispatch_output = dispatch_output._replace(
                        topk_output=topk_output
                    )
                self._maybe_log_margin_stats()
                self._maybe_verify_expert_mover(layer)
                # The first registered layer drives the swap window for the
                # whole model: later layers have not read their membership
                # yet this forward, so one window keeps the batch consistent.
                if (
                    self.kt_config.expert_swap_interval > 0
                    and _KT_EP_METHODS
                    and _KT_EP_METHODS[0] is self
                    and not torch.cuda.is_current_stream_capturing()
                ):
                    try:
                        maybe_run_expert_swap_window(self)
                    except Exception:
                        logger.exception("[kt-swap] window failed; serving continues")
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
        if self.tp_rank == 0 and self._cpu_stream is not None and not self._skip_cpu_path:
            # Use shared staging buffer (shared across all MoE layers to save GPU memory)
            assert self._shared_staging_buffer is not None, "Shared staging buffer not initialized"
            staging_buffer = self._shared_staging_buffer.get_slice(x.shape[0])

            # Copy to staging buffer on main stream
            staging_buffer.copy_(x, non_blocking=True)

            # SGLANG_DISABLE_KT_CPU_STREAM=1 collapses cpu_stream onto main stream.
            _no_cpu_stream = self._kt_no_cpu_stream
            if not _no_cpu_stream:
                # Fork to cpu_stream (waits for staging copy to complete)
                self._cpu_stream.wait_stream(torch.cuda.current_stream(x.device))
            from contextlib import nullcontext as _ctx_null
            _stream_ctx = _ctx_null() if _no_cpu_stream else torch.cuda.stream(self._cpu_stream)
            with _stream_ctx:
                # Submit uses staging_buffer, so GPU can modify original x freely
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
            _stream_ctx = _ctx_null() if _no_cpu_stream else torch.cuda.stream(self._cpu_stream)
            with _stream_ctx:
                # Use staging_buffer for sync to get correct buffer reference
                _kt_t_sync_pre = time.perf_counter() if _kt_t_apply_start is not None else None
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
    )


def maybe_run_expert_swap_window(anchor: "KTEPWrapperMethod") -> None:
    """Quiesce and re-cut expert membership, every N eager forwards.

    Called from the FIRST registered layer's apply(). Two properties make that
    a legal place. It only runs eagerly (prefill), so Python is actually
    executing and a device sync is allowed — under a captured decode replay
    none of this code runs at all. And membership is consumed exclusively
    *inside* apply(): the margin override, the id remap, and kt-kernel's own
    mask read. A later layer's tables are therefore not yet read this forward,
    and the anchor layer re-reads its own below, so the whole model sees one
    consistent membership for the batch.

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
    if _KT_SWAP_STATE["eager_forwards"] % cfg.expert_swap_interval:
        return

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
    if not entries:
        return

    mover = _get_or_create_expert_mover(anchor)
    if mover is None:
        return

    def _move(layer, dst_row, logical_id):
        mover.move(layer, dst_row, logical_id)

    result = run_swap_window(
        entries,
        move_weights=_move,
        quiesce=lambda: torch.cuda.synchronize(anchor.gpu_experts_mask_cuda.device),
    )
    _KT_SWAP_STATE["windows"] += 1
    _KT_SWAP_STATE["swaps"] += result.swaps_applied
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


class _PerLayerMover:
    """One CheckpointExpertMover per layer (permute indices are per-shape)."""

    def __init__(self, weight_path, tp_rank, tp_size):
        self._by_layer = {}
        self._args = (weight_path, tp_rank, tp_size)

    def move(self, layer, dst_row, logical_id):
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
        mover(layer, dst_row, logical_id)


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
