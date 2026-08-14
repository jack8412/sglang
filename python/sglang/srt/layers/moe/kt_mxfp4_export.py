# SPDX-License-Identifier: Apache-2.0
"""Shared MXFP4 expert-byte extraction + per-target swizzle backends.

One data movement, several target layouts (k3-hybrid F1/F2): kt-kernel's
``write_weight_scale_to_buffer`` export emits, per expert and per GPU-TP
shard,

  * raw FP4 nibble-packed weight bytes — a straight ``memcpy`` of the
    resident BufferB (``operators/amx/fp4-moe.hpp::write_weights_to_buffer``),
    laid out ``w13 = [gate | up]`` and ``w2 = down``; and
  * group scales as **bf16 expanded from the resident E8M0 codes**
    (``fast_e8m0_to_bf16``) — the export keeps its bf16 contract across the
    fp32-scale and E8M0-resident layouts.

kt-kernel's E8M0<->bf16 convention is a lossless bijection (``code << 7`` are
the exact bf16 bits; code 0 -> +0.0, code 255 -> +inf, per the repo's helpers
rather than the OCP NaN), so this module *recovers* the raw codes exactly —
with a hard assertion — instead of re-deriving them numerically.

Consumers:
  * F1 (layerwise prefill, K3): trtllm-gen slot preparation for the resident
    ``Mxfp4MoEMethod`` (the SiTU-capable B200 kernel). The marlin target
    (DSV4's ``DeepSeekMxfp4MoEMethod``) stays in ``v4_marlin_moe`` and is
    delegated to, not duplicated.
  * F2 (dynamic expert update): per-expert re-swizzle into the resident GPU
    method's shuffled parameter slots. Valid per-expert because both the
    trtllm shuffle and the block-scale interleave permute within one expert.
"""

import math
from typing import Optional, Sequence

import msgspec
import torch

# E8M0 code semantics (kt-kernel convention, NOT the OCP-spec NaN at 255).
_E8M0_MANTISSA_MASK = 0x7F
_MXFP4_GROUP_SIZE = 32


class Mxfp4ExpertBytes(msgspec.Struct):
    """Raw exported bytes for one expert (one GPU-TP shard).

    ``w13`` is ``[gate | up]`` nibble-packed uint8 of logical shape
    ``[2 * intermediate, hidden // 2]``; ``w2`` is ``[hidden,
    intermediate // 2]``. Scales are E8M0 codes (uint8), one per 32-element
    k-group: ``[2 * intermediate, hidden // 32]`` and
    ``[hidden, intermediate // 32]``.
    """

    w13: torch.Tensor
    w13_scale_e8m0: torch.Tensor
    w2: torch.Tensor
    w2_scale_e8m0: torch.Tensor


def bf16_scales_to_e8m0(scales_bf16: torch.Tensor) -> torch.Tensor:
    """Recover the resident E8M0 codes from the export's bf16 scale buffer.

    Exact inverse of kt-kernel's ``e8m0_to_bf16`` (bits are ``code << 7``).
    Fails loudly if any value is not representable — a corrupted buffer, a
    wrong dtype, or an fp32-scale wheel (whose scales are arbitrary bf16
    values) all trip the assertion rather than silently rounding.
    """
    if scales_bf16.dtype != torch.bfloat16:
        raise TypeError(f"expected bf16 export scales, got {scales_bf16.dtype}")
    bits = scales_bf16.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    if not bool((bits & _E8M0_MANTISSA_MASK).eq(0).all()) or not bool(
        (bits >> 15).eq(0).all()
    ):
        raise ValueError(
            "export scale buffer contains values outside the E8M0 range "
            "(mantissa/sign bits set) — not an E8M0-resident kt-kernel export"
        )
    return (bits >> 7).to(torch.uint8)


def e8m0_to_bf16(codes: torch.Tensor) -> torch.Tensor:
    """kt-kernel's expansion convention (code 0 -> +0.0, 255 -> +inf)."""
    return (codes.to(torch.int16) << 7).view(torch.bfloat16)


def dequant_mxfp4_ref(
    packed: torch.Tensor, scales_e8m0: torch.Tensor
) -> torch.Tensor:
    """Pure-Python fp32 reference: unpack FP4 nibbles and apply E8M0 group
    scales. Built on the repo's canonical MXFP4QuantizeUtil nibble
    convention so layout tests compare against the same decode the kernels
    are validated with."""
    from sglang.srt.layers.quantization.mxfp4_tensor import MXFP4QuantizeUtil

    n, half_k = packed.shape
    k = half_k * 2
    out = MXFP4QuantizeUtil.dequantize(
        packed, dtype=torch.float32, scale=scales_e8m0, block_sizes=[32]
    ).reshape(n, k)
    # The util computes exp2(code - 127) for every code; kt's convention maps
    # code 0 to +0.0 (not 2^-127) — mask those groups to match.
    zero_groups = scales_e8m0 == 0
    if bool(zero_groups.any()):
        keep = (~zero_groups).to(torch.float32)
        out = out * keep.repeat_interleave(_MXFP4_GROUP_SIZE, dim=1)[:, :k]
    return out


def expert_bytes_digest(bytes_: Mxfp4ExpertBytes) -> str:
    """Order-stable content digest over all four tensors (test hook: a single
    perturbed scale byte must change it)."""
    import hashlib

    h = hashlib.sha256()
    for t in (bytes_.w13, bytes_.w13_scale_e8m0, bytes_.w2, bytes_.w2_scale_e8m0):
        h.update(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Target-layout backends. Extraction happens once (above); each backend only
# permutes into its kernel's storage order.
# ---------------------------------------------------------------------------


def swizzle_marlin(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    out=None,
):
    """DSV4 target: delegate to the existing marlin preparation (accepts
    E8M0 or bf16 scales via its SRC_IS_E8M0 kernel switch)."""
    from sglang.srt.layers.quantization.v4_marlin_moe import prepare_v4_mxfp4_marlin

    return prepare_v4_mxfp4_marlin(w13, w13_scale, w2, w2_scale, out=out)


class TrtllmPermuteIndices(msgspec.Struct):
    """Shape-derived permutations for the trtllm-gen shuffled layout.

    Computed once per (shape, device) from a sample expert; both the weight
    shuffle and the scale interleave are per-expert-independent, which is
    what makes single-expert slot writes (F2) equivalent to the full
    process_weights_after_loading pass."""

    w13_weight: torch.Tensor
    w13_scale: torch.Tensor
    w2_weight: torch.Tensor
    w2_scale: torch.Tensor


def trtllm_permute_indices(
    w13_sample: torch.Tensor,
    w13_scale_sample: torch.Tensor,
    w2_sample: torch.Tensor,
    w2_scale_sample: torch.Tensor,
    *,
    epilogue_tile_m: int = 128,
    w13_gate_up_halves: bool = False,
) -> TrtllmPermuteIndices:
    """Row-permutation indices for the trtllm-gen shuffled layout.

    With ``w13_gate_up_halves=True`` the source w13 rows are ``[gate | up]``
    halves (the kt-export orientation, matching K3's non-interleaved
    checkpoints).  The trtllm-gen fused gated-act epilogue consumes
    ``(up_i, gate_i)`` row *pairs* before the shuffle, so the pair reorder
    from ``Mxfp4MoEMethod.process_weights_after_loading``
    (``mxfp4.py`` L716-730) is composed into the w13 weight/scale indices:
    ``x[pair][shuffle] == x[pair[shuffle]]``.
    """
    from sglang.srt.layers.quantization.mxfp4 import (
        _get_flashinfer_mxfp4_device_permute_indices,
    )

    w13_weight_indices = _get_flashinfer_mxfp4_device_permute_indices(
        w13_sample.view(torch.uint8), epilogue_tile_m
    )
    w13_scale_indices = _get_flashinfer_mxfp4_device_permute_indices(
        w13_scale_sample.view(torch.uint8), epilogue_tile_m, num_elts_per_sf=16
    )
    if w13_gate_up_halves:
        rows = w13_sample.shape[0]
        half = rows // 2
        pair = torch.empty(
            rows, dtype=torch.long, device=w13_weight_indices.device
        )
        pair[0::2] = torch.arange(half, device=pair.device) + half  # up (w3)
        pair[1::2] = torch.arange(half, device=pair.device)  # gate (w1)
        w13_weight_indices = pair[w13_weight_indices]
        w13_scale_indices = pair[w13_scale_indices]
    return TrtllmPermuteIndices(
        w13_weight=w13_weight_indices,
        w13_scale=w13_scale_indices,
        w2_weight=_get_flashinfer_mxfp4_device_permute_indices(
            w2_sample.view(torch.uint8), epilogue_tile_m
        ),
        w2_scale=_get_flashinfer_mxfp4_device_permute_indices(
            w2_scale_sample.view(torch.uint8), epilogue_tile_m, num_elts_per_sf=16
        ),
    )


class TrtllmBatchedSwizzle(msgspec.Struct):
    """Forward swizzle for a WHOLE LAYER's experts, in four indexed gathers.

    Granularity is the entire point. Measured on this node, one expert's TP8
    shard swizzles in ~52 us, which is launch-bound rather than
    bandwidth-bound: at 2.19 MB per expert that is only ~42 GB/s. Issued once
    per expert, split prefill's 276 cold experts x 92 layers would cost ~1.32 s
    per forward against a ~2.07 s copy floor -- +64%, which is not affordable.
    Issued once per LAYER over all its experts it costs ~1.0 ms per layer,
    ~0.092 s per forward, +4.4%. Same bytes, 14x apart.

    Both parts are expressed as plain index maps so a layer is four gathers:

      weights  out[e] = raw[e][rows]                 -> raw[:, rows, :]
      scales   out[e] = interleave(raw[e][rows])     -> raw.view(E,-1)[:, map]

    The scale map folds the row permutation and the interleave together, so
    ``nvfp4_block_scale_interleave`` is never called on the hot path -- it has
    no batched form, and calling it per expert is exactly the cost being
    avoided. The fold is exact because the interleave is a pure byte
    permutation (:func:`recover_interleave_map` asserts that).
    """

    w13_rows: torch.Tensor
    w2_rows: torch.Tensor
    w13_scale_map: torch.Tensor
    w2_scale_map: torch.Tensor
    w13_scale_out_shape: tuple
    w2_scale_out_shape: tuple


def _fold_permute_into_interleave(
    row_indices: torch.Tensor, scale_shape, device
) -> tuple:
    """Flat map for ``interleave(x[row_indices])``, plus the output shape.

    ``interleave(y).flat[q] == y.flat[src[q]]`` and
    ``y.flat[j] == x.flat[row_indices[j // C] * C + j % C]``, so composing them
    gives one gather over the raw expert.
    """
    src = recover_interleave_map(scale_shape, device)
    cols = int(scale_shape[1])
    rows_for = row_indices.to(device)[torch.div(src, cols, rounding_mode="floor")]
    folded = rows_for * cols + (src % cols)
    return folded, (int(src.numel()),)


def trtllm_batched_swizzle(
    indices: TrtllmPermuteIndices,
    *,
    w13_scale_shape,
    w2_scale_shape,
    device,
) -> TrtllmBatchedSwizzle:
    """Build the per-layer swizzle maps. Shape-derived, so cache per shape."""
    w13_map, w13_out = _fold_permute_into_interleave(
        indices.w13_scale, w13_scale_shape, device
    )
    w2_map, w2_out = _fold_permute_into_interleave(
        indices.w2_scale, w2_scale_shape, device
    )
    return TrtllmBatchedSwizzle(
        w13_rows=indices.w13_weight.to(device),
        w2_rows=indices.w2_weight.to(device),
        w13_scale_map=w13_map,
        w2_scale_map=w2_map,
        w13_scale_out_shape=w13_out,
        w2_scale_out_shape=w2_out,
    )


def apply_batched_swizzle(
    *,
    plan: TrtllmBatchedSwizzle,
    raw_w13: torch.Tensor,
    raw_w13_scale: torch.Tensor,
    raw_w2: torch.Tensor,
    raw_w2_scale: torch.Tensor,
) -> tuple:
    """Swizzle ``[num_experts, ...]`` raw stacks. Four gathers, no per-expert loop.

    Returns tensors shaped like the resident device buffers the MoE kernel
    reads, in ``WEIGHT_NAMES`` order.
    """
    e = raw_w13.shape[0]
    w13 = raw_w13.view(torch.uint8)[:, plan.w13_rows, :].contiguous()
    w2 = raw_w2.view(torch.uint8)[:, plan.w2_rows, :].contiguous()
    w13_scale = (
        raw_w13_scale.reshape(e, -1).view(torch.uint8)[:, plan.w13_scale_map]
        .reshape((e,) + plan.w13_scale_out_shape)
        .contiguous()
    )
    w2_scale = (
        raw_w2_scale.reshape(e, -1).view(torch.uint8)[:, plan.w2_scale_map]
        .reshape((e,) + plan.w2_scale_out_shape)
        .contiguous()
    )
    return w13, w13_scale, w2, w2_scale


class TrtllmInverseIndices(msgspec.Struct):
    """The inverse of :class:`TrtllmPermuteIndices`, for reading a slot back.

    Swapping demotes a GPU-resident expert, and under cold-only residency its
    CPU buffers must then be filled. Reading those bytes off the checkpoint
    costs ~17.5 MB per demotion; the GPU already holds them, just shuffled.
    These indices turn the shuffled row back into the exported layout, so the
    demoted expert can be installed from device memory instead of from disk.

    ``*_unlace`` are the inverses of ``nvfp4_block_scale_interleave``, which
    publishes none — see :func:`recover_interleave_map`.
    """

    w13_weight: torch.Tensor
    w13_scale: torch.Tensor
    w2_weight: torch.Tensor
    w2_scale: torch.Tensor
    w13_scale_unlace: torch.Tensor
    w2_scale_unlace: torch.Tensor


def recover_interleave_map(shape, device) -> torch.Tensor:
    """Recover ``src`` with ``interleave(x).flatten()[q] == x.flatten()[src[q]]``.

    ``nvfp4_block_scale_interleave`` is a fixed byte permutation with no
    published inverse, so the map is measured rather than derived: feed it
    tensors whose bytes encode each element's own flat position in base 256 —
    one pass per digit — and read the digits back out of the result.

    Raises if the result is not a permutation of ``range(n)``; every use of the
    map assumes the op neither drops nor duplicates a byte, and a layout change
    upstream must fail loudly here rather than silently install wrong weights.
    """
    from flashinfer import nvfp4_block_scale_interleave

    n = 1
    for d in shape:
        n *= int(d)
    pos = torch.arange(n, device=device)
    digits = 0
    while (1 << (8 * digits)) < max(n, 2):
        digits += 1

    src = torch.zeros(n, dtype=torch.int64, device=device)
    for j in range(digits):
        plane = ((pos >> (8 * j)) & 0xFF).to(torch.uint8).reshape(shape)
        out = nvfp4_block_scale_interleave(plane).reshape(-1).to(torch.int64)
        src = src | (out << (8 * j))

    if src.numel() != n or not torch.equal(
        torch.sort(src).values, torch.arange(n, device=device)
    ):
        raise RuntimeError(
            "nvfp4_block_scale_interleave is not a pure permutation for shape "
            f"{tuple(shape)}; the trtllm scale layout changed and unswizzling "
            "would produce wrong CPU weights"
        )
    return src


def trtllm_inverse_indices(
    indices: TrtllmPermuteIndices,
    *,
    w13_scale_shape,
    w2_scale_shape,
    device,
) -> TrtllmInverseIndices:
    """Invert one set of permute indices. Shape-derived, so cache per shape."""
    return TrtllmInverseIndices(
        w13_weight=torch.argsort(indices.w13_weight),
        w13_scale=torch.argsort(indices.w13_scale),
        w2_weight=torch.argsort(indices.w2_weight),
        w2_scale=torch.argsort(indices.w2_scale),
        w13_scale_unlace=recover_interleave_map(w13_scale_shape, device),
        w2_scale_unlace=recover_interleave_map(w2_scale_shape, device),
    )


def unswizzle_trtllm_expert(
    *,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    inverse: TrtllmInverseIndices,
    w13_scale_shape,
    w2_scale_shape,
) -> Mxfp4ExpertBytes:
    """Exact inverse of :func:`swizzle_trtllm_expert` for one resident row.

    Bitwise, on every tensor — proved against the forward path before this was
    wired in (``runs/meta/verify_unswizzle.py``). The forward is
    ``out = x[idx]`` for weights and ``out = interleave(x[idx])`` for scales,
    so the inverse un-interleaves first and then applies the inverse
    permutation.
    """

    def _unlace(flat_out, unlace, shape):
        back = torch.empty_like(flat_out)
        back[unlace] = flat_out  # out = in[src]  =>  in[src] = out
        return back.reshape(shape)

    return Mxfp4ExpertBytes(
        w13=w13.reshape(-1, w13.shape[-1]).view(torch.uint8)[inverse.w13_weight]
        .contiguous(),
        w13_scale_e8m0=_unlace(
            w13_scale.reshape(-1).view(torch.uint8),
            inverse.w13_scale_unlace,
            w13_scale_shape,
        )[inverse.w13_scale].contiguous(),
        w2=w2.reshape(-1, w2.shape[-1]).view(torch.uint8)[inverse.w2_weight]
        .contiguous(),
        w2_scale_e8m0=_unlace(
            w2_scale.reshape(-1).view(torch.uint8),
            inverse.w2_scale_unlace,
            w2_scale_shape,
        )[inverse.w2_scale].contiguous(),
    )


def swizzle_trtllm_expert(
    bytes_: Mxfp4ExpertBytes,
    indices: TrtllmPermuteIndices,
    *,
    out_w13: Optional[torch.Tensor] = None,
    out_w13_scale: Optional[torch.Tensor] = None,
    out_w2: Optional[torch.Tensor] = None,
    out_w2_scale: Optional[torch.Tensor] = None,
):
    """Permute one expert into the trtllm-gen shuffled layout, mirroring
    Mxfp4MoEMethod.process_weights_after_loading's per-expert body
    (weight.view(uint8)[indices]; nvfp4_block_scale_interleave(scale[idx])).

    With ``out_*`` given, writes in place (``copy_`` only — CUDA-graph-safe
    for F2's resident-slot updates) and returns the outputs."""
    from flashinfer import nvfp4_block_scale_interleave

    w13 = bytes_.w13.view(torch.uint8)[indices.w13_weight].contiguous()
    w13_scale = nvfp4_block_scale_interleave(
        bytes_.w13_scale_e8m0.view(torch.uint8)[indices.w13_scale].contiguous()
    )
    w2 = bytes_.w2.view(torch.uint8)[indices.w2_weight].contiguous()
    w2_scale = nvfp4_block_scale_interleave(
        bytes_.w2_scale_e8m0.view(torch.uint8)[indices.w2_scale].contiguous()
    )
    if out_w13 is not None:
        out_w13.view(torch.uint8).copy_(w13.view_as(out_w13.view(torch.uint8)))
        out_w13_scale.view(torch.uint8).copy_(
            w13_scale.view_as(out_w13_scale.view(torch.uint8))
        )
        out_w2.view(torch.uint8).copy_(w2.view_as(out_w2.view(torch.uint8)))
        out_w2_scale.view(torch.uint8).copy_(
            w2_scale.view_as(out_w2_scale.view(torch.uint8))
        )
        return out_w13, out_w13_scale, out_w2, out_w2_scale
    return w13, w13_scale, w2, w2_scale


class TrtllmPreparedWeights(msgspec.Struct):
    """One prepared full-layer image in the trtllm-gen shuffled layout.

    Mirrors ``Mxfp4MoEMethod.process_weights_after_loading``'s SM100
    flashinfer branch outputs (``mxfp4.py`` L805-830): uint8 shuffled FP4
    weight stacks plus ``nvfp4_block_scale_interleave``d scales viewed as
    ``float8_e4m3fn``.  Field naming mirrors ``V4MarlinPreparedWeights`` so
    the layerwise slot machinery can account/record both layouts uniformly.
    """

    w13: torch.Tensor
    w13_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor
    hidden_size: int
    intermediate_size: int
    num_experts: int


_TRTLLM_EPILOGUE_TILE_M = 128
# nvfp4_block_scale_interleave writes swizzled scale-factor tiles of
# 128 rows x 4 groups; inputs are padded up to those multiples.
_TRTLLM_SF_ROW_TILE = 128
_TRTLLM_SF_GROUP_TILE = 4


def _sf_interleave_numel(rows: int, k_groups: int) -> int:
    """Output element count of ``nvfp4_block_scale_interleave`` for a
    ``[rows, k_groups]`` uint8 scale matrix (128x4 tile padding)."""
    padded_rows = math.ceil(rows / _TRTLLM_SF_ROW_TILE) * _TRTLLM_SF_ROW_TILE
    padded_groups = (
        math.ceil(k_groups / _TRTLLM_SF_GROUP_TILE) * _TRTLLM_SF_GROUP_TILE
    )
    return padded_rows * padded_groups


def _trtllm_prepared_shapes(
    num_experts: int, hidden_size: int, intermediate_size: int
) -> tuple:
    """Shapes of (w13, w13_scale, w2, w2_scale) in the trtllm-gen layout.

    The shuffled stacks keep the raw geometry (the shuffle is a row
    permutation; the scale interleave is exact when its 128x4 padding is a
    no-op), which is what lets ``process_weights_after_loading`` reshape the
    interleaved scales back to ``[E, rows, k_groups]`` (``mxfp4.py``
    L806-825).  Both conditions reduce to hidden/intermediate % 128 == 0 —
    exactly ``create_weights``'s SM100 flashinfer padding (``mxfp4.py``
    L414-417) — enforced here rather than assumed.
    """
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if hidden_size % 128 or intermediate_size % 128:
        raise ValueError(
            "trtllm-gen shuffled layout requires hidden/intermediate "
            f"multiples of 128, got {hidden_size}/{intermediate_size}"
        )
    for rows, k_groups in (
        (2 * intermediate_size, hidden_size // _MXFP4_GROUP_SIZE),
        (hidden_size, intermediate_size // _MXFP4_GROUP_SIZE),
    ):
        if rows % _TRTLLM_EPILOGUE_TILE_M:
            raise ValueError(
                f"trtllm-gen shuffle needs rows % {_TRTLLM_EPILOGUE_TILE_M}"
                f" == 0, got {rows}"
            )
        if _sf_interleave_numel(rows, k_groups) != rows * k_groups:
            raise ValueError(
                "nvfp4_block_scale_interleave would pad the "
                f"[{rows}, {k_groups}] scale matrix; the prepared-slot "
                "layout requires the exact-size case"
            )
    return (
        (num_experts, 2 * intermediate_size, hidden_size // 2),
        (num_experts, 2 * intermediate_size, hidden_size // _MXFP4_GROUP_SIZE),
        (num_experts, hidden_size, intermediate_size // 2),
        (num_experts, hidden_size, intermediate_size // _MXFP4_GROUP_SIZE),
    )


def get_trtllm_mxfp4_storage_nbytes(
    *, num_experts: int, hidden_size: int, intermediate_size: int
) -> int:
    """Bytes required for one prepared trtllm-gen MXFP4 layer image.

    Kept next to ``allocate_trtllm_mxfp4`` so the lazy layerwise-prefill
    path can reserve KV-cache budget without allocating the tensors."""
    shapes = _trtllm_prepared_shapes(num_experts, hidden_size, intermediate_size)
    uint8_size = torch.tensor([], dtype=torch.uint8).element_size()
    f8_size = torch.tensor([], dtype=torch.float8_e4m3fn).element_size()
    return (
        math.prod(shapes[0]) * uint8_size
        + math.prod(shapes[1]) * f8_size
        + math.prod(shapes[2]) * uint8_size
        + math.prod(shapes[3]) * f8_size
    )


def allocate_trtllm_mxfp4(
    *,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    device: torch.device,
) -> TrtllmPreparedWeights:
    """Allocate stable caller-owned trtllm-gen storage without reading raw
    weights (dtypes per ``mxfp4.py`` L805-830: uint8 weights,
    float8_e4m3fn-viewed interleaved scales)."""
    shapes = _trtllm_prepared_shapes(num_experts, hidden_size, intermediate_size)
    return TrtllmPreparedWeights(
        w13=torch.empty(shapes[0], dtype=torch.uint8, device=device),
        w13_scale=torch.empty(
            shapes[1], dtype=torch.float8_e4m3fn, device=device
        ),
        w2=torch.empty(shapes[2], dtype=torch.uint8, device=device),
        w2_scale=torch.empty(
            shapes[3], dtype=torch.float8_e4m3fn, device=device
        ),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
    )


def prepare_trtllm_mxfp4(
    w13: torch.Tensor,
    w13_scale_bf16: torch.Tensor,
    w2: torch.Tensor,
    w2_scale_bf16: torch.Tensor,
    *,
    out: Optional[TrtllmPreparedWeights] = None,
    expert_ids: Optional[Sequence[int]] = None,
) -> TrtllmPreparedWeights:
    """Swizzle kt-export MXFP4 experts into the trtllm-gen shuffled layout
    on the current CUDA stream.

    Inputs are the raw export payload: FP4 nibble bytes with w13 as
    ``[gate | up]`` halves, and bf16 scales that are recovered to their
    exact resident E8M0 codes via ``bf16_scales_to_e8m0``.  ``expert_ids``
    restricts the swizzle to those experts (the layerwise pipeline copies
    GPU-resident experts' already-shuffled images directly, so their raw
    rows are never written); untouched ``out`` rows are preserved.  ``out``
    is optional for one-shot use and required by the layerwise
    double-buffer manager to keep storage addresses stable (``copy_``-only
    writes).
    """
    if w13.ndim != 3 or w2.ndim != 3:
        raise ValueError("export expert weights must be rank 3")
    experts = w13.shape[0]
    hidden_size = w13.shape[2] * 2
    intermediate_size = w2.shape[2] * 2
    shapes = _trtllm_prepared_shapes(experts, hidden_size, intermediate_size)
    actual_raw = (
        tuple(w13.shape),
        tuple(w13_scale_bf16.shape),
        tuple(w2.shape),
        tuple(w2_scale_bf16.shape),
    )
    if actual_raw != shapes:
        raise ValueError(
            f"inconsistent export shapes {actual_raw}, expected {shapes}"
        )

    if out is None:
        out = allocate_trtllm_mxfp4(
            num_experts=experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            device=w13.device,
        )
    else:
        actual = (
            tuple(out.w13.shape),
            tuple(out.w13_scale.shape),
            tuple(out.w2.shape),
            tuple(out.w2_scale.shape),
        )
        if actual != shapes:
            raise ValueError(
                f"prepared output shapes {actual} do not match {shapes}"
            )
        if (out.num_experts, out.hidden_size, out.intermediate_size) != (
            experts,
            hidden_size,
            intermediate_size,
        ):
            raise ValueError(
                "prepared output metadata does not match raw weights: got "
                f"E/K/N={out.num_experts}/{out.hidden_size}/"
                f"{out.intermediate_size}, expected "
                f"{experts}/{hidden_size}/{intermediate_size}"
            )

    selected = list(range(experts)) if expert_ids is None else list(expert_ids)
    if not selected:
        return out

    if expert_ids is None:
        codes13 = bf16_scales_to_e8m0(w13_scale_bf16)
        codes2 = bf16_scales_to_e8m0(w2_scale_bf16)
    else:
        # Gather first: unselected (GPU-resident) rows were never written by
        # the export and must not reach the exactness assertion.
        index = torch.tensor(selected, dtype=torch.long, device=w13.device)
        codes13 = bf16_scales_to_e8m0(w13_scale_bf16.index_select(0, index))
        codes2 = bf16_scales_to_e8m0(w2_scale_bf16.index_select(0, index))

    indices = trtllm_permute_indices(
        w13_sample=w13[selected[0]],
        w13_scale_sample=codes13[0],
        w2_sample=w2[selected[0]],
        w2_scale_sample=codes2[0],
        w13_gate_up_halves=True,
    )
    for position, expert_id in enumerate(selected):
        swizzle_trtllm_expert(
            Mxfp4ExpertBytes(
                w13=w13[expert_id],
                w13_scale_e8m0=codes13[position],
                w2=w2[expert_id],
                w2_scale_e8m0=codes2[position],
            ),
            indices,
            out_w13=out.w13[expert_id],
            out_w13_scale=out.w13_scale[expert_id],
            out_w2=out.w2[expert_id],
            out_w2_scale=out.w2_scale[expert_id],
        )
    return out


def kt_wheel_has_e8m0_resident_scales() -> bool:
    """Feature-check (not a version check): the E8M0-resident layout costs
    exactly 0.53125 B/elem vs 0.625 for fp32 scales; ask the wheel's own
    footprint binding at a K3-shaped matrix."""
    try:
        from kt_kernel import kt_kernel_ext
    except ImportError:
        return False
    n, k = 3072, 3584
    got = kt_kernel_ext.moe.mxfp4_buffer_bytes(n, k, _MXFP4_GROUP_SIZE)
    return abs(got / (n * k) - 0.53125) < 1e-6
