"""CUDA JIT: per-expert demand counters for KT margin routing, in one launch.

The swap driver needs these every forward -- promotion reads demand for
non-resident experts, demotion reads traffic served by resident ones. What it
does not need is eleven kernel launches per layer to collect them.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)

# The router emits int32 ids; the torch form only widened them because
# scatter_add_ demands an int64 index. Both are instantiated so the kernel
# takes whichever arrives rather than silently declining the common one.
_ID_DTYPES = {torch.int32: "count_i32", torch.int64: "count_i64"}


@cache_once
def _jit_kt_margin_counters_module() -> Module:
    return load_jit(
        "kimi_k3_kt_margin_counters",
        cuda_files=["kimi_k3/kt_margin_counters.cuh"],
        cuda_wrappers=[
            ("count_i32", "KtMarginCounters<int32_t>::run"),
            ("count_i64", "KtMarginCounters<int64_t>::run"),
        ],
        extra_cuda_cflags=["-O3"],
    )


def covered(topk_ids: torch.Tensor, insist: torch.Tensor, overridden: torch.Tensor) -> bool:
    return (
        topk_ids.is_cuda
        and topk_ids.dtype in _ID_DTYPES
        and insist.dtype == torch.bool
        and overridden.dtype == torch.bool
        and topk_ids.is_contiguous()
        and insist.is_contiguous()
        and overridden.is_contiguous()
        and insist.shape == topk_ids.shape
        and overridden.shape == topk_ids.shape
        and topk_ids.numel() > 0
    )


def why_not_covered(topk_ids, insist, overridden) -> str:
    """Why covered() declined, for the caller's fallback warning.

    A silent fallback is the failure mode that matters here: the torch path
    produces identical numbers, so declining the kernel costs only speed and
    shows up as a change that mysteriously did nothing. It cost one full
    measurement round (phase PS) before the dtype mismatch was found.
    """
    if not topk_ids.is_cuda:
        return "topk_ids not on CUDA"
    if topk_ids.dtype not in _ID_DTYPES:
        return f"topk_ids dtype {topk_ids.dtype} not in {sorted(map(str, _ID_DTYPES))}"
    for name, t in (("insist", insist), ("overridden", overridden)):
        if t.dtype != torch.bool:
            return f"{name} dtype {t.dtype}, expected bool"
        if t.shape != topk_ids.shape:
            return f"{name} shape {tuple(t.shape)} != topk_ids {tuple(topk_ids.shape)}"
    if not (topk_ids.is_contiguous() and insist.is_contiguous() and overridden.is_contiguous()):
        return "a tensor is not contiguous"
    if topk_ids.numel() == 0:
        return "empty batch"
    return "unknown"


def kt_margin_counters(
    insist_count: torch.Tensor,
    override_count: torch.Tensor,
    resident_count: torch.Tensor,
    topk_ids: torch.Tensor,
    insist: torch.Tensor,
    overridden: torch.Tensor,
) -> None:
    """Fold one forward into the three per-expert counters, in place.

    `topk_ids` are the ORIGINAL router ids, before any margin substitution, so
    demand is attributed to the expert the router actually asked for.

    Accumulates in place rather than returning: this runs inside graph capture,
    where the counters' addresses are baked into the captured kernel and must
    not move between replays.
    """
    fn = getattr(_jit_kt_margin_counters_module(), _ID_DTYPES[topk_ids.dtype])
    # .view(torch.uint8) reinterprets rather than copies -- torch bool is one
    # byte, and the tensor matcher maps C++ bool to uint8.
    fn(
        insist_count,
        override_count,
        resident_count,
        topk_ids.reshape(-1),
        insist.reshape(-1).view(torch.uint8),
        overridden.reshape(-1).view(torch.uint8),
    )
