"""CUDA JIT: per-expert demand counters for KT margin routing, in one launch.

The swap driver needs these every forward -- promotion reads demand for
non-resident experts, demotion reads traffic served by resident ones. What it
does not need is eleven kernel launches per layer to collect them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_kt_margin_counters_module() -> Module:
    return load_jit(
        "kimi_k3_kt_margin_counters",
        cuda_files=["kimi_k3/kt_margin_counters.cuh"],
        cuda_wrappers=[("count", "KtMarginCounters::run")],
        extra_cuda_cflags=["-O3"],
    )


def covered(topk_ids: torch.Tensor, insist: torch.Tensor, overridden: torch.Tensor) -> bool:
    return (
        topk_ids.is_cuda
        and topk_ids.dtype == torch.int64
        and insist.dtype == torch.bool
        and overridden.dtype == torch.bool
        and topk_ids.is_contiguous()
        and insist.is_contiguous()
        and overridden.is_contiguous()
        and insist.shape == topk_ids.shape
        and overridden.shape == topk_ids.shape
        and topk_ids.numel() > 0
    )


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
    # .view(torch.uint8) reinterprets rather than copies -- torch bool is one
    # byte, and the tensor matcher maps C++ bool to uint8.
    _jit_kt_margin_counters_module().count(
        insist_count,
        override_count,
        resident_count,
        topk_ids.reshape(-1),
        insist.reshape(-1).view(torch.uint8),
        overridden.reshape(-1).view(torch.uint8),
    )
