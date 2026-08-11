"""CUDA JIT: elide a KT layer's CPU-expert branch when the batch needs none.

Two ops, used together:

    kt_cpu_branch_flag(flag, topk_ids, gpu_mask)   device predicate
    with kt_conditional(flag, body_stream): ...    work that runs only if set

The conditional node is built through the driver API because sglang captures
with ``torch.cuda.CUDAGraph`` and torch has no conditional-node support. It
does not need any: ``cuStreamGetCaptureInfo`` interrogates the driver about
the capture in progress, so the node splices into the graph torch is building
without torch's participation.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_kt_cpu_branch_module() -> Module:
    return load_jit(
        "kimi_k3_kt_cpu_branch",
        cuda_files=["kimi_k3/kt_cpu_branch.cuh"],
        cuda_wrappers=[
            ("flag", "KtCpuBranchFlag::run"),
            ("cond_begin", "KtCondNode::begin"),
            ("cond_end", "KtCondNode::end"),
        ],
        extra_cuda_cflags=["-O3"],
    )


def covered(topk_ids: torch.Tensor, gpu_mask: torch.Tensor) -> bool:
    return (
        topk_ids.is_cuda
        and gpu_mask.is_cuda
        and topk_ids.dtype == torch.int64
        and gpu_mask.dtype == torch.bool
        and topk_ids.is_contiguous()
        and gpu_mask.is_contiguous()
        and topk_ids.numel() > 0
    )


def kt_cpu_branch_flag(
    flag: torch.Tensor, topk_ids: torch.Tensor, gpu_mask: torch.Tensor
) -> None:
    """flag[0] = 1 if any routed slot names a non-GPU-resident expert.

    Writes into a caller-owned tensor rather than allocating: this runs inside
    graph capture, where the flag's address is baked into the captured
    predicate kernel and must not move between replays.
    """
    # .view(torch.uint8) is a reinterpretation, not a copy: torch bool is one
    # byte. The matcher maps C++ bool to uint8, so a bool tensor is rejected.
    _jit_kt_cpu_branch_module().flag(
        flag, topk_ids.view(-1), gpu_mask.view(torch.uint8)
    )


@contextlib.contextmanager
def kt_conditional(flag: torch.Tensor, body_stream: torch.cuda.Stream):
    """Record the enclosed work into a CUDA IF body gated on `flag`.

    Must be used during capture. `body_stream` must not itself be capturing --
    it becomes the body's recording stream, so it cannot be the CPU-path
    stream that is already forked into the parent graph.
    """
    mod = _jit_kt_cpu_branch_module()
    main_stream = torch.cuda.current_stream().cuda_stream
    mod.cond_begin(main_stream, body_stream.cuda_stream, flag)
    try:
        with torch.cuda.stream(body_stream):
            yield
    finally:
        # Closed in a finally so a raised body does not leave the driver with
        # an open capture on body_stream, which would poison every later
        # capture in the process with a confusing unrelated error.
        mod.cond_end(body_stream.cuda_stream)
