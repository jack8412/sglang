from __future__ import annotations

from typing import Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit


@cache_once
def _jit_module():
    return load_jit(
        "moe_finalize_fuse_shared",
        cuda_files=["moe/moe_finalize_fuse_shared.cu"],
        extra_dependencies=["cutlass"],
        header_only=False,
    )


def moe_finalize_fuse_shared(
    gemm2_out: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    expert_weights: torch.Tensor,
    shared_output: Optional[torch.Tensor],
    top_k: int,
    enable_pdl: bool = False,
    acc_in: Optional[torch.Tensor] = None,
    acc_out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Top-k weighted unpermute of ``gemm2_out``, plus an optional shared add.

    ``acc_in`` / ``acc_out`` are fp32 ``[num_tokens, hidden_dim]`` partial sums
    that let one token's top-k be summed across SEVERAL launches over disjoint
    expert slices (see ``expert_split_moe``).  Accumulation stays in fp32 the
    whole way, so the result is rounded to bf16 exactly once -- as it is in a
    single unsplit launch.  Pass ``acc_out`` on every slice but the last, and
    ``acc_in`` on every slice but the first.

    Returns the bf16 result, or ``acc_out`` itself when this launch produces a
    partial (in which case no bf16 output is written).
    """
    assert gemm2_out.dtype == torch.bfloat16
    assert expert_weights.dtype in (torch.float32, torch.bfloat16)
    assert expanded_idx_to_permuted_idx.dtype == torch.int32
    assert gemm2_out.dim() == 2
    assert expert_weights.dim() == 2

    num_tokens, top_k_check = expert_weights.shape
    assert top_k_check == top_k
    hidden_dim = gemm2_out.shape[1]

    if shared_output is not None:
        assert shared_output.dtype == torch.bfloat16
        assert shared_output.dim() == 2
        assert shared_output.shape[0] == num_tokens
        hidden_dim = shared_output.shape[1]
        assert hidden_dim <= gemm2_out.shape[1]

    empty = gemm2_out.new_empty((0, 0), dtype=torch.bfloat16)
    if acc_out is None:
        out = torch.empty(
            num_tokens, hidden_dim, dtype=torch.bfloat16, device=gemm2_out.device
        )
    else:
        # This launch writes the fp32 partial instead of a bf16 result; the
        # accumulator carries the FINAL hidden dim, which may be narrower than
        # gemm2_out's padded one.
        assert acc_out.dtype == torch.float32
        assert shared_output is None, "shared_output must be added on the last slice"
        assert acc_out.shape[0] == num_tokens
        hidden_dim = acc_out.shape[1]
        assert hidden_dim <= gemm2_out.shape[1]
        out = empty
    if acc_in is not None:
        assert acc_in.dtype == torch.float32
        assert acc_in.shape == (num_tokens, hidden_dim)

    _jit_module().moe_finalize_fuse_shared(
        out,
        gemm2_out,
        expanded_idx_to_permuted_idx,
        expert_weights,
        empty if shared_output is None else shared_output,
        empty if acc_in is None else acc_in,
        empty if acc_out is None else acc_out,
        int(top_k),
        bool(enable_pdl),
    )
    return acc_out if acc_out is not None else out
