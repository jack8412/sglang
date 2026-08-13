# SPDX-License-Identifier: Apache-2.0
"""Evaluate a token's full expert set as two disjoint expert slices.

K3 keeps 620 of its 896 experts resident on GPU; the other 276 normally run on
CPU, which costs ~14x on prefill.  The MoE kernel already supports evaluating a
SLICE of the global expert range (``local_expert_offset`` / ``local_num_experts``
-- the mechanism expert parallelism uses), so full-expert prefill needs no
contiguous 896-row tensor: call the kernel twice over complementary slices and
combine.

The combine is exact, for two reasons that both have to hold:

* K3's router is EXTERNAL.  ``packed_topk_ids`` carries ``(expert_id << 16) |
  bf16-weight-bits``, so selection and weights are fixed before the kernel runs
  (``norm_topk_prob`` is documented unused on this path and
  ``routed_scaling_factor`` is passed as 1.0).  There is no per-call softmax to
  split, so a pick's weight is identical whichever call evaluates it.

* Finalize is a weighted gather-sum that skips dropped slots:
  ``out[t] = sum_k w[t,k] * gemm2_out[permuted_idx(t,k)]``, ``continue`` on
  ``permuted_idx == -1``.  Out-of-slice picks come back as -1, so each call
  contributes exactly its own experts and the sum over a partition equals the
  whole.

Verified bitwise against a single 896-expert call at T = 4K/16K/30K/48K, with
slot accounting exact (resident + cold = T*top_k, overlap 0).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def merge_deferred_partials(
    *,
    gemm2_a: torch.Tensor,
    idx_a: torch.Tensor,
    gemm2_b: torch.Tensor,
    idx_b: torch.Tensor,
    validate: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Concatenate two ``do_finalize=False`` results into one finalize input.

    Each call returns ``gemm2_out`` in ITS OWN permuted row space plus
    ``expanded_idx_to_permuted_idx`` with -1 for slots it does not own.  The
    two index vectors are disjoint by construction, so shifting b's valid
    indices past a's rows yields a single combined gather.

    Returns ``(gemm2_out, expanded_idx)`` ready for ``moe_finalize_fuse_shared``.

    ``validate`` asserts disjointness -- cheap relative to the GEMMs but it
    syncs, so it is off by default and used in tests and boot checks.
    """
    n_rows_a = gemm2_a.shape[0]
    idx_b_shifted = torch.where(idx_b >= 0, idx_b + n_rows_a, idx_b)
    expanded_idx = torch.where(idx_a >= 0, idx_a, idx_b_shifted)

    if validate:
        both = (idx_a >= 0) & (idx_b >= 0)
        if bool(both.any()):
            raise RuntimeError(
                f"split-moe: {int(both.sum())} slots claimed by BOTH expert "
                "slices -- the slices are not disjoint"
            )
        neither = (idx_a < 0) & (idx_b < 0)
        if bool(neither.any()):
            raise RuntimeError(
                f"split-moe: {int(neither.sum())} slots claimed by NEITHER "
                "slice -- some experts are unreachable"
            )

    return torch.cat([gemm2_a, gemm2_b], dim=0), expanded_idx


def split_slice_moe(
    *,
    situ_moe,
    packed_topk: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: Optional[torch.Tensor],
    resident: dict,
    cold: dict,
    num_experts: int,
    num_resident: int,
    top_k: int,
    intermediate_size: int,
    shared_output: Optional[torch.Tensor] = None,
    validate: bool = False,
) -> torch.Tensor:
    """Run the MoE over resident + cold expert slices and finalize once.

    ``resident`` / ``cold`` each carry ``w13``, ``w13_scale``, ``w2``,
    ``w2_scale``, ``alpha``, ``beta`` -- the per-slice weights and the
    per-expert scalar vectors, which must be sized to their OWN slice
    (the kernel indexes them by local expert index).

    ``packed_topk`` must already be in SLOT space: resident experts at
    ``[0, num_resident)`` and cold experts at ``[num_resident, num_experts)``.
    """
    from sglang.kernels.ops.moe.moe_finalize_fuse_shared import (
        moe_finalize_fuse_shared,
    )

    def _call(w, offset, n_local):
        return situ_moe.trtllm_fp4_block_scale_routed_moe(
            packed_topk_ids=packed_topk,
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale,
            gemm1_weights=w["w13"],
            gemm1_weights_scale=w["w13_scale"],
            gemm1_alpha=w["alpha"],
            gemm1_beta=w["beta"],
            gemm2_weights=w["w2"],
            gemm2_weights_scale=w["w2_scale"],
            output1_scale_scalar=None,
            output1_scale_gate_scalar=None,
            output2_scale_scalar=None,
            num_experts=num_experts,
            top_k=top_k,
            intermediate_size=intermediate_size,
            activation_type=situ_moe.ACTIVATION_SITU,
            local_expert_offset=offset,
            local_num_experts=n_local,
            do_finalize=False,
        )

    gemm2_r, weights, idx_r = _call(resident, 0, num_resident)
    gemm2_c, _, idx_c = _call(cold, num_resident, num_experts - num_resident)

    gemm2_out, expanded_idx = merge_deferred_partials(
        gemm2_a=gemm2_r, idx_a=idx_r,
        gemm2_b=gemm2_c, idx_b=idx_c,
        validate=validate,
    )
    return moe_finalize_fuse_shared(
        gemm2_out, expanded_idx, weights, shared_output, top_k
    )
