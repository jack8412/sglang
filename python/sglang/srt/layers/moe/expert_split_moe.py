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

The two slices are combined through finalize's fp32 accumulator rather than by
concatenating their gemm2 buffers: each deferred call sizes its buffer for ALL
T*top_k slots, so holding both plus a concatenation costs 3.5x the single
call's peak several times over and will not fit beside a live server.
Accumulating keeps the peak at one call's workspace plus an fp32 [T, hidden]
buffer, and -- because the accumulation never leaves fp32 -- the result is
rounded to bf16 exactly once, as an unsplit call does.

Verified against a single 896-expert call at T = 4K/16K/30K/48K, with slot
accounting exact (resident + cold = T*top_k, overlap 0).
"""

from __future__ import annotations

from typing import Optional

import torch


def _tiled_split_slice_moe(
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
    shared_output: Optional[torch.Tensor],
    token_tile: int,
) -> torch.Tensor:
    """Run ``split_slice_moe`` over token tiles, writing into one output.

    Each tile's transients are freed before the next allocates, so peak
    follows the tile, not the chunk. The output is preallocated and written
    in place -- collecting tiles and concatenating would reintroduce a
    full-size buffer and undo the saving.
    """
    num_tokens = packed_topk.shape[0]
    hidden_size = (
        shared_output.shape[1] if shared_output is not None
        else hidden_states.shape[-1] * (2 if hidden_states.dtype == torch.uint8 else 1)
    )
    out = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=hidden_states.device
    )
    for lo in range(0, num_tokens, token_tile):
        hi = min(lo + token_tile, num_tokens)
        out[lo:hi] = split_slice_moe(
            situ_moe=situ_moe,
            packed_topk=packed_topk[lo:hi],
            hidden_states=hidden_states[lo:hi],
            hidden_states_scale=(
                None if hidden_states_scale is None else hidden_states_scale[lo:hi]
            ),
            resident=resident,
            cold=cold,
            num_experts=num_experts,
            num_resident=num_resident,
            top_k=top_k,
            intermediate_size=intermediate_size,
            shared_output=(
                None if shared_output is None else shared_output[lo:hi]
            ),
            token_tile=None,          # already tiled
        )
    return out


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
    token_tile: Optional[int] = None,
) -> torch.Tensor:
    """Run the MoE over resident + cold expert slices and finalize once.

    ``token_tile`` splits the call into batches of at most that many tokens.
    A token's MoE output depends only on its own row, so tiling changes no
    value -- but both large transients (the gemm2 buffer, which the kernel
    sizes for ALL T*top_k slots, and the fp32 accumulator) scale with the
    tile rather than the chunk. The peak saving is large and costs only a few
    percent of MoE time, which is what lets a large prefill chunk coexist with
    a long-context KV pool.

    ``resident`` / ``cold`` each carry ``w13``, ``w13_scale``, ``w2``,
    ``w2_scale``, ``alpha``, ``beta`` -- the per-slice weights and the
    per-expert scalar vectors, which must be sized to their OWN slice
    (the kernel indexes them by local expert index).

    ``packed_topk`` must already be in SLOT space: resident experts at
    ``[0, num_resident)`` and cold experts at ``[num_resident, num_experts)``.
    """
    num_tokens = packed_topk.shape[0]
    if token_tile and num_tokens > token_tile:
        return _tiled_split_slice_moe(
            situ_moe=situ_moe, packed_topk=packed_topk,
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale,
            resident=resident, cold=cold, num_experts=num_experts,
            num_resident=num_resident, top_k=top_k,
            intermediate_size=intermediate_size,
            shared_output=shared_output,
            token_tile=token_tile,
        )

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

    # Each deferred call allocates a gemm2 buffer sized for ALL T*top_k slots
    # (it cannot know how many land in its own slice), so holding both at once
    # -- let alone torch.cat'ing them into a third -- costs 3.5x the single
    # call's peak and does not fit beside a live server.  Instead the resident
    # slice is finalized into an fp32 accumulator and its gemm2 buffer freed
    # before the cold call allocates; the cold slice then adds into that
    # accumulator and rounds once.  Peak is one call's workspace plus the
    # accumulator, and the result rounds to bf16 exactly as an unsplit call
    # would.
    num_tokens = packed_topk.shape[0]
    if shared_output is not None:
        hidden_size = shared_output.shape[1]
    else:
        # uint8 hidden states are fp4 pairs -- two elements per byte.
        hidden_size = hidden_states.shape[-1]
        if hidden_states.dtype == torch.uint8:
            hidden_size *= 2

    gemm2_r, weights, idx_r = _call(resident, 0, num_resident)
    acc = torch.empty(
        num_tokens, hidden_size, dtype=torch.float32, device=gemm2_r.device
    )
    moe_finalize_fuse_shared(
        gemm2_r, idx_r, weights, None, top_k, acc_out=acc
    )
    del gemm2_r

    gemm2_c, _, idx_c = _call(cold, num_resident, num_experts - num_resident)
    return moe_finalize_fuse_shared(
        gemm2_c, idx_c, weights, shared_output, top_k, acc_in=acc
    )
