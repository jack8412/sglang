# SPDX-License-Identifier: Apache-2.0
"""Pinned host cache of cold-expert weights, in the resident GPU layout.

Full-expert prefill needs every layer's 276 CPU-resident ("cold") experts on
the GPU, one layer ahead of compute.  Producing them at runtime is too slow:
kt-kernel's export of one layer's cold set measures 57-68 ms (~38.7 ms of it
irreducible CPU work reading AMX buffers and expanding E8M0 scales to bf16),
against a ~33 ms compute window.  Re-reading the checkpoint is worse -- the
down-projection is column-sliced, so a 688 KB result pages in 5.5 MB.

So the shards are built ONCE at boot and held pinned, in exactly the layout a
resident GPU row holds (TP-sliced, trtllm-gen shuffled).  Runtime then costs
only a pinned H2D: 16.2 ms per layer, inside the window.

The same rows serve expert SWAPS.  A promotion is a copy from a cache row into
the resident GPU row, and a demotion copies the outgoing row back -- neither
touches the checkpoint.  That makes this cache authoritative state rather than
a derivable cache; see ``ColdExpertStore`` for the two invariants that follow.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

# The four trtllm-gen resident parameters, in the order kt's export contract
# and ``swizzle_trtllm_expert`` both use.
WEIGHT_NAMES = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")


class ColdExpertStore:
    """Pinned host rows for every layer's cold experts, GPU-layout ready.

    Rows are addressed by ``(layer_idx, cold_slot)`` where ``cold_slot`` is a
    dense index into ``[0, num_cold)``.  The slot->logical assignment is a
    bijection maintained with **slot reuse**: when a cold expert is promoted,
    the expert demoted in the same swap takes its vacated slot.  A swap
    therefore rewrites one row per affected expert and never reindexes.

    Two invariants this class exists to enforce:

    1. **Read-before-overwrite.**  Promotion and demotion contend for the same
       slot within one swap.  :meth:`stage_row` copies the outgoing row out
       before :meth:`write_row` overwrites it; doing it the other way destroys
       the expert being promoted.
    2. **Authoritative, not derivable.**  After any demotion a slot holds a row
       that no longer matches what the checkpoint says belongs there, so the
       store must never be silently rebuilt mid-run.  :attr:`dirty` records
       that a swap has happened, and any on-disk reuse must refuse a dirty
       store.
    """

    def __init__(
        self,
        *,
        layer_indices: Sequence[int],
        per_expert_shapes: Dict[str, Tuple[Tuple[int, ...], torch.dtype]],
        num_cold: int,
        slot_to_logical: Dict[int, List[int]],
    ):
        self._layers = sorted(layer_indices)
        self._shapes = dict(per_expert_shapes)
        self._num_cold = int(num_cold)
        # layer -> [logical_id per cold slot]; mutated in place on swap.
        self._slot_to_logical = {k: list(v) for k, v in slot_to_logical.items()}
        self._logical_to_slot = {
            layer: {lid: slot for slot, lid in enumerate(ids)}
            for layer, ids in self._slot_to_logical.items()
        }
        # (layer, name) -> pinned [num_cold, *shape]
        self._rows: Dict[Tuple[int, str], torch.Tensor] = {}
        self.dirty = False

    # -- allocation --------------------------------------------------------

    def allocate(self) -> int:
        """Allocate the pinned rows.  Returns total bytes."""
        total = 0
        for layer in self._layers:
            for name, (shape, dtype) in self._shapes.items():
                t = torch.empty(
                    (self._num_cold,) + tuple(shape), dtype=dtype, pin_memory=True
                )
                self._rows[(layer, name)] = t
                total += t.numel() * t.element_size()
        logger.info(
            "[cold-store] allocated %d layers x %d experts = %.1f GiB pinned",
            len(self._layers),
            self._num_cold,
            total / (1024**3),
        )
        return total

    @property
    def num_cold(self) -> int:
        return self._num_cold

    @property
    def layers(self) -> List[int]:
        return list(self._layers)

    def layer_rows(self, layer_idx: int, name: str) -> torch.Tensor:
        """The pinned ``[num_cold, *shape]`` block for one layer+weight.

        This is the prefetch source: one bulk ``copy_`` per name per layer.
        """
        return self._rows[(layer_idx, name)]

    def row_views(self, layer_idx: int, slot: int) -> Dict[str, torch.Tensor]:
        """Per-name views of a single cold slot (the swizzle destination)."""
        return {n: self._rows[(layer_idx, n)][slot] for n in self._shapes}

    # -- slot bookkeeping --------------------------------------------------

    def slot_of(self, layer_idx: int, logical_id: int) -> Optional[int]:
        return self._logical_to_slot[layer_idx].get(logical_id)

    def logical_of(self, layer_idx: int, slot: int) -> int:
        return self._slot_to_logical[layer_idx][slot]

    def slot_logical_ids(self, layer_idx: int) -> List[int]:
        return list(self._slot_to_logical[layer_idx])

    # -- swap support ------------------------------------------------------

    def stage_row(self, layer_idx: int, slot: int) -> Dict[str, torch.Tensor]:
        """Copy a slot's rows out to fresh CPU tensors (read-before-overwrite).

        Returns detached copies, so a subsequent :meth:`write_row` into the
        same slot cannot corrupt the promoted expert.
        """
        return {
            n: self._rows[(layer_idx, n)][slot].clone() for n in self._shapes
        }

    def write_row(
        self,
        layer_idx: int,
        slot: int,
        values: Dict[str, torch.Tensor],
        *,
        logical_id: int,
    ) -> None:
        """Install ``values`` (GPU-layout bytes) at ``slot`` and rebind it."""
        for n in self._shapes:
            src = values[n]
            dst = self._rows[(layer_idx, n)][slot]
            if src.shape != dst.shape or src.dtype != dst.dtype:
                raise ValueError(
                    f"cold-store row mismatch at layer {layer_idx} slot {slot} "
                    f"{n}: got {tuple(src.shape)}/{src.dtype}, "
                    f"expected {tuple(dst.shape)}/{dst.dtype}"
                )
            dst.copy_(src.to("cpu", non_blocking=False))
        old = self._slot_to_logical[layer_idx][slot]
        if old != logical_id:
            self._logical_to_slot[layer_idx].pop(old, None)
            self._slot_to_logical[layer_idx][slot] = logical_id
            self._logical_to_slot[layer_idx][logical_id] = slot
        # Any demotion makes the store diverge from the checkpoint.
        self.dirty = True


def build_cold_store(
    *,
    layer_indices: Sequence[int],
    gpu_experts_mask: torch.Tensor,
    weight_path: str,
    tp_rank: int,
    tp_size: int,
    expert_prefix_for_layer,
    per_expert_shapes: Dict[str, Tuple[Tuple[int, ...], torch.dtype]],
    device: torch.device,
    progress_every: int = 10,
    raw_layout: bool = False,
) -> ColdExpertStore:
    """Read, TP-slice and (unless ``raw_layout``) swizzle cold experts into
    pinned rows.

    ``raw_layout`` stores CHECKPOINT-layout bytes instead of trtllm-swizzled
    ones, leaving the swizzle to ColdExpertPipeline, which does it once per
    layer on device. That is the shape the design needs if the pinned copy is
    ever to be dropped in favour of streaming out of kt's own buffers: those
    hold checkpoint layout too (fp4-moe.hpp loads by plain memcpy), so a raw
    store is the same bytes in the same layout, and proving the pipeline
    against it is the step before removing the store entirely.

    ``per_expert_shapes`` must describe whichever layout is being stored --
    raw shapes for ``raw_layout``, resident-buffer shapes otherwise.

    Cold slots are assigned in ascending logical order at build time -- the
    mirror of how residents get their dense slots (``torch.where`` +
    ``arange``) -- and thereafter maintained by slot reuse on swap.

    The swizzle runs on ``device`` (it is a CUDA kernel: there is no CPU
    implementation of ``nvfp4_block_scale_interleave``), writing into a small
    reusable device scratch that is then copied back to the pinned row.
    """
    from sglang.srt.layers.moe.kt_expert_mover import (
        CheckpointExpertReader,
        build_expert_bytes,
    )
    from sglang.srt.layers.moe.kt_mxfp4_export import (
        swizzle_trtllm_expert,
        trtllm_permute_indices,
    )

    cold_ids = torch.where(~gpu_experts_mask)[0].tolist()
    num_cold = len(cold_ids)
    layers = sorted(layer_indices)
    store = ColdExpertStore(
        layer_indices=layers,
        per_expert_shapes=per_expert_shapes,
        num_cold=num_cold,
        slot_to_logical={layer: list(cold_ids) for layer in layers},
    )
    store.allocate()

    reader = CheckpointExpertReader(weight_path)
    scratch = (
        None
        if raw_layout
        else {
            n: torch.empty(tuple(shape), dtype=dtype, device=device)
            for n, (shape, dtype) in per_expert_shapes.items()
        }
    )
    indices = None

    t_start = time.perf_counter()
    for li, layer_idx in enumerate(layers):
        prefix = expert_prefix_for_layer(layer_idx)
        for slot, logical_id in enumerate(cold_ids):
            raw = build_expert_bytes(
                reader, prefix, logical_id, tp_rank=tp_rank, tp_size=tp_size
            )
            if raw_layout:
                # Store the checkpoint-layout bytes untouched; the swizzle is
                # deferred to the pipeline, once per layer on device. Nothing
                # here touches the GPU at all, which is also why a raw store
                # builds faster than a swizzled one.
                for n, t in zip(
                    WEIGHT_NAMES,
                    (raw.w13, raw.w13_scale_e8m0, raw.w2, raw.w2_scale_e8m0),
                ):
                    store.layer_rows(layer_idx, n)[slot].copy_(t)
                continue
            on_dev = type(raw)(
                w13=raw.w13.to(device),
                w13_scale_e8m0=raw.w13_scale_e8m0.to(device),
                w2=raw.w2.to(device),
                w2_scale_e8m0=raw.w2_scale_e8m0.to(device),
            )
            if indices is None:
                indices = trtllm_permute_indices(
                    w13_sample=on_dev.w13,
                    w13_scale_sample=on_dev.w13_scale_e8m0,
                    w2_sample=on_dev.w2,
                    w2_scale_sample=on_dev.w2_scale_e8m0,
                    w13_gate_up_halves=True,
                )
            swizzle_trtllm_expert(
                on_dev,
                indices,
                out_w13=scratch["w13_weight"],
                out_w13_scale=scratch["w13_weight_scale"],
                out_w2=scratch["w2_weight"],
                out_w2_scale=scratch["w2_weight_scale"],
            )
            for n in per_expert_shapes:
                store.layer_rows(layer_idx, n)[slot].copy_(scratch[n])
        if progress_every and (li + 1) % progress_every == 0:
            done = li + 1
            el = time.perf_counter() - t_start
            logger.info(
                "[cold-store] %d/%d layers, %.1fs elapsed, ~%.0fs remaining",
                done,
                len(layers),
                el,
                el / done * (len(layers) - done),
            )

    reader.close()
    torch.cuda.synchronize(device)
    logger.info(
        "[cold-store] built %d layers x %d cold experts in %.1fs",
        len(layers),
        num_cold,
        time.perf_counter() - t_start,
    )
    return store
