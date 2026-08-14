# SPDX-License-Identifier: Apache-2.0
"""Read cold-expert weights straight out of kt-kernel's resident buffers.

The point is to stop keeping a second copy. Split prefill needs every layer's
cold experts on the GPU one layer ahead, and until now that came from a pinned
host store holding the same bytes kt already holds -- ~0.415 TB of duplication
for K3's cold set across 8 ranks. kt's buffers are checkpoint layout
(``fp4-moe.hpp::load_weights`` fills them with plain ``memcpy``: nibble-packed
FP4 weights, raw uint8 E8M0 scales), which is exactly what the GPU-side swizzle
consumes, so the duplicate exists only because the addresses were not reachable.
``expert_buffer_pointers()`` now exposes them.

Deliberately NOT via ``write_weight_scale_to_buffer``: that exports rather than
lends, and it expands the E8M0 scales to bf16 on the CPU -- ~38.7 ms of a
57-68 ms per-layer cost, and work a trtllm consumer immediately undoes because
trtllm wants those codes back as bytes.

THE LAYOUT, and why it is not a straight slice. kt partitions by NUMA node, not
by GPU rank, and it partitions the two projections on different axes. With
``per_numa = intermediate_size_per_partition``, for expert ``e``:

    gate/up        [per_numa, hidden/2]        rows, contiguous at e*wec/2
    down           [hidden, per_numa/2]        COLUMN-blocked
    gate/up scale  [per_numa, hidden/group]    rows
    down scale     [hidden, per_numa/group]    COLUMN-blocked

Concatenating the partitions along those axes rebuilds the full expert, and
slicing that for the GPU rank reproduces ``build_expert_bytes`` exactly. Doing
it that way, rather than reimplementing kt's own cpu_tp/gpu_tp branch logic,
means the two only have to agree about the checkpoint's layout -- which is the
thing both of them are already derived from.

EXPERT IDS ARE PHYSICAL. Under ``cold_only_cpu_experts`` kt allocates only the
cold experts, so a logical expert id must be resolved through the
physical-to-logical map before it indexes these buffers.
"""

from __future__ import annotations

import ctypes
import logging
from typing import Dict, List, Optional, Sequence

import torch

logger = logging.getLogger(__name__)

# Index into the per-partition pointer row returned by expert_buffer_pointers().
_GATE, _UP, _DOWN, _GATE_S, _UP_S, _DOWN_S = range(6)

# Index into the geometry vector.
_NUMA, _EXPERTS, _HIDDEN, _INTER_PER_NUMA, _GROUP = range(5)


def _wrap(addr: int, nbytes: int) -> torch.Tensor:
    """A uint8 view of kt's memory. Borrowed, never owned, never freed here."""
    buf = (ctypes.c_uint8 * nbytes).from_address(addr)
    return torch.frombuffer(buf, dtype=torch.uint8, count=nbytes)


class KtRamExpertSource:
    """Per-layer views of kt's resident expert buffers, sliced for this rank.

    One instance per MoE layer, because kt's MoE object is per layer.
    """

    def __init__(
        self,
        *,
        pointers: Sequence[Sequence[int]],
        geometry: Sequence[int],
        tp_rank: int,
        tp_size: int,
        physical_to_logical: Optional[Sequence[int]] = None,
    ):
        self.numa = int(geometry[_NUMA])
        self.experts = int(geometry[_EXPERTS])
        self.hidden = int(geometry[_HIDDEN])
        self.per_numa = int(geometry[_INTER_PER_NUMA])
        self.group = int(geometry[_GROUP])
        self.tp_rank = int(tp_rank)
        self.tp_size = int(tp_size)
        self.intermediate = self.per_numa * self.numa

        if self.intermediate % self.tp_size:
            raise ValueError(
                f"intermediate {self.intermediate} not divisible by tp_size "
                f"{self.tp_size}"
            )
        self.per_gpu = self.intermediate // self.tp_size

        # logical id -> physical slot in kt's buffers.
        self._slot_of: Dict[int, int] = (
            {int(lid): slot for slot, lid in enumerate(physical_to_logical)}
            if physical_to_logical is not None
            else {}
        )

        wec = self.per_numa * self.hidden  # elements per expert per partition
        sec = (self.hidden // self.group) * self.per_numa
        self._w_bytes = wec // 2
        self._s_bytes = sec

        # Whole-buffer uint8 views, one per partition per tensor. Wrapping is
        # free: these alias kt's memory rather than copying it.
        self._views: List[Dict[int, torch.Tensor]] = []
        for row in pointers:
            self._views.append(
                {
                    k: _wrap(int(row[k]), self.experts * self._w_bytes)
                    for k in (_GATE, _UP, _DOWN)
                }
                | {
                    k: _wrap(int(row[k]), self.experts * self._s_bytes)
                    for k in (_GATE_S, _UP_S, _DOWN_S)
                }
            )

    # -- per-expert assembly ------------------------------------------------

    def _partition_2d(self, part: int, which: int, slot: int, rows: int, cols: int):
        """One partition's block for one expert, viewed as ``[rows, cols]``."""
        per = self._w_bytes if which in (_GATE, _UP, _DOWN) else self._s_bytes
        flat = self._views[part][which]
        return flat[slot * per : (slot + 1) * per].view(rows, cols)

    def raw_shard(self, logical_id: int) -> Dict[str, torch.Tensor]:
        """This rank's TP shard of one expert, in checkpoint layout.

        Returns ``{"w13", "w13_scale", "w2", "w2_scale"}`` matching what
        ``build_expert_bytes`` produces, so the GPU-side swizzle is unchanged.
        """
        slot = self._slot_of.get(int(logical_id), int(logical_id))
        if not 0 <= slot < self.experts:
            raise KeyError(
                f"expert {logical_id} maps to slot {slot}, outside kt's "
                f"{self.experts} resident slots (cold-only allocates only the "
                "cold experts)"
            )

        h2 = self.hidden // 2
        hg = self.hidden // self.group

        # gate/up: partitions stack along the intermediate axis.
        gate = torch.cat(
            [self._partition_2d(p, _GATE, slot, self.per_numa, h2) for p in range(self.numa)],
            dim=0,
        )
        up = torch.cat(
            [self._partition_2d(p, _UP, slot, self.per_numa, h2) for p in range(self.numa)],
            dim=0,
        )
        gate_s = torch.cat(
            [self._partition_2d(p, _GATE_S, slot, self.per_numa, hg) for p in range(self.numa)],
            dim=0,
        )
        up_s = torch.cat(
            [self._partition_2d(p, _UP_S, slot, self.per_numa, hg) for p in range(self.numa)],
            dim=0,
        )
        # down: partitions stack along the INTERMEDIATE axis too, but that axis
        # is the columns here, which is what "column-blocked" means.
        down = torch.cat(
            [
                self._partition_2d(p, _DOWN, slot, self.hidden, self.per_numa // 2)
                for p in range(self.numa)
            ],
            dim=1,
        )
        down_s = torch.cat(
            [
                self._partition_2d(
                    p, _DOWN_S, slot, self.hidden, self.per_numa // self.group
                )
                for p in range(self.numa)
            ],
            dim=1,
        )

        # Now slice the full expert for this GPU rank, exactly as
        # build_expert_bytes does from the checkpoint.
        lo, hi = self.tp_rank * self.per_gpu, (self.tp_rank + 1) * self.per_gpu
        return {
            "w13": torch.cat([gate[lo:hi], up[lo:hi]], dim=0).contiguous(),
            "w13_scale": torch.cat([gate_s[lo:hi], up_s[lo:hi]], dim=0).contiguous(),
            "w2": down[:, lo // 2 : hi // 2].contiguous(),
            "w2_scale": down_s[:, lo // self.group : hi // self.group].contiguous(),
        }


def verify_against_checkpoint(
    source: "KtRamExpertSource",
    *,
    weight_path: str,
    layer_idx: int,
    expert_ids: Sequence[int],
    tp_rank: int,
    tp_size: int,
) -> bool:
    """Bitwise: do kt's buffers reproduce the checkpoint, shard for shard?

    The gate on replacing the pinned store with this source. Every way the
    mapping can be wrong -- partition concat axis, physical vs logical ids, the
    TP slice -- yields right-shaped wrong bytes, so equality is demonstrated
    against ``build_expert_bytes`` rather than argued.
    """
    from sglang.srt.layers.moe.kt_expert_mover import (
        CheckpointExpertReader,
        build_expert_bytes,
    )

    prefix = f"language_model.model.layers.{layer_idx}.block_sparse_moe.experts"
    pairs = (
        ("w13", "w13"),
        ("w13_scale", "w13_scale_e8m0"),
        ("w2", "w2"),
        ("w2_scale", "w2_scale_e8m0"),
    )
    reader = CheckpointExpertReader(weight_path)
    ok = True
    try:
        for eid in expert_ids:
            want = build_expert_bytes(
                reader, prefix, int(eid), tp_rank=tp_rank, tp_size=tp_size
            )
            try:
                got = source.raw_shard(int(eid))
            except Exception:
                logger.exception("[kt-ram] raw_shard(%d) failed", eid)
                ok = False
                continue
            for mine, theirs in pairs:
                a = got[mine].reshape(-1)
                b = getattr(want, theirs).reshape(-1).to(torch.uint8)
                if a.shape != b.shape or not torch.equal(a, b):
                    logger.error(
                        "[kt-ram] layer %d expert %d %s DIFFERS from the "
                        "checkpoint (%s vs %s, %s bytes differ)",
                        layer_idx,
                        eid,
                        mine,
                        tuple(got[mine].shape),
                        tuple(getattr(want, theirs).shape),
                        "shape" if a.shape != b.shape else int((a != b).sum()),
                    )
                    ok = False
    finally:
        reader.close()
    if ok:
        logger.info(
            "[kt-ram] layer %d: %d expert(s) reproduce the checkpoint bitwise "
            "from kt's resident buffers",
            layer_idx,
            len(expert_ids),
        )
    return ok


def build_kt_ram_source(method, *, tp_rank: int, tp_size: int):
    """Build a source for one layer's kt MoE object, or None if unavailable."""
    wrapper = getattr(method, "wrapper", None)
    moe = getattr(wrapper, "moe", None) if wrapper is not None else None
    target = moe if moe is not None else wrapper
    getter = getattr(target, "expert_buffer_pointers", None)
    if getter is None:
        logger.info(
            "[kt-ram] this kt build exposes no expert_buffer_pointers(); "
            "keeping the pinned cold store"
        )
        return None
    try:
        pointers, geometry = getter()
    except Exception:
        logger.exception("[kt-ram] expert_buffer_pointers() failed")
        return None
    return KtRamExpertSource(
        pointers=pointers,
        geometry=geometry,
        tp_rank=tp_rank,
        tp_size=tp_size,
        physical_to_logical=method._kt_physical_to_logical,
    )
