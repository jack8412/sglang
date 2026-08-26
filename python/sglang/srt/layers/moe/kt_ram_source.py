# SPDX-License-Identifier: Apache-2.0
"""Read cold-expert weights straight out of kt-kernel's resident buffers.

The point is to stop keeping a second copy. Split prefill needs every layer's
cold experts on the GPU one layer ahead, and until now that came from a pinned
host store holding the same bytes kt already holds -- ~0.415 TB of duplication
for K3's cold set across 8 ranks. kt's buffers are checkpoint layout
(``fp4-moe.hpp::load_weights`` fills them with plain ``memcpy``: nibble-packed
FP4 weights, raw uint8 E8M0 scales), which is exactly what the GPU-side swizzle
consumes, so the duplicate exists only because the addresses were not reachable.

Two access modes, one layout:

- ``KtRamExpertSource`` wraps the ABSOLUTE addresses that
  ``expert_buffer_pointers()`` exports. Valid only in the process that owns the
  kt engine (TP rank 0).
- ``KtArenaExpertSource`` indexes OFFSETS into memfd-backed arenas
  (``expert_buffer_arenas()``, KT_BUFFER_B_MEMFD=1). Offsets are address-space
  independent, so any rank that maps the fds gets the same bytes --
  ``kt_arena_share`` does the fd passing and mapping.

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
slicing that for the GPU rank gives the shard the swizzle consumes: w13 is
``[gate rows; up rows]`` for this rank's slice of the intermediate axis, and
w2 is the matching COLUMN slice, halved for the two-values-per-byte packing
and divided by the group size for its scales. This module is now the only
definition of that layout -- ``build_expert_bytes``, which used to state it
independently for a checkpoint-sourced promotion path, is gone with that
path. The implementation slices each partition's block FIRST and concatenates
only the touched pieces -- identical bytes (slice and concat commute along one axis),
but it copies the rank's 2.19 MB instead of materializing the full 17.5 MB
expert per call, which matters at ~370 promotions per swap window.

EXPERT IDS ARE BUFFER SLOTS. The ``bb_`` arrays are indexed by the same ids the
swap window and ``swap_expert_slot`` use; no physical/logical translation
happens here. Anything that reaches the CHECKPOINT must translate first, since
that side is indexed by logical id.
"""

from __future__ import annotations

import ctypes
import logging
from typing import Dict, List, Optional, Sequence

import torch

logger = logging.getLogger(__name__)

# Index into a (partition, expert) row from expert_buffer_pointers() /
# expert_buffer_arenas(): weight base then scale base for gate/up/down. These
# point into the PERSISTENT per-expert BufferB objects -- the same ones
# write_weights_to_buffer and swap_expert_slot use -- not the layer-shared
# staging arena (exporting that was the F2 segfault).
_GATE_B, _UP_B, _DOWN_B, _GATE_D, _UP_D, _DOWN_D = range(6)

_WEIGHT_KINDS = (_GATE_B, _UP_B, _DOWN_B)

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

        if len(pointers) != self.numa * self.experts:
            raise ValueError(
                f"expected {self.numa} x {self.experts} pointer rows, got "
                f"{len(pointers)}"
            )
        # rows are [partition * experts + expert]; ints, wrapped lazily so a
        # source over 896 experts does not build ~5k tensors up front.
        self._rows = [[int(v) for v in row] for row in pointers]
        del physical_to_logical  # bb_ arrays are indexed by the ids swap uses

        wec = self.per_numa * self.hidden  # elements per expert per partition
        self._w_bytes = wec // 2
        self._s_bytes = (self.hidden // self.group) * self.per_numa

    # -- mode-specific access ------------------------------------------------

    def _absent(self, expert: int) -> bool:
        """No buffers here (cold-only leaves non-resident experts without)."""
        return self._rows[expert][_GATE_B] == 0

    def _block(self, part: int, which: int, expert: int) -> torch.Tensor:
        """One partition's whole block for one expert, as flat uint8."""
        nbytes = self._w_bytes if which in _WEIGHT_KINDS else self._s_bytes
        return _wrap(self._rows[part * self.experts + expert][which], nbytes)

    # -- per-expert assembly ------------------------------------------------

    def _row_pieces(self, which: int, expert: int, cols: int) -> List[torch.Tensor]:
        """This rank's rows of a row-blocked matrix, one piece per partition hit.

        Global rows ``[lo, hi)`` of the partition-concatenated matrix; each
        partition contributes its intersection, in partition order, so
        concatenating the pieces equals slicing the concatenation.
        """
        lo, hi = self.tp_rank * self.per_gpu, (self.tp_rank + 1) * self.per_gpu
        pieces = []
        for p in range(self.numa):
            p0 = p * self.per_numa
            s, e = max(lo, p0), min(hi, p0 + self.per_numa)
            if s >= e:
                continue
            block = self._block(p, which, expert).view(self.per_numa, cols)
            pieces.append(block[s - p0 : e - p0])
        return pieces

    def _col_pieces(
        self, which: int, expert: int, cols_per_part: int, unit: int
    ) -> List[torch.Tensor]:
        """This rank's columns of a column-blocked matrix (the down matrices).

        ``unit`` converts intermediate elements to columns: 2 for the
        nibble-packed weights, ``group`` for the scales.
        """
        lo_c = self.tp_rank * self.per_gpu // unit
        hi_c = (self.tp_rank + 1) * self.per_gpu // unit
        pieces = []
        for p in range(self.numa):
            p0 = p * cols_per_part
            s, e = max(lo_c, p0), min(hi_c, p0 + cols_per_part)
            if s >= e:
                continue
            block = self._block(p, which, expert).view(self.hidden, cols_per_part)
            pieces.append(block[:, s - p0 : e - p0])
        return pieces

    def raw_shard(self, logical_id: int) -> Dict[str, torch.Tensor]:
        """This rank's TP shard of one expert, in checkpoint layout.

        Returns ``{"w13", "w13_scale", "w2", "w2_scale"}`` in the layout the
        GPU-side swizzle consumes -- see this module's docstring for the axis
        arithmetic, which lives there because nothing else states it now.

        NOT ON THE SERVING PATH: split prefill reads through
        ``ArenaDmaColdSource.layer_rows``, which DMAs straight out of the same
        arena. This is the EXECUTABLE statement of the layout, checked
        bitwise against an independently packed arena in
        test_kt_ram_source, and ``raw_shard_shapes`` -- which production DOES
        use -- is pinned against it there. Keeping it is what replaced the
        second opinion ``build_expert_bytes`` used to provide.
        """
        slot = int(logical_id)
        if not 0 <= slot < self.experts:
            raise KeyError(
                f"expert {logical_id} is outside kt's {self.experts} buffer "
                "slots"
            )
        if self._absent(slot):
            raise KeyError(
                f"expert {logical_id} is not CPU-resident here (null BufferB); "
                "under cold-only residency only cold experts have buffers"
            )

        h2 = self.hidden // 2
        hg = self.hidden // self.group

        # gate/up: partitions stack along the intermediate axis (rows), and
        # w13 = [gate rows; up rows], so one concat builds it from the pieces.
        w13 = torch.cat(
            self._row_pieces(_GATE_B, slot, h2) + self._row_pieces(_UP_B, slot, h2),
            dim=0,
        )
        w13_scale = torch.cat(
            self._row_pieces(_GATE_D, slot, hg) + self._row_pieces(_UP_D, slot, hg),
            dim=0,
        )
        # down: partitions stack along the INTERMEDIATE axis too, but that axis
        # is the columns here, which is what "column-blocked" means.
        w2 = torch.cat(
            self._col_pieces(_DOWN_B, slot, self.per_numa // 2, 2), dim=1
        ).contiguous()
        w2_scale = torch.cat(
            self._col_pieces(_DOWN_D, slot, self.per_numa // self.group, self.group),
            dim=1,
        ).contiguous()
        return {
            "w13": w13,
            "w13_scale": w13_scale,
            "w2": w2,
            "w2_scale": w2_scale,
        }

    def raw_shard_shapes(self) -> Dict[str, tuple]:
        """The shapes ``raw_shard`` returns, without touching a byte.

        Same four names, same order, every tensor uint8 (kt's arena is mapped
        as uint8 and the slicing never changes dtype). Expert-independent: the
        blocks are fixed-size per partition, so one expert's shard is every
        expert's shard.

        WHY THIS EXISTS. The split-prefill swizzle plan needs a tensor of the
        right shape and nothing else -- ``trtllm_permute_indices`` caches on
        ``tuple(x.shape)`` and never reads the data. It used to get one by
        materialising a real shard, which is a 2.19 MB copy out of mapped
        memory plus a 2.19 MB host-to-device copy, at boot, for bytes nobody
        looks at. Deriving the shapes here rather than at the call site is what
        keeps this module the ONLY statement of the axis arithmetic -- the
        alternative recomputes it beside the plan builder, where it can drift
        from ``raw_shard`` silently.
        """
        return {
            "w13": (2 * self.per_gpu, self.hidden // 2),
            "w13_scale": (2 * self.per_gpu, self.hidden // self.group),
            "w2": (self.hidden, self.per_gpu // 2),
            "w2_scale": (self.hidden, self.per_gpu // self.group),
        }


class KtArenaExpertSource(KtRamExpertSource):
    """The offset form: rows index into mapped arenas instead of addresses.

    ``arenas`` are flat uint8 views of the per-partition memfd mappings -- in
    rank 0 wrapped straight over kt's own mapping, in every other rank over
    that rank's mmap of the shipped fd. Offsets came from
    ``expert_buffer_arenas()`` and are valid in ANY mapping of the same fd,
    which is the entire reason this subclass exists.
    """

    def __init__(
        self,
        *,
        arenas: Sequence[torch.Tensor],
        offsets: Sequence[Sequence[int]],
        geometry: Sequence[int],
        tp_rank: int,
        tp_size: int,
        physical_to_logical: Optional[Sequence[int]] = None,
    ):
        super().__init__(
            pointers=offsets,
            geometry=geometry,
            tp_rank=tp_rank,
            tp_size=tp_size,
            physical_to_logical=physical_to_logical,
        )
        if len(arenas) != self.numa:
            raise ValueError(
                f"expected {self.numa} arenas (one per partition), got "
                f"{len(arenas)}"
            )
        self._arenas = [a.view(torch.uint8).reshape(-1) for a in arenas]

    def _absent(self, expert: int) -> bool:
        return self._rows[expert][_GATE_B] < 0

    def _block(self, part: int, which: int, expert: int) -> torch.Tensor:
        nbytes = self._w_bytes if which in _WEIGHT_KINDS else self._s_bytes
        off = self._rows[part * self.experts + expert][which]
        return self._arenas[part][off : off + nbytes]


def export_kt_arenas(method):
    """Rank 0: one layer's arena export, or None when not in memfd mode.

    Returns ``(fds, sizes, bases, offsets, geometry)`` exactly as
    ``expert_buffer_arenas()`` hands them over, with plain-int contents.
    """
    wrapper = getattr(method, "wrapper", None)
    moe = getattr(wrapper, "moe", None) if wrapper is not None else None
    target = moe if moe is not None else wrapper
    getter = getattr(target, "expert_buffer_arenas", None)
    if getter is None:
        return None
    try:
        fds, sizes, bases, offsets, geometry = getter()
    except Exception:
        logger.exception("[kt-ram] expert_buffer_arenas() failed")
        return None
    if not fds:
        return None  # KT_BUFFER_B_MEMFD off, or memfd fell back to malloc
    return (
        [int(f) for f in fds],
        [int(s) for s in sizes],
        [int(b) for b in bases],
        [[int(v) for v in row] for row in offsets],
        [int(g) for g in geometry],
    )
