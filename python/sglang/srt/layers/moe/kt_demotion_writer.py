# SPDX-License-Identifier: Apache-2.0
"""Rank-write demotions: every rank writes its own slice into kt's buffers.

THE PROBLEM. Under cold-only residency a demoted expert owns no CPU buffer,
so the swap window has to give it one before it becomes routable. kt's
``swap_expert_slot`` takes SIX FULL-EXPERT pointers and slices them per NUMA
partition internally, so one process has to materialize the whole expert --
which is why every previous attempt either read 12.9 GB off the checkpoint
(measured 28.8 s per window, 0.45 GB/s through 4,416 scattered mmap copies)
or all-gathered the shards across TP and deadlocked (M9/M11/M12, see
``runs/status/phase-swap-readback.status``).

THE OBSERVATION THIS MODULE IMPLEMENTS (user's, 2026-08-17): nobody needs
the whole expert in one place. The bytes are already in VRAM -- they are the
resident rows about to be overwritten -- and each rank owns a DISJOINT slice
of them. With kt's BufferB in a memfd arena every rank maps, each rank can
unswizzle its own row locally (no collective) and write it straight into the
right hole in kt's buffer:

    gate/up   rank r's rows land CONTIGUOUSLY at (r%4)*gu_w inside partition
              r//4's buffer -- one memcpy each
    down      rank r's strips land at (r%4)*rank_w2_w, stride part_w2_pitch,
              one strip per hidden row -- a strided copy

The interleave that makes w2 awkward everywhere else in this campaign is
solved by the WRITE PATTERN: no one ever reassembles eight shards. Cost is
~1.6 GB per rank, all eight in parallel, against 12.9 GB read serially.

WHY THE BYTES COME OUT IDENTICAL. kt's MXFP4 ``from_raw_mat`` is a plain
row-major memcpy at the same offsets (``fp4-moe.hpp:103``) and scales are
copied verbatim, so writing a slice at its offset is exactly what
``fill_expert_buffers`` would have written there. The layout inverse
(``unswizzle_trtllm_expert``) is proved bitwise by
``runs/meta/verify_unswizzle.py``, and kt's own
``verify_install_against_loaded`` re-checks a filled expert against the bulk
load per NUMA partition.

ORDERING, which is the whole correctness story:

    1. capture()   BEFORE any move writes a GPU row: unswizzle this rank's
                   rows for the layer's demoted experts and pull them to
                   pinned host staging. Local work only.
    2. kt move     rank 0 only, bookkeeping: the promoted expert's BufferBs
                   become the demoted expert's, unfilled. Every rank applies
                   the same move to its OWN offset table -- pure plan data,
                   identical everywhere, no communication.
    3. write()     every rank memcpys its slice into the moved buffers.
    4. barrier     before serving resumes and kt computes the expert.

Nothing here issues a CUDA collective, so a rank that fails degrades to
"this layer keeps the checkpoint path" instead of stranding seven peers in
NCCL.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

# Offsets exported by expert_buffer_arenas(), in row order.
_GATE_B, _UP_B, _DOWN_B, _GATE_D, _UP_D, _DOWN_D = range(6)


class SlotOffsets:
    """This rank's live view of where each expert's buffers sit.

    kt exports offsets once, at load. ``move_expert_slot`` then hands one
    expert's buffers to another, so the table goes stale exactly at swaps --
    which is why ``kt_arena_share`` refuses to serve reads under cold-only.
    The moves are plan data, identical on every rank, so each rank replays
    them locally and the table stays exact without any communication.
    """

    def __init__(self, rows: Sequence[Sequence[int]], *, experts: int, numa: int):
        # EVERY partition, not just this rank's. kt's move_slot_only checks
        # its preconditions on ALL partitions (it runs under do_numa_job), so
        # a rank that validated only its own would disagree with kt about
        # whether a move is legal -- and a fault confined to partition 1 would
        # make ranks 4-7 refuse while ranks 0-3 proceed, splitting the group
        # 3-vs-5. Validating the same thing kt validates keeps every rank's
        # verdict identical.
        self._by_part: List[List[Optional[List[int]]]] = []
        for part in range(numa):
            base = part * experts
            self._by_part.append(
                [
                    None
                    if int(rows[base + e][_GATE_B]) < 0
                    else [int(v) for v in rows[base + e]]
                    for e in range(experts)
                ]
            )

    def get(self, expert: int, part: int) -> Optional[List[int]]:
        return self._by_part[part][expert]

    def can_move(self, promote_id: int, demote_id: int) -> Optional[str]:
        """None if the move is legal on EVERY partition, else why not."""
        for part, table in enumerate(self._by_part):
            if table[promote_id] is None:
                return (
                    f"promoted expert {promote_id} holds no buffer in "
                    f"partition {part}"
                )
            if table[demote_id] is not None:
                return (
                    f"demoted expert {demote_id} already holds a buffer in "
                    f"partition {part}"
                )
        return None

    def apply_move(self, promote_id: int, demote_id: int) -> None:
        """Mirror kt's move on every partition, or refuse on every partition."""
        why = self.can_move(promote_id, demote_id)
        if why is not None:
            raise RuntimeError(f"slot move {promote_id}->{demote_id}: {why}")
        for table in self._by_part:
            table[demote_id] = table[promote_id]
            table[promote_id] = None


class RankShardWriter:
    """Writes this rank's slice of demoted experts into kt's memfd arena.

    One instance per process; per-layer geometry comes from the same
    ``ArenaExpertRanges`` the direct-DMA transport uses, so the two agree on
    the layout by construction rather than by comment.
    """

    def __init__(
        self,
        *,
        arena_by_layer: Dict[int, torch.Tensor],
        offsets_by_layer: Dict[int, SlotOffsets],
        geometry,  # ArenaExpertRanges-like: gu_w, gu_s, w2_*, hidden, local_rank
        shard_reader,  # _GpuResidentExpertReader: read_own_shards(layer, rows)
    ):
        self._arena = arena_by_layer
        self._offsets = offsets_by_layer
        self._g = geometry
        self._reader = shard_reader
        self._staged: Dict[int, Dict[str, torch.Tensor]] = {}
        self._staged_layer: Optional[int] = None
        self.capture_s = 0.0
        self.write_s = 0.0
        self.written = 0
        # (layer_idx, expert_id) of the most recent completed write, so the
        # window can point kt's bitwise verifier at something this path
        # actually produced.
        self.last_installed: Optional[Tuple[int, int]] = None

    # -- step 1: capture, before any GPU row is overwritten ----------------

    def capture(self, layer, layer_idx: int, rows: Sequence[int], demote_ids) -> bool:
        """Unswizzle this rank's rows for the layer's demotions, to host.

        Returns False on any failure, and the caller keeps the checkpoint
        path for the layer. Purely local: no collective, so a False here
        cannot desynchronise anything.
        """
        t0 = time.perf_counter()
        self._staged = {}
        self._staged_layer = layer_idx
        try:
            # The SAME unswizzle and shape cache the gathered path uses --
            # minus the gather. Sharing it is deliberate: a second copy of
            # this would be a second chance to get the inverse wrong, and a
            # wrong inverse yields right-shaped wrong bytes.
            shards = self._reader.read_own_shards(layer, list(rows))
            for shard, demote_id in zip(shards, demote_ids):
                # To host in one go per tensor; pinned so the D2H is a real
                # DMA rather than a staged pageable copy.
                self._staged[int(demote_id)] = {
                    "w13": _to_pinned(shard.w13),
                    "w13_scale": _to_pinned(shard.w13_scale_e8m0),
                    "w2": _to_pinned(shard.w2),
                    "w2_scale": _to_pinned(shard.w2_scale_e8m0),
                }
        except Exception:
            logger.exception(
                "[kt-rankwrite] capture failed on layer %s; this layer keeps "
                "the checkpoint path",
                layer_idx,
            )
            self._staged = {}
            return False
        finally:
            self.capture_s += time.perf_counter() - t0
        return True

    # -- step 3: write, after kt has moved the slot ------------------------

    def validate(self, layer_idx: int, swaps) -> Optional[str]:
        """Everything that can refuse, checked BEFORE any state moves.

        This is what makes the ordering safe. The write must happen AFTER
        kt's move -- writing beforehand would blit the demoted expert's bytes
        over the PROMOTED expert's live buffer, and if the layer then aborted,
        that expert stays CPU-routable with corrupted weights. But the move is
        irreversible (move_slot_only nulls the promoted entry), so nothing
        fallible may follow it. Hoisting every check here satisfies both: the
        move happens only once the write is known to be possible.

        Returns None if every pair is installable, else the first reason.
        """
        if self._staged_layer != layer_idx:
            return f"no shards captured for layer {layer_idx}"
        offsets = self._offsets.get(layer_idx)
        if offsets is None:
            return f"no offset table for layer {layer_idx}"
        for s in swaps:
            demote_id = int(s.demote)
            if demote_id not in self._staged:
                return f"expert {demote_id} was not captured"
            why = offsets.can_move(int(s.promote), demote_id)
            if why is not None:
                return why
        return None

    def write(self, layer_idx: int, promote_id: int, demote_id: int) -> bool:
        """Write this rank's slice into the buffers the demoted expert now owns.

        Call AFTER kt's move and after ``validate`` passed for this layer: the
        buffers already belong to the demoted expert, so overwriting them is
        correct rather than destructive, and everything that could refuse has
        already been checked.
        """
        if self._staged_layer != layer_idx:
            return False
        shard = self._staged.get(int(demote_id))
        if shard is None:
            return False
        t0 = time.perf_counter()
        try:
            offsets = self._offsets[layer_idx]
            row = offsets.get(int(demote_id), self._g.part)
            if row is None:
                raise RuntimeError(
                    f"expert {demote_id} holds no buffer in partition "
                    f"{self._g.part} after the move (layer {layer_idx})"
                )
            arena = self._arena[layer_idx]
            g = self._g
            lr = g.local_rank

            w13 = shard["w13"].view(torch.uint8).reshape(-1)
            w13_s = shard["w13_scale"].view(torch.uint8).reshape(-1)
            half_w, half_s = g.gu_w, g.gu_s
            # gate | up halves, each contiguous at this rank's row offset.
            _blit(arena, row[_GATE_B] + lr * half_w, w13[:half_w])
            _blit(arena, row[_UP_B] + lr * half_w, w13[half_w:])
            _blit(arena, row[_GATE_D] + lr * half_s, w13_s[:half_s])
            _blit(arena, row[_UP_D] + lr * half_s, w13_s[half_s:])
            # down: one strip per hidden row, at this rank's column offset.
            _blit_strided(
                arena,
                row[_DOWN_B] + lr * g.w2_width,
                shard["w2"].view(torch.uint8).reshape(g.hidden, g.w2_width),
                pitch=g.w2_pitch,
            )
            _blit_strided(
                arena,
                row[_DOWN_D] + lr * g.w2s_width,
                shard["w2_scale"].view(torch.uint8).reshape(g.hidden, g.w2s_width),
                pitch=g.w2s_pitch,
            )
        except Exception:
            logger.exception(
                "[kt-rankwrite] write failed for expert %s on layer %s",
                demote_id,
                layer_idx,
            )
            return False
        finally:
            self.write_s += time.perf_counter() - t0
        self.written += 1
        self.last_installed = (int(layer_idx), int(demote_id))
        return True

    def commit_move(self, layer_idx: int, promote_id: int, demote_id: int) -> None:
        """Mirror kt's slot move in this rank's offset table.

        MUST be called on EVERY rank whenever kt's ownership actually moved,
        by ANY path -- the rank-write install, and equally the checkpoint
        fallback, whose ``swap_expert_slot`` also routes through
        ``move_slot_only``. kt's state is global to the arena that every rank
        maps, so a rank that misses one move has a table that is silently one
        swap behind, and every later write lands in another expert's buffer.
        Local bookkeeping only, driven by plan data, so it stays identical
        across ranks without communication.
        """
        self._offsets[layer_idx].apply_move(int(promote_id), int(demote_id))

    def end_window(self) -> str:
        line = (
            f"[kt-rankwrite] {self.written} demotion(s): capture "
            f"{self.capture_s:.2f}s + write {self.write_s:.2f}s"
        )
        self.capture_s = self.write_s = 0.0
        self.written = 0
        self._staged = {}
        self._staged_layer = None
        return line


def _to_pinned(t: torch.Tensor) -> torch.Tensor:
    """Host copy, pinned where the platform allows it.

    Pinning makes the D2H a real DMA instead of a staged pageable copy, but
    it is an optimization, not a requirement -- and it is unavailable
    without an accelerator backend, which is exactly the case on the CPU-only
    dev box where this module's correctness tests run. Falling back keeps the
    bytes identical and the tests meaningful.
    """
    try:
        host = torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=True)
    except RuntimeError:
        host = torch.empty(t.shape, dtype=t.dtype, device="cpu")
    host.copy_(t, non_blocking=False)
    return host


def _blit(arena: torch.Tensor, offset: int, src: torch.Tensor) -> None:
    arena[offset : offset + src.numel()].copy_(src)


def _blit_strided(
    arena: torch.Tensor, offset: int, src: torch.Tensor, *, pitch: int
) -> None:
    """``src`` is [rows, width]; write row i at ``offset + i*pitch``.

    One strided view + a single copy_, not a Python loop over 3,584 rows:
    the loop is ~14k small copies per expert and would put the cost right
    back where the checkpoint path had it.
    """
    rows, width = src.shape
    dst = arena[offset : offset + (rows - 1) * pitch + width]
    dst.as_strided((rows, width), (pitch, 1)).copy_(src)
