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



class ArenaDmaWriter:
    """kt's arena <-> device in ONE hop: no pinned intermediate, no CPU memcpy.

    Both directions live here because they share the thing that costs: the
    registration. A promotion reads the same offsets a demotion writes, so
    splitting them into two objects would either register the arena twice or
    make one depend on the other's internals.

    What this replaces. The staged path reads the demoted rows off the GPU
    into a fresh pinned buffer (a real D2H DMA) and then memcpys that buffer
    into kt's arena on the CPU. The second hop is pure DRAM traffic -- a read
    and a write through the one controller, serialized against the DMA engine
    doing the first hop -- and it MEASURED as the larger of the two: at 114
    demotions per rank, capture 0.03 s against write 0.05-0.13 s; at 2,646,
    capture 2.06 s against write 3.35 s. Same ratio across a 23x range.

    Registering the arena lets the copy engine put the bytes where they belong
    itself, so the second hop stops existing rather than getting faster.

    w2's interleave costs nothing here, which is the part worth stating: the
    strips land at ``w2_pitch`` intervals inside kt's buffer, and
    cudaMemcpy2DAsync walks that stride in hardware. The host-side strided
    write it replaces is the same one the module docstring calls the awkward
    part of this campaign.

    REGISTRATION IS PER MAPPING, NOT PER EXPERT. kt makes one memfd per
    (layer, partition) and a rank reads only its own partition, so this is 92
    registrations of ~2275 MiB, once, at arm time -- against the 150,144
    per-expert ranges that made the direct-DMA transport fail with rc=2. The
    page count is what costs (~0.117 us/page, ~8 B of PTE per 4K page), so
    expect ~6 s and ~418 MB of page tables per rank for ~204 GiB.
    """

    def __init__(self, *, arena_by_layer, geometry, copy_lib, register_fn):
        self._g = geometry
        self._copy = copy_lib
        self._base: Dict[int, int] = {}
        total = 0
        for layer_idx, arena in arena_by_layer.items():
            ptr, nbytes = int(arena.data_ptr()), int(arena.numel())
            rc = register_fn(ptr, nbytes)
            if rc != 0:
                raise RuntimeError(
                    f"cudaHostRegister(layer {layer_idx}, {nbytes} B) rc={rc}"
                )
            self._base[int(layer_idx)] = ptr
            total += nbytes
        self.registered_bytes = total

    def read(self, *, layer_idx: int, row, out, stream: int) -> None:
        """kt's arena -> device: one promoted expert's slice, six copies.

        The exact inverse of :meth:`write`, against the same offsets, so a
        promotion reads back precisely the bytes a demotion put there. ``out``
        holds DEVICE tensors in checkpoint layout -- w13 and w13_scale flat,
        w2 and w2_scale as [hidden, width] -- which is the shape
        ``_swizzle_promoted_rows`` takes before the row is scattered into the
        resident slot.

        This is what makes the pinned cold store removable: the promoted bytes
        already sit in kt's buffers, and the copy engine can put them on the
        GPU without a host-side staging copy in between.
        """
        g = self._g
        lr = g.local_rank
        base = self._base[int(layer_idx)]
        w13 = out["w13"]
        w13_s = out["w13_scale"]
        self._copy.memcpy_h2d(
            w13.data_ptr(), base + row[_GATE_B] + lr * g.gu_w, g.gu_w, stream
        )
        self._copy.memcpy_h2d(
            w13.data_ptr() + g.gu_w,
            base + row[_UP_B] + lr * g.gu_w,
            g.gu_w,
            stream,
        )
        self._copy.memcpy_h2d(
            w13_s.data_ptr(), base + row[_GATE_D] + lr * g.gu_s, g.gu_s, stream
        )
        self._copy.memcpy_h2d(
            w13_s.data_ptr() + g.gu_s,
            base + row[_UP_D] + lr * g.gu_s,
            g.gu_s,
            stream,
        )
        # down: the source strips are strided by the partition pitch, the
        # destination is packed -- the copy engine walks the stride.
        self._copy.memcpy2d_h2d(
            out["w2"].data_ptr(),
            g.w2_width,
            base + row[_DOWN_B] + lr * g.w2_width,
            g.w2_pitch,
            g.w2_width,
            g.hidden,
            stream,
        )
        self._copy.memcpy2d_h2d(
            out["w2_scale"].data_ptr(),
            g.w2s_width,
            base + row[_DOWN_D] + lr * g.w2s_width,
            g.w2s_pitch,
            g.w2s_width,
            g.hidden,
            stream,
        )

    def write(self, *, layer_idx: int, row, shard, stream: int) -> None:
        """Issue one expert's six copies. Async on ``stream``."""
        g = self._g
        lr = g.local_rank
        base = self._base[int(layer_idx)]
        w13 = shard["w13"]
        w13_s = shard["w13_scale"]
        # gate | up, each contiguous at this rank's row offset -- the same two
        # offsets the host path blits to.
        self._copy.memcpy_d2h(
            base + row[_GATE_B] + lr * g.gu_w, w13.data_ptr(), g.gu_w, stream
        )
        self._copy.memcpy_d2h(
            base + row[_UP_B] + lr * g.gu_w,
            w13.data_ptr() + g.gu_w,
            g.gu_w,
            stream,
        )
        self._copy.memcpy_d2h(
            base + row[_GATE_D] + lr * g.gu_s, w13_s.data_ptr(), g.gu_s, stream
        )
        self._copy.memcpy_d2h(
            base + row[_UP_D] + lr * g.gu_s,
            w13_s.data_ptr() + g.gu_s,
            g.gu_s,
            stream,
        )
        # down: one strip per hidden row, at this rank's column offset. The
        # destination is strided; the source is a packed [hidden, width] block.
        self._copy.memcpy2d_d2h(
            base + row[_DOWN_B] + lr * g.w2_width,
            g.w2_pitch,
            shard["w2"].data_ptr(),
            g.w2_width,
            g.w2_width,
            g.hidden,
            stream,
        )
        self._copy.memcpy2d_d2h(
            base + row[_DOWN_D] + lr * g.w2s_width,
            g.w2s_pitch,
            shard["w2_scale"].data_ptr(),
            g.w2s_width,
            g.w2s_width,
            g.hidden,
            stream,
        )



class ArenaDmaColdSource:
    """Split prefill's cold stream, straight out of kt's arena.

    THIS IS WHAT DELETES THE PINNED STORE. The store exists to hand the
    pipeline a layer's 272 cold experts as packed host rows; those same bytes
    are already in kt's arena, which the demotion path has registered. So the
    copy engine can read them directly and the ~51 GiB per rank (409 GiB across
    TP8) never needs to exist.

    SIX COPIES PER LAYER, not 272. Two properties make that possible, and both
    are checked at construction rather than assumed:

      * kt bump-allocates the resident experts in order, so each kind's buffers
        sit at a CONSTANT stride -- an expert's gate block is
        ``base_gate + k * stride``. One cudaMemcpy2DAsync per kind therefore
        gathers all 272, with spitch = stride and width = the per-expert slice.
      * At cpu_tp == gpu_tp (one rank per partition) per_numa == per_gpu, so
        w2_width == w2_pitch and a rank's down-projection strips are CONTIGUOUS
        inside the block. The interleave that forces a strided read at coarser
        partitionings simply is not there, so w2 is one pitched copy like the
        rest. At cpu_tp < gpu_tp it would need one copy per expert; this class
        refuses instead of silently doing 272x the launches.

    SLOT ORDER IS ADDRESS ORDER. Staging row j must hold the expert in cold
    slot j. kt allocates blocks in ascending logical id and sglang assigns cold
    slots the same way, so block j is slot j at boot; a swap hands the block to
    the demoted expert and exchanges the slot entry in step, so the two stay
    aligned and the physical blocks never move. Sorting the live offsets by
    address therefore yields slot order at any point in the run.

    Duck-compatible with ColdExpertStore where ColdExpertPipeline touches it.
    """

    def __init__(self, *, dma, offsets_by_layer, geometry, layers, num_cold, experts):
        self._dma = dma
        self._g = geometry
        self.num_cold = int(num_cold)
        self._layers = sorted(layers)
        self.raw_layout = True  # the arena is checkpoint layout
        self.dirty = False
        g = geometry
        if g.w2_width != g.w2_pitch:
            raise ValueError(
                f"arena DMA cold source needs one rank per partition "
                f"(w2_width {g.w2_width} != w2_pitch {g.w2_pitch}); at coarser "
                f"partitionings a layer would cost 272 pitched copies, not 6"
            )
        # base address per (layer, kind) and the shared stride, derived from
        # the live offsets and verified to be a true arithmetic progression.
        self._plan: Dict[int, Tuple[int, List[int]]] = {}
        for layer_idx in self._layers:
            offs = offsets_by_layer[layer_idx]
            rows = [
                r
                for r in (offs.get(e, g.part) for e in range(experts))
                if r is not None
            ]
            if len(rows) != self.num_cold:
                raise ValueError(
                    f"layer {layer_idx}: {len(rows)} resident experts in "
                    f"partition {g.part}, expected {self.num_cold}"
                )
            bases, stride = [], None
            for kind in (_GATE_B, _UP_B, _DOWN_B, _GATE_D, _UP_D, _DOWN_D):
                vals = sorted(int(r[kind]) for r in rows)
                deltas = {b - a for a, b in zip(vals, vals[1:])}
                if len(deltas) != 1:
                    raise ValueError(
                        f"layer {layer_idx} kind {kind}: expert blocks are not "
                        f"evenly strided ({len(deltas)} distinct deltas); the "
                        f"bulk pitched read would read the wrong bytes"
                    )
                d = deltas.pop()
                if stride is None:
                    stride = d
                elif d != stride:
                    raise ValueError(
                        f"layer {layer_idx}: kind {kind} strides by {d} but "
                        f"another kind strides by {stride}"
                    )
                bases.append(vals[0])
            self._plan[int(layer_idx)] = (int(stride), bases)

    # -- ColdExpertStore-compatible lifecycle -----------------------------
    def after_enqueue(self, layer_idx: int, stream) -> None:
        return

    def reset(self) -> None:
        return

    def issue_layer_copies(self, layer_idx: int, raw, stream) -> None:
        """Gather one layer's whole cold set into ``raw``, six pitched copies."""
        g = self._g
        stride, bases = self._plan[int(layer_idx)]
        base = self._dma._base[int(layer_idx)]
        n = self.num_cold
        names = list(raw.keys())
        w13, w13_s, w2, w2_s = (raw[k] for k in names[:4])
        cp = self._dma._copy
        # gate | up halves into the packed [n, 2*gu_w] destination
        cp.memcpy2d_h2d(w13.data_ptr(), 2 * g.gu_w,
                        base + bases[0], stride, g.gu_w, n, stream)
        cp.memcpy2d_h2d(w13.data_ptr() + g.gu_w, 2 * g.gu_w,
                        base + bases[1], stride, g.gu_w, n, stream)
        cp.memcpy2d_h2d(w13_s.data_ptr(), 2 * g.gu_s,
                        base + bases[3], stride, g.gu_s, n, stream)
        cp.memcpy2d_h2d(w13_s.data_ptr() + g.gu_s, 2 * g.gu_s,
                        base + bases[4], stride, g.gu_s, n, stream)
        # down: contiguous per expert (w2_width == w2_pitch), so a whole
        # expert's block is one row of the pitched copy.
        cp.memcpy2d_h2d(w2.data_ptr(), g.hidden * g.w2_width,
                        base + bases[2], stride, g.hidden * g.w2_width, n, stream)
        cp.memcpy2d_h2d(w2_s.data_ptr(), g.hidden * g.w2s_width,
                        base + bases[5], stride, g.hidden * g.w2s_width, n, stream)


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
        dma=None,  # ArenaDmaWriter, or None for the staged host path
    ):
        self._arena = arena_by_layer
        self._offsets = offsets_by_layer
        self._g = geometry
        self._reader = shard_reader
        self._dma = dma
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
            g = self._g
            for shard, demote_id in zip(shards, demote_ids):
                if self._dma is not None:
                    # STAY ON DEVICE. The copy engine reads from here straight
                    # into kt's arena, so pulling to host first would add back
                    # exactly the hop this path exists to remove. Contiguous
                    # uint8 views because the copies are pointer arithmetic.
                    self._staged[int(demote_id)] = {
                        "w13": shard.w13.contiguous().view(torch.uint8).reshape(-1),
                        "w13_scale": shard.w13_scale_e8m0.contiguous()
                        .view(torch.uint8)
                        .reshape(-1),
                        "w2": shard.w2.contiguous()
                        .view(torch.uint8)
                        .reshape(g.hidden, g.w2_width),
                        "w2_scale": shard.w2_scale_e8m0.contiguous()
                        .view(torch.uint8)
                        .reshape(g.hidden, g.w2s_width),
                    }
                    continue
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
            g = self._g
            lr = g.local_rank

            if self._dma is not None:
                # One hop: the copy engine reads the demoted rows off the GPU
                # and lands them at kt's offsets itself. Async on the current
                # stream, so it is ordered against the window's other GPU work
                # and completed by the flush's existing device sync -- there is
                # no correctness window here because the swap window holds the
                # pipeline quiesced until it returns.
                self._dma.write(
                    layer_idx=layer_idx,
                    row=row,
                    shard=shard,
                    stream=torch.cuda.current_stream().cuda_stream,
                )
                # write_s is accumulated by the finally below on every exit,
                # including this one -- adding it here too double-counted it.
                self.written += 1
                self.last_installed = (int(layer_idx), int(demote_id))
                return True

            arena = self._arena[layer_idx]
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
