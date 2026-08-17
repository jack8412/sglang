# SPDX-License-Identifier: Apache-2.0
"""Direct-DMA cold-expert transport: DMA kt's arenas in place, no export.

The export design moved every cold byte through DRAM three times per layer
(export read + export write + DMA read, ~14.5 GB/layer against a ~500 GB/s
bus that also feeds everything else). This transport deletes the prepare
stage entirely: each rank cudaHostRegisters its READ-SET of kt's memfd
arenas (SPEC-DIRECT-DMA.md) and its copy engine reads the weights in place
-- one DRAM transit, the bottleneck moves to PCIe (~22.5 ms/layer measured
floor), and the pacing machinery disappears because weights are immutable.

Three pieces live here:

``IntervalRegistrar``
    Owns every cudaHostRegister/Unregister call for the arenas. Durable
    objects are immutable UNITS -- the exact (ptr, size) of each register
    call, because cudaHostUnregister accepts only the exact base of a prior
    register and frees that whole region (Probe A: non-base -> error 1).
    Page-granular accounting decides which units are trim-eligible; units
    are split at buffer boundaries so one live expert never pins an
    unbounded neighborhood. Registration is cheap enough (Probe A: 0.117
    us/page; 110 GB in ~3.4 s) that the design optimizes for simplicity.

``_LayerPlan`` / plan builder
    Per (layer, rank) precomputed copy programs in three op classes:
    contiguous (gate/up slices + their scales), pitched (w2 weight columns:
    one 2D copy per expert), and whole-block (w2 scales -- the exact slice
    would be a 12-byte-wide pitched copy, a DMA-efficiency trap, so the
    whole ~4x-wider block lands and one strided D2D compact produces the
    slice layout the swizzle expects). Every op is split at registration-
    unit seams: CUDA does not guarantee a copy spanning two separately
    registered ranges stays on the pinned path.

``DirectDmaSource``
    The pipeline-facing source. Duck-compatible with ExportColdSource where
    ColdExpertPipeline touches a source, plus ``issue_layer_copies`` which
    the pipeline prefers when present: the source owns the H2D issue
    because there is no host-side staging to hand back. Plans are cached
    per layer and invalidated by a table-version bump after every swap
    window -- the ONLY time logical_to_slot changes.

Swap-window contract (invariant 1 of the spec): ``window_begin_layer``
acquires the demoted experts' ranges BEFORE the tables flip and reports
per-pair success; the caller runs one fixed-shape collective over the
bitmask and drops failed pairs everywhere. Releases and trims are keyed to
the flipped table via the plan rebuild that the version bump forces.
"""

from __future__ import annotations

import bisect
import ctypes
import ctypes.util
import logging
import os
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from sglang.srt.layers.moe.expert_cold_store import WEIGHT_NAMES

logger = logging.getLogger(__name__)

PAGE = 4096
_CUDA_MEMCPY_HOST_TO_DEVICE = 1


def _page_floor(v: int) -> int:
    return v & ~(PAGE - 1)


def _page_ceil(v: int) -> int:
    return (v + PAGE - 1) & ~(PAGE - 1)


def _merge_ranges(ranges: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Page-round, sort and coalesce; overlapping inputs are legal."""
    rounded = sorted((_page_floor(lo), _page_ceil(hi)) for lo, hi in ranges)
    out: List[Tuple[int, int]] = []
    for lo, hi in rounded:
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


class IntervalRegistrar:
    """Registration-unit bookkeeping for pinned arena ranges (one per rank).

    ``register_fn(ptr, nbytes) -> rc`` / ``unregister_fn(ptr) -> rc`` are
    injected so the logic is unit-testable without CUDA; production wires
    them to torch.cuda.cudart(). Not thread-safe by design: every caller
    runs on the forward thread or inside a quiesced swap window.
    """

    # Backoff schedule for transient (rc=2) registration failures; class
    # attribute so tests can zero the sleeps. D2 measured the at-limit
    # window lasting minutes, so the schedule reaches ~18 s per unit -- the
    # pre-registration headroom reclaim (below) is what makes long windows
    # rare; this is the second line of defense, not the first.
    RETRY_DELAYS = (0.1, 0.5, 2.0, 5.0, 10.0)

    def __init__(
        self,
        *,
        register_fn: Callable[[int, int], int],
        unregister_fn: Callable[[int], int],
    ):
        self._register = register_fn
        self._unregister = unregister_fn
        # Units: parallel sorted arrays. _unit_starts is the bisect key.
        self._unit_starts: List[int] = []
        self._unit_ends: List[int] = []
        self._unit_seq: List[int] = []  # creation order, for oldest-first trim
        self._seq = 0
        # Live acquisitions: key -> merged page ranges. A key acquired twice
        # stacks (refcount), which re-demotion within one window can produce.
        self._live: Dict[object, List[List[Tuple[int, int]]]] = {}
        self._live_index_dirty = True
        self._live_index: List[Tuple[int, int]] = []

    # -- coverage ----------------------------------------------------------

    def _covered_gaps(self, lo: int, hi: int) -> List[Tuple[int, int]]:
        """Sub-ranges of [lo, hi) NOT covered by any unit."""
        gaps = []
        pos = lo
        i = bisect.bisect_right(self._unit_starts, lo) - 1
        if i >= 0 and self._unit_ends[i] > lo:
            pos = min(self._unit_ends[i], hi)
        i += 1
        while pos < hi and i < len(self._unit_starts):
            s, e = self._unit_starts[i], self._unit_ends[i]
            if s >= hi:
                break
            if s > pos:
                gaps.append((pos, s))
            pos = max(pos, min(e, hi))
            i += 1
        if pos < hi:
            gaps.append((pos, hi))
        return gaps

    def segments(self, lo: int, hi: int) -> List[Tuple[int, int, int]]:
        """Cover [lo, hi) with (seg_lo, seg_hi, unit_index) pieces.

        Raises if any byte is uncovered -- issuing a DMA from unregistered
        memory must be loud, never a silent pageable copy.
        """
        out = []
        pos = lo
        i = bisect.bisect_right(self._unit_starts, pos) - 1
        while pos < hi:
            if i < 0 or i >= len(self._unit_starts) or not (
                self._unit_starts[i] <= pos < self._unit_ends[i]
            ):
                raise RuntimeError(
                    f"[kt-dma] range [{lo:#x},{hi:#x}) not fully registered "
                    f"(hole at {pos:#x})"
                )
            seg_hi = min(self._unit_ends[i], hi)
            out.append((pos, seg_hi, i))
            pos = seg_hi
            i += 1
        return out

    def spans_single_unit(self, lo: int, hi: int) -> bool:
        i = bisect.bisect_right(self._unit_starts, lo) - 1
        return (
            i >= 0
            and self._unit_starts[i] <= lo
            and hi <= self._unit_ends[i]
        )

    # -- mutation ----------------------------------------------------------

    def _insert_unit(self, lo: int, hi: int) -> None:
        i = bisect.bisect_left(self._unit_starts, lo)
        self._unit_starts.insert(i, lo)
        self._unit_ends.insert(i, hi)
        self._unit_seq.insert(i, self._seq)
        self._seq += 1

    def _drop_unit(self, i: int) -> None:
        del self._unit_starts[i]
        del self._unit_ends[i]
        del self._unit_seq[i]

    def acquire(
        self,
        key: object,
        ranges: Sequence[Tuple[int, int]],
        *,
        boundaries: Sequence[int] = (),
    ) -> bool:
        """Ensure every byte of ``ranges`` is registered; record ``key`` live.

        New units are split at ``boundaries`` (absolute addresses -- buffer
        starts) so trim granularity stays per-buffer. All-or-nothing: on any
        registration failure the units created by THIS call are unregistered
        and False is returned, leaving state exactly as before.
        """
        merged = _merge_ranges(ranges)
        created: List[int] = []  # unit start addrs created by this call
        for lo, hi in merged:
            for g_lo, g_hi in self._covered_gaps(lo, hi):
                pieces = []
                cuts = [
                    b
                    for b in sorted(set(_page_floor(b) for b in boundaries))
                    if g_lo < b < g_hi
                ]
                prev = g_lo
                for c in cuts:
                    pieces.append((prev, c))
                    prev = c
                pieces.append((prev, g_hi))
                for p_lo, p_hi in pieces:
                    rc = self._register(p_lo, p_hi - p_lo)
                    # rc 2 (cudaErrorMemoryAllocation) is TRANSIENT here:
                    # D1's boot saw 5/8 ranks fail within one second at peak
                    # page-cache pressure (1.6 TB of shard reads still
                    # resident) while 3 ranks sailed through the identical
                    # 102 GiB -- kernel allocation pressure during the 8-way
                    # registration storm, not a cap (Probe B registered
                    # 850 GB nominal cleanly). Back off and retry: reclaim
                    # needs a moment, not a redesign.
                    if rc == 2:
                        for delay in self.RETRY_DELAYS:
                            time.sleep(delay)
                            rc = self._register(p_lo, p_hi - p_lo)
                            if rc != 2:
                                break
                    if rc != 0:
                        logger.error(
                            "[kt-dma] cudaHostRegister(%#x, %d) failed rc=%d; "
                            "rolling back this acquire",
                            p_lo,
                            p_hi - p_lo,
                            rc,
                        )
                        for c_lo in created:
                            j = bisect.bisect_left(self._unit_starts, c_lo)
                            self._unregister(c_lo)
                            self._drop_unit(j)
                        return False
                    self._insert_unit(p_lo, p_hi)
                    created.append(p_lo)
        self._live.setdefault(key, []).append(merged)
        self._live_index_dirty = True
        return True

    def release(self, key: object) -> None:
        """Drop one stacked acquisition of ``key``; unknown keys are a no-op."""
        stack = self._live.get(key)
        if not stack:
            return
        stack.pop()
        if not stack:
            del self._live[key]
        self._live_index_dirty = True

    def _rebuild_live_index(self) -> None:
        if not self._live_index_dirty:
            return
        flat: List[Tuple[int, int]] = []
        for stack in self._live.values():
            for merged in stack:
                flat.extend(merged)
        self._live_index = _merge_ranges(flat) if flat else []
        self._live_index_dirty = False

    def _unit_is_live(self, lo: int, hi: int) -> bool:
        self._rebuild_live_index()
        i = bisect.bisect_right(self._live_index, (lo, float("inf"))) - 1
        if i >= 0 and self._live_index[i][1] > lo:
            return True
        return i + 1 < len(self._live_index) and self._live_index[i + 1][0] < hi

    def registered_bytes(self) -> int:
        return sum(e - s for s, e in zip(self._unit_starts, self._unit_ends))

    def live_bytes(self) -> int:
        self._rebuild_live_index()
        return sum(e - s for s, e in self._live_index)

    def trim(self, *, budget_bytes: int) -> int:
        """Unregister dead units, oldest first, until under budget.

        QUIESCED CALLERS ONLY: nothing may be mid-DMA from these arenas.

        Single-pass by construction: one merged-live-index build, one
        two-pointer sweep classifying every unit, one sort of the dead set.
        The first cut freed one unit per O(N) rescan, which at the real
        scale (~150k units, ~4.4k dead per window) was measured at 220 s of
        pure Python inside the quiesced window -- long enough to trip the
        watchdog. This version is O(N log N) and sub-second at that scale.
        """
        total = self.registered_bytes()
        if total <= budget_bytes:
            return 0
        self._rebuild_live_index()
        live = self._live_index
        dead: List[int] = []
        j = 0
        for i in range(len(self._unit_starts)):
            s, e = self._unit_starts[i], self._unit_ends[i]
            while j < len(live) and live[j][1] <= s:
                j += 1
            if j < len(live) and live[j][0] < e:
                continue  # overlaps a live range
            dead.append(i)
        dead.sort(key=lambda i: self._unit_seq[i])
        freed = 0
        dropped: List[int] = []
        for i in dead:
            if total - freed <= budget_bytes:
                break
            rc = self._unregister(self._unit_starts[i])
            if rc != 0:
                logger.error(
                    "[kt-dma] cudaHostUnregister(%#x) rc=%d during trim; "
                    "leaving the unit registered",
                    self._unit_starts[i],
                    rc,
                )
                break
            freed += self._unit_ends[i] - self._unit_starts[i]
            dropped.append(i)
        for i in sorted(dropped, reverse=True):
            self._drop_unit(i)
        return freed

    def close(self) -> None:
        """Disarm teardown: unregister every unit, ignore errors."""
        for s in list(self._unit_starts):
            try:
                self._unregister(s)
            except Exception:
                pass
        self._unit_starts, self._unit_ends, self._unit_seq = [], [], []
        self._live.clear()
        self._live_index_dirty = True


# -- cuda runtime access ----------------------------------------------------


class CudaCopyLib:
    """ctypes handle to cudaMemcpyAsync / cudaMemcpy2DAsync.

    torch.cuda.cudart() exposes hostRegister but not the 2D copy, so the
    copies go straight to libcudart. Argtypes are declared once; the issue
    loop then pays only the bare foreign call (~1-2 us).
    """

    def __init__(self):
        self._lib = self._load()
        self._lib.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._lib.cudaMemcpyAsync.restype = ctypes.c_int
        self._lib.cudaMemcpy2DAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._lib.cudaMemcpy2DAsync.restype = ctypes.c_int

    @staticmethod
    def _load() -> ctypes.CDLL:
        # torch's own libcudart first -- it is the runtime every other CUDA
        # call in the process already goes through.
        import glob

        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        candidates = sorted(glob.glob(os.path.join(torch_lib, "libcudart*")))
        for lib_dir in (candidates or []):
            try:
                return ctypes.CDLL(lib_dir)
            except OSError:
                continue
        try:
            import nvidia.cuda_runtime as _cr  # packaged runtime

            for path in sorted(
                glob.glob(os.path.join(os.path.dirname(_cr.__file__), "lib", "libcudart*"))
            ):
                try:
                    return ctypes.CDLL(path)
                except OSError:
                    continue
        except ImportError:
            pass
        name = ctypes.util.find_library("cudart") or "libcudart.so"
        return ctypes.CDLL(name)

    def memcpy_h2d(self, dst: int, src: int, nbytes: int, stream: int) -> None:
        rc = self._lib.cudaMemcpyAsync(
            dst, src, nbytes, _CUDA_MEMCPY_HOST_TO_DEVICE, stream
        )
        if rc != 0:
            raise RuntimeError(f"cudaMemcpyAsync rc={rc}")

    def memcpy2d_h2d(
        self,
        dst: int,
        dpitch: int,
        src: int,
        spitch: int,
        width: int,
        height: int,
        stream: int,
    ) -> None:
        rc = self._lib.cudaMemcpy2DAsync(
            dst, dpitch, src, spitch, width, height,
            _CUDA_MEMCPY_HOST_TO_DEVICE, stream,
        )
        if rc != 0:
            raise RuntimeError(f"cudaMemcpy2DAsync rc={rc}")


_ERROR_CLEAR_LIB: Optional[ctypes.CDLL] = None


def _clear_sticky_cuda_error() -> None:
    """Swallow CUDA's per-thread sticky error after a failed runtime call.

    torch's cudart binding returns the raw rc WITHOUT clearing last-error, so
    a failed cudaHostRegister -- a case this transport treats as recoverable
    (arming falls back, a swap pair is skipped) -- would otherwise surface as
    a phantom 'CUDA error' at the next kernel-launch check on this thread,
    turning designed degradation into a crash mid-boot or a one-rank crash
    plus TP hang mid-window. cudaGetLastError is the only call that resets
    the state; go straight to libcudart for it.
    """
    global _ERROR_CLEAR_LIB
    try:
        if _ERROR_CLEAR_LIB is None:
            _ERROR_CLEAR_LIB = CudaCopyLib._load()
            _ERROR_CLEAR_LIB.cudaGetLastError.restype = ctypes.c_int
        _ERROR_CLEAR_LIB.cudaGetLastError()
    except Exception:
        logger.exception("[kt-dma] could not clear the sticky CUDA error")


def cudart_register_fns() -> Tuple[Callable[[int, int], int], Callable[[int], int]]:
    cudart = torch.cuda.cudart()

    def reg(ptr: int, nbytes: int) -> int:
        rc = int(cudart.cudaHostRegister(ptr, nbytes, 0))
        if rc != 0:
            _clear_sticky_cuda_error()
        return rc

    def unreg(ptr: int) -> int:
        rc = int(cudart.cudaHostUnregister(ptr))
        if rc != 0:
            _clear_sticky_cuda_error()
        return rc

    return reg, unreg


# -- per-expert range math ---------------------------------------------------


class ArenaExpertRanges:
    """Absolute source addresses of one rank's read-set, per (layer, expert).

    Wraps a ``KtArenaExpertSource`` (which already holds the mapped arenas,
    the per-(partition, expert) OFFSET rows and the geometry) and turns them
    into the address arithmetic the plans and the registrar consume:
    absolute address = this rank's mapping base + exported offset. The K3
    shape -- a rank's slice wholly inside ONE partition (per_numa a multiple
    of per_gpu) -- is asserted at construction; the general multi-piece case
    falls back to the export transport rather than guessing.
    """

    def __init__(self, source):
        from sglang.srt.layers.moe import kt_ram_source as krs

        self._src = source
        self.hidden = source.hidden
        self.per_numa = source.per_numa
        self.per_gpu = source.per_gpu
        self.group = source.group
        if self.per_numa % self.per_gpu:
            raise ValueError(
                f"per_numa {self.per_numa} not a multiple of per_gpu "
                f"{self.per_gpu}; direct-DMA needs the single-partition shape"
            )
        self.part = (source.tp_rank * self.per_gpu) // self.per_numa
        self.local_rank = (
            source.tp_rank - self.part * (self.per_numa // self.per_gpu)
        )
        # This rank only ever reads its own partition's arena.
        self._arena_base = int(source._arenas[self.part].data_ptr())
        # byte widths
        self.gu_w = self.per_gpu * self.hidden // 2
        self.gu_s = self.per_gpu * (self.hidden // self.group)
        self.w2_width = self.per_gpu // 2
        self.w2_pitch = self.per_numa // 2
        self.w2s_width = self.per_gpu // self.group
        self.w2s_pitch = self.per_numa // self.group
        self._kinds = (
            krs._GATE_B, krs._UP_B, krs._DOWN_B,
            krs._GATE_D, krs._UP_D, krs._DOWN_D,
        )

    def _addr(self, expert: int, kind_pos: int) -> int:
        src = self._src
        row = src._rows[self.part * src.experts + expert]
        off = row[self._kinds[kind_pos]]
        if off < 0:
            raise RuntimeError(f"expert {expert} absent from partition {self.part}")
        return self._arena_base + int(off)

    def op_addrs(self, expert: int) -> Dict[str, Tuple]:
        """Source addresses for the three op classes of one expert."""
        lr = self.local_rank
        gate_b = self._addr(expert, 0) + lr * self.gu_w
        up_b = self._addr(expert, 1) + lr * self.gu_w
        gate_d = self._addr(expert, 3) + lr * self.gu_s
        up_d = self._addr(expert, 4) + lr * self.gu_s
        down_b = self._addr(expert, 2)
        down_d = self._addr(expert, 5)
        return {
            "w13": ((gate_b, self.gu_w), (up_b, self.gu_w)),
            "w13_scale": ((gate_d, self.gu_s), (up_d, self.gu_s)),
            # pitched: (base + lr*width, spitch, width, height)
            "w2": (down_b + lr * self.w2_width, self.w2_pitch, self.w2_width, self.hidden),
            # whole block, contiguous: (base, nbytes); compact selects cols
            "w2_scale_block": (down_d, self.hidden * self.w2s_pitch),
        }

    def register_ranges(self, expert: int) -> List[Tuple[int, int]]:
        """Byte ranges this rank must have registered for one expert."""
        ops = self.op_addrs(expert)
        (g, gn), (u, un) = ops["w13"]
        (gd, gdn), (ud, udn) = ops["w13_scale"]
        w2_lo, w2_pitch, w2_w, w2_h = ops["w2"]
        # the pitched read spans base..base+(h-1)*pitch+w
        w2_hi = w2_lo + (w2_h - 1) * w2_pitch + w2_w
        blk, blk_n = ops["w2_scale_block"]
        return [
            (g, g + gn),
            (u, u + un),
            (gd, gd + gdn),
            (ud, ud + udn),
            (w2_lo, w2_hi),
            (blk, blk + blk_n),
        ]

    def boundaries(self, expert: int) -> List[int]:
        """Buffer-start addresses, the unit split points for this expert."""
        return [self._addr(expert, k) for k in range(6)]


# -- layer plans -------------------------------------------------------------


class _LayerPlan:
    """One layer's issue program: flat op lists, seam-split, dst-relative."""

    __slots__ = ("contig", "pitched", "row_linear", "version")

    def __init__(self):
        # (dst_name, dst_off, src_addr, nbytes)
        self.contig: List[Tuple[str, int, int, int]] = []
        # (dst_off, dpitch, src_addr, spitch, width, height) into w2_weight
        self.pitched: List[Tuple[int, int, int, int, int, int]] = []
        # seam fallback rows: (dst_off, src_addr, nbytes) into w2_weight
        self.row_linear: List[Tuple[int, int, int]] = []
        self.version = -1


def build_layer_plan(
    *,
    ranges: ArenaExpertRanges,
    registrar: IntervalRegistrar,
    expert_ids: Sequence[int],
    row_bytes: Dict[str, int],
    version: int,
) -> _LayerPlan:
    """Seam-aware plan: every emitted op lies inside ONE registration unit."""
    plan = _LayerPlan()
    plan.version = version
    w13_row = row_bytes["w13_weight"]
    w13s_row = row_bytes["w13_weight_scale"]
    w2_row = row_bytes["w2_weight"]
    w2s_blk_row = ranges.hidden * ranges.w2s_pitch

    def emit_contig(name: str, dst_off: int, src: int, nbytes: int) -> None:
        for seg_lo, seg_hi, _u in registrar.segments(src, src + nbytes):
            plan.contig.append((name, dst_off + (seg_lo - src), seg_lo, seg_hi - seg_lo))

    for j, e in enumerate(expert_ids):
        ops = ranges.op_addrs(int(e))
        (g, gn), (u, un) = ops["w13"]
        emit_contig("w13_weight", j * w13_row, g, gn)
        emit_contig("w13_weight", j * w13_row + gn, u, un)
        (gd, gdn), (ud, udn) = ops["w13_scale"]
        emit_contig("w13_weight_scale", j * w13s_row, gd, gdn)
        emit_contig("w13_weight_scale", j * w13s_row + gdn, ud, udn)
        blk, blk_n = ops["w2_scale_block"]
        emit_contig("w2_scale_block", j * w2s_blk_row, blk, blk_n)

        src, spitch, width, height = ops["w2"]
        dst0 = j * w2_row
        span_lo = src
        span_hi = src + (height - 1) * spitch + width
        if registrar.spans_single_unit(span_lo, span_hi):
            plan.pitched.append((dst0, width, src, spitch, width, height))
            continue
        # Seam inside the block: group rows by the unit containing them;
        # a row that itself straddles units degrades to linear segments.
        run_start = None
        run_unit = None

        def flush_run(upto: int) -> None:
            nonlocal run_start
            if run_start is None:
                return
            n_rows = upto - run_start
            plan.pitched.append(
                (
                    dst0 + run_start * width,
                    width,
                    src + run_start * spitch,
                    spitch,
                    width,
                    n_rows,
                )
            )
            run_start = None

        for r in range(height):
            r_lo = src + r * spitch
            segs = registrar.segments(r_lo, r_lo + width)
            if len(segs) == 1:
                unit = segs[0][2]
                if run_unit != unit:
                    flush_run(r)
                    run_start, run_unit = r, unit
                elif run_start is None:
                    run_start = r
            else:
                flush_run(r)
                run_unit = None
                for seg_lo, seg_hi, _u in segs:
                    plan.row_linear.append(
                        (dst0 + r * width + (seg_lo - r_lo), seg_lo, seg_hi - seg_lo)
                    )
        flush_run(height)
    return plan


# -- the pipeline source -----------------------------------------------------


class DirectDmaSource:
    """Pipeline source that issues its own H2D from registered kt arenas.

    Duck-compatible with ExportColdSource where ColdExpertPipeline touches a
    source (``num_cold``, ``layer_rows``, ``after_enqueue``, ``reset``), and
    additionally exposes ``issue_layer_copies`` which the pipeline uses when
    present. Nearly stateless: no flags, no epochs -- the weights are
    immutable and the only mutable input is logical_to_slot, folded in via
    ``invalidate_plans`` after each swap window.
    """

    def __init__(
        self,
        *,
        ranges_by_layer: Dict[int, ArenaExpertRanges],
        registrar: IntervalRegistrar,
        copy_lib: CudaCopyLib,
        cold_slot_expert_ids: Callable[[int], torch.Tensor],
        raw_shapes: Dict[str, tuple],
        moe_layer_indices: Sequence[int],
        num_cold: int,
        device: torch.device,
    ):
        self._ranges = ranges_by_layer
        self._registrar = registrar
        self._lib = copy_lib
        self._ids_for = cold_slot_expert_ids
        self._layers = sorted(moe_layer_indices)
        self._pos = {layer: i for i, layer in enumerate(self._layers)}
        self.num_cold = int(num_cold)
        self._raw_shapes = raw_shapes
        self._row_bytes = {
            name: self._nbytes(shape, dtype)
            for name, (shape, dtype) in raw_shapes.items()
        }
        self._plans: Dict[int, _LayerPlan] = {}
        self._version = 0
        self._issue_ms_total = 0.0
        self._issue_layers = 0
        # w2-scale whole-block landing, one per pipeline slot (NUM_SLOTS=2):
        # the exact slice would be a 12-byte-wide pitched copy; the whole
        # block is contiguous and ~4x the bytes of a tiny quantity, then one
        # strided D2D compact writes the slice layout the swizzle expects.
        any_r = next(iter(ranges_by_layer.values()))
        self._w2s_block_shape = (
            self.num_cold,
            any_r.hidden,
            any_r.w2s_pitch,
        )
        self._w2s_lo = any_r.local_rank * any_r.w2s_width
        self._w2s_hi = self._w2s_lo + any_r.w2s_width
        # Indexed by MoE-layer POSITION parity, the pipeline's own slot rule,
        # so aborts/re-primes can never desynchronise block reuse from the
        # slot the raw buffers use.
        self._w2s_blocks = [
            torch.empty(self._w2s_block_shape, dtype=torch.uint8, device=device)
            for _ in range(2)
        ]

    @staticmethod
    def _nbytes(shape, dtype) -> int:
        n = 1
        for s in shape:
            n *= int(s)
        return n * torch.empty((), dtype=dtype).element_size()

    # -- plan lifecycle ----------------------------------------------------

    def invalidate_plans(self) -> None:
        """After a swap window: logical_to_slot changed somewhere."""
        self._version += 1

    def _plan(self, layer_idx: int) -> _LayerPlan:
        plan = self._plans.get(layer_idx)
        if plan is not None and plan.version == self._version:
            return plan
        ids = self._ids_for(layer_idx)
        if len(ids) != self.num_cold:
            raise RuntimeError(
                f"layer {layer_idx}: {len(ids)} cold slots vs {self.num_cold}"
            )
        t0 = time.perf_counter()
        plan = build_layer_plan(
            ranges=self._ranges[layer_idx],
            registrar=self._registrar,
            expert_ids=ids.tolist(),
            row_bytes=self._row_bytes,
            version=self._version,
        )
        self._plans[layer_idx] = plan
        dt = (time.perf_counter() - t0) * 1e3
        if dt > 50:
            logger.info(
                "[kt-dma] layer %d plan rebuilt in %.0f ms (%d/%d/%d ops)",
                layer_idx, dt, len(plan.contig), len(plan.pitched),
                len(plan.row_linear),
            )
        return plan

    # -- pipeline hooks ----------------------------------------------------

    def issue_layer_copies(
        self,
        layer_idx: int,
        raw: Dict[str, torch.Tensor],
        stream: torch.cuda.Stream,
    ) -> None:
        """Enqueue the whole layer's H2D on ``stream``; then compact scales.

        Caller (the pipeline) is inside ``torch.cuda.stream(stream)`` with
        the slot's WAR event already waited, exactly like the ring path.
        """
        plan = self._plan(layer_idx)
        s = stream.cuda_stream
        t0 = time.perf_counter()
        bases = {name: raw[name].data_ptr() for name in WEIGHT_NAMES}
        w2s_block = self._w2s_blocks[self._pos[layer_idx] % 2]
        bases["w2_scale_block"] = w2s_block.data_ptr()
        memcpy = self._lib.memcpy_h2d
        for name, dst_off, src, nbytes in plan.contig:
            memcpy(bases[name] + dst_off, src, nbytes, s)
        w2_base = bases["w2_weight"]
        memcpy2d = self._lib.memcpy2d_h2d
        for dst_off, dpitch, src, spitch, width, height in plan.pitched:
            memcpy2d(w2_base + dst_off, dpitch, src, spitch, width, height, s)
        for dst_off, src, nbytes in plan.row_linear:
            memcpy(w2_base + dst_off, src, nbytes, s)
        # Compact: strided D2D producing the per-rank slice layout the
        # swizzle expects. Runs on the same stream, so the pipeline's
        # prefetch event still means "everything for this layer landed".
        raw_w2s = raw["w2_weight_scale"]
        dst = raw_w2s.view(torch.uint8).reshape(
            self.num_cold, self._w2s_block_shape[1], self._w2s_hi - self._w2s_lo
        )
        dst.copy_(w2s_block[:, :, self._w2s_lo : self._w2s_hi], non_blocking=True)
        self._issue_ms_total += (time.perf_counter() - t0) * 1e3
        self._issue_layers += 1

    def layer_rows(self, layer_idx: int, name: str) -> None:
        """Probe hook compatibility: plans have no host wait; keep it cheap."""
        self._plan(layer_idx)
        return None

    def after_enqueue(self, layer_idx: int, stream: torch.cuda.Stream) -> None:
        return  # no staging to recycle, no flags to publish

    def reset(self) -> None:
        # One line per pass, but only alongside the pipeline's own probe so a
        # production log stays quiet (the probe env is re-read per pass there
        # too, so the two lines always appear together).
        from sglang.srt.environ import envs

        if self._issue_layers and envs.SGLANG_DEBUG_KT_PIPELINE_OVERLAP.get():
            logger.info(
                "[kt-dma] issue %.2f ms/layer avg over %d layers; "
                "registered %.1f GiB in %d units",
                self._issue_ms_total / max(self._issue_layers, 1),
                self._issue_layers,
                self._registrar.registered_bytes() / (1 << 30),
                len(self._registrar._unit_starts),
            )
        self._issue_ms_total = 0.0
        self._issue_layers = 0

    def close(self) -> None:
        self._registrar.close()

    # -- swap-window integration ------------------------------------------

    def window_acquire_demotions(
        self, layer_idx: int, demote_ids: Sequence[int]
    ) -> List[bool]:
        """Invariant 1, step (a): register each pair's demoted expert.

        Returns per-pair local success; the CALLER runs the fixed-shape
        collective and calls ``window_release`` for pairs any rank failed.
        """
        r = self._ranges[layer_idx]
        ok = []
        for d in demote_ids:
            d = int(d)
            try:
                ok.append(
                    self._registrar.acquire(
                        (layer_idx, d),
                        r.register_ranges(d),
                        boundaries=r.boundaries(d),
                    )
                )
            except Exception:
                logger.exception(
                    "[kt-dma] acquire failed for layer %d expert %d",
                    layer_idx, d,
                )
                ok.append(False)
        return ok

    def window_release(self, layer_idx: int, expert_id: int) -> None:
        self._registrar.release((layer_idx, int(expert_id)))

    def window_trim(self, *, budget_bytes: int) -> int:
        """Quiesced-window trim to budget; returns bytes freed."""
        return self._registrar.trim(budget_bytes=budget_bytes)

    def registered_bytes(self) -> int:
        return self._registrar.registered_bytes()

    def live_bytes(self) -> int:
        return self._registrar.live_bytes()


# -- construction ------------------------------------------------------------


def cgroup_headroom_bytes() -> Optional[int]:
    """memory.max - memory.current for this container, or None.

    The decisive metric on a cgroup-limited node: D2's boot had ~500 GB of
    host MemAvailable while the CGROUP sat at its 1916 GiB memory.max
    (memory.events counted 22,901 max-limit hits) -- kernel-side charges for
    each cudaHostRegister then fail intermittently with ENOMEM (rc=2) while
    every host-wide metric looks healthy.
    """
    try:
        with open("/sys/fs/cgroup/memory.max") as f:
            raw = f.read().strip()
        if raw == "max":
            return None
        with open("/sys/fs/cgroup/memory.current") as f:
            cur = int(f.read().strip())
        return int(raw) - cur
    except (OSError, ValueError):
        return None


def reclaim_headroom_for_registration(
    *, weight_path: str, floor_bytes: int
) -> None:
    """Best-effort: drop checkpoint page cache until the cgroup has headroom.

    The registration storm charges kernel memory to the cgroup; if the
    cgroup is at memory.max the charges fail (rc=2). The checkpoint's file
    cache is the one big reclaimable charge at this point of boot (the
    weights themselves are unswappable shmem), and it is pure cache -- the
    files were fully consumed by the load. POSIX_FADV_DONTNEED is
    per-inode, immediate, and costs seconds across the whole tree.
    """
    import glob as _glob

    head = cgroup_headroom_bytes()
    if head is None or head >= floor_bytes:
        return
    t0 = time.perf_counter()
    dropped = 0
    for path in sorted(_glob.glob(os.path.join(weight_path, "*.safetensors"))):
        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                dropped += 1
            finally:
                os.close(fd)
        except OSError:
            continue
    after = cgroup_headroom_bytes()
    logger.info(
        "[kt-dma] cgroup headroom %.0f GB under the %.0f GB floor: dropped "
        "%d checkpoint files' cache in %.1f s -> headroom %.0f GB",
        head / 1e9,
        floor_bytes / 1e9,
        dropped,
        time.perf_counter() - t0,
        (after or 0) / 1e9,
    )


def boot_acquire_cold_set(
    *,
    source: DirectDmaSource,
    registrar: IntervalRegistrar,
    ranges_by_layer: Dict[int, ArenaExpertRanges],
    cold_slot_expert_ids: Callable[[int], torch.Tensor],
) -> bool:
    """Register the initial cold set (invariant 3). All-or-nothing."""
    t0 = time.perf_counter()
    for layer_idx, r in sorted(ranges_by_layer.items()):
        for e in cold_slot_expert_ids(layer_idx).tolist():
            if not registrar.acquire(
                (layer_idx, int(e)),
                r.register_ranges(int(e)),
                boundaries=r.boundaries(int(e)),
            ):
                logger.error(
                    "[kt-dma] boot registration failed at layer %d expert %d "
                    "after %.1f GiB",
                    layer_idx, e, registrar.registered_bytes() / (1 << 30),
                )
                return False
    logger.info(
        "[kt-dma] boot registration: %.1f GiB in %d units, %.1f s",
        registrar.registered_bytes() / (1 << 30),
        len(registrar._unit_starts),
        time.perf_counter() - t0,
    )
    return True
