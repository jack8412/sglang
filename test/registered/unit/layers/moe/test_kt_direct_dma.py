# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the direct-DMA registrar and plan builder.

Everything here runs CUDA-free: the registrar takes injected register/
unregister callables, and the plan builder only does address arithmetic.
The semantics the FakeCuda enforces are the ones Probe A measured on the
node (2026-08-17): overlap -> 712, unregister of a non-base -> 1, ranges
round to whole pages, adjacent disjoint registrations compose.
"""

import importlib.util
import os
import sys
import types
import unittest

import torch


def _load_kt_direct_dma():
    """Import the module under test, standalone if the package won't import.

    The dev VM lacks orjson, which sglang's package __init__ pulls in; the
    module under test only needs torch, WEIGHT_NAMES and kt_ram_source's
    constants, so fall back to loading those directly by file path with a
    stub package hierarchy.
    """
    try:
        from sglang.srt.layers.moe import kt_direct_dma

        return kt_direct_dma
    except ModuleNotFoundError:
        pass

    here = os.path.dirname(os.path.abspath(__file__))
    moe_dir = os.path.normpath(
        os.path.join(here, "../../../../../python/sglang/srt/layers/moe")
    )

    def stub(name):
        mod = sys.modules.get(name)
        if mod is None:
            mod = types.ModuleType(name)
            sys.modules[name] = mod
        return mod

    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.layers",
        "sglang.srt.layers.moe",
        "sglang.srt.environ",
    ):
        stub(name)

    class _Env:
        def get(self):
            return False

    sys.modules["sglang.srt.environ"].envs = types.SimpleNamespace(
        SGLANG_DEBUG_KT_PIPELINE_OVERLAP=_Env()
    )

    def load(mod_name, filename):
        spec = importlib.util.spec_from_file_location(
            mod_name, os.path.join(moe_dir, filename)
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        setattr(sys.modules["sglang.srt.layers.moe"], mod_name.split(".")[-1], mod)
        return mod

    names = stub("sglang.srt.layers.moe.kt_mxfp4_export")
    names.WEIGHT_NAMES = (
        "w13_weight",
        "w13_weight_scale",
        "w2_weight",
        "w2_weight_scale",
    )
    sys.modules["sglang.srt.layers.moe"].kt_mxfp4_export = names
    load("sglang.srt.layers.moe.kt_ram_source", "kt_ram_source.py")
    return load("sglang.srt.layers.moe.kt_direct_dma", "kt_direct_dma.py")


_dma = _load_kt_direct_dma()
PAGE = _dma.PAGE
ArenaExpertRanges = _dma.ArenaExpertRanges
IntervalRegistrar = _dma.IntervalRegistrar
_merge_ranges = _dma._merge_ranges
build_layer_plan = _dma.build_layer_plan


class FakeCuda:
    """Mimics cudaHostRegister/Unregister semantics per Probe A."""

    def __init__(self, fail_at=None):
        self.regions = {}  # base -> size
        self.calls = []
        # Fail every register call from this index on (persistent failure --
        # the registrar retries rc=2 with backoff, so a one-shot failure
        # would be absorbed by the retry rather than exercising rollback).
        self.fail_at = fail_at
        self._n = 0

    def register(self, ptr, nbytes):
        self.calls.append(("reg", ptr, nbytes))
        assert ptr % PAGE == 0 and nbytes % PAGE == 0, "registrar must page-align"
        self._n += 1
        if self.fail_at is not None and self._n >= self.fail_at:
            return 2  # cudaErrorMemoryAllocation
        for base, size in self.regions.items():
            if ptr < base + size and base < ptr + nbytes:
                return 712  # cudaErrorHostMemoryAlreadyRegistered
        self.regions[ptr] = nbytes
        return 0

    def unregister(self, ptr):
        self.calls.append(("unreg", ptr))
        if ptr not in self.regions:
            return 1  # cudaErrorInvalidValue
        del self.regions[ptr]
        return 0


def make_registrar(fake):
    reg = IntervalRegistrar(register_fn=fake.register, unregister_fn=fake.unregister)
    reg.RETRY_DELAYS = (0, 0, 0)  # keep persistent-failure tests fast
    return reg


class TestMergeRanges(unittest.TestCase):
    def test_rounds_and_merges(self):
        out = _merge_ranges([(10, 100), (PAGE + 1, PAGE + 2), (50, 60)])
        self.assertEqual(out, [(0, 2 * PAGE)])

    def test_disjoint_kept(self):
        out = _merge_ranges([(0, PAGE), (10 * PAGE, 11 * PAGE)])
        self.assertEqual(out, [(0, PAGE), (10 * PAGE, 11 * PAGE)])


class TestIntervalRegistrar(unittest.TestCase):
    def test_acquire_registers_gaps_only(self):
        fake = FakeCuda()
        reg = make_registrar(fake)
        self.assertTrue(reg.acquire("a", [(0, 4 * PAGE)]))
        self.assertTrue(reg.acquire("b", [(2 * PAGE, 8 * PAGE)]))
        # second acquire registered only the uncovered tail
        regs = [c for c in fake.calls if c[0] == "reg"]
        self.assertEqual(regs[0], ("reg", 0, 4 * PAGE))
        self.assertEqual(regs[1], ("reg", 4 * PAGE, 4 * PAGE))
        self.assertEqual(reg.registered_bytes(), 8 * PAGE)

    def test_boundary_splits_units(self):
        fake = FakeCuda()
        reg = make_registrar(fake)
        self.assertTrue(
            reg.acquire("a", [(0, 8 * PAGE)], boundaries=[3 * PAGE, 5 * PAGE])
        )
        regs = [c for c in fake.calls if c[0] == "reg"]
        self.assertEqual(len(regs), 3)
        self.assertEqual(
            sorted(fake.regions.items()),
            [(0, 3 * PAGE), (3 * PAGE, 2 * PAGE), (5 * PAGE, 3 * PAGE)],
        )

    def test_transient_rc2_is_retried(self):
        """One rc=2 then success: the acquire must survive (D1's boot saw
        5/8 ranks fail on exactly this -- transient kernel pressure)."""

        class OneShot(FakeCuda):
            def register(self, ptr, nbytes):
                self._n += 1
                if self._n == 1:
                    self.calls.append(("reg", ptr, nbytes))
                    return 2
                return super().register(ptr, nbytes)

        fake = OneShot()
        reg = make_registrar(fake)
        self.assertTrue(reg.acquire("a", [(0, 2 * PAGE)]))
        self.assertEqual(reg.registered_bytes(), 2 * PAGE)

    def test_failed_acquire_rolls_back(self):
        fake = FakeCuda(fail_at=2)
        reg = make_registrar(fake)
        ok = reg.acquire("a", [(0, 2 * PAGE)], boundaries=[PAGE])
        self.assertFalse(ok)
        self.assertEqual(fake.regions, {})  # first piece was unregistered
        self.assertEqual(reg.registered_bytes(), 0)
        self.assertEqual(reg.live_bytes(), 0)

    def test_segments_and_seams(self):
        fake = FakeCuda()
        reg = make_registrar(fake)
        reg.acquire("a", [(0, 2 * PAGE)])
        reg.acquire("b", [(2 * PAGE, 4 * PAGE)])
        self.assertFalse(reg.spans_single_unit(PAGE, 3 * PAGE))
        self.assertTrue(reg.spans_single_unit(0, 2 * PAGE))
        segs = reg.segments(PAGE, 3 * PAGE)
        self.assertEqual([(s, e) for s, e, _ in segs], [(PAGE, 2 * PAGE), (2 * PAGE, 3 * PAGE)])
        with self.assertRaises(RuntimeError):
            reg.segments(3 * PAGE, 5 * PAGE)  # hole beyond 4*PAGE

    def test_trim_frees_only_dead_units_oldest_first(self):
        fake = FakeCuda()
        reg = make_registrar(fake)
        reg.acquire("a", [(0, 2 * PAGE)])
        reg.acquire("b", [(4 * PAGE, 6 * PAGE)])
        reg.acquire("c", [(8 * PAGE, 10 * PAGE)])
        reg.release("a")
        reg.release("c")
        freed = reg.trim(budget_bytes=2 * PAGE)
        self.assertEqual(freed, 4 * PAGE)  # a then c; b is live and stays
        self.assertEqual(sorted(fake.regions), [4 * PAGE])
        # a's unit went first (oldest)
        unregs = [c for c in fake.calls if c[0] == "unreg"]
        self.assertEqual(unregs[0], ("unreg", 0))

    def test_stacked_acquires_keep_pages_live(self):
        fake = FakeCuda()
        reg = make_registrar(fake)
        reg.acquire("k", [(0, 2 * PAGE)])
        reg.acquire("k", [(0, 2 * PAGE)])  # re-demotion within a window
        reg.release("k")
        self.assertEqual(reg.trim(budget_bytes=0), 0)  # still live once
        reg.release("k")
        self.assertEqual(reg.trim(budget_bytes=0), 2 * PAGE)

    def test_shared_boundary_page_stays_while_neighbor_lives(self):
        fake = FakeCuda()
        reg = make_registrar(fake)
        # two acquires whose page-rounded ranges share the middle page
        reg.acquire("a", [(0, PAGE + 10)])
        reg.acquire("b", [(PAGE + 20, 3 * PAGE)])
        reg.release("a")
        # a's unit [0, 2*PAGE) overlaps b's live page -> must NOT be trimmed
        self.assertEqual(reg.trim(budget_bytes=0), 0)
        reg.release("b")
        # nothing live anymore: everything registered goes
        self.assertEqual(reg.trim(budget_bytes=0), 3 * PAGE)
        self.assertEqual(reg.registered_bytes(), 0)

    def test_trim_is_fast_at_production_scale(self):
        """The first trim cut was O(dead x total): 220 s measured at 150k
        units / 4.4k dead -- a scheduler stall inside the quiesced window.
        The single-pass version must handle that shape in well under a
        second (null unregister fn, pure bookkeeping)."""
        import time

        reg = IntervalRegistrar(
            register_fn=lambda p, n: 0, unregister_fn=lambda p: 0
        )
        n_units = 150_000
        stride = 4 * PAGE
        for i in range(n_units):
            reg.acquire(i, [(i * stride, i * stride + PAGE)])
        for i in range(0, 3 * 4_416, 3):  # ~4.4k dead, scattered
            reg.release(i)
        t0 = time.perf_counter()
        freed = reg.trim(budget_bytes=reg.live_bytes())
        dt = time.perf_counter() - t0
        self.assertGreater(freed, 0)
        self.assertLess(dt, 1.0, f"trim took {dt:.1f} s at production scale")

    def test_close_unregisters_everything(self):
        fake = FakeCuda()
        reg = make_registrar(fake)
        reg.acquire("a", [(0, 2 * PAGE)])
        reg.acquire("b", [(4 * PAGE, 5 * PAGE)])
        reg.close()
        self.assertEqual(fake.regions, {})
        self.assertEqual(reg.registered_bytes(), 0)


class _FakeArenaSource:
    """Minimal stand-in for KtArenaExpertSource with a synthetic layout."""

    def __init__(self, *, experts=6, hidden=64, per_numa=32, group=16,
                 tp_rank=0, tp_size=8, numa=2):
        self.experts = experts
        self.hidden = hidden
        self.per_numa = per_numa
        self.group = group
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.numa = numa
        self.intermediate = per_numa * numa
        self.per_gpu = self.intermediate // tp_size
        w_bytes = per_numa * hidden // 2
        s_bytes = (hidden // group) * per_numa
        # bump-pack per expert: gate_b, up_b, down_b, gate_d, up_d, down_d
        self._rows = []
        arena_size = 0
        offsets_per_part = []
        for part in range(numa):
            part_rows = []
            off = 0
            for e in range(experts):
                row = []
                for nbytes in (w_bytes, w_bytes, w_bytes, s_bytes, s_bytes, s_bytes):
                    row.append(off)
                    off += (nbytes + 63) & ~63
                # reorder to (gate_b, up_b, down_b, gate_d, up_d, down_d)
                part_rows.append(row)
            offsets_per_part.append(part_rows)
            arena_size = max(arena_size, off)
        self._arenas = [
            torch.zeros(arena_size + PAGE, dtype=torch.uint8) for _ in range(numa)
        ]
        # flatten in [part * experts + e] order like the real export
        flat = []
        for part in range(numa):
            flat.extend(offsets_per_part[part])
        self._rows = flat


class TestPlanBuilder(unittest.TestCase):
    def _setup(self, tp_rank=0):
        src = _FakeArenaSource(tp_rank=tp_rank)
        r = ArenaExpertRanges(src)
        fake = FakeCuda()
        reg = make_registrar(fake)
        return src, r, reg

    def _row_bytes(self, r):
        return {
            "w13_weight": 2 * r.gu_w,
            "w13_weight_scale": 2 * r.gu_s,
            "w2_weight": r.hidden * r.w2_width,
            "w2_weight_scale": r.hidden * r.w2s_width,
        }

    def test_register_ranges_cover_all_ops(self):
        _, r, reg = self._setup()
        self.assertTrue(reg.acquire((0, 1), r.register_ranges(1), boundaries=r.boundaries(1)))
        plan = build_layer_plan(
            ranges=r, registrar=reg, expert_ids=[1],
            row_bytes=self._row_bytes(r), version=0,
        )
        # every op must be inside registered memory (segments() would raise)
        self.assertGreater(len(plan.contig), 0)
        total_w13 = sum(n for name, _d, _s, n in plan.contig if name == "w13_weight")
        self.assertEqual(total_w13, 2 * r.gu_w)

    def test_plan_covers_full_rows_across_seams(self):
        src, r, reg = self._setup()
        # register expert 1 in two separate acquires to force a seam inside
        # its down block: first the head page, then the rest.
        ranges = r.register_ranges(1)
        (w2_lo, w2_hi) = ranges[4]
        head = (w2_lo, w2_lo + 1)
        reg.acquire("head", [head])
        self.assertTrue(reg.acquire((0, 1), ranges, boundaries=r.boundaries(1)))
        plan = build_layer_plan(
            ranges=r, registrar=reg, expert_ids=[1],
            row_bytes=self._row_bytes(r), version=0,
        )
        # pitched + row_linear bytes must together cover the whole w2 slice
        pitched_bytes = sum(w * h for _d, _dp, _s, _sp, w, h in plan.pitched)
        linear_bytes = sum(n for _d, _s, n in plan.row_linear)
        self.assertEqual(pitched_bytes + linear_bytes, r.hidden * r.w2_width)
        # and no pitched op may cross a unit seam
        for _dst, _dp, s, sp, w, h in plan.pitched:
            self.assertTrue(reg.spans_single_unit(s, s + (h - 1) * sp + w))

    def test_dst_offsets_are_disjoint_and_in_bounds(self):
        _, r, reg = self._setup(tp_rank=3)
        ids = [0, 2, 4]
        for e in ids:
            self.assertTrue(
                reg.acquire((0, e), r.register_ranges(e), boundaries=r.boundaries(e))
            )
        rb = self._row_bytes(r)
        plan = build_layer_plan(
            ranges=r, registrar=reg, expert_ids=ids, row_bytes=rb, version=0
        )
        # w13 destination coverage == num_experts * row exactly, no overlap
        seen = []
        for name, dst, _src, n in plan.contig:
            if name == "w13_weight":
                seen.append((dst, dst + n))
        seen.sort()
        for (a0, a1), (b0, b1) in zip(seen, seen[1:]):
            self.assertLessEqual(a1, b0)
        self.assertEqual(sum(b - a for a, b in seen), len(ids) * rb["w13_weight"])
        self.assertLessEqual(max(b for _a, b in seen), len(ids) * rb["w13_weight"])

    def test_local_rank_offsets_differ_between_ranks(self):
        _, r0, reg0 = self._setup(tp_rank=0)
        _, r1, reg1 = self._setup(tp_rank=1)
        a0 = r0.op_addrs(1)
        a1 = r1.op_addrs(1)
        self.assertNotEqual(a0["w13"][0][0], a1["w13"][0][0])
        self.assertEqual(a0["w2"][1], a1["w2"][1])  # same pitch
        # rank 4 lands in partition 1
        src4 = _FakeArenaSource(tp_rank=4)
        r4 = ArenaExpertRanges(src4)
        self.assertEqual(r4.part, 1)
        self.assertEqual(r4.local_rank, 0)


if __name__ == "__main__":
    unittest.main()
