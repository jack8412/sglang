# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the kt arena geometry.

Runs CUDA-free: :class:`ArenaExpertRanges` is pure address arithmetic over
kt's memfd arenas.

This file used to test a direct-DMA interval registrar and per-expert plan
builder as well. That transport is gone -- it was gated
``and not cold_only_cpu_experts``, never armed on any node, and issued ~1,632
copies per layer where ArenaDmaColdSource issues 6 -- so only the geometry
cases remain, which is what the production cold source and the rank writer
are actually built from.
"""
import importlib.util
import os
import sys
import types
import unittest

import torch


def _load_kt_arena_geometry():
    """Import the module under test, standalone if the package won't import.

    The dev VM lacks orjson, which sglang's package __init__ pulls in; the
    module under test only needs torch, WEIGHT_NAMES and kt_ram_source's
    constants, so fall back to loading those directly by file path with a
    stub package hierarchy.
    """
    try:
        from sglang.srt.layers.moe import kt_arena_geometry

        return kt_arena_geometry
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
    return load("sglang.srt.layers.moe.kt_arena_geometry", "kt_arena_geometry.py")


_dma = _load_kt_arena_geometry()
ArenaExpertRanges = _dma.ArenaExpertRanges
PAGE = _dma.PAGE



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


class TestArenaExpertRanges(unittest.TestCase):
    def test_local_rank_offsets_differ_between_ranks(self):
        """Two ranks in one partition read DIFFERENT bytes at the SAME pitch.

        This is the property the whole cold path rests on: a rank's slice is
        its own column strip of the partition's block, so the base moves with
        local_rank while the pitch does not. Getting it wrong yields
        right-shaped wrong weights, silently.
        """
        r0 = ArenaExpertRanges(_FakeArenaSource(tp_rank=0))
        r1 = ArenaExpertRanges(_FakeArenaSource(tp_rank=1))
        a0, a1 = r0.op_addrs(1), r1.op_addrs(1)
        self.assertNotEqual(a0["w13"][0][0], a1["w13"][0][0])
        self.assertEqual(a0["w2"][1], a1["w2"][1])  # same pitch

    def test_rank_maps_to_its_own_partition(self):
        """Rank 4 of 8 lands in partition 1 when there are 2 partitions."""
        r4 = ArenaExpertRanges(_FakeArenaSource(tp_rank=4))
        self.assertEqual(r4.part, 1)
        self.assertEqual(r4.local_rank, 0)


if __name__ == "__main__":
    unittest.main()
