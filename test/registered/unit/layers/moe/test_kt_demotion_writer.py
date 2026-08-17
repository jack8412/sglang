# SPDX-License-Identifier: Apache-2.0
"""The property the rank-write demotion path lives or dies on.

Eight ranks each write only their own slice into a shared arena, and the
result must equal, byte for byte, what kt's ``fill_expert_buffers`` would
have written from the FULL expert. If it does not, the failure mode is
silent: a valid-looking expert holding another rank's weights, or a hole.

Everything here is CPU-only. The arena is a plain uint8 tensor standing in
for the memfd mapping, and the reference is a direct transcription of kt's
fill (``fp4-moe.hpp``: gate/up are row-major memcpys of this partition's
rows; down is copied one hidden row at a time out of the full intermediate
dimension; scales likewise).
"""

import importlib.util
import os
import sys
import types
import unittest

import torch


def _load(mod_name, filename):
    here = os.path.dirname(os.path.abspath(__file__))
    moe_dir = os.path.normpath(
        os.path.join(here, "../../../../../python/sglang/srt/layers/moe")
    )
    for name in ("sglang", "sglang.srt", "sglang.srt.layers", "sglang.srt.layers.moe"):
        sys.modules.setdefault(name, types.ModuleType(name))
    spec = importlib.util.spec_from_file_location(
        mod_name, os.path.join(moe_dir, filename)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


try:
    from sglang.srt.layers.moe import kt_demotion_writer as _dw
except ModuleNotFoundError:  # dev VM without orjson: load standalone
    _dw = _load("sglang.srt.layers.moe.kt_demotion_writer", "kt_demotion_writer.py")

SlotOffsets = _dw.SlotOffsets
RankShardWriter = _dw.RankShardWriter

# K3 geometry, shrunk on the hidden axis so the test is instant. The ratios
# that matter are preserved: 2 NUMA partitions, 4 GPU ranks per partition.
NUMA, TP_SIZE, GROUP = 2, 8, 32
HIDDEN, PER_NUMA = 256, 1536
EXPERTS = 6
INTERMEDIATE = PER_NUMA * NUMA          # 3072
PER_GPU = INTERMEDIATE // TP_SIZE       # 384
GU_W = PER_GPU * HIDDEN // 2            # per-rank gate (or up) bytes
GU_S = PER_GPU * (HIDDEN // GROUP)
W2_WIDTH = PER_GPU // 2                 # per-rank down strip, bytes
W2_PITCH = PER_NUMA // 2                # partition row width
W2S_WIDTH = PER_GPU // GROUP
W2S_PITCH = PER_NUMA // GROUP
GU_BLOCK = PER_NUMA * HIDDEN // 2       # partition gate buffer size
GUS_BLOCK = PER_NUMA * (HIDDEN // GROUP)
W2_BLOCK = HIDDEN * W2_PITCH
W2S_BLOCK = HIDDEN * W2S_PITCH
PER_EXPERT = 2 * (GU_BLOCK + GUS_BLOCK) + W2_BLOCK + W2S_BLOCK


class _Geom:
    """Stands in for ArenaExpertRanges (only the fields the writer reads)."""

    def __init__(self, rank):
        self.hidden = HIDDEN
        self.per_gpu = PER_GPU
        self.group = GROUP
        self.part = rank // (TP_SIZE // NUMA)
        self.local_rank = rank % (TP_SIZE // NUMA)
        self.gu_w = GU_W
        self.gu_s = GU_S
        self.w2_width = W2_WIDTH
        self.w2_pitch = W2_PITCH
        self.w2s_width = W2S_WIDTH
        self.w2s_pitch = W2S_PITCH


def _full_expert(seed):
    """One expert's six FULL tensors, as the checkpoint holds them."""
    g = torch.Generator().manual_seed(seed)

    def r(*shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, generator=g)

    return {
        "gate": r(INTERMEDIATE, HIDDEN // 2),
        "up": r(INTERMEDIATE, HIDDEN // 2),
        "down": r(HIDDEN, INTERMEDIATE // 2),
        "gate_s": r(INTERMEDIATE, HIDDEN // GROUP),
        "up_s": r(INTERMEDIATE, HIDDEN // GROUP),
        "down_s": r(HIDDEN, INTERMEDIATE // GROUP),
    }


def _rank_shard(full, rank):
    """What unswizzle_trtllm_expert hands this rank: build_expert_bytes form."""
    lo, hi = rank * PER_GPU, (rank + 1) * PER_GPU
    return types.SimpleNamespace(
        w13=torch.cat([full["gate"][lo:hi], full["up"][lo:hi]], dim=0).contiguous(),
        w13_scale_e8m0=torch.cat(
            [full["gate_s"][lo:hi], full["up_s"][lo:hi]], dim=0
        ).contiguous(),
        w2=full["down"][:, lo // 2 : hi // 2].contiguous(),
        w2_scale_e8m0=full["down_s"][:, lo // GROUP : hi // GROUP].contiguous(),
    )


def _kt_reference(full, part):
    """kt's fill_expert_buffers, transcribed, for one NUMA partition."""
    lo, hi = part * PER_NUMA, (part + 1) * PER_NUMA
    return {
        "gate": full["gate"][lo:hi].reshape(-1),
        "up": full["up"][lo:hi].reshape(-1),
        "down": full["down"][:, lo // 2 : hi // 2].reshape(-1),
        "gate_s": full["gate_s"][lo:hi].reshape(-1),
        "up_s": full["up_s"][lo:hi].reshape(-1),
        "down_s": full["down_s"][:, lo // GROUP : hi // GROUP].reshape(-1),
    }


class _FakeReader:
    """Returns the pre-computed shard for whichever rank we are simulating."""

    def __init__(self, shards_by_row):
        self._by_row = shards_by_row

    def read_own_shards(self, layer, rows):
        return [self._by_row[r] for r in rows]


def _offset_rows(experts, numa, resident_ids):
    """Bump-packed offsets exactly like kt's arena allocator, -1 if absent."""
    rows = []
    for _part in range(numa):
        off = 0
        part_rows = []
        for e in range(experts):
            if e not in resident_ids:
                part_rows.append([-1] * 6)
                continue
            # order: gate_b, up_b, down_b, gate_d, up_d, down_d
            g_b, off = off, off + GU_BLOCK
            u_b, off = off, off + GU_BLOCK
            d_b, off = off, off + W2_BLOCK
            g_d, off = off, off + GUS_BLOCK
            u_d, off = off, off + GUS_BLOCK
            d_d, off = off, off + W2S_BLOCK
            part_rows.append([g_b, u_b, d_b, g_d, u_d, d_d])
        rows.extend(part_rows)
    return rows


class TestSlotOffsets(unittest.TestCase):
    def test_move_transfers_the_buffer_and_leaves_a_hole(self):
        rows = _offset_rows(EXPERTS, NUMA, resident_ids={0, 1, 2})
        off = SlotOffsets(rows, experts=EXPERTS, part=0)
        promoted = off.get(1)
        self.assertIsNotNone(promoted)
        self.assertIsNone(off.get(4))
        off.apply_move(1, 4)
        self.assertEqual(off.get(4), promoted)
        self.assertIsNone(off.get(1))

    def test_move_refuses_an_absent_promoted_or_occupied_demoted(self):
        rows = _offset_rows(EXPERTS, NUMA, resident_ids={0, 1})
        off = SlotOffsets(rows, experts=EXPERTS, part=0)
        with self.assertRaises(RuntimeError):
            off.apply_move(3, 4)  # promoted holds nothing here
        with self.assertRaises(RuntimeError):
            off.apply_move(0, 1)  # demoted already holds one

    def test_partitions_are_read_independently(self):
        rows = _offset_rows(EXPERTS, NUMA, resident_ids={0, 1, 2})
        p0 = SlotOffsets(rows, experts=EXPERTS, part=0)
        p1 = SlotOffsets(rows, experts=EXPERTS, part=1)
        # Same layout per partition, but they are separate objects: a move on
        # one must not disturb the other (each rank owns only its own).
        self.assertEqual(p0.get(2), p1.get(2))
        p0.apply_move(2, 5)
        self.assertIsNone(p0.get(2))
        self.assertIsNotNone(p1.get(2))


class TestEightRanksReconstructTheExpert(unittest.TestCase):
    """THE property: disjoint per-rank writes == kt's own fill."""

    def _run(self, *, promote_id, demote_id, resident_ids):
        full = _full_expert(seed=1234)
        shards = {rank: _rank_shard(full, rank) for rank in range(TP_SIZE)}
        rows = _offset_rows(EXPERTS, NUMA, resident_ids=resident_ids)
        arena = {
            part: torch.zeros(EXPERTS * PER_EXPERT, dtype=torch.uint8)
            for part in range(NUMA)
        }

        for rank in range(TP_SIZE):
            geom = _Geom(rank)
            writer = RankShardWriter(
                arena_by_layer={7: arena[geom.part]},
                offsets_by_layer={
                    7: SlotOffsets(rows, experts=EXPERTS, part=geom.part)
                },
                geometry=geom,
                # row index is arbitrary; map it to this rank's shard
                shard_reader=_FakeReader({0: shards[rank]}),
            )
            self.assertTrue(writer.capture(object(), 7, [0], [demote_id]))
            self.assertTrue(writer.write(7, promote_id, demote_id))
        return full, arena, rows

    def test_reconstructs_bitwise(self):
        promote_id, demote_id = 1, 4
        full, arena, rows = self._run(
            promote_id=promote_id, demote_id=demote_id, resident_ids={0, 1, 2}
        )
        for part in range(NUMA):
            ref = _kt_reference(full, part)
            base = rows[part * EXPERTS + promote_id]  # buffers the demoted got
            a = arena[part]
            g_b, u_b, d_b, g_d, u_d, d_d = base
            for name, off, want in (
                ("gate", g_b, ref["gate"]),
                ("up", u_b, ref["up"]),
                ("down", d_b, ref["down"]),
                ("gate_s", g_d, ref["gate_s"]),
                ("up_s", u_d, ref["up_s"]),
                ("down_s", d_d, ref["down_s"]),
            ):
                got = a[off : off + want.numel()]
                self.assertTrue(
                    torch.equal(got, want),
                    f"partition {part} {name}: bytes differ "
                    f"({int((got != want).sum())} of {want.numel()})",
                )

    def test_writes_touch_nothing_outside_the_moved_buffers(self):
        promote_id, demote_id = 1, 4
        _full, arena, rows = self._run(
            promote_id=promote_id, demote_id=demote_id, resident_ids={0, 1, 2}
        )
        for part in range(NUMA):
            a = arena[part]
            for other in (0, 2):
                base = rows[part * EXPERTS + other]
                span = a[base[0] : base[0] + PER_EXPERT]
                self.assertEqual(
                    int(span.sum()),
                    0,
                    f"partition {part}: writing expert {demote_id} disturbed "
                    f"expert {other}'s buffers",
                )

    def test_a_missing_rank_leaves_a_detectable_hole(self):
        """Sanity on the test itself: if one rank skips, it must NOT pass."""
        full = _full_expert(seed=99)
        shards = {rank: _rank_shard(full, rank) for rank in range(TP_SIZE)}
        rows = _offset_rows(EXPERTS, NUMA, resident_ids={0, 1, 2})
        arena = {
            part: torch.zeros(EXPERTS * PER_EXPERT, dtype=torch.uint8)
            for part in range(NUMA)
        }
        for rank in range(TP_SIZE):
            if rank == 5:
                continue  # the hole
            geom = _Geom(rank)
            writer = RankShardWriter(
                arena_by_layer={7: arena[geom.part]},
                offsets_by_layer={
                    7: SlotOffsets(rows, experts=EXPERTS, part=geom.part)
                },
                geometry=geom,
                shard_reader=_FakeReader({0: shards[rank]}),
            )
            writer.capture(object(), 7, [0], [4])
            writer.write(7, 1, 4)
        ref = _kt_reference(full, part=1)  # rank 5 lives in partition 1
        base = rows[1 * EXPERTS + 1]
        got = arena[1][base[0] : base[0] + ref["gate"].numel()]
        self.assertFalse(torch.equal(got, ref["gate"]))


if __name__ == "__main__":
    unittest.main()
