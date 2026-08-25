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
import re
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
        off = SlotOffsets(rows, experts=EXPERTS, numa=NUMA)
        promoted = off.get(1, 0)
        self.assertIsNotNone(promoted)
        self.assertIsNone(off.get(4, 0))
        off.apply_move(1, 4)
        self.assertEqual(off.get(4, 0), promoted)
        self.assertIsNone(off.get(1, 0))

    def test_move_refuses_an_absent_promoted_or_occupied_demoted(self):
        rows = _offset_rows(EXPERTS, NUMA, resident_ids={0, 1})
        off = SlotOffsets(rows, experts=EXPERTS, numa=NUMA)
        with self.assertRaises(RuntimeError):
            off.apply_move(3, 4)  # promoted holds nothing here
        with self.assertRaises(RuntimeError):
            off.apply_move(0, 1)  # demoted already holds one

    def test_a_move_applies_to_every_partition(self):
        """kt's move_slot_only runs under do_numa_job, i.e. on ALL partitions.

        A table that tracked only this rank's partition would disagree with kt
        about whether a move is legal, and a fault confined to partition 1
        would make ranks 4-7 refuse while ranks 0-3 proceeded -- a 3-vs-5
        split of the group.
        """
        rows = _offset_rows(EXPERTS, NUMA, resident_ids={0, 1, 2})
        off = SlotOffsets(rows, experts=EXPERTS, numa=NUMA)
        before = [off.get(2, p) for p in range(NUMA)]
        off.apply_move(2, 5)
        for p in range(NUMA):
            self.assertIsNone(off.get(2, p))
            self.assertEqual(off.get(5, p), before[p])

    def test_can_move_refuses_when_any_partition_refuses(self):
        rows = _offset_rows(EXPERTS, NUMA, resident_ids={0, 1, 2})
        off = SlotOffsets(rows, experts=EXPERTS, numa=NUMA)
        self.assertIsNone(off.can_move(1, 4))
        # make partition 1 alone illegal, exactly the split-risk case
        off._by_part[1][1] = None
        self.assertIsNotNone(off.can_move(1, 4))
        with self.assertRaises(RuntimeError):
            off.apply_move(1, 4)


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
                    7: SlotOffsets(rows, experts=EXPERTS, numa=NUMA)
                },
                geometry=geom,
                # row index is arbitrary; map it to this rank's shard
                shard_reader=_FakeReader({0: shards[rank]}),
            )
            self.assertTrue(writer.capture(object(), 7, [0], [demote_id]))
            swaps = [types.SimpleNamespace(promote=promote_id, demote=demote_id)]
            self.assertIsNone(writer.validate(7, swaps))
            writer.commit_move(7, promote_id, demote_id)
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

    def test_write_requires_the_move_to_have_happened(self):
        """Move first, then write -- the reverse corrupts a live expert.

        Writing before the move blits the demoted expert's bytes over the
        PROMOTED expert's buffer, which is still CPU-routable until the tables
        flip; an aborted layer then leaves it serving corrupted weights. So
        write() addresses the DEMOTED expert and refuses until ownership has
        actually moved.
        """
        full = _full_expert(seed=7)
        rows = _offset_rows(EXPERTS, NUMA, resident_ids={0, 1, 2})
        offsets = SlotOffsets(rows, experts=EXPERTS, numa=NUMA)
        writer = RankShardWriter(
            arena_by_layer={7: torch.zeros(EXPERTS * PER_EXPERT, dtype=torch.uint8)},
            offsets_by_layer={7: offsets},
            geometry=_Geom(0),
            shard_reader=_FakeReader({0: _rank_shard(full, 0)}),
        )
        writer.capture(object(), 7, [0], [4])
        # before the move the demoted expert owns nothing -> refuse
        self.assertFalse(writer.write(7, 1, 4))
        writer.commit_move(7, 1, 4)
        self.assertTrue(writer.write(7, 1, 4))

    def test_validate_refuses_before_anything_moves(self):
        """Every refusal must be known BEFORE the irreversible kt move.

        move_slot_only nulls the promoted entry, so nothing fallible may run
        after it. validate() is what lets the install commit only once the
        write is known to be possible.
        """
        rows = _offset_rows(EXPERTS, NUMA, resident_ids={0, 1, 2})
        offsets = SlotOffsets(rows, experts=EXPERTS, numa=NUMA)
        writer = RankShardWriter(
            arena_by_layer={7: torch.zeros(EXPERTS * PER_EXPERT, dtype=torch.uint8)},
            offsets_by_layer={7: offsets},
            geometry=_Geom(0),
            shard_reader=_FakeReader({}),
        )
        swaps = [types.SimpleNamespace(promote=1, demote=4)]
        # nothing captured for this layer
        self.assertIsNotNone(writer.validate(7, swaps))
        # captured, but the promoted expert holds no buffer -> still refused
        writer._staged_layer = 7
        writer._staged[4] = {}
        self.assertIsNotNone(
            writer.validate(7, [types.SimpleNamespace(promote=3, demote=4)])
        )
        self.assertIsNotNone(offsets.get(1, 0))  # untouched

    def test_a_fallback_move_must_be_committed_or_later_writes_corrupt(self):
        """The checkpoint fallback moves kt's slot too.

        If a rank misses that move, its table is one swap behind forever and
        the NEXT write lands in whatever expert now owns those bytes. Here:
        window 1 installs 1->4 by the fallback (bookkeeping only), window 2
        rank-writes 2->5. With the commit, the bytes land in expert 2's
        buffer; without it the table would still think expert 1 is available
        and the offsets would disagree with kt.
        """
        full = _full_expert(seed=11)
        rows = _offset_rows(EXPERTS, NUMA, resident_ids={0, 1, 2})
        arena = torch.zeros(EXPERTS * PER_EXPERT, dtype=torch.uint8)
        offsets = SlotOffsets(rows, experts=EXPERTS, numa=NUMA)
        writer = RankShardWriter(
            arena_by_layer={7: arena},
            offsets_by_layer={7: offsets},
            geometry=_Geom(0),
            shard_reader=_FakeReader({0: _rank_shard(full, 0)}),
        )
        # window 1: checkpoint fallback installed 1 -> 4; only bookkeeping here
        writer.commit_move(7, 1, 4)
        self.assertIsNone(offsets.get(1, 0))
        self.assertIsNotNone(offsets.get(4, 0))

        # window 2: rank-write 2 -> 5 must use expert 2's buffer
        writer.capture(object(), 7, [0], [5])
        expected_base = offsets.get(2, 0)
        writer.commit_move(7, 2, 5)
        self.assertTrue(writer.write(7, 2, 5))
        self.assertEqual(offsets.get(5, 0), expected_base)
        want = _kt_reference(full, part=0)["gate"]
        got = arena[expected_base[0] : expected_base[0] + want.numel()]
        # only rank 0's slice was written, so compare just that span
        self.assertTrue(torch.equal(got[: GU_W], want[: GU_W]))

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
                    7: SlotOffsets(rows, experts=EXPERTS, numa=NUMA)
                },
                geometry=geom,
                shard_reader=_FakeReader({0: shards[rank]}),
            )
            writer.capture(object(), 7, [0], [4])
            writer.commit_move(7, 1, 4)
            writer.write(7, 1, 4)
        ref = _kt_reference(full, part=1)  # rank 5 lives in partition 1
        base = rows[1 * EXPERTS + 1]
        got = arena[1][base[0] : base[0] + ref["gate"].numel()]
        self.assertFalse(torch.equal(got, ref["gate"]))


class TestDirectDmaMatchesTheHostPath(unittest.TestCase):
    """The DMA offsets must land exactly where the host blits land.

    Direct DMA replaces the pinned-staging plus host memcpy with one
    device-to-arena copy per range, and w2's interleave stops being a strided
    CPU write and becomes a cudaMemcpy2DAsync stride. Nothing about that is
    visible if it is wrong: a mis-computed offset or a swapped pitch/width
    writes a valid-looking expert holding the wrong bytes, which is the same
    silent failure the whole module docstring is about.

    So both paths write the same shard into their own arena and the two arenas
    are compared byte for byte. CUDA-gated because cudaHostRegister and the
    copy engine are the things under test; there is no CPU stand-in for them
    that would prove anything."""

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_dma_and_host_writes_are_bitwise_equal(self):
        ArenaDmaWriter = _dw.ArenaDmaWriter
        _blit, _blit_strided = _dw._blit, _dw._blit_strided
        from sglang.srt.layers.moe.kt_arena_geometry import (
            CudaCopyLib,
            cudart_register_fns,
        )

        # One expert's buffers laid out back to back, in kt's row order:
        # gate | up | down | gate_s | up_s | down_s.
        off_gate = 0
        off_up = off_gate + GU_BLOCK
        off_down = off_up + GU_BLOCK
        off_gate_s = off_down + W2_BLOCK
        off_up_s = off_gate_s + GUS_BLOCK
        off_down_s = off_up_s + GUS_BLOCK
        row = [off_gate, off_up, off_down, off_gate_s, off_up_s, off_down_s]
        total = off_down_s + W2S_BLOCK

        host_arena = torch.zeros(total, dtype=torch.uint8)
        dma_arena = torch.zeros(total, dtype=torch.uint8)

        reg_fn, unreg_fn = cudart_register_fns()
        rc = reg_fn(int(dma_arena.data_ptr()), int(dma_arena.numel()))
        self.assertEqual(rc, 0, f"cudaHostRegister rc={rc}")
        try:
            full = _full_expert(seed=7)
            for rank in range(TP_SIZE):
                g = _Geom(rank)
                if g.part != 0:
                    continue  # one partition's arena, as a rank sees it
                sh = _rank_shard(full, rank)
                lr = g.local_rank

                # -- host path, the proven one
                w13 = sh.w13.reshape(-1)
                _blit(host_arena, row[0] + lr * g.gu_w, w13[: g.gu_w])
                _blit(host_arena, row[1] + lr * g.gu_w, w13[g.gu_w :])
                w13s = sh.w13_scale_e8m0.reshape(-1)
                _blit(host_arena, row[3] + lr * g.gu_s, w13s[: g.gu_s])
                _blit(host_arena, row[4] + lr * g.gu_s, w13s[g.gu_s :])
                _blit_strided(
                    host_arena,
                    row[2] + lr * g.w2_width,
                    sh.w2.reshape(g.hidden, g.w2_width),
                    pitch=g.w2_pitch,
                )
                _blit_strided(
                    host_arena,
                    row[5] + lr * g.w2s_width,
                    sh.w2_scale_e8m0.reshape(g.hidden, g.w2s_width),
                    pitch=g.w2s_pitch,
                )

                # -- DMA path, from device
                dev = torch.device("cuda")
                gpu_shard = {
                    "w13": sh.w13.reshape(-1).contiguous().to(dev),
                    "w13_scale": sh.w13_scale_e8m0.reshape(-1)
                    .contiguous()
                    .to(dev),
                    "w2": sh.w2.reshape(g.hidden, g.w2_width)
                    .contiguous()
                    .to(dev),
                    "w2_scale": sh.w2_scale_e8m0.reshape(g.hidden, g.w2s_width)
                    .contiguous()
                    .to(dev),
                }
                writer = ArenaDmaWriter(
                    arena_by_layer={},
                    geometry=g,
                    copy_lib=CudaCopyLib(),
                    register_fn=lambda p, n: 0,
                )
                writer._base[0] = int(dma_arena.data_ptr())
                writer.write(
                    layer_idx=0,
                    row=row,
                    shard=gpu_shard,
                    stream=torch.cuda.current_stream().cuda_stream,
                )
            torch.cuda.synchronize()
            self.assertTrue(
                torch.equal(host_arena, dma_arena),
                f"{int((host_arena != dma_arena).sum())} bytes differ",
            )
        finally:
            unreg_fn(int(dma_arena.data_ptr()))


class TestDirectDmaRoundTrip(unittest.TestCase):
    """A promotion must read back exactly what a demotion wrote.

    The two directions share offsets but not code paths -- write uses
    memcpy_d2h / memcpy2d_d2h, read uses the h2d pair with source and
    destination pitches swapped. A sign error or a transposed pitch/width in
    either produces right-shaped wrong bytes, which is the failure mode this
    whole module exists to prevent, and no shape check would catch it.

    So: write a known shard through the DMA path, read it back through the DMA
    path, and require the bytes to be identical. Doing it for every rank of a
    partition also proves the slices do not overlap -- rank 2's read must not
    see rank 1's write."""

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_write_then_read_is_identity(self):
        ArenaDmaWriter = _dw.ArenaDmaWriter
        from sglang.srt.layers.moe.kt_arena_geometry import (
            CudaCopyLib,
            cudart_register_fns,
        )

        off_gate = 0
        off_up = off_gate + GU_BLOCK
        off_down = off_up + GU_BLOCK
        off_gate_s = off_down + W2_BLOCK
        off_up_s = off_gate_s + GUS_BLOCK
        off_down_s = off_up_s + GUS_BLOCK
        row = [off_gate, off_up, off_down, off_gate_s, off_up_s, off_down_s]
        arena = torch.zeros(off_down_s + W2S_BLOCK, dtype=torch.uint8)

        reg_fn, unreg_fn = cudart_register_fns()
        rc = reg_fn(int(arena.data_ptr()), int(arena.numel()))
        self.assertEqual(rc, 0, f"cudaHostRegister rc={rc}")
        try:
            dev = torch.device("cuda")
            full = _full_expert(seed=11)
            written = {}
            for rank in range(TP_SIZE):
                g = _Geom(rank)
                if g.part != 0:
                    continue
                sh = _rank_shard(full, rank)
                src = {
                    "w13": sh.w13.reshape(-1).contiguous().to(dev),
                    "w13_scale": sh.w13_scale_e8m0.reshape(-1).contiguous().to(dev),
                    "w2": sh.w2.reshape(g.hidden, g.w2_width).contiguous().to(dev),
                    "w2_scale": sh.w2_scale_e8m0.reshape(g.hidden, g.w2s_width)
                    .contiguous()
                    .to(dev),
                }
                dma = ArenaDmaWriter(
                    arena_by_layer={}, geometry=g,
                    copy_lib=CudaCopyLib(), register_fn=lambda p, n: 0,
                )
                dma._base[0] = int(arena.data_ptr())
                dma.write(layer_idx=0, row=row, shard=src,
                          stream=torch.cuda.current_stream().cuda_stream)
                written[rank] = (g, src, dma)
            torch.cuda.synchronize()

            for rank, (g, src, dma) in written.items():
                out = {
                    "w13": torch.zeros_like(src["w13"]),
                    "w13_scale": torch.zeros_like(src["w13_scale"]),
                    "w2": torch.zeros_like(src["w2"]),
                    "w2_scale": torch.zeros_like(src["w2_scale"]),
                }
                dma.read(layer_idx=0, row=row, out=out,
                         stream=torch.cuda.current_stream().cuda_stream)
                torch.cuda.synchronize()
                for name in ("w13", "w13_scale", "w2", "w2_scale"):
                    self.assertTrue(
                        torch.equal(out[name], src[name]),
                        f"rank {rank} {name}: read back "
                        f"{int((out[name] != src[name]).sum())} differing bytes",
                    )
        finally:
            unreg_fn(int(arena.data_ptr()))


class TestDirectDmaRoundTripContiguous(unittest.TestCase):
    """Same round trip, but at ONE RANK PER PARTITION.

    That geometry (per_numa == per_gpu, so w2_pitch == w2_width) takes a
    different branch: the down strips are contiguous, so the copies drop the 2D
    descriptor entirely. It is also the geometry production runs, and the
    strided test above cannot reach it -- its _Geom has 2 partitions, pitch 768
    against width 192.

    The branch exists because the pitched form cost 7.05 ms per expert there:
    3,584 row transactions of 192 bytes to move a block that is one copy. So
    this asserts the fast path still round-trips bitwise, not just that it is
    faster."""

    class _Geom8:
        """per_numa == per_gpu: one rank per partition, contiguous down."""

        def __init__(self):
            self.hidden = HIDDEN
            self.per_gpu = PER_GPU
            self.group = GROUP
            self.part = 0
            self.local_rank = 0
            self.gu_w = GU_W
            self.gu_s = GU_S
            self.w2_width = PER_GPU // 2
            self.w2_pitch = PER_GPU // 2          # == width: contiguous
            self.w2s_width = PER_GPU // GROUP
            self.w2s_pitch = PER_GPU // GROUP

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_contiguous_branch_round_trips(self):
        ArenaDmaWriter = _dw.ArenaDmaWriter
        from sglang.srt.layers.moe.kt_arena_geometry import (
            CudaCopyLib,
            cudart_register_fns,
        )

        g = self._Geom8()
        self.assertEqual(g.w2_pitch, g.w2_width, "fixture must hit the fast path")
        gu_block = GU_W
        w2_block = HIDDEN * g.w2_width
        gus_block = GU_S
        w2s_block = HIDDEN * g.w2s_width
        row = [0, gu_block, 2 * gu_block, 2 * gu_block + w2_block,
               2 * gu_block + w2_block + gus_block,
               2 * gu_block + w2_block + 2 * gus_block]
        total = row[5] + w2s_block
        arena = torch.zeros(total, dtype=torch.uint8)

        reg_fn, unreg_fn = cudart_register_fns()
        rc = reg_fn(int(arena.data_ptr()), int(arena.numel()))
        self.assertEqual(rc, 0, f"cudaHostRegister rc={rc}")
        try:
            dev = torch.device("cuda")
            torch.manual_seed(23)
            src = {
                "w13": torch.randint(0, 255, (2 * GU_W,), dtype=torch.uint8).to(dev),
                "w13_scale": torch.randint(0, 255, (2 * GU_S,), dtype=torch.uint8).to(dev),
                "w2": torch.randint(
                    0, 255, (HIDDEN, g.w2_width), dtype=torch.uint8
                ).to(dev),
                "w2_scale": torch.randint(
                    0, 255, (HIDDEN, g.w2s_width), dtype=torch.uint8
                ).to(dev),
            }
            dma = ArenaDmaWriter(
                arena_by_layer={}, geometry=g,
                copy_lib=CudaCopyLib(), register_fn=lambda p, n: 0,
            )
            dma._base[0] = int(arena.data_ptr())
            st = torch.cuda.current_stream().cuda_stream
            dma.write(layer_idx=0, row=row, shard=src, stream=st)
            torch.cuda.synchronize()
            out = {k: torch.zeros_like(v) for k, v in src.items()}
            dma.read(layer_idx=0, row=row, out=out, stream=st)
            torch.cuda.synchronize()
            for name in src:
                self.assertTrue(
                    torch.equal(out[name], src[name]),
                    f"{name}: {int((out[name] != src[name]).sum())} bytes differ",
                )
        finally:
            unreg_fn(int(arena.data_ptr()))


class TestColdSourceSatisfiesThePipelineProtocol(unittest.TestCase):
    """Every method the pipeline calls unguarded, the arena source must have.

    This exists because of a real failure. ``ArenaDmaColdSource`` shipped
    without ``layer_rows``, and nothing noticed: the pipeline only calls it
    from inside ``if self._probe is not None``, so the gap was invisible until
    a server booted with the overlap probe enabled and every one of
    the eight ranks died with AttributeError partway through a benchmark --
    after a 17-minute boot.

    So do not hardcode the method list. Derive it from the pipeline source, and
    treat a name as REQUIRED unless the pipeline guards it with hasattr (which
    is how the optional ``issue_layer_copies`` fast path is dispatched). A new
    unguarded ``self._source.foo()`` then fails here rather than on the node.
    """

    def _pipeline_source(self):
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.normpath(
            os.path.join(
                here,
                "../../../../../python/sglang/srt/layers/moe/expert_pipeline.py",
            )
        )
        with open(path) as fh:
            return fh.read()

    def test_arena_source_implements_every_unguarded_store_call(self):
        src = self._pipeline_source()
        called = set(re.findall(r"self\._source\.([A-Za-z_][A-Za-z0-9_]*)", src))
        optional = set(
            re.findall(r'hasattr\(\s*self\._source\s*,\s*"([^"]+)"', src)
        )
        required = called - optional

        self.assertIn(
            "layer_rows",
            required,
            "the regression this test exists for: layer_rows is called "
            "unguarded (under the overlap probe), so it is REQUIRED",
        )

        missing = sorted(
            name for name in required if not hasattr(_dw.ArenaDmaColdSource, name)
        )
        self.assertEqual(
            missing,
            [],
            f"ArenaDmaColdSource is missing {missing}; the pipeline calls "
            f"these without a hasattr guard (required={sorted(required)}, "
            f"optional={sorted(optional)})",
        )

    def test_layer_rows_is_cheap_and_returns_none(self):
        """The probe charges its host-side wait to gather_wait; ours is zero.

        A source that reads out of kt's registered arena has no host gather, so
        the honest measurement is 0 ms. Returning None (rather than a tensor)
        is what marks it as "nothing was gathered" -- and it must not blow up
        on a layer index it has never planned.
        """
        src = _dw.ArenaDmaColdSource.__new__(_dw.ArenaDmaColdSource)
        self.assertIsNone(src.layer_rows(0, "w13_weight"))
        self.assertIsNone(src.layer_rows(9999, "w2_weight"))


if __name__ == "__main__":
    unittest.main()