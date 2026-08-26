"""kt-RAM expert source tests (srt/layers/moe/kt_ram_source + kt_arena_share).

The source's one job is byte fidelity: whatever mode it runs in (absolute
pointers in the kt-owning process, or offsets into fd-passed read-only
mappings in every other rank), raw_shard must reproduce exactly the
cat-then-slice construction its docstring specifies -- the construction
the source was proved bitwise against the checkpoint on the node.
These tests pin the modes to a NumPy reference of that construction, and pin
the share protocol (send_fds -> header/meta -> mmap PROT_READ -> source) end
to end over a real memfd, including the shared-page property the whole design
rests on.
"""

import ctypes
import mmap
import os
import pickle
import socket
import unittest

import numpy as np
import torch

from sglang.srt.layers.moe import kt_arena_share
from sglang.srt.layers.moe.kt_ram_source import (
    KtArenaExpertSource,
    KtRamExpertSource,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

# Small but non-degenerate: per_gpu (=16) is a multiple of the scale group, so
# every rank's scale slice is non-empty, and two ranks share partition 0 while
# two share partition 1 -- both single-partition slice paths get exercised.
NUMA, EXPERTS, HIDDEN, PER_NUMA, GROUP = 2, 8, 64, 32, 8
TP_SIZE = 4
INTERMEDIATE = PER_NUMA * NUMA
W_BYTES = PER_NUMA * HIDDEN // 2
S_BYTES = (HIDDEN // GROUP) * PER_NUMA
GEOMETRY = [NUMA, EXPERTS, HIDDEN, PER_NUMA, GROUP]
KINDS = (
    ("gate_b", W_BYTES),
    ("up_b", W_BYTES),
    ("down_b", W_BYTES),
    ("gate_d", S_BYTES),
    ("up_d", S_BYTES),
    ("down_d", S_BYTES),
)


def _make_blocks(seed=7):
    rng = np.random.default_rng(seed)
    return {
        (p, e, kind): rng.integers(0, 256, nb, dtype=np.uint8)
        for p in range(NUMA)
        for e in range(EXPERTS)
        for kind, nb in KINDS
    }


def _reference_shard(blocks, e, tp_rank):
    """The specified construction: concat partitions, THEN slice the rank."""
    h2, hg = HIDDEN // 2, HIDDEN // GROUP
    cat = np.concatenate
    gate = cat([blocks[(p, e, "gate_b")].reshape(PER_NUMA, h2) for p in range(NUMA)], 0)
    up = cat([blocks[(p, e, "up_b")].reshape(PER_NUMA, h2) for p in range(NUMA)], 0)
    gate_s = cat([blocks[(p, e, "gate_d")].reshape(PER_NUMA, hg) for p in range(NUMA)], 0)
    up_s = cat([blocks[(p, e, "up_d")].reshape(PER_NUMA, hg) for p in range(NUMA)], 0)
    down = cat(
        [blocks[(p, e, "down_b")].reshape(HIDDEN, PER_NUMA // 2) for p in range(NUMA)], 1
    )
    down_s = cat(
        [blocks[(p, e, "down_d")].reshape(HIDDEN, PER_NUMA // GROUP) for p in range(NUMA)],
        1,
    )
    per_gpu = INTERMEDIATE // TP_SIZE
    lo, hi = tp_rank * per_gpu, (tp_rank + 1) * per_gpu
    return {
        "w13": cat([gate[lo:hi], up[lo:hi]], 0),
        "w13_scale": cat([gate_s[lo:hi], up_s[lo:hi]], 0),
        "w2": np.ascontiguousarray(down[:, lo // 2 : hi // 2]),
        "w2_scale": np.ascontiguousarray(down_s[:, lo // GROUP : hi // GROUP]),
    }


def _pack_arenas(blocks):
    """Bump-pack per partition, 64-rounded, in (p outer, e inner) row order."""

    def r64(v):
        return (v + 63) & ~63

    arenas, offsets = [], []
    total = EXPERTS * sum(r64(nb) for _, nb in KINDS)
    for p in range(NUMA):
        arena = np.zeros(total, dtype=np.uint8)
        used = 0
        for e in range(EXPERTS):
            row = []
            for kind, nb in KINDS:
                arena[used : used + nb] = blocks[(p, e, kind)]
                row.append(used)
                used += r64(nb)
            offsets.append(row)
        arenas.append(arena)
    return arenas, offsets


class TestRawShardModes(CustomTestCase):
    """Red if either access mode stops reproducing the specified layout, if
    the two modes diverge from each other, or if absent-expert slots stop
    failing loudly."""

    def _check(self, source, blocks, tp_rank):
        for e in range(EXPERTS):
            got = source.raw_shard(e)
            want = _reference_shard(blocks, e, tp_rank)
            for k in ("w13", "w13_scale", "w2", "w2_scale"):
                self.assertEqual(tuple(got[k].shape), want[k].shape, (e, k))
                self.assertTrue(
                    np.array_equal(got[k].numpy(), want[k]), (e, k, "bytes differ")
                )

    def test_pointer_mode_matches_reference(self):
        blocks = _make_blocks()
        keep, rows = [], []
        for p in range(NUMA):
            for e in range(EXPERTS):
                row = []
                for kind, _ in KINDS:
                    data = blocks[(p, e, kind)].tobytes()
                    buf = ctypes.create_string_buffer(data, len(data))
                    keep.append(buf)
                    row.append(ctypes.addressof(buf))
                rows.append(row)
        for r in range(TP_SIZE):
            self._check(
                KtRamExpertSource(
                    pointers=rows, geometry=GEOMETRY, tp_rank=r, tp_size=TP_SIZE
                ),
                blocks,
                r,
            )

    def test_arena_mode_matches_reference(self):
        blocks = _make_blocks()
        arenas, offsets = _pack_arenas(blocks)
        tensors = [torch.from_numpy(a) for a in arenas]
        for r in range(TP_SIZE):
            self._check(
                KtArenaExpertSource(
                    arenas=tensors,
                    offsets=offsets,
                    geometry=GEOMETRY,
                    tp_rank=r,
                    tp_size=TP_SIZE,
                ),
                blocks,
                r,
            )

    def test_shapes_accessor_matches_the_real_shard(self):
        """raw_shard_shapes must state what raw_shard actually produces.

        This is the only cross-check left on the shard geometry. Production no
        longer calls raw_shard at all -- the split-prefill pipeline reads
        through ArenaDmaColdSource.issue_layer_copies -- but the swizzle plan is built
        from raw_shard_shapes, and a wrong shape there permutes the resident
        rows by the wrong map: right-shaped, finite, silently wrong weights for
        every promoted expert.

        raw_shard remains the executable statement of the layout (its bytes are
        checked against an independently packed arena above), so pinning the
        accessor against it is what keeps the arithmetic honest now that
        build_expert_bytes, which used to state it a second time, is gone.

        Every rank and every expert, because the shapes are claimed to be
        expert-independent and the per-rank slicing is where the arithmetic can
        go wrong.
        """
        blocks = _make_blocks(seed=23)
        arenas, offsets = _pack_arenas(blocks)
        tensors = [torch.from_numpy(a) for a in arenas]
        for r in range(TP_SIZE):
            src = KtArenaExpertSource(
                arenas=tensors,
                offsets=offsets,
                geometry=GEOMETRY,
                tp_rank=r,
                tp_size=TP_SIZE,
            )
            claimed = src.raw_shard_shapes()
            self.assertEqual(
                set(claimed), {"w13", "w13_scale", "w2", "w2_scale"}
            )
            for e in range(EXPERTS):
                got = src.raw_shard(e)
                for k, t in got.items():
                    self.assertEqual(tuple(t.shape), tuple(claimed[k]), (r, e, k))
                    self.assertEqual(t.dtype, torch.uint8, (r, e, k))

    def test_absent_expert_raises(self):
        blocks = _make_blocks()
        arenas, offsets = _pack_arenas(blocks)
        offsets = [list(row) for row in offsets]
        offsets[0] = [-1] * 6
        src = KtArenaExpertSource(
            arenas=[torch.from_numpy(a) for a in arenas],
            offsets=offsets,
            geometry=GEOMETRY,
            tp_rank=0,
            tp_size=TP_SIZE,
        )
        with self.assertRaises(KeyError):
            src.raw_shard(0)


class TestFdShareProtocol(CustomTestCase):
    """The consumer half of kt_arena_share against a real memfd: header+fds,
    metadata, read-only mapping, byte fidelity, and page sharing. Red if fd
    passing loses fds, if the mapping stops being the same physical pages, or
    if a read-only mapping stops feeding the source."""

    def test_end_to_end_over_socketpair(self):
        blocks = _make_blocks(seed=11)
        arenas, offsets = _pack_arenas(blocks)

        fds, maps_rw = [], []
        for a in arenas:
            fd = os.memfd_create("kt_test_arena")
            os.ftruncate(fd, len(a))
            m = mmap.mmap(fd, len(a))
            m[:] = a.tobytes()
            fds.append(fd)
            maps_rw.append(m)

        srv, cli = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            meta = pickle.dumps(
                {
                    "sizes": [len(a) for a in arenas],
                    "offsets": offsets,
                    "geometry": GEOMETRY,
                }
            )
            header = kt_arena_share._HEADER.pack(3, len(fds), len(meta))
            socket.send_fds(srv, [header], fds)
            srv.sendall(meta)

            (layer_idx, n_fds, meta_len), got_fds = (
                kt_arena_share._recv_header_and_fds(cli)
            )
            self.assertEqual((layer_idx, n_fds, meta_len), (3, len(fds), len(meta)))
            got_meta = pickle.loads(kt_arena_share._recv_exact(cli, meta_len))
            maps_ro = [
                mmap.mmap(fd, size, prot=mmap.PROT_READ)
                for fd, size in zip(got_fds, got_meta["sizes"])
            ]
            for fd in got_fds:
                os.close(fd)
            tensors = [kt_arena_share._wrap_mapping(m) for m in maps_ro]
            src = KtArenaExpertSource(
                arenas=tensors,
                offsets=got_meta["offsets"],
                geometry=got_meta["geometry"],
                tp_rank=1,
                tp_size=TP_SIZE,
            )
            got = src.raw_shard(2)
            want = _reference_shard(blocks, 2, 1)
            for k in ("w13", "w13_scale", "w2", "w2_scale"):
                self.assertTrue(np.array_equal(got[k].numpy(), want[k]), k)

            # Same physical pages, not a copy: a write through the producer's
            # mapping must be visible through the consumer's.
            probe = got_meta["offsets"][0][0]
            maps_rw[0][probe] = (maps_rw[0][probe] + 1) % 256
            self.assertEqual(tensors[0][probe].item(), maps_rw[0][probe])
        finally:
            srv.close()
            cli.close()
            for fd in fds:
                os.close(fd)


if __name__ == "__main__":
    unittest.main()
