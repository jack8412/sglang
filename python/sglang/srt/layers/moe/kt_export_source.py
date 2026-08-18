# SPDX-License-Identifier: Apache-2.0
"""Cold-expert source fed by kt's batched raw export -- no big shared mapping.

The design that replaces the 1.45 TB memfd arena (user decision): kt keeps
plain ANONYMOUS weight memory, and the process that owns the bytes moves
them. Once per layer, rank 0 hands kt's worker pool an expert-id list plus
destination pointers (``write_raw_experts_to_buffer``: plain C++ memcpy of
nibble-packed weights and raw u8 scales, zero Python in the loop -- the
GIL-bound Python gather measured 86 ms/layer live; the C++ export 52 ms under
a full reclaim storm), and the pool fills every rank's slice of a small
pinned ring. Each rank then DMAs its own ring stage on its copy stream and
the existing GPU swizzle produces the resident layout.

RINGS. One POSIX-shm segment per rank (~1.8 GiB: NUM_STAGE x 276 experts x
2.2 MB), the exact pattern of kt's activation rings: every rank CREATES its
own segment and cudaHostRegisters it locally (pinning is per-process); rank 0
OPENS the peers' segments by broadcast name and hands kt raw pointers into
them. Plus a 64-byte pacing header per ring (padded so the data blocks
stay 64-aligned for the exporter's non-temporal stores).

PACING, and why it cannot desynchronise. The export list for layer L is
derived from the LIVE logical_to_slot table, identical on every rank, and
rank 0 exports THE SAME layer sequence every rank consumes, so there is no
per-rank decision anywhere -- the M9/M11/M12 rule by construction. The flags
are plain shm int64s:

    ready[stage]    (written by rank 0, read by all)  = layer POSITION whose
                    bytes are complete in that stage, published after
                    cpu_infer.sync() returns for that export.
    consumed[stage] (written by each rank, read by rank 0) = layer position
                    whose H2D from that stage has COMPLETED on the copy
                    stream, so rank 0 never overwrites bytes still being
                    DMA'd (the WAR gate).

A rank that dies or stalls trips the peer's bounded wait loudly; nothing
blocks forever and nothing falls back to stale bytes.

Swap-window promotions ride the same machinery: windows run quiesced between
batches, so stage 0 is free and a synchronous batched export of the <=8
promoted experts per layer costs ~1 ms/layer.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid as uuid_mod
from multiprocessing import shared_memory
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from sglang.srt.layers.moe.expert_cold_store import WEIGHT_NAMES

logger = logging.getLogger(__name__)

NUM_STAGE = 3
_HEADER_SLOTS = 2 * NUM_STAGE  # ready[stage] ..., consumed[stage] ...
# Padded to 64 so every data block behind it is 64-aligned -- the exporter's
# non-temporal stores require aligned destinations and silently fall back to
# cached copies (re-adding the RFO DRAM transit) if the rings are not.
_HEADER_BYTES = 64
assert _HEADER_SLOTS * 8 <= _HEADER_BYTES
_WAIT_TIMEOUT_S = 120.0
_EMPTY = -1
# Published by rank 0 when a promotion export raises, so peers fail over in
# ~0.2 ms instead of burning the full timeout per swapped layer. Distinct
# from _EMPTY, from prefill gpos (>= 0) and from promotion tokens (<= -100).
_POISON = -2


def _ring_nbytes(num_cold: int, raw_shapes: Dict[str, tuple]) -> int:
    per_expert = sum(
        int(np.prod(shape)) * torch.empty((), dtype=dtype).element_size()
        for shape, dtype in raw_shapes.values()
    )
    return _HEADER_BYTES + NUM_STAGE * num_cold * per_expert


class ColdRing:
    """One rank's shm ring: header + [stage][name][num_cold, *shape] blocks."""

    def __init__(
        self,
        *,
        shm: shared_memory.SharedMemory,
        num_cold: int,
        raw_shapes: Dict[str, tuple],
        owns: bool,
    ):
        self.shm = shm
        self.owns = owns
        self.num_cold = num_cold
        buf = np.frombuffer(shm.buf, dtype=np.uint8)
        self.header = np.frombuffer(
            shm.buf, dtype=np.int64, count=_HEADER_SLOTS
        )
        # [stage][name] -> torch uint8 view [num_cold, prod(shape)] and the
        # per-(stage, expert_pos) data_ptrs kt's export consumes.
        self.blocks: List[Dict[str, torch.Tensor]] = []
        off = _HEADER_BYTES
        self._per_name_nbytes = {
            n: int(np.prod(shape))
            * torch.empty((), dtype=dtype).element_size()
            for n, (shape, dtype) in raw_shapes.items()
        }
        for _stage in range(NUM_STAGE):
            stage_views = {}
            for n, (shape, dtype) in raw_shapes.items():
                nbytes = self._per_name_nbytes[n] * num_cold
                t = torch.from_numpy(buf[off : off + nbytes]).view(
                    num_cold, self._per_name_nbytes[n]
                )
                stage_views[n] = t
                off += nbytes
            self.blocks.append(stage_views)
        self._raw_shapes = raw_shapes

    @classmethod
    def create(cls, *, name: str, num_cold: int, raw_shapes) -> "ColdRing":
        shm = shared_memory.SharedMemory(
            name=name, create=True, size=_ring_nbytes(num_cold, raw_shapes)
        )
        ring = cls(shm=shm, num_cold=num_cold, raw_shapes=raw_shapes, owns=True)
        ring.header[:] = _EMPTY
        return ring

    @classmethod
    def open(cls, *, name: str, num_cold: int, raw_shapes) -> "ColdRing":
        shm = shared_memory.SharedMemory(name=name, create=False)
        return cls(shm=shm, num_cold=num_cold, raw_shapes=raw_shapes, owns=False)

    def register_pinned(self) -> None:
        """cudaHostRegister the whole segment in THIS process (DMA source)."""
        base = np.frombuffer(self.shm.buf, dtype=np.uint8)
        t = torch.from_numpy(base)
        rc = torch.cuda.cudart().cudaHostRegister(
            t.data_ptr(), t.numel(), 0
        )
        if int(rc) != 0:
            raise RuntimeError(f"cudaHostRegister(cold ring) failed rc={int(rc)}")

    def stage_tensor(self, stage: int, name: str, shape, dtype) -> torch.Tensor:
        """The stage block shaped for the pipeline's raw-landing copy."""
        return self.blocks[stage][name].view(torch.uint8).reshape(
            (self.num_cold,) + tuple(shape)
        )

    def stage_ptrs(self, stage: int, name: str) -> List[int]:
        base = self.blocks[stage][name].data_ptr()
        step = self._per_name_nbytes[name]
        return [base + i * step for i in range(self.num_cold)]

    def close(self) -> None:
        """Best-effort teardown for the disarm path: drop views + mapping.

        The cudaHostRegister pinning is process-lifetime by design; what this
        releases is the shm handle so the segment can actually die. A live
        exported view makes SharedMemory.close raise BufferError -- treated as
        "leaks until process exit", which the rare disarm path tolerates.
        """
        self.blocks = []
        self.header = None
        try:
            self.shm.close()
        except (BufferError, OSError):
            logger.info("[kt-export] ring shm close deferred to process exit")


class ExportColdSource:
    """Pipeline source: kt's C++ pool prepares layer L+2, ranks DMA at L+1.

    Duck-compatible with ColdExpertStore/ArenaColdSource where the pipeline
    touches a source: ``num_cold``, ``layer_rows``, ``after_enqueue``,
    ``reset``. Look-ahead lives inside ``layer_rows`` exactly like the arena
    source's did.
    """
    # Bytes of contiguous staging this source wants the pipeline to hand it.
    # 0 means "issue your own copies"; a source that can land its layer in one
    # contiguous H2D sets it and gets a scratch buffer per slot.
    staging_nbytes: int = 0

    def __init__(
        self,
        *,
        tp_rank: int,
        tp_size: int,
        my_ring: ColdRing,
        peer_rings: Optional[List[ColdRing]],  # rank 0 only, index == rank
        cold_slot_expert_ids: Callable[[int], torch.Tensor],
        raw_shapes: Dict[str, tuple],
        moe_layer_indices: Sequence[int],
        num_cold: int,
        # rank 0: layer_idx -> the kt NativeMoEWrapper whose pool exports it;
        # peers: None (they only ever wait on their ring's flags).
        wrappers_by_layer: Optional[Dict[int, object]] = None,
    ):
        self._tp_rank = tp_rank
        self._tp_size = tp_size
        self._ring = my_ring
        self._peer_rings = peer_rings
        self._ids_for = cold_slot_expert_ids
        self._raw_shapes = raw_shapes
        self._layers = sorted(moe_layer_indices)
        self._pos = {layer: i for i, layer in enumerate(self._layers)}
        self._num_pos = len(self._layers)
        # Pass epoch: positions repeat every prefill pass but the shm flags
        # persist, so published pacing values are GLOBAL positions
        # (pass_gen * num_pos + pos) -- monotone forever. Without this the
        # second pass of the server's life deterministically raised "pacing
        # broke" (stale ready from pass N-1 trips the monotonicity guard).
        # Every rank bumps the epoch in reset(), which runs at the same
        # plan-deterministic point of every pass: lockstep, no collectives.
        self._pass_gen = 0
        # Promotion tokens: unique per CALL, in lockstep across ranks by the
        # same plan-determinism -- a per-layer token was ABA-prone across
        # windows (a stale equal header let peers consume before the export).
        self._promo_seq = 0
        self.num_cold = int(num_cold)
        self._wrappers = wrappers_by_layer
        # H2D-completion events per stage, for publishing consumed[stage]
        # without blocking the forward thread.
        self._stage_events: List[Optional[torch.cuda.Event]] = [None] * NUM_STAGE
        self._stage_pending: List[int] = [_EMPTY] * NUM_STAGE

        if tp_rank == 0:
            # One worker serializes exports in layer order; cpu_infer's sync
            # blocks, so it must never run on the forward thread.
            self._queue: "queue.Queue" = queue.Queue()
            self._inflight: set = set()
            self._lock = threading.Lock()
            self._worker = threading.Thread(
                target=self._export_worker, name="kt-cold-export", daemon=True
            )
            self._worker.start()

    # -- rank 0: the export side ------------------------------------------

    def _export_worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return  # close() sentinel
            gpos, layer_idx = item
            if gpos // self._num_pos != self._pass_gen:
                # Stale enqueue from a pass that reset mid-flight.
                with self._lock:
                    self._inflight.discard(gpos)
                continue
            try:
                self._export_layer(gpos, layer_idx)
            except Exception:
                logger.exception(
                    "[kt-export] export for layer %d (gpos %d) failed; "
                    "consumers will time out loudly rather than read stale "
                    "bytes",
                    layer_idx,
                    gpos,
                )

    def _gpos(self, layer_idx: int) -> int:
        return self._pass_gen * self._num_pos + self._pos[layer_idx]

    def _export_layer(self, gpos: int, layer_idx: int) -> None:
        stage = gpos % NUM_STAGE
        # WAR: before overwriting a stage, its CURRENT occupant must have
        # been consumed by that ring's rank -- UNLESS the occupant belongs to
        # a previous epoch. Prior-epoch bytes can never be read again
        # (consumers only wait on current-epoch positions), and gating on
        # them wedged the first smoke: a lookahead export that finished after
        # its pass ended left ready > consumed with nobody left to consume.
        # Negative occupants (empty, promotion tokens, poison) need no gate
        # either: promotion rows are cloned out before the window's next
        # layer's export.
        deadline = time.monotonic() + _WAIT_TIMEOUT_S
        gen_floor = (gpos // self._num_pos) * self._num_pos
        for r in self._peer_rings:
            while True:
                cur = int(r.header[stage])
                if cur < gen_floor or int(r.header[NUM_STAGE + stage]) >= cur:
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"stage {stage}: occupant {cur} never consumed; "
                        "refusing to overwrite"
                    )
                time.sleep(0.0005)

        ids = self._ids_for(layer_idx).tolist()
        if len(ids) != self.num_cold:
            raise RuntimeError(
                f"layer {layer_idx}: {len(ids)} cold slots vs ring {self.num_cold}"
            )
        # Pointers [expert_pos * tp + rank] across ALL rings, kt's contract.
        ptrs = {n: [] for n in WEIGHT_NAMES}
        per_ring = {
            n: [r.stage_ptrs(stage, n) for r in self._peer_rings]
            for n in WEIGHT_NAMES
        }
        for i in range(self.num_cold):
            for rank in range(self._tp_size):
                for n in WEIGHT_NAMES:
                    ptrs[n].append(per_ring[n][rank][i])

        w = self._wrappers[layer_idx]
        w.submit_write_raw_experts_to_buffer(
            self._tp_size, ids,
            ptrs[WEIGHT_NAMES[0]], ptrs[WEIGHT_NAMES[1]],
            ptrs[WEIGHT_NAMES[2]], ptrs[WEIGHT_NAMES[3]],
        )
        w.sync_write_raw_experts_to_buffer()
        # Publish AFTER the bytes are complete, on every ring: peers poll
        # their own header only.
        for r in self._peer_rings:
            r.header[stage] = gpos
        with self._lock:
            self._inflight.discard(gpos)

    def _kick(self, layer_idx: int) -> None:
        if self._tp_rank != 0:
            return
        gpos = self._gpos(layer_idx)
        stage = gpos % NUM_STAGE
        ring0 = self._peer_rings[0]
        if ring0.header[stage] == gpos:
            return  # already exported this pass (stale values are SMALLER)
        with self._lock:
            if gpos in self._inflight:
                return
            self._inflight.add(gpos)
        self._queue.put((gpos, layer_idx))

    # -- every rank: the consume side --------------------------------------

    def _publish_consumed(self) -> None:
        """Fold completed H2D events into consumed[]; never blocks."""
        for stage in range(NUM_STAGE):
            ev = self._stage_events[stage]
            if ev is not None and ev.query():
                self._ring.header[NUM_STAGE + stage] = self._stage_pending[stage]
                self._stage_events[stage] = None

    def layer_rows(self, layer_idx: int, name: str) -> torch.Tensor:
        gpos = self._gpos(layer_idx)
        pos = self._pos[layer_idx]
        stage = gpos % NUM_STAGE
        self._publish_consumed()
        # Look-ahead BEFORE waiting, so a cold start overlaps.
        for ahead in self._layers[pos : pos + NUM_STAGE]:
            self._kick(ahead)
        deadline = time.monotonic() + _WAIT_TIMEOUT_S
        while self._ring.header[stage] != gpos:
            if self._ring.header[stage] > gpos:
                raise RuntimeError(
                    f"cold ring stage {stage} advanced past gpos {gpos}: "
                    f"holds {int(self._ring.header[stage])} -- pacing broke"
                )
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"cold ring: gpos {gpos} never became ready "
                    "(rank 0 export stalled or died)"
                )
            # Keep folding H2D completions while spinning: rank 0's WAR gate
            # may be waiting on THIS thread's consumed[] publish (the review's
            # circular-wait finding), and only this thread can publish it.
            self._publish_consumed()
            time.sleep(0.0002)
        shape, dtype = self._raw_shapes[name]
        return self._ring.stage_tensor(stage, name, shape, dtype)

    def after_enqueue(self, layer_idx: int, stream: torch.cuda.Stream) -> None:
        gpos = self._gpos(layer_idx)
        stage = gpos % NUM_STAGE
        ev = torch.cuda.Event()
        ev.record(stream)
        self._stage_events[stage] = ev
        self._stage_pending[stage] = gpos
        self._publish_consumed()

    def reset(self) -> None:
        """Pass boundary, called with both streams synchronized.

        Folds pending H2D events, then bumps the pass epoch; every rank does
        this at the same plan point, so epochs stay in lockstep with no
        communication. No acknowledgment of unconsumed ready values is needed
        (or safe -- it raced in-flight lookahead exports): the WAR gate skips
        prior-epoch occupants by construction.
        """
        for stage in range(NUM_STAGE):
            if self._stage_events[stage] is not None:
                self._ring.header[NUM_STAGE + stage] = self._stage_pending[stage]
                self._stage_events[stage] = None
        self._pass_gen += 1

    def close(self) -> None:
        """Teardown for the disarm path: stop the worker, drop the rings."""
        if self._tp_rank == 0 and self._peer_rings is not None:
            self._queue.put(None)
            self._worker.join(timeout=5.0)
            for r in self._peer_rings:
                if r is not self._ring:
                    r.close()
        self._ring.close()

    # -- swap-window promotions --------------------------------------------

    def export_experts_sync(
        self, layer_idx: int, expert_ids: Sequence[int]
    ) -> Dict[str, torch.Tensor]:
        """Rank-0-driven batched export of a few experts, consumed by ALL
        ranks from stage 0 of their rings. Callers run quiesced between
        batches (the swap window), so the stage is free by construction; the
        ready flag round-trips through the same headers.
        """
        n = len(expert_ids)
        if n == 0 or n > self.num_cold:
            raise ValueError(f"bad promotion batch size {n}")
        stage = 0
        # Unique per CALL and in lockstep across ranks (every rank executes
        # the identical plan-deterministic window sequence): a per-layer token
        # was ABA-prone -- a stale equal header from a previous window let
        # peers consume stage 0 before rank 0's export had even started.
        self._promo_seq += 1
        token = -100 - self._promo_seq
        if self._tp_rank == 0:
            try:
                ptrs = {m: [] for m in WEIGHT_NAMES}
                for i in range(n):
                    for rank in range(self._tp_size):
                        for m in WEIGHT_NAMES:
                            ptrs[m].append(
                                self._peer_rings[rank].stage_ptrs(stage, m)[i]
                            )
                w = self._wrappers[layer_idx]
                w.submit_write_raw_experts_to_buffer(
                    self._tp_size, [int(e) for e in expert_ids],
                    ptrs[WEIGHT_NAMES[0]], ptrs[WEIGHT_NAMES[1]],
                    ptrs[WEIGHT_NAMES[2]], ptrs[WEIGHT_NAMES[3]],
                )
                w.sync_write_raw_experts_to_buffer()
            except Exception:
                # Peers must not burn the full timeout per swapped layer to
                # learn this; poison fails them over in ~0.2 ms.
                for r in self._peer_rings:
                    r.header[stage] = _POISON
                raise
            for r in self._peer_rings:
                r.header[stage] = token
        deadline = time.monotonic() + _WAIT_TIMEOUT_S
        while self._ring.header[stage] != token:
            if self._ring.header[stage] == _POISON:
                raise RuntimeError("rank 0 promotion export failed (poisoned)")
            if time.monotonic() > deadline:
                raise RuntimeError("promotion export never became ready")
            time.sleep(0.0002)
        # CALLERS MUST CLONE the returned views before their next layer's
        # export overwrites stage 0 (_begin_layer clones before its per-layer
        # collective, which then acts as the cross-rank consumption barrier).
        out = {}
        for m in WEIGHT_NAMES:
            shape, dtype = self._raw_shapes[m]
            out[m] = self._ring.stage_tensor(stage, m, shape, dtype)[:n]
        return out


def _rank_node_cpus(tp_rank: int, tp_size: int) -> Optional[list]:
    """CPUs of the socket whose kt partition serves this rank, or None."""
    try:
        import os

        nodes = sorted(
            int(d[4:])
            for d in os.listdir("/sys/devices/system/node")
            if d.startswith("node") and d[4:].isdigit()
        )
        parts = max(1, min(len(nodes), tp_size))
        node = nodes[tp_rank * parts // tp_size]
        with open(f"/sys/devices/system/node/node{node}/cpulist") as f:
            spec = f.read().strip()
        cpus = set()
        for part in spec.split(","):
            if "-" in part:
                lo, hi = part.split("-")
                cpus.update(range(int(lo), int(hi) + 1))
            elif part:
                cpus.add(int(part))
        return sorted(cpus) or None
    except OSError:
        return None


def _create_and_register_ring(
    *, name: str, num_cold: int, raw_shapes, tp_rank: int, tp_size: int
) -> ColdRing:
    """Create + pin one ring with its pages on the rank's own socket.

    The bulk pages are first-touched by cudaHostRegister's fault-in, in THIS
    process -- not by kt's partition-bound writers -- so the register must run
    under socket affinity or the export pays cross-UPI writes: measured 64.4
    vs 30.7 ms/layer for exactly this, a clean 2x.
    """
    import contextlib
    import os

    cpus = _rank_node_cpus(tp_rank, tp_size)

    @contextlib.contextmanager
    def bound():
        if cpus is None:
            logger.info("[kt-export] no NUMA binding for ring pages")
            yield
            return
        old = os.sched_getaffinity(0)
        try:
            os.sched_setaffinity(0, cpus)
        except OSError:
            yield
            return
        try:
            yield
        finally:
            os.sched_setaffinity(0, old)

    with bound():
        ring = ColdRing.create(name=name, num_cold=num_cold, raw_shapes=raw_shapes)
        ring.register_pinned()
    return ring


def build_cold_rings(
    *,
    tp_rank: int,
    tp_size: int,
    num_cold: int,
    raw_shapes: Dict[str, tuple],
) -> tuple:
    """Create this rank's ring, exchange names, open peers' on rank 0.

    Boot-time collectives only (a uuid broadcast + one barrier so creation
    precedes opening), the same shape the activation rings already use.
    """
    import torch.distributed as dist

    from sglang.srt.distributed import get_tp_group

    if tp_size == 1 or not dist.is_initialized():
        ring = _create_and_register_ring(
            name=f"kt_coldring_r0_{uuid_mod.uuid4().hex[:8]}",
            num_cold=num_cold,
            raw_shapes=raw_shapes,
            tp_rank=tp_rank,
            tp_size=max(tp_size, 1),
        )
        return ring, [ring]

    holder = [uuid_mod.uuid4().hex[:8] if tp_rank == 0 else None]
    dist.broadcast_object_list(
        holder, src=get_tp_group().first_rank, group=get_tp_group().cpu_group
    )
    uid = holder[0]
    # Local failures (shm create, cudaHostRegister) must NOT skip the
    # barriers below -- a rank that bails early strands the group in a
    # collective until the gloo timeout. Every rank traverses every
    # collective, then raises; finalize's arming consensus turns one rank's
    # raise into a unanimous fallback.
    err = None
    my_ring = None
    try:
        my_ring = _create_and_register_ring(
            name=f"kt_coldring_r{tp_rank}_{uid}",
            num_cold=num_cold,
            raw_shapes=raw_shapes,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
    except Exception as exc:
        logger.exception("[kt-export] ring create/register failed")
        err = exc
    # Everyone must have created before rank 0 opens.
    dist.barrier(group=get_tp_group().cpu_group)
    peer_rings = None
    if tp_rank == 0 and err is None:
        try:
            peer_rings = []
            for r in range(tp_size):
                if r == 0:
                    peer_rings.append(my_ring)
                else:
                    peer_rings.append(
                        ColdRing.open(
                            name=f"kt_coldring_r{r}_{uid}",
                            num_cold=num_cold,
                            raw_shapes=raw_shapes,
                        )
                    )
        except Exception as exc:
            # A peer that failed to CREATE leaves rank 0's open failing too;
            # both raise after the barriers and the consensus disarms all.
            logger.exception("[kt-export] rank 0 could not open a peer ring")
            err = exc
            peer_rings = None
    # Named segments outlive crashes; unlink now so the LAST unmap frees them
    # (the mappings themselves stay valid), matching _create_cpu_buffers.
    dist.barrier(group=get_tp_group().cpu_group)
    if my_ring is not None:
        try:
            my_ring.shm.unlink()
        except FileNotFoundError:
            pass
    if err is not None:
        raise err
    return my_ring, peer_rings
