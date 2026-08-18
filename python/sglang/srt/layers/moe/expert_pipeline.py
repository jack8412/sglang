# SPDX-License-Identifier: Apache-2.0
"""Double-buffered prefetch of cold-expert weights during full-expert prefill.

While layer N's MoE runs on the compute stream, layer N+2's cold experts are
copied from the pinned host store into the device buffer it will use.  Two
device buffers alternate, so a layer's weights are never overwritten while it
is still reading them.

The event protocol is a direct port of ``dwdp/weight_manager.py``, which
solves the same producer/consumer problem for MNNVL peer weights:

    prefetch(L)  : copy_stream waits consume[slot]  -> copies -> record prefetch[slot]
    wait(L)      : compute_stream waits prefetch[slot]
    done(L)      : record consume[slot] on compute_stream, then prefetch(L+2)

Layer L+2 is the one prefetched, not L+1: with two slots, L+2 is the next
layer that reuses L's slot, so that is the copy the consume event gates.

Buffer index is the layer's POSITION in the sorted MoE-layer list, not the
layer id -- K3's early layers are dense, so ``layer_idx % 2`` would alternate
incorrectly.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Sequence

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.expert_cold_store import WEIGHT_NAMES, ColdExpertStore

logger = logging.getLogger(__name__)


class ArenaColdSource:
    """Feed the pipeline from this rank's mapping of kt's memfd arenas.

    Replaces the ~0.415 TB pinned cold store: the bytes already sit in kt's
    resident buffers (checkpoint layout), mapped read-only into every rank by
    kt_arena_share. What remains is getting each layer's ~600 MB rank-shard
    onto the H2D path at prefill cadence, and the arena is pageable shmem, so
    a direct async copy would fall off the DMA fast path. This stages instead:
    a thread pool gathers layer L's cold shards into one of three small pinned
    buffers a couple of layers ahead of L's H2D. The CPU is idle during split
    prefill -- every routed expert runs on GPU -- so the gather is free
    concurrency, and the pinned staging (3 x ~600 MB) keeps the copy floor
    without registering the multi-hundred-GB arena mapping (that direct-DMA
    variant stays a measured follow-up, not a prerequisite).

    Duck-compatible with ColdExpertStore where ColdExpertPipeline touches it
    (``num_cold``, ``layer_rows``) plus the lifecycle hooks the pipeline calls
    on both (``after_enqueue``, ``reset``).

    SLOT ORDER IS THE TABLES'. Staging row ``j`` holds the expert whose
    ``logical_to_slot`` entry is ``num_gpu + j`` -- supplied per layer by
    ``cold_slot_expert_ids`` -- so routing and weights derive from the same
    table. Swaps exchange that table pairwise at window boundaries; a gather
    snapshots it at issue time, and passes never straddle a window (the
    window quiesces between batches), so a pass is internally consistent.
    """
    # Bytes of contiguous staging this source wants the pipeline to hand it.
    # 0 means "issue your own copies"; a source that can land its layer in one
    # contiguous H2D sets it and gets a scratch buffer per slot.
    staging_nbytes: int = 0

    NUM_STAGE = 3
    # NOT more workers = faster: the per-expert slicing holds the GIL and the
    # convoy of blocked workers was MEASURED slower at 8 threads than at 2
    # (133 ms vs 43 ms per layer on the dev stack); the memcpys themselves
    # release the GIL, so two threads already keep two copies in flight.
    WORKERS = 2

    # raw_shard keys, in WEIGHT_NAMES order.
    _RAW_KEYS = ("w13", "w13_scale", "w2", "w2_scale")

    def __init__(
        self,
        *,
        sources_by_layer: Dict[int, object],
        cold_slot_expert_ids: Callable[[int], torch.Tensor],
        raw_shapes: Dict[str, tuple],
        moe_layer_indices: Sequence[int],
        num_cold: int,
    ):
        self._sources = dict(sources_by_layer)
        self._ids_for = cold_slot_expert_ids
        self._layers = sorted(moe_layer_indices)
        self._pos = {layer: i for i, layer in enumerate(self._layers)}
        self.num_cold = int(num_cold)

        # Every rank's arena reads are socket-local by geometry (a rank's TP
        # slice lives inside ONE per_numa block); bind the staging pages and
        # the gather threads to that socket or roughly half the ~215 GB/s
        # aggregate gather traffic crosses UPI for nothing. Best effort:
        # binding failures leave today's unbound behavior.
        self._node_cpus = self._arena_node_cpus()

        with self._bound_to_arena_node():
            # device="cpu" is explicit because construction can run under a
            # cuda default device; a device-less empty would silently build
            # GPU "staging" and the first gather would die (or worse).
            self._staging: List[Dict[str, torch.Tensor]] = [
                {
                    n: torch.empty(
                        (self.num_cold,) + tuple(shape),
                        dtype=dtype,
                        device="cpu",
                        pin_memory=True,
                    )
                    for n, (shape, dtype) in raw_shapes.items()
                }
                for _ in range(self.NUM_STAGE)
            ]
        # Staging row views, prebuilt: they depend only on (stage, row), and
        # rebuilding 4 views per expert inside the gather loop is ~1-2 ms of
        # GIL-held work per layer that the forward thread ends up waiting on.
        self._out_views: List[list] = [
            [
                {
                    key: stage_buf[name][i].view(torch.uint8)
                    for key, name in zip(self._RAW_KEYS, WEIGHT_NAMES)
                }
                for i in range(self.num_cold)
            ]
            for stage_buf in self._staging
        ]
        # layer -> list of in-flight chunk futures; removed on consumption.
        self._futures: Dict[int, list] = {}
        # Which layer's gather most recently claimed each stage (debug aid).
        self._stage_layer: List[Optional[int]] = [None] * self.NUM_STAGE
        # WAR gate per stage: recorded on the copy stream after the previous
        # occupant's H2D was enqueued; a gather reusing the stage must wait it
        # so the DMA is not reading rows the pool is overwriting.
        self._stage_free: List[Optional[torch.cuda.Event]] = [None] * self.NUM_STAGE
        self._pool = ThreadPoolExecutor(
            max_workers=self.WORKERS,
            thread_name_prefix="kt-cold-gather",
            initializer=self._bind_worker_thread,
        )

        nbytes = sum(
            t.numel() * t.element_size()
            for buf in self._staging
            for t in buf.values()
        )
        logger.info(
            "[cold-pipeline] arena source: %d stages x %d cold experts = "
            "%.2f GiB pinned staging, %d gather threads",
            self.NUM_STAGE,
            self.num_cold,
            nbytes / (1024**3),
            self.WORKERS,
        )

    def _stage(self, layer_idx: int) -> int:
        return self._pos[layer_idx] % self.NUM_STAGE

    def _arena_node_cpus(self) -> Optional[list]:
        """CPUs of the socket holding this rank's arena partition, or None."""
        try:
            src = next(iter(self._sources.values()))
            node = (src.tp_rank * src.per_gpu) // src.per_numa
            with open(f"/sys/devices/system/node/node{node}/cpulist") as f:
                spec = f.read().strip()
            cpus = set()
            for part in spec.split(","):
                if "-" in part:
                    lo, hi = part.split("-")
                    cpus.update(range(int(lo), int(hi) + 1))
                elif part:
                    cpus.add(int(part))
            cpus &= os.sched_getaffinity(0)
            if not cpus:
                raise RuntimeError("empty intersection with process affinity")
            return sorted(cpus)
        except Exception as exc:
            logger.info(
                "[cold-pipeline] no NUMA binding for the gather (%s); "
                "cross-socket staging traffic possible",
                exc,
            )
            return None

    def _bound_to_arena_node(self):
        """Context manager: pin the calling thread to the arena's socket."""
        import contextlib

        if self._node_cpus is None:
            return contextlib.nullcontext()

        @contextlib.contextmanager
        def bind():
            old = os.sched_getaffinity(0)
            os.sched_setaffinity(0, self._node_cpus)
            try:
                yield
            finally:
                os.sched_setaffinity(0, old)

        return bind()

    def _bind_worker_thread(self) -> None:
        if self._node_cpus is not None:
            try:
                os.sched_setaffinity(0, self._node_cpus)
            except OSError:
                pass

    def begin_gather(self, layer_idx: int) -> None:
        """Start one layer's gather on the pool; idempotent per layer."""
        if layer_idx in self._futures:
            return
        stage = self._stage(layer_idx)
        ids = self._ids_for(layer_idx)
        if len(ids) != self.num_cold:
            raise RuntimeError(
                f"layer {layer_idx}: {len(ids)} cold slots, staging holds "
                f"{self.num_cold}"
            )
        free_ev = self._stage_free[stage]
        self._stage_layer[stage] = layer_idx
        source = self._sources[layer_idx]
        out_views = self._out_views[stage]
        ids_list = ids.tolist()

        def gather(lo: int, hi: int) -> None:
            if free_ev is not None:
                free_ev.synchronize()
            for i in range(lo, hi):
                source.raw_shard_into(ids_list[i], out_views[i])

        step = -(-self.num_cold // self.WORKERS)
        self._futures[layer_idx] = [
            self._pool.submit(gather, lo, min(lo + step, self.num_cold))
            for lo in range(0, self.num_cold, step)
        ]

    def layer_rows(self, layer_idx: int, name: str) -> torch.Tensor:
        """The layer's packed cold rows for one name; blocks until gathered.

        Look-ahead is kicked BEFORE waiting on this layer, so a cold start
        (prime, first layer of every pass) overlaps the next layers' gathers
        with this one's wait instead of serializing them; in steady state a
        gather has ~2 layer periods of head start and the wait is near zero.
        """
        i = self._pos[layer_idx]
        if layer_idx not in self._futures:
            self.begin_gather(layer_idx)  # cold start (prime, or a miss)
        for ahead in self._layers[i + 1 : i + self.NUM_STAGE]:
            self.begin_gather(ahead)
        for f in self._futures[layer_idx]:
            f.result()  # propagate a gather failure loudly, never stale bytes
        return self._staging[self._stage(layer_idx)][name]

    def after_enqueue(self, layer_idx: int, stream: torch.cuda.Stream) -> None:
        """Mark the stage reusable once the just-enqueued H2D completes."""
        stage = self._stage(layer_idx)
        ev = torch.cuda.Event()
        ev.record(stream)
        self._stage_free[stage] = ev
        self._futures.pop(layer_idx, None)

    def reset(self) -> None:
        """Drain in-flight gathers; called with both streams synchronized."""
        for futs in self._futures.values():
            for f in futs:
                try:
                    f.result()
                except Exception:
                    logger.exception("[cold-pipeline] gather failed during reset")
        self._futures.clear()
        self._stage_layer = [None] * self.NUM_STAGE
        self._stage_free = [None] * self.NUM_STAGE


class _OverlapProbe:
    """Per-layer copy/compute timing for the prefetch, read after the pass.

    The number that decides whether prefetch is working is not the copy
    duration on its own -- it is how long the COMPUTE stream sat blocked
    waiting for it.  ``wait_prefetch`` issues a stream wait, so bracketing
    that wait with two events on the compute stream measures the stall
    directly: if the copy landed early the two events are adjacent, and if it
    did not, the gap is exactly the time prefetch failed to hide.

    Events are recorded during the pass and only read once it has finished,
    so nothing here synchronises the hot path.
    """

    def __init__(self, num_layers: int):
        def events():
            return [
                torch.cuda.Event(enable_timing=True) for _ in range(num_layers)
            ]

        self._copy_begin, self._copy_end = events(), events()
        self._stall_begin, self._stall_end = events(), events()
        self._compute_end = events()
        self._copied: set = set()
        self._computed: set = set()
        # Host-side, per layer: how long prefetch_layer waited for the arena
        # gather BEFORE enqueuing the H2D. Without it that wait hides inside
        # the copy interval (the copy stream sits idle between copy_begin and
        # the late-enqueued copies) and a gather-headroom deficit reads as an
        # H2D-bandwidth regression.
        self._gather_wait_ms: Dict[int, float] = {}

    def copy_begin(self, pos, stream):
        self._copy_begin[pos].record(stream)

    def copy_end(self, pos, stream):
        self._copy_end[pos].record(stream)
        self._copied.add(pos)

    def stall_begin(self, pos, stream):
        self._stall_begin[pos].record(stream)

    def stall_end(self, pos, stream):
        self._stall_end[pos].record(stream)

    def compute_end(self, pos, stream):
        self._compute_end[pos].record(stream)
        self._computed.add(pos)

    def gather_wait(self, pos, ms):
        self._gather_wait_ms[pos] = ms

    def summarize(self) -> Optional[str]:
        """One line per pass. Caller must have synchronised both streams."""
        rows = []
        for pos in sorted(self._computed & self._copied):
            try:
                copy = self._copy_begin[pos].elapsed_time(self._copy_end[pos])
                stall = self._stall_begin[pos].elapsed_time(self._stall_end[pos])
                compute = self._stall_end[pos].elapsed_time(self._compute_end[pos])
            except RuntimeError:
                continue                # event never recorded this pass
            rows.append((pos, copy, stall, compute))
        self._copied.clear()
        self._computed.clear()
        if not rows:
            return None

        n = len(rows)
        copy = [r[1] for r in rows]
        stall = [r[2] for r in rows]
        compute = [r[3] for r in rows]
        tot_stall = sum(stall)
        tot_compute = sum(compute)
        # Margin: the compute window a copy had to hide under, minus the copy.
        margins = [comp - cp for cp, comp in zip(copy, compute)]
        worst = min(range(n), key=lambda i: margins[i])
        # The stall share is quoted against stall + MoE compute -- the window
        # this probe can see. It is NOT a share of the forward, which also
        # contains attention, the dense path and communication; dividing by
        # the forward needs a number the pipeline does not have.
        gw = [self._gather_wait_ms.get(r[0], 0.0) for r in rows]
        self._gather_wait_ms.clear()
        return (
            f"[cold-pipeline] {n} layers | "
            f"copy {sum(copy)/n:.1f} ms avg (max {max(copy):.1f}), "
            f"{sum(copy)/1000:.2f} s total | "
            f"gather-wait {sum(gw)/n:.1f} ms avg (max {max(gw):.1f}) | "
            f"moe {tot_compute/n:.1f} ms avg | "
            f"STALL {tot_stall/n:.2f} ms avg, {max(stall):.1f} max, "
            f"{tot_stall:.0f} ms total = "
            f"{100*tot_stall/max(tot_stall+tot_compute, 1e-9):.0f}% of stall+moe "
            f"| worst margin {margins[worst]:+.1f} ms at layer pos {rows[worst][0]}"
        )


class ColdExpertPipeline:
    """Streams each layer's cold experts to device, one layer ahead.

    ``device_buffers`` is ``[slot][name] -> [num_cold, *shape]`` on device;
    two slots, alternating by MoE-layer position.
    """

    NUM_SLOTS = 2

    def __init__(
        self,
        *,
        store,  # ColdExpertStore or ArenaColdSource (num_cold/layer_rows/hooks)
        device: torch.device,
        per_expert_shapes: Dict[str, tuple],
        moe_layer_indices: Sequence[int],
        swizzle_plan=None,
        raw_shapes: Optional[Dict[str, tuple]] = None,
    ):
        self._store = store
        self._device = device
        self._layers = sorted(moe_layer_indices)
        self._pos = {layer: i for i, layer in enumerate(self._layers)}
        # Dynamic-swizzle mode: the store holds CHECKPOINT-layout bytes and the
        # trtllm layout is produced here, once per layer, on device.
        #
        # The point of doing it per layer is measured, not stylistic: one
        # expert's TP8 shard swizzles in ~52 us -- launch-bound, only ~42 GB/s
        # for 2.19 MB -- so 276 cold experts x 92 layers issued per expert is
        # ~1.32 s per forward against a ~2.07 s copy floor. Issued once per
        # layer it is ~1.0 ms, ~0.092 s per forward. Same bytes, 14x apart.
        self._swizzle_plan = swizzle_plan

        self._buffers: List[Dict[str, torch.Tensor]] = [
            {
                n: torch.empty(
                    (store.num_cold,) + tuple(shape), dtype=dtype, device=device
                )
                for n, (shape, dtype) in per_expert_shapes.items()
            }
            for _ in range(self.NUM_SLOTS)
        ]
        # Raw landing buffers, only in dynamic-swizzle mode. One per slot, so a
        # layer's raw block can land while the previous layer's swizzled block
        # is still being read.
        self._raw_buffers: Optional[List[Dict[str, torch.Tensor]]] = (
            [
                {
                    n: torch.empty(
                        (store.num_cold,) + tuple(shape), dtype=dtype, device=device
                    )
                    for n, (shape, dtype) in raw_shapes.items()
                }
                for _ in range(self.NUM_SLOTS)
            ]
            if swizzle_plan is not None and raw_shapes is not None
            else None
        )
        # Contiguous landing ground, when the source asks for one. Per slot, for
        # the same reason the raw buffers are: layer n+1's H2D is in flight while
        # layer n is still being unpacked and read.
        self._staging: Optional[List[torch.Tensor]] = (
            [
                torch.empty(store.staging_nbytes, dtype=torch.uint8, device=device)
                for _ in range(self.NUM_SLOTS)
            ]
            if store.staging_nbytes and self._raw_buffers is not None
            else None
        )
        if self._staging is not None:
            logger.info(
                "[cold-pipeline] contiguous staging: %d slots x %.0f MiB device "
                "-- one H2D per layer instead of six pitched copies",
                self.NUM_SLOTS,
                store.staging_nbytes / (1 << 20),
            )

        # Which layer currently occupies each slot (None = never filled).
        self._slot_layer: List[Optional[int]] = [None] * self.NUM_SLOTS

        self._copy_stream = torch.cuda.Stream(device=device)
        self._prefetch_events = [torch.cuda.Event() for _ in range(self.NUM_SLOTS)]
        self._consume_events = [torch.cuda.Event() for _ in range(self.NUM_SLOTS)]
        # Pre-record consume events so the first prefetch does not stall.
        cur = torch.cuda.current_stream(device)
        for ev in self._consume_events:
            ev.record(cur)

        # Toggleable per PASS, not fixed at boot: reset() re-reads the env, so
        # `SGLANG_DEBUG_KT_PIPELINE_OVERLAP=1` exported into a running
        # server's environment... cannot work cross-process -- but the env CAN
        # be flipped via the /set_envs debug route or a config reload without
        # a 17-minute reboot. Costing a boot per decomposition was the bug.
        self._probe = (
            _OverlapProbe(len(self._layers))
            if envs.SGLANG_DEBUG_KT_PIPELINE_OVERLAP.get()
            else None
        )

        nbytes = sum(
            t.numel() * t.element_size()
            for buf in self._buffers
            for t in buf.values()
        )
        logger.info(
            "[cold-pipeline] %d slots x %d cold experts = %.2f GiB device",
            self.NUM_SLOTS,
            store.num_cold,
            nbytes / (1024**3),
        )

    # -- layer bookkeeping -------------------------------------------------

    def _slot(self, layer_idx: int) -> int:
        return self._pos[layer_idx] % self.NUM_SLOTS

    def _next_layer(self, layer_idx: int) -> Optional[int]:
        i = self._pos[layer_idx] + 1
        return self._layers[i] if i < len(self._layers) else None

    # -- the pipeline ------------------------------------------------------

    def prefetch_layer(self, layer_idx: int) -> None:
        """Copy one layer's cold experts into its slot on the copy stream."""
        slot = self._slot(layer_idx)
        if self._probe is not None:
            # Resolve the (arena) gather BEFORE copy_begin and time it
            # host-side: one name waits the layer's whole gather, so the
            # in-loop layer_rows calls return instantly and copy_begin ->
            # copy_end goes back to measuring the transfer alone. A plain
            # store pays a dict lookup.
            t0 = time.perf_counter()
            self._store.layer_rows(layer_idx, WEIGHT_NAMES[0])
            self._probe.gather_wait(
                self._pos[layer_idx], (time.perf_counter() - t0) * 1e3
            )
        with torch.cuda.stream(self._copy_stream):
            # WAR: the slot's previous occupant must be done being read.
            self._copy_stream.wait_event(self._consume_events[slot])
            # After the WAR wait, so this times the transfer and not the
            # queueing behind the previous occupant.
            if self._probe is not None:
                self._probe.copy_begin(self._pos[layer_idx], self._copy_stream)
            dst = self._buffers[slot]
            if self._raw_buffers is None:
                for name in WEIGHT_NAMES:
                    dst[name].copy_(
                        self._store.layer_rows(layer_idx, name), non_blocking=True
                    )
            else:
                # Land the checkpoint-layout bytes, then swizzle the whole
                # layer into the resident-layout buffer. Both stay on the copy
                # stream, so the existing prefetch event still means exactly
                # "this layer's weights are ready to read" and wait_prefetch
                # needs no change.
                self._swizzle_into(slot, layer_idx, dst)
            if self._probe is not None:
                self._probe.copy_end(self._pos[layer_idx], self._copy_stream)
            self._prefetch_events[slot].record(self._copy_stream)
        # After the copies are enqueued: an arena source uses this to recycle
        # its staging slot once the DMA completes; the store's is a no-op.
        self._store.after_enqueue(layer_idx, self._copy_stream)
        self._slot_layer[slot] = layer_idx

    def _swizzle_into(
        self, slot: int, layer_idx: int, dst: Dict[str, torch.Tensor]
    ) -> None:
        """H2D the raw block, then swizzle it into ``dst`` in four gathers."""
        from sglang.srt.layers.moe.kt_mxfp4_export import apply_batched_swizzle

        raw = self._raw_buffers[slot]

        if hasattr(self._store, "issue_layer_copies"):
            # The source owns the H2D issue: there is no host staging to hand
            # back, because the bytes are read straight out of kt's registered
            # arena. Enqueued on the SAME copy stream so the prefetch event's
            # meaning is unchanged. Duck-typed rather than isinstance so both
            # the direct-DMA transport and the arena cold source qualify
            # without this file importing either.
            self._store.issue_layer_copies(
                layer_idx,
                raw,
                self._copy_stream,
                staging=None if self._staging is None else self._staging[slot],
            )
        else:
            for name in WEIGHT_NAMES:
                raw[name].copy_(
                    self._store.layer_rows(layer_idx, name), non_blocking=True
                )
        out = apply_batched_swizzle(
            plan=self._swizzle_plan,
            raw_w13=raw[WEIGHT_NAMES[0]],
            raw_w13_scale=raw[WEIGHT_NAMES[1]],
            raw_w2=raw[WEIGHT_NAMES[2]],
            raw_w2_scale=raw[WEIGHT_NAMES[3]],
        )
        for name, produced in zip(WEIGHT_NAMES, out):
            target = dst[name]
            target.view(torch.uint8).reshape(-1).copy_(
                produced.reshape(-1).view(torch.uint8)
            )

    def wait_prefetch(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Block the compute stream until this layer's weights have landed.

        Returns the device buffers to hand to the MoE call.
        """
        slot = self._slot(layer_idx)
        if self._slot_layer[slot] != layer_idx:
            raise RuntimeError(
                f"cold-pipeline: layer {layer_idx} expects slot {slot} but it "
                f"holds layer {self._slot_layer[slot]} -- prefetch order broke"
            )
        cur = torch.cuda.current_stream(self._device)
        if self._probe is not None:
            self._probe.stall_begin(self._pos[layer_idx], cur)
        cur.wait_event(self._prefetch_events[slot])
        if self._probe is not None:
            self._probe.stall_end(self._pos[layer_idx], cur)
        return self._buffers[slot]

    def record_compute_and_prefetch_next(self, layer_idx: int) -> None:
        """Release this layer's slot and start the layer two ahead."""
        slot = self._slot(layer_idx)
        cur = torch.cuda.current_stream(self._device)
        if self._probe is not None:
            self._probe.compute_end(self._pos[layer_idx], cur)
        self._consume_events[slot].record(cur)
        nxt = self._next_layer(layer_idx)
        if nxt is not None:
            nxt2 = self._next_layer(nxt)
            if nxt2 is not None:
                self.prefetch_layer(nxt2)

    def prime(self) -> None:
        """Fill both slots at the start of a prefill pass."""
        for layer in self._layers[: self.NUM_SLOTS]:
            self.prefetch_layer(layer)

    def reset(self) -> None:
        """Forget slot occupancy so the next prefill re-primes cleanly."""
        torch.cuda.current_stream(self._device).synchronize()
        self._copy_stream.synchronize()
        # Both streams are idle here, so the PREVIOUS pass's events are all
        # complete and readable -- this is the one place the summary costs
        # nothing extra.
        if self._probe is not None:
            line = self._probe.summarize()
            if line is not None:
                logger.info("%s", line)
        self._slot_layer = [None] * self.NUM_SLOTS
        # Re-evaluate the probe toggle at every pass boundary so enabling the
        # decomposition never costs a reboot.
        want_probe = envs.SGLANG_DEBUG_KT_PIPELINE_OVERLAP.get()
        if want_probe and self._probe is None:
            self._probe = _OverlapProbe(len(self._layers))
        elif not want_probe and self._probe is not None:
            self._probe = None
        # Both streams are idle (synchronized above), so the source can drain
        # its gather threads without racing any in-flight DMA.
        self._store.reset()
