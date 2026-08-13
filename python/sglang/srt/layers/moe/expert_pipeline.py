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
from typing import Dict, List, Optional, Sequence

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.expert_cold_store import WEIGHT_NAMES, ColdExpertStore

logger = logging.getLogger(__name__)


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
        return (
            f"[cold-pipeline] {n} layers | "
            f"copy {sum(copy)/n:.1f} ms avg (max {max(copy):.1f}) | "
            f"compute {tot_compute/n:.1f} ms avg | "
            f"STALL {tot_stall/n:.2f} ms avg, {max(stall):.1f} max, "
            f"{tot_stall:.0f} ms total = {100*tot_stall/max(tot_stall+tot_compute, 1e-9):.0f}% "
            f"of the pass | worst margin {margins[worst]:+.1f} ms at layer pos "
            f"{rows[worst][0]}"
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
        store: ColdExpertStore,
        device: torch.device,
        per_expert_shapes: Dict[str, tuple],
        moe_layer_indices: Sequence[int],
    ):
        self._store = store
        self._device = device
        self._layers = sorted(moe_layer_indices)
        self._pos = {layer: i for i, layer in enumerate(self._layers)}

        self._buffers: List[Dict[str, torch.Tensor]] = [
            {
                n: torch.empty(
                    (store.num_cold,) + tuple(shape), dtype=dtype, device=device
                )
                for n, (shape, dtype) in per_expert_shapes.items()
            }
            for _ in range(self.NUM_SLOTS)
        ]
        # Which layer currently occupies each slot (None = never filled).
        self._slot_layer: List[Optional[int]] = [None] * self.NUM_SLOTS

        self._copy_stream = torch.cuda.Stream(device=device)
        self._prefetch_events = [torch.cuda.Event() for _ in range(self.NUM_SLOTS)]
        self._consume_events = [torch.cuda.Event() for _ in range(self.NUM_SLOTS)]
        # Pre-record consume events so the first prefetch does not stall.
        cur = torch.cuda.current_stream(device)
        for ev in self._consume_events:
            ev.record(cur)

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
        with torch.cuda.stream(self._copy_stream):
            # WAR: the slot's previous occupant must be done being read.
            self._copy_stream.wait_event(self._consume_events[slot])
            # After the WAR wait, so this times the transfer and not the
            # queueing behind the previous occupant.
            if self._probe is not None:
                self._probe.copy_begin(self._pos[layer_idx], self._copy_stream)
            dst = self._buffers[slot]
            for name in WEIGHT_NAMES:
                dst[name].copy_(
                    self._store.layer_rows(layer_idx, name), non_blocking=True
                )
            if self._probe is not None:
                self._probe.copy_end(self._pos[layer_idx], self._copy_stream)
            self._prefetch_events[slot].record(self._copy_stream)
        self._slot_layer[slot] = layer_idx

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
