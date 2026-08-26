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

from sglang.srt.layers.moe.kt_mxfp4_export import WEIGHT_NAMES

logger = logging.getLogger(__name__)


class ColdExpertPipeline:
    """Streams each layer's cold experts to device, one layer ahead.

    The copy stream carries the H2D transfer ONLY. In dynamic-swizzle mode the
    gather that turns checkpoint layout into trtllm layout runs on the COMPUTE
    stream instead, because the copy stream is the floor (~25 ms/layer against
    4-7 ms of MoE) and the compute stream has the slack to hide it.
    """

    NUM_SLOTS = 2

    def __init__(
        self,
        *,
        source,  # a cold source: num_cold / issue_layer_copies / lifecycle
        device: torch.device,
        per_expert_shapes: Dict[str, tuple],
        moe_layer_indices: Sequence[int],
        swizzle_plan=None,
        raw_shapes: Optional[Dict[str, tuple]] = None,
    ):
        self._source = source
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

        # WHICH buffer the copy stream writes decides how many of each is
        # needed: the copy stream fills raw[slot] and the compute stream
        # gathers raw -> resident. Resident is written AND read by the compute
        # stream, in order, so ONE is enough; raw needs NUM_SLOTS, because
        # layer L's raw is being gathered while L+1's is still landing. The
        # copy-stream-written buffer is what _slot() indexes. 2 raw + 1
        # resident costs 1.66 GiB/rank where 2 + 2 cost 2.22.
        #
        # There is no second mode. A caller with no swizzle plan used to write
        # resident directly (the pre-swizzled store); finalize_split_prefill
        # now raises rather than building this without one, since kt's arena is
        # checkpoint layout and nothing else can produce a resident row from it.
        self._buffers: List[Dict[str, torch.Tensor]] = [
            {
                n: torch.empty(
                    (source.num_cold,) + tuple(shape), dtype=dtype, device=device
                )
                for n, (shape, dtype) in per_expert_shapes.items()
            }
        ]
        self._raw_buffers: List[Dict[str, torch.Tensor]] = [
            {
                n: torch.empty(
                    (source.num_cold,) + tuple(shape), dtype=dtype, device=device
                )
                for n, (shape, dtype) in raw_shapes.items()
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

        def _gib(bufs) -> float:
            return sum(
                t.numel() * t.element_size() for buf in bufs for t in buf.values()
            ) / (1024**3)

        raw_gib = _gib(self._raw_buffers)
        logger.info(
            "[cold-pipeline] %d cold experts: %d resident + %d raw = %.2f GiB "
            "device (%.2f resident + %.2f raw)",
            source.num_cold,
            len(self._buffers),
            len(self._raw_buffers),
            _gib(self._buffers) + raw_gib,
            _gib(self._buffers),
            raw_gib,
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
            # TRANSFER ONLY. The swizzle is a gather kernel and this stream is
            # the floor, so running it here puts compute on the critical path;
            # it moves to wait_prefetch on the compute stream, which idles most
            # of every layer. The prefetch event therefore means "the raw block
            # has landed", not "resident is ready".
            self._issue_raw(slot, layer_idx)
            self._prefetch_events[slot].record(self._copy_stream)
        # After the copies are enqueued: an arena source uses this to recycle
        # its staging slot once the DMA completes; the store's is a no-op.
        self._source.after_enqueue(layer_idx, self._copy_stream)
        self._slot_layer[slot] = layer_idx

    def _issue_raw(self, slot: int, layer_idx: int) -> None:
        """Enqueue this layer's checkpoint-layout block into its raw slot."""
        # The source owns the H2D issue: there is no host staging to hand
        # back, because the bytes are read straight out of kt's registered
        # arena. Enqueued on the SAME copy stream so the prefetch event's
        # meaning is unchanged.
        #
        # Unguarded, and that is deliberate: this used to sit behind a hasattr
        # so the deleted direct-DMA transport and the arena source could both
        # qualify, with a layer_rows host-copy fallback for sources that had
        # neither. There is one source now, and an unguarded call is what makes
        # test_arena_source_implements_every_unguarded_store_call treat this as
        # REQUIRED rather than optional.
        self._source.issue_layer_copies(
            layer_idx, self._raw_buffers[slot], self._copy_stream
        )

    def _swizzle_into(self, slot: int, dst: Dict[str, torch.Tensor]) -> None:
        """Gather the raw block into ``dst`` on the CALLER's stream.

        Four gathers, no per-expert loop. Runs on the compute stream: the same
        kernels, moved off the bottleneck.
        """
        from sglang.srt.layers.moe.kt_mxfp4_export import apply_batched_swizzle

        raw = self._raw_buffers[slot]
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
        cur.wait_event(self._prefetch_events[slot])
        # One resident buffer is enough because this stream both writes and
        # reads it: the previous layer's MoE was enqueued before this call, so
        # it has already read the buffer by the time this gather overwrites it.
        dst = self._buffers[0]
        self._swizzle_into(slot, dst)
        # The raw slot dies as soon as the gather has read it -- earlier than
        # the MoE, so the copy stream waits less than it used to.
        self._consume_events[slot].record(cur)
        return dst

    def record_compute_and_prefetch_next(self, layer_idx: int) -> None:
        """Start the layer two ahead. This layer's slot is already released.

        No consume event here: wait_prefetch records it as soon as the gather
        has read the raw slot, which is earlier than the MoE finishes and is
        what lets the copy stream run ahead. The plain-store mode that had to
        wait until here is gone.
        """
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
        self._slot_layer = [None] * self.NUM_SLOTS
        # Both streams are idle (synchronized above), so the source can drain
        # its gather threads without racing any in-flight DMA.
        self._source.reset()
