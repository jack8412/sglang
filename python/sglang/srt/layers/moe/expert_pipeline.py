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

from sglang.srt.layers.moe.expert_cold_store import WEIGHT_NAMES, ColdExpertStore

logger = logging.getLogger(__name__)


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
            dst = self._buffers[slot]
            for name in WEIGHT_NAMES:
                dst[name].copy_(
                    self._store.layer_rows(layer_idx, name), non_blocking=True
                )
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
        torch.cuda.current_stream(self._device).wait_event(
            self._prefetch_events[slot]
        )
        return self._buffers[slot]

    def record_compute_and_prefetch_next(self, layer_idx: int) -> None:
        """Release this layer's slot and start the layer two ahead."""
        slot = self._slot(layer_idx)
        self._consume_events[slot].record(torch.cuda.current_stream(self._device))
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
        self._slot_layer = [None] * self.NUM_SLOTS
