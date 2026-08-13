"""Double-buffered cold expert prefetch pipeline for prefill.

While layer N computes, layer N+1's cold experts are loaded from CPU and
swizzled into VMM-backed pages on a separate CUDA stream.  After layer N
finishes, its cold experts are unmapped to free VRAM.

Modeled on ``dwdp/weight_manager.py`` — same double-buffered stream+event
pattern, but the data source is CPU (raw checkpoint / KT AMX buffer)
instead of a peer GPU, and the destination is VMM-backed expert pages
instead of prefetched MNNVL pages.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch

from sglang.srt.layers.moe.expert_vmm import ExpertVmmAllocator

logger = logging.getLogger(__name__)


class ExpertPipeline:
    """Orchestrates per-layer cold expert loading across all MoE layers.

    Double-buffered: while layer N's weights are being consumed from one
    slot, layer N+2's cold experts are being loaded into the other slot.
    """

    def __init__(
        self,
        allocators: Dict[int, ExpertVmmAllocator],
        cold_expert_ids: List[int],
        copy_stream: torch.cuda.Stream,
        device: torch.device,
    ):
        """Args:
            allocators: ``{layer_idx: ExpertVmmAllocator}`` for all MoE layers.
            cold_expert_ids: Logical expert IDs that are cold (not resident).
            copy_stream: Dedicated CUDA stream for CPU→GPU copies + swizzle.
            device: CUDA device.
        """
        self._allocators = allocators
        self._cold_ids = cold_expert_ids
        self._copy_stream = copy_stream
        self._device = device

        # Double-buffered events: prefetch_done[2] + consumed[2].
        self._prefetch_events: List[torch.cuda.Event] = [
            torch.cuda.Event() for _ in range(2)
        ]
        self._consume_events: List[torch.cuda.Event] = [
            torch.cuda.Event() for _ in range(2)
        ]
        # Pre-record consume events so the first prefetch doesn't stall.
        current = torch.cuda.current_stream(device)
        for ev in self._consume_events:
            ev.record(current)

        self._moe_layer_indices = sorted(allocators.keys())
        self._moe_layer_set = set(self._moe_layer_indices)

        # Track which layers have cold experts currently mapped.
        self._mapped_layers: set = set()

    def _buf_idx(self, layer_idx: int) -> int:
        pos = self._moe_layer_indices.index(layer_idx)
        return pos % 2

    def next_moe_layer(self, layer_idx: int) -> Optional[int]:
        pos = self._moe_layer_indices.index(layer_idx)
        if pos + 1 < len(self._moe_layer_indices):
            return self._moe_layer_indices[pos + 1]
        return None

    def prefetch_layer(
        self,
        layer_idx: int,
        cold_data_source,
    ) -> None:
        """Map cold expert pages for ``layer_idx`` and load data from CPU.

        ``cold_data_source`` is a callable that, given (layer_idx, expert_id),
        returns a dict ``{weight_name: cpu_tensor}`` of raw (un-shuffled)
        TP-sharded expert data.
        """
        buf_idx = self._buf_idx(layer_idx)
        allocator = self._allocators[layer_idx]

        with torch.cuda.stream(self._copy_stream):
            # WAR: wait for compute to finish reading this slot before overwriting.
            self._copy_stream.wait_event(self._consume_events[buf_idx])

            for expert_id in self._cold_ids:
                raw = cold_data_source(layer_idx, expert_id)
                allocator.map_cold(expert_id, raw)
                # Swizzle from raw into the VMM-backed pages.
                # The swizzle writes into the same VMM tensor slices that
                # map_cold just backed — done on the copy stream so the
                # compute stream sees the swizzled data after the event.
                # TODO: call swizzle_trtllm_expert here once integrated
                # with the kt_mxfp4_export module.

            self._prefetch_events[buf_idx].record(self._copy_stream)

        self._mapped_layers.add(layer_idx)

    def wait_prefetch(self, layer_idx: int) -> None:
        """Compute stream waits for the prefetch of ``layer_idx`` to complete."""
        buf_idx = self._buf_idx(layer_idx)
        compute_stream = torch.cuda.current_stream(self._device)
        compute_stream.wait_event(self._prefetch_events[buf_idx])

    def record_compute_and_prefetch_next(
        self,
        layer_idx: int,
        cold_data_source,
    ) -> None:
        """Signal that layer's data is consumed and prefetch layer 2 ahead."""
        buf_idx = self._buf_idx(layer_idx)
        compute_stream = torch.cuda.current_stream(self._device)
        self._consume_events[buf_idx].record(compute_stream)

        # Unmap the previous layer's cold experts to free VRAM.
        prev_layer = self._prev_moe_layer(layer_idx)
        if prev_layer is not None and prev_layer in self._mapped_layers:
            self._allocators[prev_layer].unmap_all_cold()
            self._mapped_layers.discard(prev_layer)

        # Prefetch the layer 2 ahead (reuses the same buffer slot).
        next_layer = self.next_moe_layer(layer_idx)
        if next_layer is not None:
            next_next = self.next_moe_layer(next_layer)
            if next_next is not None:
                self.prefetch_layer(next_next, cold_data_source)

    def _prev_moe_layer(self, layer_idx: int) -> Optional[int]:
        pos = self._moe_layer_indices.index(layer_idx)
        if pos > 0:
            return self._moe_layer_indices[pos - 1]
        return None

    def prefetch_first_layers(self, cold_data_source) -> None:
        """Prefetch the first two MoE layers at the start of a prefill."""
        if len(self._moe_layer_indices) >= 1:
            self.prefetch_layer(self._moe_layer_indices[0], cold_data_source)
        if len(self._moe_layer_indices) >= 2:
            self.prefetch_layer(self._moe_layer_indices[1], cold_data_source)

    def unmap_all(self) -> None:
        """Unmap all cold experts across all layers (prefill→decode transition)."""
        for layer_idx in list(self._mapped_layers):
            self._allocators[layer_idx].unmap_all_cold()
        self._mapped_layers.clear()
