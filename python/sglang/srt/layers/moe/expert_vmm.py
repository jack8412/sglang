"""VMM-backed expert weight allocation for the prefill expert pipeline.

Reserves virtual address space for the full [num_experts, ...] weight tensor
per layer, backs the resident experts at boot, and dynamically maps/unmaps
cold expert pages during prefill.  The kernel sees one contiguous tensor —
no ID remapping, no margin routing, no kernel changes.

Reuses the VMM primitives from ``dwdp/vmm.py`` (cuMemCreate, cuMemMap,
cuMemUnmap, cuMemSetAccess, tensor_from_ptr).
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
from cuda.bindings import driver as cuda

from sglang.srt.distributed.device_communicators.vmm_utils import (
    check_drv,
    make_rw_access_desc,
)
from sglang.srt.layers.moe.dwdp.vmm import (
    create_local_handle,
    free_va,
    map_handle,
    reserve_va,
    set_access,
    tensor_from_ptr,
    unmap_va,
    release_handle,
)

logger = logging.getLogger(__name__)

# B200 granularity is 2 MiB.  Query at runtime for portability.
def _get_granularity(device_id: int) -> int:
    prop = cuda.CUmemAllocationProp()
    prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device_id
    result = cuda.cuMemGetAllocationGranularity(
        prop, cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
    )
    return int(check_drv(result, "cuMemGetAllocationGranularity"))


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


class ExpertVmmAllocator:
    """Manages VMM-backed expert weight tensors for one layer.

    Reserves VA for ``[num_experts, *per_expert_shape]`` per weight name.
    At construction, backs the resident experts' pages and copies their data.
    Cold expert pages are left unbacked — mapped on demand via
    :meth:`map_cold` and freed via :meth:`unmap_cold`.
    """

    def __init__(
        self,
        device_id: int,
        layer_idx: int,
        weight_specs: Dict[str, Tuple[torch.Size, torch.dtype]],
        num_experts: int,
        gpu_experts_mask: torch.Tensor,
    ):
        """Args:
            device_id: CUDA device index.
            layer_idx: Layer index (for logging).
            weight_specs: ``{name: (full_shape_excluding_expert_dim, dtype)}``
                e.g. ``{"w13_weight": ((768, 3584), torch.int8), ...}``
            num_experts: Total number of experts (896).
            gpu_experts_mask: Bool tensor ``[num_experts]``, True = resident.
        """
        self.device_id = device_id
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.granularity = _get_granularity(device_id)
        self.gpu_experts_mask = gpu_experts_mask

        # Per weight name: VA base, per-expert byte offset, per-expert size
        # (aligned), full tensor shape, dtype.
        self._weight_specs: Dict[str, dict] = {}
        self._va_bases: Dict[str, int] = {}
        self._va_sizes: Dict[str, int] = {}
        # Per expert: list of (handle or 0) — 0 means unbacked.
        self._handles: Dict[str, List[int]] = {}
        self._tensors: Dict[str, torch.Tensor] = {}

        for name, (shape, dtype) in weight_specs.items():
            itemsize = dtype.itemsize
            numel = 1
            for d in shape:
                numel *= d
            per_expert_bytes = numel * itemsize
            per_expert_aligned = _align_up(per_expert_bytes, self.granularity)
            total_bytes = per_expert_aligned * num_experts

            va_base = reserve_va(total_bytes, self.granularity)

            self._va_bases[name] = va_base
            self._va_sizes[name] = total_bytes
            self._handles[name] = [0] * num_experts
            self._weight_specs[name] = {
                "shape": (num_experts,) + tuple(shape),
                "dtype": dtype,
                "per_expert_bytes": per_expert_bytes,
                "per_expert_aligned": per_expert_aligned,
            }

        self._resident_mapped = False
        self._cold_mapped: set = set()  # expert IDs currently backed

    def map_resident(self, expert_id: int, weight_data: Dict[str, torch.Tensor]) -> None:
        """Back one resident expert's VA and copy its (already-shuffled) data."""
        for name, data in weight_data.items():
            self._map_one(expert_id, name, data)

    def _map_one(self, expert_id: int, name: str, data: torch.Tensor) -> None:
        """Create a physical handle, map it at the expert's VA, copy data."""
        spec = self._weight_specs[name]
        aligned = spec["per_expert_aligned"]
        offset = expert_id * aligned
        va = self._va_bases[name] + offset

        handle = create_local_handle(aligned, self.device_id)
        map_handle(va, aligned, handle)
        set_access(va, aligned, self.device_id)
        self._handles[name][expert_id] = handle

        # Copy weight data into the mapped pages.
        dst = tensor_from_ptr(
            ptr=va,
            shape=spec["shape"][1:],  # exclude expert dim
            dtype=spec["dtype"],
            device_id=self.device_id,
        )
        dst.copy_(data)

    def map_cold(self, expert_id: int, raw_data: Dict[str, torch.Tensor]) -> None:
        """Back a cold expert's VA with fresh physical pages and copy raw data.

        ``raw_data`` is the raw (un-shuffled) TP-sharded expert on CPU.
        Swizzling into trtllm format happens separately on GPU after this copy.
        """
        for name, data in raw_data.items():
            self._map_one(expert_id, name, data)
        self._cold_mapped.add(expert_id)

    def unmap_cold(self, expert_id: int) -> None:
        """Unback a cold expert's VA, freeing physical VRAM."""
        for name in self._weight_specs:
            handle = self._handles[name][expert_id]
            if handle != 0:
                spec = self._weight_specs[name]
                aligned = spec["per_expert_aligned"]
                offset = expert_id * aligned
                va = self._va_bases[name] + offset
                unmap_va(va, aligned)
                release_handle(handle)
                self._handles[name][expert_id] = 0
        self._cold_mapped.discard(expert_id)

    def unmap_all_cold(self) -> None:
        """Unback all cold experts (called at prefill→decode transition)."""
        for expert_id in list(self._cold_mapped):
            self.unmap_cold(expert_id)

    def get_tensor(self, name: str) -> torch.Tensor:
        """Return the full ``[num_experts, ...]`` torch tensor backed by VMM VA."""
        if name not in self._tensors:
            spec = self._weight_specs[name]
            self._tensors[name] = tensor_from_ptr(
                ptr=self._va_bases[name],
                shape=spec["shape"],
                dtype=spec["dtype"],
                device_id=self.device_id,
            )
        return self._tensors[name]

    def get_expert_slice(self, name: str, expert_id: int) -> torch.Tensor:
        """Return a view of one expert's weight slice in the VMM tensor."""
        spec = self._weight_specs[name]
        aligned = spec["per_expert_aligned"]
        offset = expert_id * aligned
        return tensor_from_ptr(
            ptr=self._va_bases[name] + offset,
            shape=spec["shape"][1:],
            dtype=spec["dtype"],
            device_id=self.device_id,
        )

    def destroy(self) -> None:
        """Release all handles and free the VA reservation."""
        for name in list(self._handles.keys()):
            for expert_id in range(self.num_experts):
                handle = self._handles[name][expert_id]
                if handle != 0:
                    spec = self._weight_specs[name]
                    aligned = spec["per_expert_aligned"]
                    offset = expert_id * aligned
                    va = self._va_bases[name] + offset
                    unmap_va(va, aligned)
                    release_handle(handle)
                    self._handles[name][expert_id] = 0
            free_va(self._va_bases[name], self._va_sizes[name])
            del self._handles[name]
            del self._va_bases[name]
            del self._va_sizes[name]
        self._tensors.clear()
        self._cold_mapped.clear()
