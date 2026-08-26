# SPDX-License-Identifier: Apache-2.0
"""Read one MXFP4 expert's weights out of the checkpoint, in kt's layout.

WHAT IS LEFT HERE. This module used to carry ``CheckpointExpertMover``, which
wrote a checkpoint-sourced expert straight into a resident GPU row -- the
promotion and demotion fallback for swaps. Both are gone: a promoted expert's
bytes come out of kt's arena by DMA and a demoted expert's are written back
into it by each rank's own shard writer, so no swap touches the checkpoint at
all.

What survives is the READER, and it survives for a different job: the
split-prefill swizzle plan is derived from one 2.2 MB sample read of a single
expert, once per process, to recover the trtllm permute indices. The layout
knowledge that made the mover correct is the same knowledge that makes that
sample correct, which is why it stays in one place.

Sourcing from the checkpoint rather than from kt-kernel's export is
deliberate. K3 stores every expert individually
(``experts.<i>.w1|w2|w3.weight_packed`` / ``.weight_scale``), so one expert is
~17.5 MB of addressable byte ranges — no layer-wide read — and the on-disk
scales are *already* E8M0 codes, so the bf16 round trip the export path needs
disappears.
"""

import json
import logging
import os
from typing import Dict, Tuple

import torch

logger = logging.getLogger(__name__)

# Checkpoint names, relative to a layer's expert prefix. w1 = gate, w3 = up,
# w2 = down; w13 is the [gate | up] concatenation the trtllm epilogue wants.
_GATE, _UP, _DOWN = "w1", "w3", "w2"


class CheckpointExpertReader:
    """Reads one expert's six MXFP4 tensors, lazily and per-expert.

    safetensors files are opened once and memory-mapped, so repeated reads hit
    page cache rather than the device, and a read touches only the byte ranges
    of the tensors it names.
    """

    def __init__(self, weight_path: str):
        self.weight_path = weight_path
        index_files = [
            f for f in os.listdir(weight_path) if f.endswith(".index.json")
        ]
        if not index_files:
            raise FileNotFoundError(f"no *.index.json under {weight_path}")
        with open(os.path.join(weight_path, index_files[0])) as f:
            self.weight_map: Dict[str, str] = json.load(f)["weight_map"]
        self._handles: Dict[str, object] = {}

    def _get(self, name: str) -> torch.Tensor:
        shard = self.weight_map.get(name)
        if shard is None:
            raise KeyError(f"{name} not in checkpoint index")
        handle = self._handles.get(shard)
        if handle is None:
            from safetensors import safe_open

            handle = safe_open(
                os.path.join(self.weight_path, shard), framework="pt", device="cpu"
            )
            self._handles[shard] = handle
        return handle.get_tensor(name)

    def read_expert(
        self, expert_prefix: str, expert_id: int
    ) -> Tuple[torch.Tensor, ...]:
        """Return (gate, gate_scale, up, up_scale, down, down_scale)."""
        base = f"{expert_prefix}.{expert_id}"
        return tuple(
            self._get(f"{base}.{proj}.{suffix}")
            for proj in (_GATE, _UP, _DOWN)
            for suffix in ("weight_packed", "weight_scale")
        )

    def close(self) -> None:
        self._handles.clear()


def build_expert_bytes(
    reader: CheckpointExpertReader,
    expert_prefix: str,
    expert_id: int,
    *,
    tp_rank: int,
    tp_size: int,
):
    """Assemble one expert's TP shard in the orientation the swizzle expects.

    gate/up are ``[intermediate, hidden]``-shaped, so a TP shard is a
    contiguous ROW slice; down is ``[hidden, intermediate]``, so its shard is a
    COLUMN slice — halved again because two FP4 values share a byte, and
    likewise for its scales at one code per 32-element group. Getting either
    slice wrong yields a valid-looking tensor of the right shape holding
    another rank's weights -- wrong outputs rather than a failure, which is why
    the arithmetic is spelled out here rather than inlined at the call site.
    """
    from sglang.srt.layers.moe.kt_mxfp4_export import Mxfp4ExpertBytes

    gate, gate_s, up, up_s, down, down_s = reader.read_expert(
        expert_prefix, expert_id
    )

    inter = gate.shape[0]
    if inter % tp_size:
        raise ValueError(f"intermediate {inter} not divisible by tp_size {tp_size}")
    per = inter // tp_size
    lo, hi = tp_rank * per, (tp_rank + 1) * per

    w13 = torch.cat([gate[lo:hi], up[lo:hi]], dim=0).contiguous()
    w13_scale = torch.cat([gate_s[lo:hi], up_s[lo:hi]], dim=0).contiguous()
    # down: [hidden, intermediate] -> columns for this rank. Packed bytes hold
    # two values each and scales one code per 32 values, so both column ranges
    # scale down accordingly.
    w2 = down[:, lo // 2 : hi // 2].contiguous()
    w2_scale = down_s[:, lo // 32 : hi // 32].contiguous()

    return Mxfp4ExpertBytes(
        w13=w13,
        w13_scale_e8m0=w13_scale.to(torch.uint8),
        w2=w2,
        w2_scale_e8m0=w2_scale.to(torch.uint8),
    )
