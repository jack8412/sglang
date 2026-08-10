# SPDX-License-Identifier: Apache-2.0
"""Move one MXFP4 expert's weights from the checkpoint into a resident GPU row.

This is the only step of a swap that physically rewrites weights, so it is
kept apart from the bookkeeping and made verifiable on its own: ``verify_row``
rebuilds an expert that is *already* resident and compares byte-for-byte
against the row the production loader produced. If that matches, the read,
the TP slice, the gate/up assembly and the swizzle are all correct together —
which is the only claim worth making, since a mistake anywhere in that chain
shows up as slightly-wrong outputs rather than a failure.

Sourcing from the checkpoint rather than from kt-kernel's export is
deliberate. K3 stores every expert individually
(``experts.<i>.w1|w2|w3.weight_packed`` / ``.weight_scale``), so one expert is
~17.5 MB of addressable byte ranges — no layer-wide read — and the on-disk
scales are *already* E8M0 codes, so the bf16 round trip the export path needs
disappears. It also keeps the promotion path independent of whether kt-kernel
happens to hold the expert, which matters once kt stores only the cold set.
"""

import json
import logging
import os
from typing import Dict, Optional, Tuple

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
    another rank's weights, which is exactly the class of error ``verify_row``
    exists to catch.
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


class CheckpointExpertMover:
    """A ``MoveWeightsFn``: write expert ``logical_id`` into resident row.

    The trtllm permutation indices depend only on shapes, so they are computed
    once per layer and reused — recomputing them per swap would dominate the
    cost of the copy itself.
    """

    def __init__(
        self,
        weight_path: str,
        *,
        expert_prefix_for_layer,
        tp_rank: int,
        tp_size: int,
        param_names: Tuple[str, ...],
    ):
        self.reader = CheckpointExpertReader(weight_path)
        self.expert_prefix_for_layer = expert_prefix_for_layer
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.param_names = param_names
        self._indices = None

    def _permute_indices(self, bytes_):
        from sglang.srt.layers.moe.kt_mxfp4_export import trtllm_permute_indices

        if self._indices is None:
            self._indices = trtllm_permute_indices(
                w13_sample=bytes_.w13,
                w13_scale_sample=bytes_.w13_scale_e8m0,
                w2_sample=bytes_.w2,
                w2_scale_sample=bytes_.w2_scale_e8m0,
                w13_gate_up_halves=True,
            )
        return self._indices

    def _swizzle_into(self, layer, dst_row: int, bytes_, *, out=None) -> None:
        from sglang.srt.layers.moe.kt_mxfp4_export import swizzle_trtllm_expert

        w13_n, w13_s_n, w2_n, w2_s_n = self.param_names
        dst = out or {
            "w13": getattr(layer, w13_n).data[dst_row],
            "w13_scale": getattr(layer, w13_s_n).data[dst_row],
            "w2": getattr(layer, w2_n).data[dst_row],
            "w2_scale": getattr(layer, w2_s_n).data[dst_row],
        }
        device = dst["w13"].device
        on_dev = type(bytes_)(
            w13=bytes_.w13.to(device),
            w13_scale_e8m0=bytes_.w13_scale_e8m0.to(device),
            w2=bytes_.w2.to(device),
            w2_scale_e8m0=bytes_.w2_scale_e8m0.to(device),
        )
        swizzle_trtllm_expert(
            on_dev,
            self._permute_indices(on_dev),
            out_w13=dst["w13"],
            out_w13_scale=dst["w13_scale"],
            out_w2=dst["w2"],
            out_w2_scale=dst["w2_scale"],
        )

    def __call__(self, layer, dst_row: int, logical_id: int) -> None:
        prefix = self.expert_prefix_for_layer(layer)
        bytes_ = build_expert_bytes(
            self.reader,
            prefix,
            logical_id,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
        )
        self._swizzle_into(layer, dst_row, bytes_)

    def verify_row(self, layer, dst_row: int, logical_id: int) -> bool:
        """Bitwise check against a row the production loader already filled.

        Rebuilds ``logical_id`` into scratch buffers shaped like the resident
        row and compares every byte with the live row. Call it for an expert
        that IS currently resident at ``dst_row``: equality proves the whole
        chain — checkpoint read, TP slice, gate/up assembly, swizzle — agrees
        with how the model was actually loaded. It never writes to the layer.
        """
        prefix = self.expert_prefix_for_layer(layer)
        bytes_ = build_expert_bytes(
            self.reader, prefix, logical_id, tp_rank=self.tp_rank, tp_size=self.tp_size
        )
        w13_n, w13_s_n, w2_n, w2_s_n = self.param_names
        live = {
            "w13": getattr(layer, w13_n).data[dst_row],
            "w13_scale": getattr(layer, w13_s_n).data[dst_row],
            "w2": getattr(layer, w2_n).data[dst_row],
            "w2_scale": getattr(layer, w2_s_n).data[dst_row],
        }
        scratch = {k: torch.zeros_like(v) for k, v in live.items()}
        self._swizzle_into(layer, dst_row, bytes_, out=scratch)

        ok = True
        for name, ref in live.items():
            got = scratch[name]
            same = torch.equal(got.view(torch.uint8), ref.view(torch.uint8))
            if not same:
                diff = (got.view(torch.uint8) != ref.view(torch.uint8)).sum().item()
                logger.error(
                    "[kt-swap-verify] layer expert %d row %d: %s MISMATCH "
                    "(%d/%d bytes differ)",
                    logical_id,
                    dst_row,
                    name,
                    diff,
                    ref.numel(),
                )
                ok = False
        if ok:
            logger.info(
                "[kt-swap-verify] expert %d row %d: bitwise match on all four "
                "params — mover agrees with the production loader",
                logical_id,
                dst_row,
            )
        return ok
