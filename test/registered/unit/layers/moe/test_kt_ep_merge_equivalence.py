"""Merge-equivalence test for srt/layers/moe/kt_ep_wrapper.py.

Proves the KT hybrid contract on dummy weights: the wrapper's
mask -> remap -> GPU apply -> CPU sync -> add pipeline produces exactly the
monolithic MoE result, with each side applying routing weights for its own
disjoint expert set (kt-kernel returns a pre-weighted CPU contribution).
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.moe import kt_ep_wrapper as ktw
from sglang.srt.layers.moe.kt_ep_wrapper import (
    KTMOE_WRAPPER_BASE_CTOR_PARAMS,
    KTMOE_WRAPPER_SITU_CTOR_PARAMS,
    KTConfig,
    KTEPWrapperMethod,
)
from sglang.srt.runtime_context import get_parallel, reset_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

NUM_EXPERTS = 64
TOP_K = 8
HIDDEN = 128  # latent-space width for K3-shaped layers
INTERMEDIATE = 96
NUM_TOKENS = 17


def _silu_mlp(x, w13, w2):
    """Reference expert: silu(x @ w1) * (x @ w3) @ w2, w13 = [w1; w3]."""
    gate_up = x @ w13.t()
    gate, up = gate_up.chunk(2, dim=-1)
    return (torch.nn.functional.silu(gate) * up) @ w2.t()


def _reference_moe(x, topk_weights, topk_ids, w13, w2, expert_filter=None):
    """Monolithic reference: sum over top-k experts of weight * expert(x).

    expert_filter: optional bool mask [num_experts]; experts whose entry is
    False contribute zero (used to split the GPU-side / CPU-side sums).
    """
    out = torch.zeros_like(x)
    for t in range(x.shape[0]):
        for k in range(topk_ids.shape[1]):
            e = int(topk_ids[t, k])
            if e < 0:
                continue
            if expert_filter is not None and not bool(expert_filter[e]):
                continue
            out[t] += topk_weights[t, k] * _silu_mlp(x[t : t + 1], w13[e], w2[e])[0]
    return out


class _MockKTMoEWrapper:
    """CPU-expert engine mock with the real ctor contract.

    The signature is derived from the module's canonical ctor-param constants
    so this mock cannot drift from the wheel contract (amendment C1)."""

    _ALLOWED = KTMOE_WRAPPER_BASE_CTOR_PARAMS | KTMOE_WRAPPER_SITU_CTOR_PARAMS

    # Class-level stash: weights for ALL experts + the CPU-resident mask,
    # injected by the test before wrapper construction.
    w13 = None
    w2 = None
    gpu_mask = None
    instances = []

    def __init__(self, **kwargs):
        unknown = set(kwargs) - self._ALLOWED
        assert not unknown, f"mock ctor got params outside the contract: {unknown}"
        self.ctor_kwargs = kwargs
        self._pending = None
        type(self).instances.append(self)

    def submit_forward(self, x, topk_ids, topk_weights, cuda_stream):
        self._pending = (x.clone(), topk_ids.clone(), topk_weights.clone())

    def sync_forward(self, ref_tensor, cuda_stream):
        x, topk_ids, topk_weights = self._pending
        self._pending = None
        # kt-kernel applies routing weights for CPU-resident experts itself:
        # the returned tensor is the pre-weighted CPU contribution.
        cpu_resident = ~type(self).gpu_mask
        return _reference_moe(
            x, topk_weights, topk_ids, type(self).w13, type(self).w2,
            expert_filter=cpu_resident,
        ).to(ref_tensor.dtype)

    def load_weights(self, physical_to_logical_map_cpu):
        pass


class _FakeGpuMethod:
    """GPU quant method double: computes the reference over the REMAPPED
    dense GPU expert slots, skipping -1 (masked CPU experts) — mirroring what
    a real GPU MoE kernel does with the wrapper's masked dispatch."""

    def __init__(self, logical_to_gpu_index):
        # dense slot -> logical expert id
        num_gpu = int((logical_to_gpu_index >= 0).sum())
        self.slot_to_logical = torch.full((num_gpu,), -1, dtype=torch.long)
        for logical, slot in enumerate(logical_to_gpu_index.tolist()):
            if slot >= 0:
                self.slot_to_logical[slot] = logical

    def create_moe_runner(self, layer, moe_runner_config):
        pass

    def apply(self, layer, dispatch_output):
        x = dispatch_output.hidden_states
        topk_weights, topk_ids, _ = dispatch_output.topk_output
        out = torch.zeros_like(x)
        for t in range(x.shape[0]):
            for k in range(topk_ids.shape[1]):
                slot = int(topk_ids[t, k])
                if slot < 0:
                    continue
                e = int(self.slot_to_logical[slot])
                assert e >= 0, "GPU method received a slot with no logical owner"
                out[t] += topk_weights[t, k] * _silu_mlp(
                    x[t : t + 1], _MockKTMoEWrapper.w13[e], _MockKTMoEWrapper.w2[e]
                )[0]
        return SimpleNamespace(hidden_states=out)


class TestKTEPMergeEquivalence(CustomTestCase):
    def tearDown(self):
        reset_context()
        _MockKTMoEWrapper.instances.clear()

    def _run_hybrid(self, num_gpu_experts, seed=0):
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed)

        # Dummy weights for all experts + a non-contiguous GPU mask.
        w13 = torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN, generator=gen) * 0.1
        w2 = torch.randn(NUM_EXPERTS, HIDDEN, INTERMEDIATE, generator=gen) * 0.1
        perm = torch.randperm(NUM_EXPERTS, generator=gen)
        gpu_mask = torch.zeros(NUM_EXPERTS, dtype=torch.bool)
        gpu_mask[perm[:num_gpu_experts]] = True

        _MockKTMoEWrapper.w13 = w13
        _MockKTMoEWrapper.w2 = w2
        _MockKTMoEWrapper.gpu_mask = gpu_mask

        x = torch.randn(NUM_TOKENS, HIDDEN, generator=gen)
        logits = torch.randn(NUM_TOKENS, NUM_EXPERTS, generator=gen)
        topk_weights, topk_ids = torch.topk(torch.softmax(logits, -1), TOP_K, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)
        topk_ids = topk_ids.to(torch.int32)

        monolithic = _reference_moe(x, topk_weights, topk_ids, w13, w2)

        kt_config = KTConfig(
            layer_idx=0,
            gpu_experts_mask=gpu_mask,
            cpuinfer_threads=2,
            threadpool_count=1,
            weight_path="/nonexistent-dummy",
            chunked_prefill_size=64,
            max_deferred_experts_per_token=0,
            method="MXFP4",
            num_layers=1,
        )

        with patch.multiple(
            ktw,
            KTRANSFORMERS_AVAILABLE=True,
            KTMoEWrapper=_MockKTMoEWrapper,
            KT_WHEEL_SUPPORTS_SITU=True,
            create=True,
        ), get_parallel().override(tp_rank=0, tp_size=1):
            method = KTEPWrapperMethod(MagicMock(), kt_config)
            fake_gpu = _FakeGpuMethod(method.logical_to_gpu_index)
            method.gpu_method = fake_gpu
            # Minimal runtime state normally set up in create_weights /
            # create_moe_runner — built by hand so the test needs no CUDA:
            method.wrapper = _MockKTMoEWrapper(
                layer_idx=0,
                num_experts=NUM_EXPERTS,
                num_experts_per_tok=TOP_K,
                hidden_size=HIDDEN,
                moe_intermediate_size=INTERMEDIATE,
                gpu_experts_mask=gpu_mask,
                cpuinfer_threads=2,
                threadpool_count=1,
                weight_path="/nonexistent-dummy",
                chunked_prefill_size=64,
                method="MXFP4",
            )
            method.moe_runner_config = SimpleNamespace(
                activation="silu", routed_scaling_factor=None
            )
            method.gpu_experts_mask_cuda = gpu_mask.clone()
            method.logical_to_gpu_index_cuda = method.logical_to_gpu_index.clone()
            method.global_num_experts = NUM_EXPERTS
            method.gpu_prefill_token_threshold = 0
            method._cpu_stream = None
            method._sync_done_event = None

            hybrid = self._apply_hybrid(method, x, topk_weights, topk_ids)

        return monolithic, hybrid, gpu_mask

    def _apply_hybrid(self, method, x, topk_weights, topk_ids):
        """Drive the wrapper's semantic pipeline without CUDA streams: the
        same mask/remap function and the same merge (gpu + pre-weighted cpu),
        matching KTEPWrapperMethod.apply's hybrid path step-for-step."""
        method.wrapper.submit_forward(x, topk_ids, topk_weights, None)
        masked = ktw.mask_and_remap_expert_ids(
            topk_ids.clone(),
            method.gpu_experts_mask_cuda,
            method.logical_to_gpu_index_cuda,
        )
        dispatch = SimpleNamespace(
            hidden_states=x,
            topk_output=(topk_weights, masked, None),
        )
        gpu_out = method.gpu_method.apply(None, dispatch).hidden_states
        cpu_out = method.wrapper.sync_forward(x, None)
        return gpu_out + cpu_out

    def test_merge_equals_monolithic(self):
        """Derived property (Phase-1 plan): the hybrid pipeline's
        mask -> dense-remap -> GPU partial sum + pre-weighted CPU partial sum
        must reproduce the monolithic MoE bit-for-bit in exact arithmetic.
        Red if the remap stops being a bijection, if either side starts
        applying routing weights for the other's experts (double count), or
        if masked (-1) slots leak into the GPU sum. Includes the num_gpu=0
        (all-CPU) and num_gpu=all (KT-attached but empty CPU set) edges."""
        for num_gpu in (0, 20, 44, NUM_EXPERTS):
            with self.subTest(num_gpu_experts=num_gpu):
                monolithic, hybrid, _ = self._run_hybrid(num_gpu)
                torch.testing.assert_close(hybrid, monolithic, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
