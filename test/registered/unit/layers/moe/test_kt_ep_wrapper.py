"""Unit tests for srt/layers/moe/kt_ep_wrapper.py — no server, no CUDA.

Covers, on CPU only:

- the logical<->GPU expert-index partition/bijection derived in
  ``KTEPWrapperMethod.__init__`` and consumed by ``mask_and_remap_expert_ids``;
- the front-loading / uniform / random GPU-expert placement mask generators;
- the KTMoEWrapper ctor-kwarg resolution in ``create_weights`` for the
  Kimi-K3 SiTU path (situ_beta/situ_linear_beta channel, swiglu zeroing,
  method allow-list, wheel-capability and gemm1_alpha rails), driven through
  a recording mock whose accepted params are generated from the module's
  canonical ctor-contract constants so mock and wheel cannot drift;
- the activation allow-list gate in ``_submit_with_staged_input``;
- the ``ServerArgs._handle_kt`` safety rails and the
  ``validate_kimi_k3_kt`` launch gates, on dummy-boundary ServerArgs.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.arg_groups.kimi_k3_hook import validate_kimi_k3_kt
from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.moe import kt_ep_wrapper as ktw
from sglang.srt.layers.moe.kt_ep_wrapper import (
    KTMOE_WRAPPER_BASE_CTOR_PARAMS,
    KTMOE_WRAPPER_SITU_CTOR_PARAMS,
    KTConfig,
    KTEPWrapperMethod,
    generate_front_loading_masks,
    generate_random_masks,
    generate_uniform_masks,
    mask_and_remap_expert_ids,
)
from sglang.srt.runtime_context import get_parallel, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

NUM_EXPERTS = 64
HIDDEN = 128
INTERMEDIATE = 96

_KT_CTOR_PARAM_UNION = KTMOE_WRAPPER_BASE_CTOR_PARAMS | KTMOE_WRAPPER_SITU_CTOR_PARAMS


class _RecordingKTMoEWrapper:
    """KTMoEWrapper stand-in for the ctor-contract tests (amendment C1).

    The set of accepted ctor params is generated FROM the module's canonical
    contract constants, so if the constants change the mock follows — the mock
    cannot drift from the wheel contract the module enforces at import time.
    """

    last_kwargs = None

    def __init__(self, **kwargs):
        unknown = set(kwargs) - _KT_CTOR_PARAM_UNION
        if unknown:
            raise TypeError(
                f"ctor params outside the canonical KTMoEWrapper contract: "
                f"{sorted(unknown)}"
            )
        type(self).last_kwargs = dict(kwargs)


class _FakeLayer:
    """Carries exactly the attributes KTEPWrapperMethod.create_weights reads."""

    def __init__(
        self,
        moe_runner_config,
        top_k=8,
        intermediate_size_per_partition=INTERMEDIATE,
        moe_tp_size=2,
        num_experts=NUM_EXPERTS,
    ):
        self.top_k = top_k
        self.intermediate_size_per_partition = intermediate_size_per_partition
        self.moe_tp_size = moe_tp_size
        self.moe_runner_config = moe_runner_config
        # _scoped_layer_num_local_experts overrides both counts around the
        # wrapped method's create/load/apply delegations.
        self.num_local_experts = num_experts
        self.num_experts = num_experts
        self._device_anchor = torch.zeros(1)

    def parameters(self):
        # create_weights only does next(layer.parameters()).device.
        return iter([self._device_anchor])


def _random_gpu_mask(num_experts, num_gpu, seed):
    """Random bool mask with exactly num_gpu True entries."""
    gen = torch.Generator().manual_seed(seed)
    mask = torch.zeros(num_experts, dtype=torch.bool)
    mask[torch.randperm(num_experts, generator=gen)[:num_gpu]] = True
    return mask


def _make_kt_config(gpu_mask, method="MXFP4", layer_idx=0, num_layers=6, max_deferred=3):
    return KTConfig(
        layer_idx=layer_idx,
        gpu_experts_mask=gpu_mask,
        cpuinfer_threads=8,
        threadpool_count=2,
        numa_nodes=[4, 5],
        weight_path="/nonexistent-dummy",
        chunked_prefill_size=64,
        max_deferred_experts_per_token=max_deferred,
        method=method,
        num_layers=num_layers,
    )


def _dummy_server_args(**fields):
    """Dummy-boundary ServerArgs: __post_init__ early-returns for
    model_path="dummy", so the dataclass fields (all defaulted) stay
    assignable and _handle_kt / validate_kimi_k3_kt can be driven directly."""
    server_args = ServerArgs(model_path="dummy")
    for name, value in fields.items():
        setattr(server_args, name, value)
    return server_args


class TestMaskAndRemapPartition(CustomTestCase):
    """The GPU/CPU expert partition and the dense-remap bijection are the
    correctness core of the hybrid dispatch: a broken mapping silently sends
    tokens to the wrong expert weights."""

    def tearDown(self):
        reset_context()

    def _build_method(self, gpu_mask):
        # Instantiate through the real class so the mapping derivation under
        # test is the production one, not a test-side replica.
        with patch.multiple(ktw, KTRANSFORMERS_AVAILABLE=True, create=True), \
                get_parallel().override(tp_rank=0, tp_size=1):
            return KTEPWrapperMethod(
                gpu_method=MagicMock(), kt_config=_make_kt_config(gpu_mask)
            )

    def test_partition_and_bijection(self):
        for num_experts, num_gpu in ((64, 16), (896, 620), (10, 10), (12, 0)):
            with self.subTest(num_experts=num_experts, num_gpu=num_gpu):
                mask = _random_gpu_mask(num_experts, num_gpu, seed=num_experts)
                method = self._build_method(mask)
                l2g = method.logical_to_gpu_index
                g2l = method.gpu_index_to_logical

                # (1) every expert is exactly one of GPU-resident (has a dense
                # slot) or CPU-resident (-1) — partition matches the mask.
                self.assertEqual(method.num_gpu_experts, num_gpu)
                self.assertTrue(torch.equal(l2g >= 0, mask))

                # (2) restricted to the GPU set, the remap is a bijection onto
                # [0, num_gpu): the slots are a permutation of arange and the
                # inverse table round-trips back to the logical ids.
                gpu_slots = l2g[mask].to(torch.int64)
                self.assertTrue(
                    torch.equal(gpu_slots.sort().values, torch.arange(num_gpu))
                )
                self.assertTrue(
                    torch.equal(g2l[gpu_slots].to(torch.int64), torch.where(mask)[0])
                )

    def test_mask_and_remap_expert_ids(self):
        # NOTE: mask_and_remap_expert_ids is @torch.compile(dynamic=True)
        # decorated; it is called normally here, so the first call pays a
        # one-off CPU inductor compile on CI (covered by est_time).
        gen = torch.Generator().manual_seed(3)
        for num_experts, num_gpu in ((64, 16), (896, 620), (12, 0)):
            with self.subTest(num_experts=num_experts, num_gpu=num_gpu):
                mask = _random_gpu_mask(num_experts, num_gpu, seed=num_gpu + 1)
                method = self._build_method(mask)
                topk_ids = torch.randint(
                    0, num_experts, (33, 8), generator=gen, dtype=torch.int32
                )
                topk_ids_before = topk_ids.clone()

                out = mask_and_remap_expert_ids(
                    topk_ids, method.gpu_experts_mask, method.logical_to_gpu_index
                )

                gpu_pos = mask[topk_ids_before.to(torch.int64)]
                # CPU-resident ids -> exactly -1 (and nothing else is -1).
                self.assertTrue((out[~gpu_pos] == -1).all())
                if num_gpu > 0:
                    dense = out[gpu_pos].to(torch.int64)
                    self.assertTrue(((dense >= 0) & (dense < num_gpu)).all())
                    # Dense slots round-trip to the original logical ids.
                    self.assertTrue(
                        torch.equal(
                            method.gpu_index_to_logical[dense],
                            topk_ids_before[gpu_pos],
                        )
                    )
                else:
                    self.assertTrue((out == -1).all())
                # Out-of-place contract: the routing input is not mutated.
                self.assertTrue(torch.equal(topk_ids, topk_ids_before))


class TestPlacementMaskGenerators(CustomTestCase):
    """Placement-budget math for the static GPU-expert masks: dense-prefix
    layers must be all-True (bypass the KT wrapper) and the per-layer True
    counts must spend exactly the configured budget."""

    def test_front_loading_fills_earlier_layers_first(self):
        masks = generate_front_loading_masks(
            num_layers=6,
            num_experts=64,
            num_gpu_experts=100,
            first_k_dense_replace=2,
            moe_layer_freq=1,
        )
        self.assertEqual(masks.shape, (6, 64))
        self.assertEqual(masks.dtype, torch.bool)
        # Dense prefix bypasses the wrapper entirely.
        self.assertTrue(masks[0].all())
        self.assertTrue(masks[1].all())
        # Earlier MoE layers saturate before later ones get anything.
        self.assertEqual(masks[2:].sum(dim=1).tolist(), [64, 36, 0, 0])
        self.assertEqual(int(masks[2:].sum()), 100)

    def test_uniform_split_with_remainder(self):
        masks = generate_uniform_masks(
            num_layers=6,
            num_experts=64,
            num_gpu_experts=90,
            first_k_dense_replace=2,
            moe_layer_freq=1,
        )
        self.assertTrue(masks[0].all())
        self.assertTrue(masks[1].all())
        # 90 over 4 MoE layers: 22 each, first 2 layers absorb the remainder.
        self.assertEqual(masks[2:].sum(dim=1).tolist(), [23, 23, 22, 22])
        self.assertEqual(int(masks[2:].sum()), 90)

    def test_uniform_respects_moe_layer_freq(self):
        masks = generate_uniform_masks(
            num_layers=6,
            num_experts=64,
            num_gpu_experts=10,
            first_k_dense_replace=2,
            moe_layer_freq=2,
        )
        # MoE layers are those with idx >= first_k AND idx % freq == 0: {2, 4}.
        # Every other layer counts as dense and is all-True.
        for dense_layer in (0, 1, 3, 5):
            self.assertTrue(masks[dense_layer].all())
        self.assertEqual(int(masks[2].sum()), 5)
        self.assertEqual(int(masks[4].sum()), 5)

    def test_random_is_deterministic_and_on_budget(self):
        kwargs = dict(
            num_layers=6,
            num_experts=64,
            num_gpu_experts=100,
            first_k_dense_replace=2,
            moe_layer_freq=1,
            seed=42,
        )
        first = generate_random_masks(**kwargs)
        second = generate_random_masks(**kwargs)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(first[0].all())
        self.assertTrue(first[1].all())
        self.assertEqual(int(first[2:].sum()), 100)


class TestSituCtorContract(CustomTestCase):
    """create_weights must resolve the Kimi-K3 SiTU activation into the
    dedicated situ_beta/situ_linear_beta ctor channel (zeroing the legacy
    swiglu pair) and fail closed on unsupported methods, missing wheel
    support, or missing config values."""

    def tearDown(self):
        # create_weights leases the process-global "kt_staging" buffer.
        reset_context()
        _RecordingKTMoEWrapper.last_kwargs = None

    def test_mock_tracks_module_ctor_constants(self):
        # The mock accepts exactly the params the module constants declare:
        # the full union constructs, anything outside it is rejected. If the
        # constants change, the mock follows automatically.
        _RecordingKTMoEWrapper(**{name: None for name in _KT_CTOR_PARAM_UNION})
        with self.assertRaises(TypeError):
            _RecordingKTMoEWrapper(layer_idx=0, not_in_the_contract=1)

    def _run_create_weights(
        self,
        *,
        activation="situ",
        gemm1_alpha=4.0,
        gemm1_clamp_limit=25.0,
        method="MXFP4",
        wheel_supports_situ=True,
        layer_idx=0,
    ):
        """Drive KTEPWrapperMethod.create_weights on CPU.

        Mocked, in addition to the KTMoEWrapper ctor itself: the module's
        get_stream binding (real one creates a CUDA side stream) and
        torch.cuda.Event (rank-0 sync event) — everything else, including the
        shared staging buffer lease, runs for real on CPU tensors.
        """
        mask = _random_gpu_mask(NUM_EXPERTS, 40, seed=11)
        kt_config = _make_kt_config(mask, method=method, layer_idx=layer_idx)
        layer = _FakeLayer(
            moe_runner_config=MoeRunnerConfig(
                activation=activation,
                gemm1_alpha=gemm1_alpha,
                gemm1_clamp_limit=gemm1_clamp_limit,
            ),
        )
        gpu_method = MagicMock()
        _RecordingKTMoEWrapper.last_kwargs = None
        with patch.multiple(
            ktw,
            KTRANSFORMERS_AVAILABLE=True,
            KTMoEWrapper=_RecordingKTMoEWrapper,
            KT_WHEEL_SUPPORTS_SITU=wheel_supports_situ,
            get_stream=MagicMock(),
            create=True,
        ), patch("torch.cuda.Event", MagicMock()), get_parallel().override(
            tp_rank=0, tp_size=1
        ):
            method_obj = KTEPWrapperMethod(gpu_method=gpu_method, kt_config=kt_config)
            method_obj.create_weights(
                layer=layer,
                num_experts=NUM_EXPERTS,
                hidden_size=HIDDEN,
                intermediate_size_per_partition=INTERMEDIATE,
                params_dtype=torch.float32,
            )
        return method_obj, layer, gpu_method

    def test_situ_ctor_kwargs(self):
        method_obj, layer, gpu_method = self._run_create_weights()
        ctor_kwargs = _RecordingKTMoEWrapper.last_kwargs
        self.assertIsNotNone(ctor_kwargs)
        # SiTU rides its dedicated ctor channel; the legacy swiglu pair is
        # zeroed so the kt-kernel side cannot double-apply an epilogue.
        self.assertEqual(ctor_kwargs["situ_beta"], 4.0)
        self.assertEqual(ctor_kwargs["situ_linear_beta"], 25.0)
        self.assertEqual(ctor_kwargs["swiglu_alpha"], 0.0)
        self.assertEqual(ctor_kwargs["swiglu_limit"], 0.0)
        self.assertEqual(ctor_kwargs["method"], "MXFP4")
        # Full intermediate size is re-derived from the layer's TP partition.
        self.assertEqual(
            ctor_kwargs["moe_intermediate_size"],
            layer.intermediate_size_per_partition * layer.moe_tp_size,
        )
        self.assertEqual(ctor_kwargs["hidden_size"], HIDDEN)
        # The KT wrapper sees the GLOBAL expert count ...
        self.assertEqual(ctor_kwargs["num_experts"], NUM_EXPERTS)
        self.assertEqual(ctor_kwargs["num_experts_per_tok"], layer.top_k)
        self.assertEqual(ctor_kwargs["max_deferred_experts_per_token"], 3)
        # ... while the wrapped GPU method sees only the dense GPU count.
        self.assertEqual(
            gpu_method.create_weights.call_args.kwargs["num_experts"],
            method_obj.num_gpu_experts,
        )

    def test_final_layer_forces_zero_deferred(self):
        # layer_idx == num_layers - 1 (6 in the fixture config) must always
        # submit with zero deferred experts.
        self._run_create_weights(layer_idx=5)
        self.assertEqual(
            _RecordingKTMoEWrapper.last_kwargs["max_deferred_experts_per_token"], 0
        )

    def test_situ_rejects_unsupported_method(self):
        with self.assertRaises(ValueError) as raised:
            self._run_create_weights(method="LLAMAFILE")
        message = str(raised.exception)
        self.assertIn("situ", message)
        # The error surfaces the method allow-list.
        self.assertIn("MXFP4", message)
        self.assertIn("AMXINT4", message)
        # The wrapper ctor never ran.
        self.assertIsNone(_RecordingKTMoEWrapper.last_kwargs)

    def test_situ_requires_wheel_support(self):
        with self.assertRaises(RuntimeError) as raised:
            self._run_create_weights(wheel_supports_situ=False)
        self.assertIn("situ_beta", str(raised.exception))
        self.assertIsNone(_RecordingKTMoEWrapper.last_kwargs)

    def test_situ_requires_gemm1_alpha(self):
        with self.assertRaises(ValueError) as raised:
            self._run_create_weights(gemm1_alpha=None)
        self.assertIn("gemm1_alpha", str(raised.exception))
        self.assertIsNone(_RecordingKTMoEWrapper.last_kwargs)

    def test_staged_submit_activation_gate(self):
        mask = _random_gpu_mask(NUM_EXPERTS, 16, seed=5)
        with patch.multiple(ktw, KTRANSFORMERS_AVAILABLE=True, create=True), \
                get_parallel().override(tp_rank=0, tp_size=1):
            method_obj = KTEPWrapperMethod(
                gpu_method=MagicMock(), kt_config=_make_kt_config(mask)
            )

        method_obj.moe_runner_config = MoeRunnerConfig(activation="gelu")
        with self.assertRaises(ValueError) as raised:
            method_obj._submit_with_staged_input(
                layer=MagicMock(),
                dispatch_output=MagicMock(),
                staged_hidden_states=torch.zeros(2, 4),
            )
        message = str(raised.exception)
        self.assertIn("gelu", message)
        self.assertIn("silu", message)
        self.assertIn("situ", message)

        # Allowed activations pass the gate; with wrapper=None (create_weights
        # never ran) the call is then a clean no-op, which isolates the gate.
        for allowed in ("silu", "situ"):
            method_obj.moe_runner_config = MoeRunnerConfig(activation=allowed)
            self.assertIsNone(
                method_obj._submit_with_staged_input(
                    layer=MagicMock(),
                    dispatch_output=MagicMock(),
                    staged_hidden_states=torch.zeros(2, 4),
                )
            )


class TestServerArgsKTRails(CustomTestCase):
    """ServerArgs._handle_kt safety rails, driven directly on dummy-boundary
    ServerArgs (all kt_* fields are defaulted dataclass fields)."""

    def test_ratio_without_weight_path_warns_only(self):
        server_args = _dummy_server_args(kt_gpu_experts_ratio=0.5)
        with self.assertLogs("sglang.srt.server_args", level="WARNING") as logs:
            server_args._handle_kt()
        self.assertTrue(any("no effect" in line for line in logs.output))

    def test_numa_threadpool_mismatch_rejected(self):
        server_args = _dummy_server_args(
            kt_weight_path="/nonexistent-dummy",
            kt_numa_nodes=[0, 1, 2],
            kt_threadpool_count=2,
        )
        with self.assertRaisesRegex(ValueError, "kt-numa-nodes"):
            server_args._handle_kt()

    def test_ratio_out_of_range_rejected(self):
        server_args = _dummy_server_args(
            kt_weight_path="/nonexistent-dummy",
            kt_gpu_experts_ratio=1.5,
        )
        with self.assertRaisesRegex(ValueError, "kt-gpu-experts-ratio"):
            server_args._handle_kt()

    def test_frequency_strategy_needs_activation_data(self):
        server_args = _dummy_server_args(
            kt_weight_path="/nonexistent-dummy",
            kt_expert_placement_strategy="frequency",
            init_expert_location="trivial",
        )
        with self.assertRaisesRegex(ValueError, "frequency"):
            server_args._handle_kt()

    def test_dynamic_expert_update_accepts_mxfp4(self):
        """Rail-removal pin (F2): dynamic+MXFP4 passes config validation;
        the E8M0-resident wheel requirement is enforced at wrapper init
        (feature-check — see test_kt_dynamic_update_mxfp4), not here. Red if
        someone reintroduces a config-time rejection."""
        server_args = _dummy_server_args(
            kt_weight_path="/nonexistent-dummy",
            kt_enable_dynamic_expert_update=True,
            kt_method="MXFP4",
        )
        server_args._handle_kt()  # must not raise

    def test_valid_combo_flips_shared_experts_fusion(self):
        server_args = _dummy_server_args(
            kt_weight_path="/nonexistent-dummy",
            kt_numa_nodes=[4, 5],
            kt_threadpool_count=2,
            kt_gpu_experts_ratio=0.7,
            kt_expert_placement_strategy="uniform",
            kt_method="MXFP4",
        )
        self.assertFalse(server_args.disable_shared_experts_fusion)
        with self.assertLogs("sglang.srt.server_args", level="WARNING"):
            server_args._handle_kt()
        self.assertTrue(server_args.disable_shared_experts_fusion)


class TestKimiK3KTGates(CustomTestCase):
    """validate_kimi_k3_kt gates KT to plain TP: EP a2a backends, DP
    attention, and TBO all break the rank-0 CPU-expert latent-space merge."""

    def test_kt_off_never_gates(self):
        # With KT disabled the gate must not fire even when every conflicting
        # feature is on (pins the early return).
        server_args = _dummy_server_args(
            moe_a2a_backend="deepep",
            enable_dp_attention=True,
            enable_two_batch_overlap=True,
        )
        validate_kimi_k3_kt(server_args)

    def test_deepep_rejected(self):
        server_args = _dummy_server_args(
            kt_weight_path="/nonexistent-dummy",
            moe_a2a_backend="deepep",
        )
        with self.assertRaisesRegex(ValueError, "moe-a2a-backend"):
            validate_kimi_k3_kt(server_args)

    def test_dp_attention_rejected(self):
        server_args = _dummy_server_args(
            kt_weight_path="/nonexistent-dummy",
            enable_dp_attention=True,
        )
        with self.assertRaisesRegex(ValueError, "dp-attention"):
            validate_kimi_k3_kt(server_args)

    def test_two_batch_overlap_rejected(self):
        server_args = _dummy_server_args(
            kt_weight_path="/nonexistent-dummy",
            enable_two_batch_overlap=True,
        )
        with self.assertRaisesRegex(ValueError, "two-batch-overlap"):
            validate_kimi_k3_kt(server_args)

    def test_plain_tp_accepted(self):
        server_args = _dummy_server_args(
            kt_weight_path="/nonexistent-dummy",
            moe_a2a_backend="none",
            enable_dp_attention=False,
            enable_two_batch_overlap=False,
        )
        validate_kimi_k3_kt(server_args)


if __name__ == "__main__":
    unittest.main()
