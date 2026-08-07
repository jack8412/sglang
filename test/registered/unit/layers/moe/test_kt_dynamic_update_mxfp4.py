"""F2 plumbing tests: MXFP4 dynamic expert update (srt/layers/moe/kt_ep_wrapper).

Covers the CPU-provable dynamic-update requirements: the wheel feature
assertion, no-op-when-disabled semantics, and the in-place mask/mapping
contract (CUDA-graph safety). The GPU-side promote-and-serve fire is
checklist item G3 in the RUNBOOK.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.moe import kt_ep_wrapper as ktw
from sglang.srt.layers.moe.kt_ep_wrapper import KTConfig, KTEPWrapperMethod
from sglang.srt.runtime_context import get_parallel, reset_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _kt_config(**overrides):
    fields = dict(
        layer_idx=0,
        gpu_experts_mask=torch.ones(8, dtype=torch.bool),
        cpuinfer_threads=2,
        threadpool_count=1,
        weight_path="/dummy",
        chunked_prefill_size=64,
        max_deferred_experts_per_token=0,
        method="MXFP4",
        num_layers=1,
    )
    fields.update(overrides)
    return KTConfig(**fields)


class _MockKTMoEWrapper:
    def __init__(self, **kwargs):
        pass


class TestWheelFeatureAssertion(CustomTestCase):
    """Bug-class guard (bookkeeping): dynamic+MXFP4 on a wheel without the
    E8M0-resident layout must fail at construction, not corrupt scales at the
    first promote. Red if the feature-check is dropped or becomes a version
    check that a differently-numbered wheel slips past."""

    def tearDown(self):
        reset_context()

    def _construct(self, *, dynamic: bool, wheel_has_e8m0: bool):
        with patch.multiple(
            ktw,
            KTRANSFORMERS_AVAILABLE=True,
            KTMoEWrapper=_MockKTMoEWrapper,
            create=True,
        ), patch(
            "sglang.srt.layers.moe.kt_mxfp4_export.kt_wheel_has_e8m0_resident_scales",
            return_value=wheel_has_e8m0,
        ), get_parallel().override(tp_rank=0, tp_size=1):
            return KTEPWrapperMethod(
                MagicMock(),
                _kt_config(kt_enable_dynamic_expert_update=dynamic),
            )

    def test_dynamic_mxfp4_requires_e8m0_wheel(self):
        with self.assertRaisesRegex(ValueError, "E8M0"):
            self._construct(dynamic=True, wheel_has_e8m0=False)

    def test_dynamic_mxfp4_accepts_e8m0_wheel(self):
        method = self._construct(dynamic=True, wheel_has_e8m0=True)
        self.assertIsNotNone(method)

    def test_static_mxfp4_ignores_wheel_layout(self):
        # Static placement must keep working on the fp32-scale wheel.
        method = self._construct(dynamic=False, wheel_has_e8m0=False)
        self.assertIsNotNone(method)


class TestPromotionNoOp(CustomTestCase):
    """Negative-branch completeness: the promotion hook must be a strict
    no-op (False, no state touched) when disabled or without a direct-copy
    source. Red if the gate inverts or a marlin-prepared slot starts being
    treated as a trtllm source."""

    def tearDown(self):
        reset_context()

    def _method(self, *, dynamic: bool):
        with patch.multiple(
            ktw,
            KTRANSFORMERS_AVAILABLE=True,
            KTMoEWrapper=_MockKTMoEWrapper,
            create=True,
        ), patch(
            "sglang.srt.layers.moe.kt_mxfp4_export.kt_wheel_has_e8m0_resident_scales",
            return_value=True,
        ), get_parallel().override(tp_rank=0, tp_size=1):
            return KTEPWrapperMethod(
                MagicMock(),
                _kt_config(kt_enable_dynamic_expert_update=dynamic),
            )

    def test_disabled_returns_false_without_touching_slot(self):
        method = self._method(dynamic=False)
        slot = MagicMock()
        result = method._maybe_promote_experts_from_slot(
            layer=MagicMock(), slot=slot, dispatch_output=MagicMock()
        )
        self.assertFalse(result)
        slot.prepared_params.__bool__.assert_not_called()

    def test_marlin_prepared_slot_returns_false(self):
        method = self._method(dynamic=True)
        slot = SimpleNamespace(prepared_params=None)
        with get_parallel().override(tp_rank=0, tp_size=1):
            result = method._maybe_promote_experts_from_slot(
                layer=MagicMock(), slot=slot, dispatch_output=MagicMock()
            )
        self.assertFalse(result)


class TestMaskUpdateInPlace(CustomTestCase):
    """Derived property (CUDA-graph contract): a promotion must update
    gpu_experts_mask_cuda / logical_to_gpu_index_cuda strictly in place —
    same storage, new values — because decode graphs captured their
    addresses. Red if any update path rebinds those tensors."""

    def tearDown(self):
        reset_context()

    def test_update_flow_preserves_cuda_tensor_storage(self):
        num_experts = 8
        with patch.multiple(
            ktw,
            KTRANSFORMERS_AVAILABLE=True,
            KTMoEWrapper=_MockKTMoEWrapper,
            create=True,
        ), patch(
            "sglang.srt.layers.moe.kt_mxfp4_export.kt_wheel_has_e8m0_resident_scales",
            return_value=True,
        ), get_parallel().override(tp_rank=0, tp_size=1):
            mask = torch.zeros(num_experts, dtype=torch.bool)
            mask[:4] = True
            method = KTEPWrapperMethod(
                MagicMock(),
                _kt_config(
                    gpu_experts_mask=mask.clone(),
                    kt_enable_dynamic_expert_update=True,
                ),
            )
            method.global_num_experts = num_experts
            method.gpu_experts_mask_cuda = mask.clone()
            method.logical_to_gpu_index_cuda = method.logical_to_gpu_index.clone()
            method.wrapper = MagicMock()
            mask_ptr = method.gpu_experts_mask_cuda.data_ptr()
            index_ptr = method.logical_to_gpu_index_cuda.data_ptr()

            selected = torch.tensor([4, 5, 6, 7], dtype=torch.int64)
            plan = ktw.Mxfp4DynUpdatePlan(
                param_names=ktw._MXFP4_TRTLLM_RESIDENT_PARAM_NAMES,
                disabled_reason=None,
            )
            source_ctx = SimpleNamespace(_is_mxfp4_quant=True)
            with patch.object(
                method, "_mxfp4_dyn_update_plan_for", return_value=plan
            ), patch.object(
                ktw, "select_top_experts_from_batch", return_value=selected
            ), patch.object(
                ktw, "copy_experts_weights_mxfp4"
            ) as copy_fn, patch.object(
                ktw.dist, "is_initialized", return_value=False
            ), patch.object(
                ktw, "update_kt_wrapper_masks"
            ) as mask_update:
                dispatch = SimpleNamespace(
                    topk_output=SimpleNamespace(
                        topk_ids=torch.zeros(2, 4, dtype=torch.int32)
                    )
                )
                method._update_gpu_experts_from_batch(
                    layer=MagicMock(),
                    ctx=source_ctx,
                    dispatch_output=dispatch,
                )

            copy_fn.assert_called_once()
            self.assertEqual(method.gpu_experts_mask_cuda.data_ptr(), mask_ptr)
            self.assertEqual(
                method.logical_to_gpu_index_cuda.data_ptr(), index_ptr
            )
            expected_mask = torch.zeros(num_experts, dtype=torch.bool)
            expected_mask[4:] = True
            self.assertTrue(
                torch.equal(method.gpu_experts_mask_cuda, expected_mask)
            )
            # Promoted experts occupy dense GPU slots; demoted ones are -1.
            self.assertTrue(
                (method.logical_to_gpu_index_cuda[selected] >= 0).all()
            )
            self.assertTrue(
                (method.logical_to_gpu_index_cuda[:4] == -1).all()
            )
            mask_update.assert_called_once()


if __name__ == "__main__":
    unittest.main()
