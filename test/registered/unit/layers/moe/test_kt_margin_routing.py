"""Margin-routing P1 tests (srt/layers/moe/kt_ep_wrapper).

Covers the CPU-provable margin-routing requirements from
SPEC-MARGIN-ROUTING.md: the override/insist split derivation, the
distinct-alternative assignment, the degenerate-layer rails, and the
config plumbing. The GPU-side gates (bit-exactness at margin unset,
gsm8k/acceptance at margin > 0) are node checklist items.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.moe import kt_ep_wrapper as ktw
from sglang.srt.layers.moe.kt_ep_wrapper import (
    KTConfig,
    KTEPWrapperMethod,
    _margin_override_topk_ids_impl,
)
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


class TestMarginOverrideDerivation(CustomTestCase):
    """Derived property: the override/insist split and the substitute
    assignment. Red if the lead comparison drifts off the best UNSELECTED
    resident, if resident picks stop being excluded as alternatives, or if
    multiple overrides in one token collapse onto one alternative (the
    cumsum-rank derivation)."""

    # Experts 0-3 GPU-resident, 4-7 CPU.
    MASK = torch.tensor([True, True, True, True, False, False, False, False])

    def test_insist_and_override_split(self):
        # Token: picks e4 (logit 5.0, big lead -> insist), e1 (resident,
        # untouched), e5 (logit 2.1, lead 0.6 over best unselected resident
        # e3 @ 1.5 -> override at margin 1.0).  e0 @ 1.0 and e2 @ 0.5 are
        # weaker alternatives; e1 is selected so it must NOT be the
        # alternative even though its logit 2.0 beats e3.
        topk_ids = torch.tensor([[4, 1, 5]])
        logits = torch.tensor([[1.0, 2.0, 0.5, 1.5, 5.0, 2.1, 0.0, 0.0]])
        new_ids, insist, override = _margin_override_topk_ids_impl(
            topk_ids, logits, self.MASK, 1.0
        )
        self.assertEqual(new_ids.tolist(), [[4, 1, 3]])
        self.assertEqual(insist.tolist(), [[True, False, False]])
        self.assertEqual(override.tolist(), [[False, False, True]])

    def test_multiple_overrides_get_distinct_alternatives(self):
        # Both CPU picks are within margin; they must land on the token's
        # 1st and 2nd best unselected residents (e2 @ 2.5, e3 @ 2.4), not
        # both on e2.
        topk_ids = torch.tensor([[6, 7, 0]])
        logits = torch.tensor([[3.0, 0.1, 2.5, 2.4, 0.0, 0.0, 2.6, 2.7]])
        new_ids, insist, override = _margin_override_topk_ids_impl(
            topk_ids, logits, self.MASK, 1.0
        )
        self.assertEqual(new_ids.tolist(), [[2, 3, 0]])
        self.assertEqual(override.tolist(), [[True, True, False]])
        self.assertEqual(insist.tolist(), [[False, False, False]])

    def test_counters_index_original_ids(self):
        # The masks align with the ORIGINAL topk_ids (true router
        # preference): an overridden slot reports the overridden expert, not
        # its substitute.  Red if a refactor moves the counting after the
        # rewrite.
        topk_ids = torch.tensor([[6, 7, 0]])
        logits = torch.tensor([[3.0, 0.1, 2.5, 2.4, 0.0, 0.0, 2.6, 2.7]])
        new_ids, _, override = _margin_override_topk_ids_impl(
            topk_ids, logits, self.MASK, 1.0
        )
        overridden_originals = topk_ids[override].tolist()
        self.assertEqual(overridden_originals, [6, 7])
        self.assertNotIn(6, new_ids.tolist()[0])
        self.assertNotIn(7, new_ids.tolist()[0])


class TestMarginDegenerateRails(CustomTestCase):
    """Completeness / negative-branch contracts. Red if the -inf rails are
    dropped: an all-CPU layer (layer_concentrated) or an
    alternatives-exhausted token would silently substitute a NON-resident
    expert picked out of the -inf pool, i.e. still-CPU work counted as an
    override."""

    def test_no_residents_no_override(self):
        mask = torch.zeros(8, dtype=torch.bool)
        topk_ids = torch.tensor([[4, 5, 6]])
        logits = torch.rand(1, 8)
        new_ids, insist, override = _margin_override_topk_ids_impl(
            topk_ids, logits, mask, 100.0
        )
        self.assertEqual(new_ids.tolist(), topk_ids.tolist())
        self.assertFalse(override.any().item())
        self.assertEqual(insist.tolist(), [[True, True, True]])

    def test_alternatives_exhausted_slots_fall_back_to_insist(self):
        # Only ONE unselected resident exists (e0; e1 is selected), but two
        # slots want an override: the second must stay an insist with its
        # original id, never a -inf-pool substitute.
        mask = torch.tensor([True, True, False, False, False, False, False, False])
        topk_ids = torch.tensor([[4, 5, 1]])
        logits = torch.tensor([[2.9, 0.5, 0.0, 0.0, 3.0, 2.95, 0.0, 0.0]])
        new_ids, insist, override = _margin_override_topk_ids_impl(
            topk_ids, logits, mask, 10.0
        )
        self.assertEqual(new_ids.tolist(), [[0, 5, 1]])
        self.assertEqual(override.tolist(), [[True, False, False]])
        self.assertEqual(insist.tolist(), [[False, True, False]])

    def test_padded_slots_ignored(self):
        mask = torch.tensor([True, True, True, True, False, False, False, False])
        topk_ids = torch.tensor([[4, -1, -1]])
        logits = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.1, 0.0, 0.0, 0.0]])
        new_ids, insist, override = _margin_override_topk_ids_impl(
            topk_ids, logits, mask, 10.0
        )
        self.assertEqual(new_ids.tolist()[0][1:], [-1, -1])
        self.assertFalse(insist[0, 1:].any().item())
        self.assertFalse(override[0, 1:].any().item())


class TestMarginConfigPlumbing(CustomTestCase):
    """Critical-path bookkeeping: the flag travels ServerArgs -> KTConfig ->
    wrapper state, and OFF means no margin state at all. Red if someone adds
    the field to one carrier and not the other, or flips the default away
    from bit-exact."""

    def tearDown(self):
        reset_context()

    def _construct(self, **kt_overrides):
        with patch.multiple(
            ktw,
            KTRANSFORMERS_AVAILABLE=True,
            KTMoEWrapper=_MockKTMoEWrapper,
            create=True,
        ), get_parallel().override(tp_rank=0, tp_size=1):
            return KTEPWrapperMethod(MagicMock(), _kt_config(**kt_overrides))

    def test_default_is_off(self):
        self.assertIsNone(KTConfig.__dataclass_fields__["routing_margin"].default)
        method = self._construct()
        self.assertIsNone(method._margin)
        self.assertIsNone(method._margin_insist_count)
        self.assertIsNone(method._margin_override_count)

    def test_margin_carried_to_wrapper(self):
        method = self._construct(routing_margin=0.25)
        self.assertEqual(method._margin, 0.25)


if __name__ == "__main__":
    unittest.main()
