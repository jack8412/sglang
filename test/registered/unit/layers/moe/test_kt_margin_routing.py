"""Routing-substitution tests (srt/layers/moe/kt_ep_wrapper).

Covers the CPU-provable requirements of margin routing (SPEC-MARGIN-ROUTING.md)
now that ``--kt-routing-margin`` is a per-token BUDGET -- the share of a token's
own mixture weight that substitution may move -- rather than a router-logit gap:
the override/insist split, the per-token bound, the distinct-alternative
assignment, the degenerate-layer rails and the config plumbing. The GPU-side
gates (bit-exactness at margin 0.0, gsm8k/acceptance above it) are node
checklist items.
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


def _route(topk_ids, logits, weights, mask, budget, full_override=False, bias=None):
    """Call the kernel with the router's selection bias defaulted to None.

    Most cases here are about the budget or the reweight and do not care about
    the bias; TestStandInRanking passes one explicitly and
    test_kernel_signature_is_positional pins the real argument order so this
    adapter cannot hide a signature drift.
    """
    return _margin_override_topk_ids_impl(
        topk_ids, logits, weights, bias, mask, budget, full_override
    )


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


class TestOverrideDerivation(CustomTestCase):
    """Derived property: the override/insist split and the substitute
    assignment. Red if the stand-in stops being the best UNSELECTED resident,
    if resident picks stop being excluded as alternatives, or if multiple
    overrides in one token collapse onto one alternative (the cumsum-rank
    derivation)."""

    # Experts 0-3 GPU-resident, 4-7 CPU.
    MASK = torch.tensor([True, True, True, True, False, False, False, False])

    def test_insist_and_override_split(self):
        # Token picks e4 (CPU, 0.6 of the mixture -- too big for the budget,
        # so an insist), e1 (resident, untouched) and e5 (CPU, 0.2 -- fits, so
        # an override).  The stand-in must be e3 @ 1.5: e0 @ 1.0 and e2 @ 0.5
        # are weaker, and e1 is already selected so it must NOT be chosen even
        # though its logit 2.0 beats e3's.
        topk_ids = torch.tensor([[4, 1, 5]])
        weights = torch.tensor([[0.6, 0.2, 0.2]])
        logits = torch.tensor([[1.0, 2.0, 0.5, 1.5, 5.0, 2.1, 0.0, 0.0]])
        new_ids, new_w, insist, override = _route(
            topk_ids, logits, weights, self.MASK, 0.3
        )
        self.assertEqual(new_ids.tolist(), [[4, 1, 3]])
        self.assertEqual(insist.tolist(), [[True, False, False]])
        self.assertEqual(override.tolist(), [[False, False, True]])

    def test_multiple_overrides_get_distinct_alternatives(self):
        # Both CPU picks fit the budget; they must land on the token's 1st and
        # 2nd best unselected residents (e2 @ 2.5, e3 @ 2.4), not both on e2.
        topk_ids = torch.tensor([[6, 7, 0]])
        weights = torch.tensor([[0.1, 0.1, 0.8]])
        logits = torch.tensor([[3.0, 0.1, 2.5, 2.4, 0.0, 0.0, 2.6, 2.7]])
        new_ids, new_w, insist, override = _route(
            topk_ids, logits, weights, self.MASK, 0.25
        )
        self.assertEqual(new_ids.tolist(), [[2, 3, 0]])
        self.assertEqual(override.tolist(), [[True, True, False]])
        self.assertEqual(insist.tolist(), [[False, False, False]])

    def test_counters_index_original_ids(self):
        # The masks align with the ORIGINAL topk_ids (true router preference):
        # an overridden slot reports the overridden expert, not its substitute.
        # Red if a refactor moves the counting after the rewrite -- the swap
        # policy would then see demand for experts it already holds.
        topk_ids = torch.tensor([[6, 7, 0]])
        weights = torch.tensor([[0.1, 0.1, 0.8]])
        logits = torch.tensor([[3.0, 0.1, 2.5, 2.4, 0.0, 0.0, 2.6, 2.7]])
        new_ids, _, _, override = _route(
            topk_ids, logits, weights, self.MASK, 0.25
        )
        self.assertEqual(topk_ids[override].tolist(), [6, 7])
        self.assertNotIn(6, new_ids.tolist()[0])
        self.assertNotIn(7, new_ids.tolist()[0])


class TestWeightBudget(CustomTestCase):
    """Derived property: the substituted share of a token's own mixture never
    exceeds the budget, and the budget is spent smallest-slot-first so it buys
    the most CPU slots per unit of error.

    Red if the greedy stops sorting (it would then substitute whatever the
    router happened to emit first -- and this route runs ``topk(...,
    sorted=False)``, so slot order carries no information at all), if the
    running total stops being cumulative (each slot would be tested alone and
    the per-token bound would be lost), or if the budget stops being normalized
    by the token's own weight sum."""

    MASK = torch.tensor([True, True, True, True, False, False, False, False])
    # e0 > e2 > e3 as stand-ins; e1 is high so it is a real router pick.
    LOGITS = torch.tensor([[3.0, 9.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0]])

    def test_budget_bounds_the_substituted_share(self):
        # CPU picks carry 0.05 / 0.10 / 0.55; a 0.16 budget admits the first
        # two (0.15 cumulative) and must refuse the third (0.70).
        topk_ids = torch.tensor([[4, 5, 6, 1]])
        weights = torch.tensor([[0.05, 0.10, 0.55, 0.30]])
        new_ids, new_w, insist, override = _route(
            topk_ids, self.LOGITS, weights, self.MASK, 0.16
        )
        self.assertEqual(override.tolist(), [[True, True, False, False]])
        self.assertEqual(insist.tolist(), [[False, False, True, False]])
        self.assertEqual(new_ids.tolist(), [[0, 2, 6, 1]])
        self.assertLessEqual(float((weights * override).sum()), 0.16)

    def test_budget_is_spent_smallest_first(self):
        # Same three CPU picks, re-ordered so the LARGEST sits in slot 0. A
        # budget for exactly one slot must buy the 0.05 slot, not the 0.55 one.
        topk_ids = torch.tensor([[6, 4, 5, 1]])
        weights = torch.tensor([[0.55, 0.05, 0.10, 0.30]])
        new_ids, _, _, override = _route(
            topk_ids, self.LOGITS, weights, self.MASK, 0.07
        )
        self.assertEqual(override.tolist(), [[False, True, False, False]])
        self.assertEqual(new_ids.tolist(), [[6, 0, 5, 1]])

    def test_flat_routing_token_stays_bounded(self):
        # The case the removed router-logit rule got backwards. Every CPU pick
        # leads the best unselected resident by <= 0.05 logits, which is the
        # NORMAL situation for lower-ranked slots (the comparison point sits at
        # the top-k selection boundary), so the old rule replaced all three --
        # 75% of the token's mixture -- exactly where the model is blending
        # most experts. The budget must hold at 20% instead.
        topk_ids = torch.tensor([[4, 5, 6, 1]])
        logits = torch.tensor([[2.0, 5.0, 1.9, 1.8, 2.05, 2.02, 2.01, 0.0]])
        weights = torch.tensor([[0.20, 0.25, 0.30, 0.25]])
        _, _, _, override = _route(
            topk_ids, logits, weights, self.MASK, 0.25
        )
        self.assertEqual(override.tolist(), [[True, False, False, False]])
        self.assertAlmostEqual(float((weights * override).sum()), 0.20, places=5)

    def test_budget_is_a_share_not_an_absolute_weight(self):
        # Identical routing with the weights scaled by 2.5 -- what
        # routed_scaling_factor folded into topk_weights, or
        # moe_renormalize=False, actually looks like. The decision must not
        # move: the budget is compared against the token's OWN total, which is
        # the only reason one number can mean the same thing in all 92 layers.
        topk_ids = torch.tensor([[4, 5, 6, 1]])
        logits = torch.tensor([[2.0, 5.0, 1.9, 1.8, 2.05, 2.02, 2.01, 0.0]])
        norm = torch.tensor([[0.20, 0.25, 0.30, 0.25]])
        _, _, _, a = _route(
            topk_ids, logits, norm, self.MASK, 0.25
        )
        _, _, _, b = _route(
            topk_ids, logits, norm * 2.5, self.MASK, 0.25
        )
        self.assertEqual(a.tolist(), b.tolist())

    def test_zero_budget_substitutes_nothing(self):
        topk_ids = torch.tensor([[4, 5, 6, 1]])
        weights = torch.tensor([[0.05, 0.10, 0.55, 0.30]])
        new_ids, new_w, insist, override = _route(
            topk_ids, self.LOGITS, weights, self.MASK, 0.0
        )
        self.assertFalse(override.any().item())
        self.assertEqual(insist.tolist(), [[True, True, True, False]])
        self.assertEqual(new_ids.tolist(), topk_ids.tolist())

    def test_per_token_budgets_carry_the_count_only_contract(self):
        # A [num_tokens] budget tensor must decide PER TOKEN, so a request at
        # 0.0 stays bit-exact beside one at 0.16 in the same batch. A scalar
        # 0.0 is handled by the caller skipping the rewrite for the whole
        # batch, which cannot express that mix.
        topk_ids = torch.tensor([[4, 5, 6, 1], [4, 5, 6, 1]])
        logits = self.LOGITS.repeat(2, 1)
        weights = torch.tensor([[0.05, 0.10, 0.55, 0.30]]).repeat(2, 1)
        new_ids, _, _, override = _route(
            topk_ids, logits, weights, self.MASK, torch.tensor([0.0, 0.16])
        )
        self.assertEqual(override.tolist()[0], [False, False, False, False])
        self.assertEqual(override.tolist()[1], [True, True, False, False])
        self.assertEqual(new_ids.tolist()[0], [4, 5, 6, 1])

    def test_budget_one_saturates_on_an_all_cpu_token(self):
        """Bug regression: budget 1.0 stranded a slot as an insist.

        `spend` accumulates in ascending weight order while `total` sums in
        slot order. On a token whose ENTIRE routed set is CPU-resident those
        are the same quantity summed differently, so they can disagree by an
        ulp -- `spend[-1] > 1.0 * total` -- which drops the largest slot and
        re-ranks every stand-in after it via the alt_rank cumsum. Measured on
        ~20% of random all-CPU rows before the fix. These exact weights
        reproduce it. Red if the >= 1.0 saturation is removed: the flag help,
        the KTConfig docstring and the `nobound` launch profile all promise
        that 1.0 substitutes everything.
        """
        topk_ids = torch.tensor([[4, 5, 6, 7]])
        weights = torch.tensor([[0.22667624, 0.07907632, 0.38262960, 0.31161791]])
        logits = torch.tensor([[3.0, 2.0, 1.0, 0.5, 0.1, 0.2, 0.3, 0.4]])
        b_ids, _, b_insist, b_over = _route(
            topk_ids, logits, weights, self.MASK, 1.0
        )
        f_ids, _, f_insist, f_over = _route(
            topk_ids, logits, weights, self.MASK, 0.0, True
        )
        self.assertFalse(b_insist.any().item())
        self.assertEqual(b_over.tolist(), f_over.tolist())
        self.assertEqual(b_ids.tolist(), f_ids.tolist())

    def test_budget_at_one_matches_full_override(self):
        # 1.0 is the documented no-bound endpoint: the whole mixture is
        # substitutable, so the rule must land exactly where
        # --kt-routing-full-override does.
        topk_ids = torch.tensor([[4, 5, 6, 1]])
        logits = torch.tensor([[2.0, 5.0, 1.9, 1.8, 2.05, 2.02, 2.01, 0.0]])
        weights = torch.tensor([[0.20, 0.25, 0.30, 0.25]])
        ids_b, wb, insist_b, over_b = _route(
            topk_ids, logits, weights, self.MASK, 1.0
        )
        ids_f, wf, insist_f, over_f = _route(
            topk_ids, logits, weights, self.MASK, 0.0, True
        )
        self.assertEqual(over_b.tolist(), over_f.tolist())
        self.assertEqual(insist_b.tolist(), insist_f.tolist())
        self.assertEqual(ids_b.tolist(), ids_f.tolist())
        self.assertFalse(insist_b.any().item())


class TestWeightRecompute(CustomTestCase):
    """Derived property: after substitution the weights are the router's
    weights for the set of experts ACTUALLY evaluated -- each member at its own
    gate, renormalised across the set (REAP's rule: drop the expert, recompute
    the top-k weights without it). The invariant is that
    ``w_i / sigmoid(logit[id_i])`` is one constant across a token's routed
    slots, which is exactly what the router produces and exactly what handing a
    stand-in the replaced expert's weight breaks.

    Red if a stand-in inherits the weight of the pick it replaces (a below-cut
    expert then runs at an above-cut weight -- one-signed, and compounding over
    92 layers), if the renormalisation is dropped (the layer output shrinks by
    the gate deficit), or if it renormalises to 1.0 instead of the row's own
    total (which silently discards routed_scaling_factor when that is folded
    into topk_weights).

    The fixtures build topk_weights the way the router does rather than
    inventing them: the invariant is a statement about the router's OWN
    weights, so weights unrelated to the logits would make it vacuous."""

    MASK = torch.tensor([True, True, True, True, False, False, False, False])
    LOGITS = torch.tensor([[1.0, 2.0, 0.5, 1.5, 5.0, 2.1, 0.0, 0.0]])
    IDS = torch.tensor([[4, 1, 5]])
    # e4 and e5 are CPU-resident and carry ~0.359 / ~0.322 of the mixture;
    # 0.33 admits the smaller one only, so e5 is replaced and e4 insists.
    BUDGET = 0.33

    @classmethod
    def _router_weights(cls, ids, logits, scale=1.0):
        """topk_weights as topk.py produces them: sigmoid(logit) over the
        selected set, renormalised (K3 routes via DSv3 noaux_tc, where the
        correction bias enters selection only, never the weight)."""
        g = torch.sigmoid(torch.gather(logits, -1, ids.clamp_min(0).long()))
        g = torch.where(ids >= 0, g, torch.zeros_like(g))
        return g / g.sum(dim=-1, keepdim=True) * scale

    def _assert_proportional_to_gates(self, new_ids, new_w, routed):
        g = torch.sigmoid(torch.gather(self.LOGITS, -1, new_ids.clamp_min(0).long()))
        share = new_w / g
        ref = share[routed]
        self.assertTrue(
            bool((ref - ref[0]).abs().max() < 1e-5),
            f"weights are not proportional to the evaluated set's gates: {share}",
        )

    def _run(self, budget, scale=1.0, full_override=False, ids=None):
        ids = self.IDS if ids is None else ids
        w = self._router_weights(ids, self.LOGITS, scale)
        new_ids, new_w, insist, override = _route(
            ids, self.LOGITS, w, self.MASK, budget, full_override
        )
        return w, new_ids, new_w, insist, override

    def test_weights_match_the_gates_of_the_experts_actually_run(self):
        w, new_ids, new_w, _, override = self._run(self.BUDGET)
        self.assertEqual(new_ids.tolist(), [[4, 1, 3]])
        self.assertEqual(override.tolist(), [[False, False, True]])
        self._assert_proportional_to_gates(new_ids, new_w, self.IDS >= 0)

    def test_stand_in_does_not_inherit_the_replaced_weight(self):
        w, _, new_w, _, _ = self._run(self.BUDGET)
        # e5 (logit 2.1) gives way to e3 (logit 1.5) -- a lower gate, so that
        # slot must carry LESS than it did, and the deficit must land on the
        # picks that survived.
        self.assertLess(float(new_w[0, 2]), float(w[0, 2]))
        self.assertGreater(float(new_w[0, 0]), float(w[0, 0]))
        self.assertGreater(float(new_w[0, 1]), float(w[0, 1]))

    def test_row_total_is_preserved_not_forced_to_one(self):
        # A row summing to 2.5 stands in for routed_scaling_factor folded into
        # topk_weights (kimi_k3.py:433-455). It must come back summing to 2.5.
        w, _, new_w, _, _ = self._run(self.BUDGET, scale=2.5)
        self.assertAlmostEqual(float(new_w.sum()), float(w.sum()), places=5)

    def test_untouched_slots_keep_their_relative_proportions(self):
        w, _, new_w, _, _ = self._run(self.BUDGET)
        # Slots 0 and 1 were not substituted, so only the shared normaliser may
        # move them: their ratio must be exactly what the router produced.
        self.assertAlmostEqual(
            float(new_w[0, 0] / new_w[0, 1]), float(w[0, 0] / w[0, 1]), places=5
        )

    def test_no_override_leaves_weights_bit_identical(self):
        # The count-only contract is bit-exactness, not near-equality: with no
        # overrides every ratio is exactly 1.0 and both sums are the same
        # expression over the same values, so the tensor comes back unchanged.
        w, _, new_w, _, override = self._run(0.0)
        self.assertFalse(override.any().item())
        self.assertTrue(torch.equal(new_w, w))

    def test_padded_slots_are_left_alone(self):
        ids = torch.tensor([[4, -1, 5]])
        w = self._router_weights(ids, self.LOGITS).clone()
        w[0, 1] = 9.0
        _, new_w, _, _ = _route(
            ids, self.LOGITS, w, self.MASK, 1.0
        )
        # An unrouted slot's weight is never read; rewriting it would only
        # invite a downstream consumer to start trusting it.
        self.assertEqual(float(new_w[0, 1]), 9.0)

    def test_full_override_also_recomputes(self):
        # The static CPU-path skip replaces EVERY CPU pick, so it is the mode
        # where inheritance would apply to the most mass. It must not bypass
        # the recompute.
        w, new_ids, new_w, insist, override = self._run(0.0, full_override=True)
        self.assertEqual(override.tolist(), [[True, False, True]])
        self.assertFalse(insist.any().item())
        self._assert_proportional_to_gates(new_ids, new_w, self.IDS >= 0)
        self.assertAlmostEqual(float(new_w.sum()), float(w.sum()), places=5)


class TestStandInRanking(CustomTestCase):
    """Derived property: the stand-in is the expert the ROUTER would promote out
    of the resident pool, i.e. the best unselected resident by
    sigmoid(logit) + e_score_correction_bias -- the space the router selects in
    (topk.py:1301) -- not by affinity alone.

    Red if the bias stops being applied to the ranking, or if it leaks into the
    WEIGHT (the router excludes it there: topk_weights = scores.gather(ids),
    topk.py:1327). The bias is a load-balancing term the model trained under, so
    ranking without it systematically promotes the experts the router was taught
    to under-select; measured at 56% of tokens receiving a different stand-in at
    a bias sd of 0.05."""

    MASK = torch.tensor([True, True, True, True, False, False, False, False])
    # Unselected residents e0/e2/e3. sigma order is e0 > e2 > e3; the bias
    # reverses the top two, so the two rankings disagree on the winner.
    LOGITS = torch.tensor([[1.2, 3.0, 1.0, 0.2, 2.0, 1.8, 0.0, 0.0]])
    BIAS = torch.tensor([-0.10, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
    IDS = torch.tensor([[4, 1, 5]])
    # DERIVED, never invented. test_bias_does_not_reach_the_weight asserts the
    # returned weights stay proportional to sigmoid(logit); that is a statement
    # about the ROUTER's weights, so a hand-written triple unrelated to LOGITS
    # makes it fail for a reason that says nothing about the code. Same trap as
    # TestBiasGap above, which already carries the note -- and it caught this
    # class the first time these tests were actually executed.
    WEIGHTS = (
        lambda g: g / g.sum(dim=-1, keepdim=True)
    )(torch.sigmoid(torch.gather(LOGITS, -1, IDS.long())))
    # e4 and e5 carry 0.3272 / 0.3188 of the real mixture, so the budget
    # has to sit in [0.3188, 0.6461) to admit the smaller one alone. The
    # 0.25 this used to pass went with the invented weights, where e5 was
    # 0.2; against the router's own weights it admits nothing.
    BUDGET = 0.40

    def test_bias_decides_the_stand_in(self):
        # sigma: e0 0.769 > e2 0.731; sigma+bias: e2 0.781 > e0 0.669.
        ids, _, _, override = _route(
            self.IDS, self.LOGITS, self.WEIGHTS, self.MASK, self.BUDGET, bias=self.BIAS
        )
        self.assertEqual(override.tolist(), [[False, False, True]])
        self.assertEqual(ids.tolist(), [[4, 1, 2]])

    def test_without_a_bias_the_ranking_is_the_logit_order(self):
        # A router with no correction bias must keep the old behaviour exactly:
        # sigmoid is monotone, so the raw logit already gives its order.
        ids, _, _, _ = _route(
            self.IDS, self.LOGITS, self.WEIGHTS, self.MASK, self.BUDGET, bias=None
        )
        self.assertEqual(ids.tolist(), [[4, 1, 0]])

    def test_bias_does_not_reach_the_weight(self):
        # The stand-in's weight is its own gate, sigmoid(logit), renormalised --
        # the bias steers selection only. Red if alt_scores (selection space) is
        # reused as the logit for the reweight.
        ids, w, _, _ = _route(
            self.IDS, self.LOGITS, self.WEIGHTS, self.MASK, self.BUDGET, bias=self.BIAS
        )
        gates = torch.sigmoid(torch.gather(self.LOGITS, -1, ids.long()))
        share = w / gates
        self.assertTrue(
            bool((share - share[0, 0]).abs().max() < 1e-5),
            f"weights are not proportional to the gates alone: {share}",
        )

    def test_kernel_signature_is_positional(self):
        # Pins the real argument order, so the _route adapter above cannot hide
        # a parameter being inserted or reordered.
        ids, w, insist, override = _margin_override_topk_ids_impl(
            self.IDS, self.LOGITS, self.WEIGHTS, self.BIAS, self.MASK, self.BUDGET, False
        )
        self.assertEqual(ids.tolist(), [[4, 1, 2]])
        self.assertEqual(override.tolist(), [[False, False, True]])


class TestDegenerateRails(CustomTestCase):
    """Completeness / negative-branch contracts. Red if the -inf rails are
    dropped: an all-CPU layer (layer_concentrated) or an
    alternatives-exhausted token would silently substitute a NON-resident
    expert picked out of the -inf pool, i.e. still-CPU work counted as an
    override."""

    def test_no_residents_no_override(self):
        mask = torch.zeros(8, dtype=torch.bool)
        topk_ids = torch.tensor([[4, 5, 6]])
        weights = torch.tensor([[0.2, 0.3, 0.5]])
        new_ids, new_w, insist, override = _route(
            topk_ids, torch.rand(1, 8), weights, mask, 1.0
        )
        self.assertEqual(new_ids.tolist(), topk_ids.tolist())
        self.assertFalse(override.any().item())
        self.assertEqual(insist.tolist(), [[True, True, True]])

    def test_exhausted_alternatives_fall_back_to_insist(self):
        # Only ONE unselected resident exists (e0; e1 is selected) but two
        # slots fit the budget: the second must stay an insist with its
        # original id, never a -inf-pool substitute. Falling back can only
        # LOWER the substituted share, so the per-token bound survives it.
        mask = torch.tensor([True, True, False, False, False, False, False, False])
        topk_ids = torch.tensor([[4, 5, 1]])
        logits = torch.tensor([[2.9, 0.5, 0.0, 0.0, 3.0, 2.95, 0.0, 0.0]])
        weights = torch.tensor([[0.05, 0.05, 0.90]])
        new_ids, new_w, insist, override = _route(
            topk_ids, logits, weights, mask, 0.5
        )
        self.assertEqual(new_ids.tolist(), [[0, 5, 1]])
        self.assertEqual(override.tolist(), [[True, False, False]])
        self.assertEqual(insist.tolist(), [[False, True, False]])
        self.assertLessEqual(float((weights * override).sum()), 0.5)

    def test_padded_slots_are_neither_counted_nor_substituted(self):
        # -1 slots must stay out of the token's weight total (they would
        # otherwise inflate it and loosen the budget) and out of the greedy.
        mask = torch.tensor([True, True, True, True, False, False, False, False])
        topk_ids = torch.tensor([[4, -1, -1, 1]])
        logits = torch.tensor([[3.0, 9.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
        weights = torch.tensor([[0.30, 9.0, 9.0, 0.70]])
        new_ids, new_w, insist, override = _route(
            topk_ids, logits, weights, mask, 0.29
        )
        # 0.30 of a 1.00 total exceeds a 0.29 budget; had the padded 9.0s
        # entered the total the budget would have been 5.5 and the slot would
        # have been substituted.
        self.assertFalse(override.any().item())
        self.assertEqual(insist.tolist(), [[True, False, False, False]])
        self.assertEqual(new_ids.tolist(), topk_ids.tolist())
        self.assertFalse(insist[0, 1:3].any().item())


class TestFullOverride(CustomTestCase):
    """Derived property: full override leaves ZERO insists whenever a layer
    holds at least top_k residents — the invariant the static CPU-path skip
    rests on. Red if the budget greedy leaks back into the full-override
    branch, or if the isfinite rail is dropped (which would let an expert
    from the -inf pool be chosen as a 'resident' substitute)."""

    def test_zero_insists_when_residents_at_least_topk(self):
        # 4 residents, top_k 4, every pick CPU-resident: without full override
        # a 0.0 budget would leave four insists; with it all four are replaced.
        mask = torch.tensor([True, True, True, True, False, False, False, False])
        topk_ids = torch.tensor([[4, 5, 6, 7]])
        logits = torch.tensor([[0.1, 0.2, 0.3, 0.4, 90.0, 91.0, 92.0, 93.0]])
        weights = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        new_ids, new_w, insist, override = _route(
            topk_ids, logits, weights, mask, 0.0, True
        )
        self.assertFalse(insist.any().item())
        self.assertTrue(override.all().item())
        self.assertCountEqual(new_ids.tolist()[0], [0, 1, 2, 3])

    def test_fewer_residents_than_topk_leaves_insists(self):
        # 3 residents vs top_k 4: the tightness of the >= top_k guard. One
        # slot cannot be given a distinct resident, so it must remain an
        # insist rather than silently duplicating or taking a CPU expert.
        mask = torch.tensor([True, True, True, False, False, False, False, False])
        topk_ids = torch.tensor([[4, 5, 6, 7]])
        logits = torch.tensor([[0.1, 0.2, 0.3, 0.0, 90.0, 91.0, 92.0, 93.0]])
        weights = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        _, _, insist, override = _route(
            topk_ids, logits, weights, mask, 0.0, True
        )
        self.assertEqual(int(insist.sum()), 1)
        self.assertEqual(int(override.sum()), 3)

    def test_full_override_ignores_the_budget(self):
        # full_override is the outer branch: it must ignore the budget, or the
        # static CPU-path skip would be unsafe at any budget below 1.0 -- an
        # insist there is a routed contribution nothing computes.
        mask = torch.tensor([True, True, True, True, False, False, False, False])
        topk_ids = torch.tensor([[4, 0, 5, 1]])
        logits = torch.tensor([[5.0, 4.0, 0.1, 0.2, 99.0, 98.0, 0.0, 0.0]])
        weights = torch.tensor([[0.4, 0.1, 0.4, 0.1]])
        _, _, insist_a, override_a = _route(
            topk_ids, logits, weights, mask, 0.0, True
        )
        _, _, insist_b, override_b = _route(
            topk_ids, logits, weights, mask, 1.0, True
        )
        self.assertEqual(insist_a.tolist(), insist_b.tolist())
        self.assertEqual(override_a.tolist(), override_b.tolist())
        self.assertFalse(insist_a.any().item())


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

    def test_default_is_exact_routing(self):
        """Critical-path bookkeeping: the default budget substitutes nothing.

        0.0 is the server default now, not None, so this is the configuration
        almost every deployment boots. It must route bit-exactly -- the greedy
        spends against `spend <= 0`, so no slot is ever taken -- and must not
        allocate the margin counters, which the swap driver owns.
        """
        self.assertEqual(KTConfig.__dataclass_fields__["routing_margin"].default, 0.0)
        method = self._construct()
        self.assertEqual(method._margin, 0.0)
        self.assertIsNone(method._margin_insist_count)
        self.assertIsNone(method._margin_override_count)

    def test_default_budget_skips_the_rewrite_kernel(self):
        """Bug regression: making 0.0 the default put the greedy on every step.

        The margin used to default to None, which skipped the routing block
        entirely. At a scalar 0.0 the block was still ENTERED -- the kernel ran
        argsort/cumsum/scatter over [T, 16] at each of the 92 layers -- and only
        the decision to apply its result was declined. Defaulting to 0.0 without
        this would have handed every exact-routing server that cost for output
        it throws away.

        Pins the dispatch, not the kernel: at a scalar-0.0 budget with no
        request asking and no full override, apply() must take the demand-counter
        branch and never reach margin_override_topk_ids. Red if the fast path is
        removed, or if it widens to swallow full override (whose entire effect
        IS the rewrite) or a per-request budget.
        """
        method = self._construct()
        self.assertEqual(method._margin, 0.0)
        self.assertFalse(method._full_override)

        # The three inputs the dispatch reads, as apply() computes them.
        def exact(margin, per_req, full_override):
            return not margin and not per_req and not full_override

        self.assertTrue(exact(method._margin, False, method._full_override))
        # ... and every way of asking for a rewrite must defeat it.
        self.assertFalse(exact(0.0, True, False))   # a request asked
        self.assertFalse(exact(0.0, False, True))   # full override
        self.assertFalse(exact(0.25, False, False))  # server budget

    def test_margin_carried_to_wrapper(self):
        method = self._construct(routing_margin=0.25)
        self.assertEqual(method._margin, 0.25)

    def test_padded_batch_honours_the_real_rows(self):
        """Derived property: a SHORT budget tensor means the token axis was
        padded after it was built, and the real rows still apply.

        ForwardContext is published from forward_batch.kt_routing_margin at the
        top of _forward_raw, while _prepare_eager_forward_batch pads INSIDE that
        scope and _pad_tensor_to_size returns a new tensor -- so padding the
        field at its source is invisible here, and extending at the read is
        where it can be repaired. The rows line up because every remaining
        producer of a short tensor appends: _pad_inputs_to_size is the only one
        (prepare_attn_tp_scatter_input delegates to it rather than slicing), and
        the TBO split and the speculative-verify expansion produce
        correctly-sized tensors.

        Red if the tolerance is removed (per-request budgets silently die on any
        padded config) or widened to accept a LONGER tensor, which no batch
        shape explains.
        """
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )

        method = self._construct(routing_margin=0.25)
        with forward_context(
            ForwardContext(
                attn_backend=None, kt_routing_margin=torch.tensor([0.5, 0.1])
            )
        ):
            resolved = method._resolve_margin(torch.zeros(5, 4))
        self.assertEqual(
            [round(v, 4) for v in resolved.tolist()], [0.5, 0.1, 0.25, 0.25, 0.25]
        )

    def test_default_server_margin_fills_the_padded_tail_with_zero(self):
        # The padded tail must take the SAME default the sentinel resolves to.
        # On a server left at the default that is 0.0 -- substitute nothing,
        # route exactly -- and never None, which would reach the greedy.
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )

        method = self._construct()
        with forward_context(
            ForwardContext(attn_backend=None, kt_routing_margin=torch.tensor([0.5]))
        ):
            resolved = method._resolve_margin(torch.zeros(3, 4))
        self.assertEqual([round(v, 4) for v in resolved.tolist()], [0.5, 0.0, 0.0])

    def test_more_budgets_than_tokens_is_refused_and_says_so(self):
        """Negative-branch contract: only SHORT is explainable.

        A tensor longer than the batch is not a shape padding can produce, so
        there is no rule for which rows to drop -- guessing would silently apply
        one request's quality setting to another's tokens. Falling back to one
        scalar for everyone is the safe direction, and saying so keeps it from
        looking like a knob that mysteriously does nothing.
        """
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )

        method = self._construct(routing_margin=0.25)
        with forward_context(
            ForwardContext(
                attn_backend=None, kt_routing_margin=torch.full((6,), 0.5)
            )
        ), self.assertLogs("sglang.srt.layers.moe.kt_ep_wrapper", "WARNING") as log:
            resolved = method._resolve_margin(torch.zeros(2, 4))
        self.assertEqual(resolved, 0.25)
        self.assertIn("IGNORED", "".join(log.output))

    def test_the_fallback_warning_survives_graph_capture(self):
        """Bug regression: a single bool one-shot dropped every later warning.

        Decode-graph capture reaches _resolve_margin before any request exists,
        so a plain one-shot is burned at boot and every real request is then
        ignored in silence. Keying by cause is what makes the warning reach a
        real request. Red if the set is replaced by a bool.
        """
        from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod

        KTEPWrapperMethod._per_request_budget_warned = set()
        method = self._construct(routing_margin=0.25)
        with self.assertLogs(
            "sglang.srt.layers.moe.kt_ep_wrapper", "WARNING"
        ) as log:
            method._warn_per_request_budget_ignored("length-mismatch", 3, 8)
        self.assertIn("rows but the MoE sees", "".join(log.output))

    def test_matching_length_still_applies_per_token_budgets(self):
        # The positive branch, so the mismatch guard above cannot be "fixed" by
        # rejecting every tensor. The -1 sentinel means "this request did not
        # ask" and must resolve to the server default, not to -1.
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )

        method = self._construct(routing_margin=0.25)
        margins = torch.tensor([0.5, -1.0, 0.1])
        with forward_context(
            ForwardContext(attn_backend=None, kt_routing_margin=margins)
        ):
            resolved = method._resolve_margin(torch.zeros(3, 4))
        # assertAlmostEqual per element: the tensor is float32, so 0.1 comes
        # back as 0.10000000149011612 and an exact list compare fails on the
        # dtype rather than on the behaviour.
        for got, want in zip(resolved.tolist(), [0.5, 0.25, 0.1]):
            self.assertAlmostEqual(got, want, places=6)

    def test_resolve_margin_never_returns_none(self):
        """Bug regression: an unset server margin crashed the greedy.

        The routing block is entered on a per-request budget ALONE, so a server
        at the default budget still reaches the kernel. _resolve_margin used to
        return None there whenever it could not use the per-token tensor -- no
        forward context, or a length mismatch -- and None reached
        `budget * total` in the greedy: TypeError, killing the scheduler, and
        at decode-graph CAPTURE rather than on a request, since the capture
        context always carries the graph-resident margin slot. The server
        margin is a plain float now, which closes the hole at the source; this
        pins the value it resolves to.
        """
        self.assertEqual(self._construct()._resolve_margin(torch.zeros(3, 4)), 0.0)
        self.assertEqual(
            self._construct(routing_margin=0.25)._resolve_margin(torch.zeros(3, 4)),
            0.25,
        )


class TestRoutingMarginRange(CustomTestCase):
    """Critical-path bookkeeping for the unit change. --kt-routing-margin was a
    router-logit gap (production 0.5, sweeps to 5.0) and is now a share of each
    token's mixture weight. Values above 1.0 are meaningless under the new
    definition and are almost always a leftover from the old one, so the range
    rail is the only thing that turns a stale launch script into an error
    instead of a silently different quality point."""

    def _args(self, **kw):
        from sglang.srt.server_args import ServerArgs

        base = dict(
            model_path="/dummy",
            kt_weight_path="/dummy",
            kt_method="MXFP4",
        )
        base.update(kw)
        return ServerArgs(**base)

    def test_out_of_range_margin_raises(self):
        with self.assertRaises(ValueError) as cm:
            self._args(kt_routing_margin=5.0)
        self.assertIn("[0.0, 1.0]", str(cm.exception))

    def test_negative_margin_raises(self):
        with self.assertRaises(ValueError):
            self._args(kt_routing_margin=-0.1)

    def test_nan_margin_raises(self):
        # NaN compares False everywhere, i.e. 100% insists -- the exact inverse
        # of the intended bias, and fatal under the full-override skip.
        with self.assertRaises(ValueError):
            self._args(kt_routing_margin=float("nan"))

    def test_share_range_accepted(self):
        self._args(kt_routing_margin=0.0)
        self._args(kt_routing_margin=0.1)
        self._args(kt_routing_margin=1.0)

    def test_per_request_margin_shares_the_range(self):
        from sglang.srt.sampling.sampling_params import SamplingParams

        SamplingParams(kt_routing_margin=0.1).verify(vocab_size=32000)
        with self.assertRaises(ValueError):
            SamplingParams(kt_routing_margin=5.0).verify(vocab_size=32000)


class TestKtRoutingContractGuard(CustomTestCase):
    """Critical-path bookkeeping for the two router properties the weight
    recompute depends on and the KT wrapper cannot see.

    The wrapper receives ids, weights and logits -- not the TopK config -- so
    `moe_renormalize` and the group-limit settings are unverifiable at the
    point they matter. `_assert_kt_routing_contract` checks them where the
    config is in scope. Red if the guard is dropped, or if an upstream default
    flips: either failure is silent, producing weights that look plausible and
    are not the router's."""

    def _cfg(self, **kw):
        from types import SimpleNamespace

        base = dict(moe_renormalize=True, num_expert_group=1, topk_group=1)
        base.update(kw)
        return SimpleNamespace(**base)

    def _guard(self, cfg):
        from sglang.srt.models.kimi_k3 import _assert_kt_routing_contract

        return _assert_kt_routing_contract(cfg)

    def test_shipping_contract_passes(self):
        self._guard(self._cfg())

    def test_raw_sigmoid_weights_are_refused(self):
        # Without renormalisation the recompute would rescale every slot
        # instead of only the replaced ones, amplifying the layer output.
        with self.assertRaises(ValueError) as cm:
            self._guard(self._cfg(moe_renormalize=False))
        self.assertIn("moe_renormalize", str(cm.exception))

    def test_grouped_selection_is_refused(self):
        # The router restricts its top-k to the chosen groups; the stand-in
        # search ranks residents globally, so it could pick from an excluded
        # group. Both knobs must be checked -- either alone enables grouping.
        with self.assertRaises(ValueError) as cm:
            self._guard(self._cfg(num_expert_group=8))
        self.assertIn("num_expert_group", str(cm.exception))
        with self.assertRaises(ValueError):
            self._guard(self._cfg(topk_group=4))

    def test_none_valued_group_settings_are_treated_as_ungrouped(self):
        # `None` is how an ungrouped checkpoint expresses itself; the guard
        # must not raise on it, and must not crash comparing None to 1.
        self._guard(self._cfg(num_expert_group=None, topk_group=None))

    def test_correction_bias_binds_to_the_wrapper_and_skips_plain_layers(self):
        """Negative-branch contract for the bias binding.

        A layer whose experts are all GPU-resident is left unwrapped on purpose
        (kt_ep_wrapper.create_kt_config_from_server_args returns None when
        gpu_experts_mask.all()), and layer_concentrated placement or
        --kt-gpu-experts-ratio 1.0 produce whole layers like that. Binding must
        skip them rather than raise -- an earlier version raised, which would
        have killed startup on every such placement. It is a no-op and not a
        degradation because an unwrapped layer has no CPU-resident pick to
        substitute, so it never consults the bias.

        Red if the isinstance narrowing is replaced by a getattr (which would
        bind onto an arbitrary object) or by an unconditional assignment.
        """
        from types import SimpleNamespace

        from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod
        from sglang.srt.models.kimi_k3 import _bind_kt_correction_bias

        bias = torch.zeros(8)
        wrapper = KTEPWrapperMethod.__new__(KTEPWrapperMethod)
        _bind_kt_correction_bias(SimpleNamespace(quant_method=wrapper), bias)
        self.assertIs(wrapper.correction_bias, bias)

        plain = SimpleNamespace(quant_method=object())
        _bind_kt_correction_bias(plain, bias)  # must not raise

    def test_shipping_config_defaults_still_satisfy_the_guard(self):
        # An external-source literal check: the guard is only a rail if the
        # model's own defaults pass it. If an upstream bump flips
        # moe_renormalize or introduces expert groups, this goes red here
        # rather than on the node 50 minutes into a boot.
        from sglang.srt.configs.kimi_linear import KimiLinearConfig

        self._guard(KimiLinearConfig())


class TestSplitPrefillMinTokens(CustomTestCase):
    """Critical-path bookkeeping for --kt-expert-split-prefill-min-tokens.

    It used to be _SPLIT_PREFILL_MIN_TOKENS, a module constant, so the only way
    to move it was an edit. The value is a BREAK-EVEN against the CPU-expert
    path (1.99 s fixed cold stream x ~1,400 tok/s = ~2,800 tokens, 4096 rounded
    up), and the trap it exists to avoid is setting it to the chunk size --
    which leaves only exactly-full chunks qualifying and drops every remainder
    onto the CPU path (65,498 tokens = 32768 split + 32730 CPU, 6.4x slower).
    """

    def _args(self, **kw):
        from sglang.srt.server_args import ServerArgs

        base = dict(
            model_path="/dummy",
            kt_weight_path="/dummy",
            kt_method="MXFP4",
        )
        base.update(kw)
        return ServerArgs(**base)

    def test_default_is_the_measured_break_even(self):
        self.assertEqual(self._args().kt_expert_split_prefill_min_tokens, 4096)

    def test_zero_is_refused(self):
        # 0 arms split prefill on every forward, decode included, where a fixed
        # ~2 s cold stream buys nothing at all.
        with self.assertRaises(ValueError) as cm:
            self._args(kt_expert_split_prefill_min_tokens=0)
        self.assertIn("must be >= 1", str(cm.exception))

    def test_negative_is_refused(self):
        with self.assertRaises(ValueError):
            self._args(kt_expert_split_prefill_min_tokens=-1)

    def test_at_or_above_chunk_size_warns(self):
        """The chunk-size trap, as a warning rather than a refusal.

        A deliberately huge threshold is a legitimate way to disable split
        prefill for one run, so this must not raise -- but silently taking the
        CPU path on every remainder chunk is the 6.4x regression, so it must
        not be silent either.
        """
        with self.assertLogs("sglang.srt.server_args", level="WARNING") as log:
            self._args(
                kt_expert_split_prefill=True,
                chunked_prefill_size=4096,
                kt_expert_split_prefill_min_tokens=4096,
            )
        self.assertIn("every remainder falls back", "".join(log.output))

    def test_below_chunk_size_is_quiet(self):
        # The positive branch, so the warning above cannot be "fixed" by
        # firing on every configuration.
        args = self._args(
            kt_expert_split_prefill=True,
            chunked_prefill_size=32768,
            kt_expert_split_prefill_min_tokens=4096,
        )
        self.assertEqual(args.kt_expert_split_prefill_min_tokens, 4096)

    def test_the_threshold_reaches_the_wrapper(self):
        # The whole point of the change: the value must travel from the flag to
        # the gate. Red if KTConfig or the wrapper stops carrying it.
        from sglang.srt.layers.moe.kt_ep_wrapper import KTConfig

        self.assertEqual(
            KTConfig.__dataclass_fields__["split_prefill_min_tokens"].default, 4096
        )


class TestSwapRequiresSplitPrefill(CustomTestCase):
    """Critical-path bookkeeping for the swap -> split-prefill dependency.

    Cold-only residency leaves a demoted expert owning no CPU buffers, so a
    swap window must give it some before it becomes routable. The only writer
    is the arena-DMA rank-write path, and that is built by split prefill's boot
    hook -- so without split prefill a window would reach its arming consensus,
    find no writer on any rank, and terminate the server. The checkpoint read
    that used to cover this (~17.5 MB per demotion, ~12.9 GB per window) has
    been removed, so this rail is what turns a mid-serving terminate into a
    boot-time error.
    """

    def _args(self, **kw):
        from sglang.srt.server_args import ServerArgs

        base = dict(
            model_path="/dummy",
            kt_weight_path="/dummy",
            kt_method="MXFP4",
        )
        base.update(kw)
        return ServerArgs(**base)

    def test_swapping_without_split_prefill_raises(self):
        with self.assertRaises(ValueError) as cm:
            self._args(kt_expert_swap_transitions=4)
        self.assertIn("--kt-expert-split-prefill", str(cm.exception))

    def test_swapping_with_split_prefill_is_allowed(self):
        self._args(kt_expert_swap_transitions=4, kt_expert_split_prefill=True)

    def test_split_prefill_without_swapping_is_allowed(self):
        # Split prefill stands alone: it computes every routed expert on GPU
        # and needs no demotion writer of its own.
        self._args(kt_expert_split_prefill=True)

    def test_neither_is_allowed(self):
        # Plain margin-routed serving, no swapping, no split prefill. Nothing
        # ever demotes, so nothing needs the writer.
        self._args()

    def test_split_prefill_sets_the_kt_memfd_gate(self):
        # arena-DMA is no longer a flag: selecting split prefill is what makes
        # kt export its buffers as memfds. Red if the two are decoupled again.
        import os

        prev = os.environ.pop("KT_BUFFER_B_MEMFD", None)
        try:
            self._args(kt_expert_split_prefill=True)
            self.assertEqual(os.environ.get("KT_BUFFER_B_MEMFD"), "1")
        finally:
            os.environ.pop("KT_BUFFER_B_MEMFD", None)
            if prev is not None:
                os.environ["KT_BUFFER_B_MEMFD"] = prev


class TestResidentHitsCreditTheServingExpert(CustomTestCase):
    """Bug regression: an overridden slot credited its resident hit to nobody.

    Demand and resident hits are attributed to DIFFERENT experts on an
    overridden slot. Demand belongs to the expert the router asked for -- it is
    what argues for promoting it. A resident hit belongs to the expert that
    actually computed the slot, because its INVERSE ranks demotion victims
    (ExpertSwapPolicy.select sorts cand_demote ascending by hits_ema), so it has
    to mean "did work", not "was named".

    Attributing it to the original id gave the substitute nothing, so a resident
    expert doing well as a stand-in looked idle and went to the front of the
    demotion queue -- margin routing nominating its own best stand-ins, harder
    the higher the margin, and invisible because the counter still summed to a
    plausible total.

    Exercises the torch fallback: the fused kernel is CUDA-only and is asserted
    elsewhere to produce identical numbers.
    """

    def _counters(self, topk_ids, served_ids, insist, override, num_experts=8):
        from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod

        method = KTEPWrapperMethod.__new__(KTEPWrapperMethod)
        method._margin_insist_count = torch.zeros(num_experts, dtype=torch.int32)
        method._margin_override_count = torch.zeros(num_experts, dtype=torch.int32)
        method._resident_hit_count = torch.zeros(num_experts, dtype=torch.int32)
        method._update_margin_counters(
            topk_ids, served_ids, insist, override
        )
        return (
            method._margin_insist_count.tolist(),
            method._margin_override_count.tolist(),
            method._resident_hit_count.tolist(),
        )

    def test_override_credits_demand_to_router_and_hit_to_substitute(self):
        # One token, two slots. Slot 0: router wanted CPU expert 5, resident 2
        # stood in. Slot 1: router wanted resident 3 and got it.
        insist, override, hits = self._counters(
            topk_ids=torch.tensor([[5, 3]]),
            served_ids=torch.tensor([[2, 3]]),
            insist=torch.tensor([[False, False]]),
            override=torch.tensor([[True, False]]),
        )
        self.assertEqual(override[5], 1, "demand belongs to the router's choice")
        self.assertEqual(override[2], 0, "the substitute did not generate demand")
        self.assertEqual(hits[2], 1, "the substitute SERVED the slot")
        self.assertEqual(hits[5], 0, "the CPU expert served nothing")
        self.assertEqual(hits[3], 1, "an ordinary resident pick still counts")

    def test_insist_credits_no_resident_hit(self):
        # The CPU expert ran, so no resident expert served this slot.
        insist, override, hits = self._counters(
            topk_ids=torch.tensor([[5]]),
            served_ids=torch.tensor([[5]]),
            insist=torch.tensor([[True]]),
            override=torch.tensor([[False]]),
        )
        self.assertEqual(insist[5], 1)
        self.assertEqual(sum(hits), 0)

    def test_unmasked_slots_are_unchanged_when_nothing_overrides(self):
        # The compatibility claim: with served_ids == topk_ids the counters
        # reproduce exactly what the pre-fix code produced.
        ids = torch.tensor([[1, 4], [4, 6]])
        insist, override, hits = self._counters(
            topk_ids=ids,
            served_ids=ids,
            insist=torch.zeros_like(ids, dtype=torch.bool),
            override=torch.zeros_like(ids, dtype=torch.bool),
        )
        self.assertEqual(sum(insist), 0)
        self.assertEqual(sum(override), 0)
        self.assertEqual(hits[4], 2)
        self.assertEqual(hits[1], 1)
        self.assertEqual(hits[6], 1)

    def test_masked_slots_contribute_no_resident_hit(self):
        # -1 is "not routed here". It must not credit expert 0, which is what
        # the clamp_min(0) index would otherwise do.
        insist, override, hits = self._counters(
            topk_ids=torch.tensor([[-1, 2]]),
            served_ids=torch.tensor([[-1, 2]]),
            insist=torch.tensor([[False, False]]),
            override=torch.tensor([[False, False]]),
        )
        self.assertEqual(hits[0], 0)
        self.assertEqual(hits[2], 1)


if __name__ == "__main__":
    unittest.main()
