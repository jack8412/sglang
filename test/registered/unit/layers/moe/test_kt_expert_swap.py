"""Swap-policy tests (srt/layers/moe/kt_expert_swap).

Covers the decisions the policy is responsible for: measuring recent traffic
rather than launch history, refusing unprofitable or noise-level swaps, and
not letting a completed swap immediately reverse itself.
"""

import unittest

import torch

from sglang.srt.layers.moe.kt_expert_swap import ExpertSwapPolicy
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _cum(pairs, n=8):
    t = torch.zeros(n, dtype=torch.int64)
    for i, v in pairs.items():
        t[i] = v
    return t


class TestSwapSelection(CustomTestCase):
    """Derived property: the greedy pairing and its dead band. Red if the
    hysteresis comparison drops (thrashing returns), if the pairing stops
    being disjoint, or if the budget stops binding."""

    # experts 0-3 resident, 4-7 offloaded
    MASK = torch.tensor([True] * 4 + [False] * 4)

    def _policy(self, **kw):
        kw.setdefault("ema_alpha", 1.0)  # latest interval only, for determinism
        kw.setdefault("min_demand", 1.0)
        return ExpertSwapPolicy(8, **kw)

    def test_promotes_high_demand_over_unused_resident(self):
        p = self._policy()
        p.observe(_cum({}), _cum({}))  # baseline
        p.observe(_cum({4: 100}), _cum({0: 50, 1: 40, 2: 30, 3: 0}))
        swaps = p.select(self.MASK)
        self.assertEqual(len(swaps), 1)
        self.assertEqual((swaps[0].promote, swaps[0].demote), (4, 3))

    def test_hysteresis_blocks_marginal_swap(self):
        # demand 60 vs incumbent hits 50: a real but small edge. At the
        # default 2x dead band this must NOT swap -- the transfer would cost
        # more than the gain and invite a swap back next interval.
        p = self._policy(hysteresis=2.0)
        p.observe(_cum({}), _cum({}))
        p.observe(_cum({4: 60}), _cum({0: 50, 1: 50, 2: 50, 3: 50}))
        self.assertEqual(p.select(self.MASK), [])

    def test_budget_caps_swaps_per_evaluation(self):
        p = self._policy(max_swaps=2)
        p.observe(_cum({}), _cum({}))
        p.observe(_cum({4: 90, 5: 80, 6: 70, 7: 60}), _cum({}))
        self.assertEqual(len(p.select(self.MASK)), 2)

    def test_pairs_are_disjoint(self):
        p = self._policy(max_swaps=4)
        p.observe(_cum({}), _cum({}))
        p.observe(_cum({4: 90, 5: 80}), _cum({}))
        swaps = p.select(self.MASK)
        self.assertEqual(len({s.promote for s in swaps}), len(swaps))
        self.assertEqual(len({s.demote for s in swaps}), len(swaps))

    def test_min_demand_floor_rejects_noise(self):
        p = self._policy(min_demand=10.0)
        p.observe(_cum({}), _cum({}))
        p.observe(_cum({4: 3}), _cum({}))
        self.assertEqual(p.select(self.MASK), [])


class TestCounterDeltaSemantics(CustomTestCase):
    """Derived property: cumulative counters must be differenced, and the
    first observation is a baseline only. Red if a refactor feeds raw
    cumulative values into the EMA — the resident set would then be pinned to
    whatever the traffic looked like shortly after launch."""

    MASK = torch.tensor([True] * 4 + [False] * 4)

    def test_first_observation_is_baseline_only(self):
        p = ExpertSwapPolicy(8, ema_alpha=1.0, min_demand=1.0)
        p.observe(_cum({4: 10_000}), _cum({}))  # whole history, must not count
        self.assertEqual(p.select(self.MASK), [])

    def test_only_the_delta_counts(self):
        p = ExpertSwapPolicy(8, ema_alpha=1.0, min_demand=1.0)
        p.observe(_cum({4: 1000, 5: 0}), _cum({}))
        # Since the baseline, expert 5 saw 40 and expert 4 only 5: 5 must win
        # despite 4's far larger lifetime total.
        p.observe(_cum({4: 1005, 5: 40}), _cum({}))
        swaps = p.select(self.MASK)
        self.assertEqual(swaps[0].promote, 5)

    def test_counter_reset_does_not_produce_negative_delta(self):
        p = ExpertSwapPolicy(8, ema_alpha=1.0, min_demand=1.0)
        p.observe(_cum({4: 500}), _cum({}))
        p.observe(_cum({4: 7}), _cum({}))  # counters restarted
        self.assertGreaterEqual(p.demand_ema[4].item(), 0.0)


class TestPostSwapHygiene(CustomTestCase):
    """Completeness contract: after a swap the two experts' histories describe
    a world that no longer exists. Red if note_swapped stops clearing them —
    the pair would qualify to swap straight back on the next evaluation."""

    MASK = torch.tensor([True] * 4 + [False] * 4)

    def test_swapped_pair_history_is_cleared(self):
        p = ExpertSwapPolicy(8, ema_alpha=1.0, min_demand=1.0)
        p.observe(_cum({}), _cum({}))
        p.observe(_cum({4: 100}), _cum({3: 0}))
        swaps = p.select(self.MASK)
        p.note_swapped(swaps)
        self.assertEqual(p.demand_ema[4].item(), 0.0)
        self.assertEqual(p.hits_ema[3].item(), 0.0)
        # With membership now flipped, nothing should qualify immediately.
        flipped = self.MASK.clone()
        flipped[4], flipped[3] = True, False
        self.assertEqual(p.select(flipped), [])


class TestStateRoundTrip(CustomTestCase):
    """Critical-path bookkeeping: the persisted state is the next launch's
    seed. Red if a field is added to the EMA set without being serialised, or
    if a mismatched expert count is silently accepted."""

    def test_round_trip_preserves_decisions(self):
        p = ExpertSwapPolicy(8, ema_alpha=1.0, min_demand=1.0)
        p.observe(_cum({}), _cum({}))
        p.observe(_cum({4: 100, 5: 50}), _cum({0: 5}))
        before = p.select(torch.tensor([True] * 4 + [False] * 4))

        q = ExpertSwapPolicy(8, ema_alpha=1.0, min_demand=1.0)
        q.load_state_dict(p.state_dict())
        after = q.select(torch.tensor([True] * 4 + [False] * 4))
        self.assertEqual(
            [(s.promote, s.demote) for s in before],
            [(s.promote, s.demote) for s in after],
        )

    def test_expert_count_mismatch_raises(self):
        p = ExpertSwapPolicy(8)
        q = ExpertSwapPolicy(16)
        with self.assertRaises(ValueError):
            q.load_state_dict(p.state_dict())


if __name__ == "__main__":
    unittest.main()
