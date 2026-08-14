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


def _tables(num_experts=8, resident=(0, 1, 2, 3)):
    from sglang.srt.layers.moe.kt_expert_swap import SwapTables

    mask = torch.zeros(num_experts, dtype=torch.bool)
    l2g = torch.full((num_experts,), -1, dtype=torch.int32)
    for row, e in enumerate(resident):
        mask[e] = True
        l2g[e] = row
    g2l = torch.tensor(list(resident), dtype=torch.int32)
    return SwapTables(
        gpu_experts_mask=mask,
        gpu_experts_mask_cuda=mask.clone(),
        logical_to_gpu_index=l2g,
        logical_to_gpu_index_cuda=l2g.clone(),
        gpu_index_to_logical=g2l,
        pinned_mask=mask.clone(),
    )


class TestTableUpdate(CustomTestCase):
    """Derived property: a swap must leave all four membership tables agreeing,
    with the promoted expert in exactly the demoted expert's row. Red if the
    row is re-densified (which would move OTHER experts' rows without moving
    their weights) or if any table is missed — both produce tokens computed
    against the wrong expert, silently."""

    def test_promoted_takes_the_demoted_row(self):
        from sglang.srt.layers.moe.kt_expert_swap import (
            ExpertSwap,
            apply_swaps_to_tables,
            assert_tables_consistent,
        )

        t = _tables()
        rows = apply_swaps_to_tables(t, [ExpertSwap(6, 2, 100.0, 0.0)])
        self.assertEqual(rows, [2])  # expert 2 lived in row 2
        self.assertTrue(t.gpu_experts_mask[6])
        self.assertFalse(t.gpu_experts_mask[2])
        self.assertEqual(int(t.logical_to_gpu_index[6]), 2)
        self.assertEqual(int(t.logical_to_gpu_index[2]), -1)
        self.assertEqual(int(t.gpu_index_to_logical[2]), 6)
        # every other expert's row is untouched
        self.assertEqual(int(t.logical_to_gpu_index[0]), 0)
        self.assertEqual(int(t.logical_to_gpu_index[3]), 3)
        assert_tables_consistent(t, 4)

    def test_device_and_pinned_copies_track(self):
        from sglang.srt.layers.moe.kt_expert_swap import (
            ExpertSwap,
            apply_swaps_to_tables,
        )

        t = _tables()
        ptrs = (t.gpu_experts_mask_cuda.data_ptr(), t.pinned_mask.data_ptr())
        apply_swaps_to_tables(t, [ExpertSwap(5, 1, 50.0, 0.0)])
        self.assertTrue(bool(t.gpu_experts_mask_cuda[5]))
        self.assertFalse(bool(t.gpu_experts_mask_cuda[1]))
        self.assertTrue(bool(t.pinned_mask[5]))
        # in-place: decode graphs captured these addresses, kt C++ holds the
        # pinned pointer
        self.assertEqual(
            (t.gpu_experts_mask_cuda.data_ptr(), t.pinned_mask.data_ptr()), ptrs
        )

    def test_rejects_incoherent_swap(self):
        from sglang.srt.layers.moe.kt_expert_swap import (
            ExpertSwap,
            apply_swaps_to_tables,
        )

        t = _tables()
        with self.assertRaises(ValueError):  # demote a non-resident expert
            apply_swaps_to_tables(t, [ExpertSwap(6, 7, 1.0, 0.0)])
        with self.assertRaises(ValueError):  # promote an already-resident one
            apply_swaps_to_tables(t, [ExpertSwap(0, 1, 1.0, 0.0)])

    def test_invariant_catches_desync(self):
        from sglang.srt.layers.moe.kt_expert_swap import assert_tables_consistent

        t = _tables()
        t.gpu_experts_mask[7] = True  # mask says resident, no row assigned
        with self.assertRaises(AssertionError):
            assert_tables_consistent(t, 4)


class TestSwapWindow(CustomTestCase):
    """Critical-path bookkeeping for the window driver. Red if weights stop
    being written BEFORE the tables flip (a window where a row is advertised
    as one expert while holding another's weights), or if a failing layer
    stops being isolated."""

    def _entry(self, policy=None):
        from sglang.srt.layers.moe.kt_expert_swap import ExpertSwapPolicy

        p = policy or ExpertSwapPolicy(8, ema_alpha=1.0, min_demand=1.0, max_swaps=2)
        p.observe(_cum({}), _cum({}))
        # Expert 2 must be the unambiguous demotion victim: give every other
        # resident real traffic, so the choice does not depend on tie-break
        # order among equally-unused experts.
        p.observe(_cum({6: 100}), _cum({0: 50, 1: 40, 3: 30}))
        return {
            "policy": p,
            "tables": _tables(),
            "layer": object(),
            "num_gpu_experts": 4,
            "layer_idx": 40,
        }

    def test_weights_move_before_tables_flip(self):
        from sglang.srt.layers.moe.kt_expert_swap import run_swap_window

        entry = self._entry()
        order = []

        def move(layer, row, expert, demoted):
            # tables must still show the OLD occupant at this point
            order.append(
                (row, expert, int(entry["tables"].gpu_index_to_logical[row]))
            )

        res = run_swap_window([entry], move_weights=move)
        self.assertEqual(res.swaps_applied, 1)
        row, promoted, occupant_at_write = order[0]
        self.assertEqual((row, promoted), (2, 6))
        self.assertEqual(occupant_at_write, 2)  # flip had not happened yet
        self.assertEqual(int(entry["tables"].gpu_index_to_logical[2]), 6)

    def test_failing_layer_is_isolated_not_fatal(self):
        from sglang.srt.layers.moe.kt_expert_swap import run_swap_window

        entry = self._entry()

        def boom(layer, row, expert, demoted):
            raise RuntimeError("export failed")

        res = run_swap_window([entry], move_weights=boom)
        self.assertEqual(res.swaps_applied, 0)
        self.assertEqual(res.skipped_layers, 1)
        # tables untouched, so the rewritten row is still advertised as its
        # previous occupant and nothing routes to a half-updated expert
        self.assertEqual(int(entry["tables"].gpu_index_to_logical[2]), 2)
        self.assertTrue(bool(entry["tables"].gpu_experts_mask[2]))

    def test_finish_layer_runs_after_moves_but_before_the_flip(self):
        """The hook batched movers flush through must land inside the window.

        A mover that only RECORDS in move_weights and applies the copies in
        bulk has to be given a point to flush before the tables flip. Flushing
        lazily on the next layer's first move instead put the write after the
        flip, so a failure left the tables advertising experts whose rows still
        held the previous occupants.
        """
        from sglang.srt.layers.moe.kt_expert_swap import run_swap_window

        entry = self._entry()
        seq = []

        run_swap_window(
            [entry],
            move_weights=lambda l, r, e, d: seq.append("move"),
            finish_layer=lambda: seq.append(
                f"flush@{int(entry['tables'].gpu_index_to_logical[2])}"
            ),
        )
        # flushed after the move, while the row still reads as its old occupant
        self.assertEqual(seq, ["move", "flush@2"])
        self.assertEqual(int(entry["tables"].gpu_index_to_logical[2]), 6)

    def test_failing_finish_layer_leaves_the_tables_alone(self):
        from sglang.srt.layers.moe.kt_expert_swap import run_swap_window

        entry = self._entry()

        def boom():
            raise RuntimeError("bulk copy failed")

        res = run_swap_window(
            [entry],
            move_weights=lambda l, r, e, d: None,
            finish_layer=boom,
        )
        self.assertEqual(res.swaps_applied, 0)
        self.assertEqual(res.skipped_layers, 1)
        # The whole point: a failed bulk write must NOT leave row 2 advertised
        # as expert 6 while it still holds expert 2.
        self.assertEqual(int(entry["tables"].gpu_index_to_logical[2]), 2)
        self.assertTrue(bool(entry["tables"].gpu_experts_mask[2]))

    def test_quiesce_runs_before_any_mutation(self):
        from sglang.srt.layers.moe.kt_expert_swap import run_swap_window

        entry = self._entry()
        seq = []
        run_swap_window(
            [entry],
            move_weights=lambda l, r, e, d: seq.append("move"),
            quiesce=lambda: seq.append("quiesce"),
        )
        self.assertEqual(seq[0], "quiesce")


if __name__ == "__main__":
    unittest.main()
