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

    def test_ties_break_by_ascending_expert_id(self):
        """Derived property: with every demand equal and every hit equal, the
        pairing is decided purely by the tie-break, and it must be ascending
        expert id on BOTH sides -- lowest-id cold expert to lowest-id resident.

        This is not cosmetic. Every TP rank runs this selection independently
        on its own copy of the counters and they must reach the same pairs, or
        the ranks' placements diverge and each computes a different model,
        silently. The python implementation got this from sort() being stable;
        a whole-tensor rewrite gets it only from an explicitly stable argsort,
        and nothing else in this file would notice if it were dropped."""
        p = self._policy(max_swaps=4)
        p.observe(_cum({}), _cum({}))
        # all four cold experts equally in demand, all four residents equally
        # unused: only the tie-break can order this.
        p.observe(_cum({4: 10, 5: 10, 6: 10, 7: 10}), _cum({}))
        swaps = p.select(self.MASK)
        self.assertEqual(
            [(s.promote, s.demote) for s in swaps], [(4, 0), (5, 1), (6, 2), (7, 3)]
        )

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

    def test_slot_table_exchanges_the_pair(self):
        """Split prefill routes through logical_to_slot; a swap must exchange
        exactly the swapped pair's entries there (promoted takes the demoted
        one's resident slot == its row; demoted takes the promoted one's cold
        slot) and touch nothing else. Red if the slot table goes stale again —
        the pre-arena builds shipped that bug, and it misroutes both experts
        on every split prefill after the first acting window."""
        from sglang.srt.layers.moe.kt_expert_swap import (
            ExpertSwap,
            apply_swaps_to_tables,
        )

        num_experts, resident = 8, (0, 1, 2, 3)
        l2s = torch.empty(num_experts, dtype=torch.int32)
        for row, e in enumerate(resident):
            l2s[e] = row
        for j, e in enumerate(e for e in range(num_experts) if e not in resident):
            l2s[e] = len(resident) + j
        t = _tables()._replace(logical_to_slot=l2s, logical_to_slot_cuda=l2s.clone())
        before = l2s.clone()

        apply_swaps_to_tables(t, [ExpertSwap(6, 2, 100.0, 0.0)])
        self.assertEqual(int(t.logical_to_slot[6]), int(before[2]))
        self.assertEqual(int(t.logical_to_slot[2]), int(before[6]))
        for e in range(num_experts):
            if e not in (2, 6):
                self.assertEqual(int(t.logical_to_slot[e]), int(before[e]), e)
        self.assertTrue(torch.equal(t.logical_to_slot_cuda, t.logical_to_slot))

    def test_slot_table_absent_is_tolerated(self):
        """Methods without split prefill build no slot table; apply must not
        require one."""
        from sglang.srt.layers.moe.kt_expert_swap import (
            ExpertSwap,
            apply_swaps_to_tables,
        )

        t = _tables()
        self.assertIsNone(t.logical_to_slot)
        apply_swaps_to_tables(t, [ExpertSwap(6, 2, 100.0, 0.0)])

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

    def test_rejected_batch_leaves_every_table_untouched(self):
        """A batch containing one illegal pair must apply NONE of it.

        The per-swap loop this replaced raised on the offending pair with the
        preceding pairs already written, leaving the four tables disagreeing
        about experts that were never meant to move -- and a swap window that
        dies with half-flipped tables is exactly the silently-wrong-weights
        state the invariant check exists to catch. Red if validation moves back
        inside the write loop."""
        from sglang.srt.layers.moe.kt_expert_swap import (
            ExpertSwap,
            apply_swaps_to_tables,
        )

        t = _tables()
        before = (
            t.gpu_experts_mask.clone(),
            t.logical_to_gpu_index.clone(),
            t.gpu_index_to_logical.clone(),
        )
        with self.assertRaises(ValueError):
            apply_swaps_to_tables(
                t,
                [
                    ExpertSwap(6, 2, 100.0, 0.0),  # legal, and must NOT be applied
                    ExpertSwap(7, 5, 100.0, 0.0),  # demotes a non-resident
                ],
            )
        self.assertTrue(torch.equal(t.gpu_experts_mask, before[0]))
        self.assertTrue(torch.equal(t.logical_to_gpu_index, before[1]))
        self.assertTrue(torch.equal(t.gpu_index_to_logical, before[2]))

    def test_invariant_catches_desync(self):
        from sglang.srt.layers.moe.kt_expert_swap import assert_tables_consistent

        t = _tables()
        t.gpu_experts_mask[7] = True  # mask says resident, no row assigned
        with self.assertRaises(AssertionError):
            assert_tables_consistent(t, 4)

    def test_invariant_catches_broken_round_trip(self):
        """A desync that leaves the resident COUNT and the row PERMUTATION
        intact, and disagrees only about which expert owns which row: l2g sends
        2 -> row 2, but g2l says row 2 holds expert 3. That is the state in
        which a token is computed against the wrong expert's weights.

        Distinct from test_invariant_catches_desync, which trips the count
        check and so never reaches the round trip. Red if the round-trip check
        degrades to vacuously true -- the live failure mode when it is written
        whole-tensor, where a dtype or indexing slip compares the wrong things
        and passes everything."""
        from sglang.srt.layers.moe.kt_expert_swap import assert_tables_consistent

        t = _tables()
        assert_tables_consistent(t, 4)  # consistent to begin with
        t.gpu_index_to_logical[2] = 3
        t.gpu_index_to_logical[3] = 2
        with self.assertRaises(AssertionError):
            assert_tables_consistent(t, 4)


class TestTablesOnCuda(CustomTestCase):
    """Bug regression: the membership tables are not guaranteed to be CPU.

    SwapTables documents them as CPU and every test here built them that way,
    so a whole-tensor rewrite of assert_tables_consistent shipped with
    torch.arange() on the default device and killed a live server:

        RuntimeError: Expected all tensors to be on the same device, but got
        other is on cpu, different from other tensors on cuda:5

    The per-expert version it replaced was device-agnostic for free, because
    .item() pulls a scalar off any device. Whole-tensor ops are not, and no
    CPU-only test can tell the difference -- which is the whole point of this
    one."""

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_invariant_and_apply_accept_cuda_tables(self):
        from sglang.srt.layers.moe.kt_expert_swap import (
            ExpertSwap,
            apply_swaps_to_tables,
            assert_tables_consistent,
        )

        t = _tables()
        cuda = torch.device("cuda")
        t = t._replace(
            gpu_experts_mask=t.gpu_experts_mask.to(cuda),
            logical_to_gpu_index=t.logical_to_gpu_index.to(cuda),
            gpu_index_to_logical=t.gpu_index_to_logical.to(cuda),
        )
        assert_tables_consistent(t, 4)
        rows = apply_swaps_to_tables(t, [ExpertSwap(6, 2, 100.0, 0.0)])
        self.assertEqual(rows, [2])
        assert_tables_consistent(t, 4)
        self.assertTrue(bool(t.gpu_experts_mask[6]))
        self.assertEqual(int(t.gpu_index_to_logical[2]), 6)


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

    def test_failing_layer_propagates_and_leaves_the_tables_alone(self):
        """A failed move must RAISE, and must not touch the tables.

        The raise is the fail-fast policy: skipping the layer and continuing
        was the shipped behaviour and it was a silent-wrong-weights bug --
        move_weights only RECORDS, so the window-end drain then flushed the
        skipped layer's staged promotions into rows whose tables never
        flipped. This test asserted the old contract and kept passing until
        the policy changed under it.

        The other half is unchanged and is the real invariant: whatever the
        window does on the way out, row 2 must still be advertised as its
        previous occupant, so nothing routes to a half-updated expert."""
        from sglang.srt.layers.moe.kt_expert_swap import run_swap_window

        entry = self._entry()

        def boom(layer, row, expert, demoted):
            raise RuntimeError("export failed")

        with self.assertRaises(RuntimeError):
            run_swap_window([entry], move_weights=boom)
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

        with self.assertRaises(RuntimeError):
            run_swap_window(
                [entry],
                move_weights=lambda l, r, e, d: None,
                finish_layer=boom,
            )
        # The whole point, and unchanged by fail-fast: a failed bulk write must
        # NOT leave row 2 advertised as expert 6 while it still holds expert 2.
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


class TestDemotionPrefetchGate(CustomTestCase):
    """Do not read the checkpoint for demotions rank-write will never read.

    The swap window can prefetch a demoted expert's bytes off the checkpoint
    (~12.9 GB per window) so the fetch is off the critical path. Under rank-
    write demotion nothing ever consumes them: each rank captures its own slice
    from its own GPU rows.

    The gate for that used to key on whether a PREVIOUS window had armed
    rank-write -- a value that does not exist until a window has run. So every
    boundary before the first window prefetched in full. V14 logged 17 such
    passes and every window reported "prefetch 0 hit / 0 miss": ~219 GB read
    off disk, none of it consumed, and worse than free because it lands right
    after a plan change and so misses cache.

    The writer is constructed at boot, so its presence is the signal that is
    actually available when the decision is made.
    """

    def setUp(self):
        from sglang.srt.layers.moe import kt_ep_wrapper

        self.mod = kt_ep_wrapper
        self.saved = dict(kt_ep_wrapper._KT_SWAP_STATE)
        kt_ep_wrapper._KT_SWAP_STATE.clear()
        self.addCleanup(
            lambda: (
                kt_ep_wrapper._KT_SWAP_STATE.clear(),
                kt_ep_wrapper._KT_SWAP_STATE.update(self.saved),
            )
        )

    def test_writer_present_before_any_window_suppresses_prefetch(self):
        """The regression: decided at boot, not after the first window."""
        self.mod._KT_SWAP_STATE["rank_writer"] = object()
        self.assertNotIn("rank_write_armed", self.mod._KT_SWAP_STATE)
        self.assertTrue(self.mod._rank_write_owns_demotions())

    def test_no_writer_leaves_the_prefetch_enabled(self):
        """Without rank-write the checkpoint IS the source; keep prefetching."""
        self.assertFalse(self.mod._rank_write_owns_demotions())
        self.mod._KT_SWAP_STATE["rank_writer"] = None
        self.assertFalse(self.mod._rank_write_owns_demotions())

    def test_armed_still_counts_on_its_own(self):
        """Arming remains sufficient, so the two signals cannot disagree."""
        self.mod._KT_SWAP_STATE["rank_write_armed"] = True
        self.assertTrue(self.mod._rank_write_owns_demotions())


if __name__ == "__main__":
    unittest.main()
