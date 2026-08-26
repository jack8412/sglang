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


def _vec(pairs, n=8, dtype=torch.float64):
    t = torch.zeros(n, dtype=dtype)
    for i, v in pairs.items():
        t[i] = v
    return t


def _window(p, scores, counts):
    """One window where each listed expert fired `count` times contributing
    `score` each, which is what the device accumulators would hold."""
    total = {k: scores[k] * counts[k] for k in counts}
    p.observe(_vec(total), _vec(counts))


MASK = torch.tensor([True] * 4 + [False] * 4)


class TestReapRanking(CustomTestCase):
    """Derived property: placement follows damage-per-activation, and the
    greedy pairing, budget and dead band that turn scores into swaps. Red if
    the ranking picks up a frequency term, if the pairing stops being disjoint,
    or if the hysteresis comparison drops (thrashing returns)."""

    def _policy(self, **kw):
        kw.setdefault("ema_alpha", 1.0)
        kw.setdefault("min_evidence", 1.0)
        return ExpertSwapPolicy(8, **kw)

    def test_rare_high_value_expert_outranks_a_frequent_low_value_one(self):
        # THE reason REAP divides by |X_k|. Expert 5 fires 5,000 times giving
        # 1.0 each; expert 4 fires 10 times giving 40.0 each. Any statistic
        # that counts activations promotes 5; REAP promotes 4.
        p = self._policy()
        _window(p, {4: 40.0, 5: 1.0}, {4: 10, 5: 5000})
        swaps = p.select(MASK)
        self.assertEqual(swaps[0].promote, 4)

    def test_promotes_over_the_weakest_resident(self):
        p = self._policy()
        _window(p, {0: 5.0, 1: 4.0, 2: 3.0, 3: 0.5, 4: 90.0}, dict.fromkeys([0, 1, 2, 3, 4], 10))
        swaps = p.select(MASK)
        self.assertEqual((swaps[0].promote, swaps[0].demote), (4, 3))

    def test_hysteresis_blocks_a_marginal_swap(self):
        p = self._policy(hysteresis=2.0)
        _window(p, {0: 5.0, 1: 5.0, 2: 5.0, 3: 5.0, 4: 6.0}, dict.fromkeys(range(5), 10))
        self.assertEqual(p.select(MASK), [])

    def test_budget_caps_swaps_per_evaluation(self):
        p = self._policy(max_swaps=2)
        _window(p, {4: 9.0, 5: 8.0, 6: 7.0, 7: 6.0}, dict.fromkeys([4, 5, 6, 7], 10))
        self.assertEqual(len(p.select(MASK)), 2)

    def test_pairs_are_disjoint(self):
        p = self._policy(max_swaps=4)
        _window(p, {4: 9.0, 5: 8.0}, {4: 10, 5: 10})
        swaps = p.select(MASK)
        self.assertEqual(len({s.promote for s in swaps}), len(swaps))
        self.assertEqual(len({s.demote for s in swaps}), len(swaps))

    def test_ties_break_by_ascending_expert_id(self):
        # Every rank runs this independently; if equal scores did not resolve
        # the same way everywhere the placements would silently diverge.
        p = self._policy(max_swaps=4)
        _window(p, {4: 9.0, 5: 9.0, 6: 9.0, 7: 9.0}, dict.fromkeys([4, 5, 6, 7], 10))
        self.assertEqual([s.promote for s in p.select(MASK)], [4, 5, 6, 7])

    def test_an_unmeasured_non_resident_is_never_promoted(self):
        # No evidence is not the same as a low score.
        p = self._policy(min_evidence=1.0)
        _window(p, {0: 1.0}, {0: 10})
        self.assertEqual(p.select(MASK), [])

    def test_an_unmeasured_resident_is_the_first_demoted(self):
        # Zero score in the demote direction is correct: an expert nothing has
        # routed to is the safest row to take.
        p = self._policy()
        _window(p, {0: 9.0, 1: 9.0, 2: 9.0, 4: 50.0}, {0: 10, 1: 10, 2: 10, 4: 10})
        swaps = p.select(MASK)
        self.assertEqual(swaps[0].demote, 3)


class TestWindowSemantics(CustomTestCase):
    """Derived property: windows combine as a POOLED mean -- sum and count
    decayed together -- not as an average of per-window means. Red if the
    count is dropped anywhere, which would let a one-activation window move a
    score as far as a five-thousand-activation one."""

    def test_windows_are_weighted_by_their_activation_count(self):
        p = ExpertSwapPolicy(8, ema_alpha=0.5)
        _window(p, {4: 100.0}, {4: 1})     # one sample, huge
        _window(p, {4: 1.0}, {4: 100})     # a hundred samples, small
        # Pooled: (0.5*100 + 100) / (0.5*1 + 100) = 150 / 100.5
        self.assertAlmostEqual(p.score[4].item(), 150.0 / 100.5, places=6)
        # An average of the two window means would be ~50.5.
        self.assertLess(p.score[4].item(), 5.0)

    def test_a_window_is_a_delta_not_a_running_total(self):
        # The device accumulators are zeroed after every snapshot, so there is
        # no baseline to subtract and no first-call special case.
        p = ExpertSwapPolicy(8, ema_alpha=1.0)
        _window(p, {4: 7.0}, {4: 3})
        self.assertAlmostEqual(p.score[4].item(), 7.0)

    def test_an_unmeasured_expert_keeps_its_score(self):
        p = ExpertSwapPolicy(8, ema_alpha=0.5)
        _window(p, {4: 20.0}, {4: 10})
        _window(p, {}, {})  # nothing routed to it this window
        self.assertAlmostEqual(p.score[4].item(), 20.0)

    def test_negative_counts_are_refused(self):
        p = ExpertSwapPolicy(8)
        with self.assertRaises(ValueError):
            p.observe(_vec({}), _vec({0: -1}))

    def test_shape_mismatch_is_refused(self):
        p = ExpertSwapPolicy(8)
        with self.assertRaises(ValueError):
            p.observe(_vec({}, n=4), _vec({}, n=4))


class TestPostSwapHygiene(CustomTestCase):
    """Completeness contract: a swapped pair must not trade straight back.
    REAP needs no history reset to get this -- the score is a property of the
    expert, not of where it lives -- so this guards that the mask flip alone
    is sufficient. Red if select() ever ranks a demotion by something that
    inverts when an expert changes side."""

    def test_swapped_pair_does_not_immediately_trade_back(self):
        p = ExpertSwapPolicy(8, ema_alpha=1.0, min_evidence=1.0)
        _window(p, {0: 9.0, 1: 9.0, 2: 9.0, 3: 0.1, 4: 90.0},
                dict.fromkeys([0, 1, 2, 3, 4], 10))
        swaps = p.select(MASK)
        self.assertEqual((swaps[0].promote, swaps[0].demote), (4, 3))
        flipped = MASK.clone()
        flipped[4], flipped[3] = True, False
        self.assertEqual(p.select(flipped), [])


class TestStateRoundTrip(CustomTestCase):
    """Critical-path bookkeeping: the persisted state is the next launch's
    seed. Red if a field is added to the score set without being serialised,
    or if a mismatched expert count is silently accepted."""

    def test_round_trip_preserves_decisions(self):
        p = ExpertSwapPolicy(8, ema_alpha=1.0, min_evidence=1.0)
        _window(p, {0: 2.0, 4: 90.0, 5: 50.0}, {0: 10, 4: 10, 5: 10})
        before = p.select(MASK)

        q = ExpertSwapPolicy(8, ema_alpha=1.0, min_evidence=1.0)
        q.load_state_dict(p.state_dict())
        self.assertEqual(
            [(s.promote, s.demote) for s in before],
            [(s.promote, s.demote) for s in q.select(MASK)],
        )
        # Both halves of the mean, not just the ranking they happened to imply.
        self.assertAlmostEqual(q.score[4].item(), 90.0)
        self.assertAlmostEqual(q.reap_count[4].item(), 10.0)

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

        p = policy or ExpertSwapPolicy(
            8, ema_alpha=1.0, min_evidence=1.0, max_swaps=2
        )
        # Expert 2 must be the unambiguous demotion victim: give every other
        # resident a real score, so the choice does not depend on tie-break
        # order among equally-unmeasured experts.
        _window(p, {0: 9.0, 1: 8.0, 3: 7.0, 6: 90.0}, dict.fromkeys([0, 1, 3, 6], 10))
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


class TestFlushMovesFixedWidthBatch(CustomTestCase):
    """A reused, full-width batch buffer must not leak stale rows.

    _flush_moves used to torch.stack a fresh batch per layer, so its shape
    followed len(items). That shape varies between layers, the caching
    allocator could not reuse the blocks, and each miss forced a
    free/synchronize -- ~6.4 ms apiece against an allocator nearly full at
    mem-fraction 0.89, which is ~4.7 s of a 5.19 s swap window while the DMA
    itself is 0.075 ms/expert.

    The fix is split prefill's shape: one buffer set, allocated once, ALWAYS
    processed at full width so every layer presents the same shape, with only
    the live prefix scattered. That introduces a failure mode worth pinning --
    rows beyond len(items) hold whatever the PREVIOUS layer left there, so if
    the scatter ever widened past the prefix a layer would silently inherit
    another layer's experts.
    """

    def _run_layer(self, buf, values, dst, rows):
        """One layer: fill the prefix, process full width, scatter the prefix."""
        n = len(values)
        for i, v in enumerate(values):
            buf[i].copy_(torch.full_like(buf[i], v))
        processed = buf * 2  # stands in for the swizzle: full width, same shape
        idx = torch.tensor(rows, dtype=torch.long)
        dst.index_copy_(0, idx, processed[:n])

    def test_stale_rows_are_never_scattered(self):
        cap, width = 8, 4
        buf = torch.zeros(cap, width, dtype=torch.float32)
        dst = torch.zeros(32, width, dtype=torch.float32)

        # layer A fills all 8 slots
        self._run_layer(buf, list(range(1, 9)), dst, list(range(8)))
        # layer B uses only 3 -- slots 3..7 still hold layer A's values
        self._run_layer(buf, [100, 200, 300], dst, [10, 11, 12])

        torch.testing.assert_close(dst[10], torch.full((width,), 200.0))
        torch.testing.assert_close(dst[11], torch.full((width,), 400.0))
        torch.testing.assert_close(dst[12], torch.full((width,), 600.0))
        # nothing else moved: rows 13.. must still be zero, i.e. layer A's
        # leftovers in buf[3:] did not ride along
        self.assertEqual(float(dst[13:].abs().sum()), 0.0)

    def test_shape_is_constant_across_layers(self):
        """The point of the change: the allocator sees one shape, not many."""
        cap, width = 8, 4
        buf = torch.zeros(cap, width, dtype=torch.float32)
        shapes = set()
        for n in (8, 3, 5, 8, 1):
            processed = buf * 2
            shapes.add(tuple(processed.shape))
            self.assertEqual(processed[:n].shape[0], n)
        self.assertEqual(len(shapes), 1, f"batch shape varied: {shapes}")


if __name__ == "__main__":
    unittest.main()
