"""REAP measurement tests (srt/layers/moe/kt_ep_wrapper._ReapStage).

Checks the statistic the swap policy is fed: that it is the L2 norm of the
SUMMED expert output weighted by the mixture weight, accumulated per expert
with its own activation count, and that the things which have corrupted it
before -- padded decode rows, slots the launch did not compute, unbounded
accumulators -- cannot reach it.
"""

import unittest
from contextlib import contextmanager
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _Group:
    """Single-rank stand-in: the collectives reduce to the identity."""

    world_size = 1
    rank_in_group = 0
    first_rank = 0
    device_group = None


@contextmanager
def _single_rank():
    with patch(
        "sglang.srt.layers.moe.kt_ep_wrapper.get_tp_group", return_value=_Group()
    ):
        yield


def _stage(*, layers=2, experts=4, top_k=2, hidden=8, max_tokens=3):
    from sglang.srt.layers.moe.kt_ep_wrapper import _ReapStage

    with _single_rank():
        return _ReapStage(
            num_layers=layers,
            num_experts=experts,
            top_k=top_k,
            hidden=hidden,
            max_tokens=max_tokens,
            width=1,
            device=torch.device("cpu"),
        )


def _gemm2(rows, hidden, scale=1.0):
    """Distinct, exactly-representable rows so norms are unambiguous."""
    out = torch.zeros(rows, hidden, dtype=torch.bfloat16)
    for r in range(rows):
        out[r] = torch.full((hidden,), (r + 1) * 0.5 * scale, dtype=torch.bfloat16)
    return out


class TestReapStatistic(CustomTestCase):
    """Derived property: the accumulated value is sum(w * ||f||) and the
    divisor is the expert's own activation count. Red if the norm is taken
    over the wrong axis, if the weight is dropped or applied twice, or if the
    count stops matching the number of contributions."""

    def test_accumulates_weight_times_the_row_norm(self):
        st = _stage()
        hidden = 8
        gemm2 = _gemm2(4, hidden)
        # token 0 -> slots (expert 1 via row 0, expert 3 via row 2)
        idx = torch.tensor([[0, 2]], dtype=torch.int64)
        weights = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
        ids = torch.tensor([[1, 3]], dtype=torch.int64)
        with _single_rank():
            st.stage(
                layer_idx=0,
                gemm2_out=gemm2,
                permuted_idx=idx,
                weights=weights,
                served_ids=ids,
            )
            st.combine(1)

        n0 = gemm2[0].float().norm().item()
        n2 = gemm2[2].float().norm().item()
        self.assertAlmostEqual(st.num[0, 1].item(), 0.25 * n0, places=3)
        self.assertAlmostEqual(st.num[0, 3].item(), 0.75 * n2, places=3)
        self.assertEqual(st.count[0, 1].item(), 1)
        self.assertEqual(st.count[0, 3].item(), 1)

    def test_repeated_activations_sum_and_count(self):
        st = _stage()
        gemm2 = _gemm2(4, 8)
        # Both tokens route slot 0 to expert 2, so it accumulates twice.
        idx = torch.tensor([[0, 1], [0, 1]], dtype=torch.int64)
        weights = torch.tensor([[0.5, 0.5], [0.25, 0.75]], dtype=torch.float32)
        ids = torch.tensor([[2, 3], [2, 3]], dtype=torch.int64)
        with _single_rank():
            st.stage(
                layer_idx=1,
                gemm2_out=gemm2,
                permuted_idx=idx,
                weights=weights,
                served_ids=ids,
            )
            st.combine(2)
        n0 = gemm2[0].float().norm().item()
        self.assertAlmostEqual(st.num[1, 2].item(), (0.5 + 0.25) * n0, places=3)
        self.assertEqual(st.count[1, 2].item(), 2)

    def test_a_slot_the_launch_did_not_compute_contributes_nothing(self):
        # -1 is a CPU expert, masked out before dispatch. It has no row, and
        # crediting it would both invent a score and inflate the divisor.
        st = _stage()
        gemm2 = _gemm2(4, 8)
        idx = torch.tensor([[-1, 1]], dtype=torch.int64)
        weights = torch.tensor([[0.9, 0.1]], dtype=torch.float32)
        ids = torch.tensor([[2, 3]], dtype=torch.int64)
        with _single_rank():
            st.stage(
                layer_idx=0,
                gemm2_out=gemm2,
                permuted_idx=idx,
                weights=weights,
                served_ids=ids,
            )
            st.combine(1)
        self.assertEqual(st.num[0, 2].item(), 0.0)
        self.assertEqual(st.count[0, 2].item(), 0)
        self.assertEqual(st.count[0, 3].item(), 1)


class TestPaddedRowsAreNotMeasured(CustomTestCase):
    """Bug regression: a captured decode graph pads up to its capture size and
    sglang deliberately leaves the padded tail of input_ids unzeroed, so those
    rows carry the PREVIOUS replay's tokens and produce real routing and real
    expert outputs. Counting them puts a fixed, workload-independent sample
    into |X_k| on every step -- the divisor REAP exists for. Red if combine()
    stops slicing to the real batch size."""

    def test_only_the_real_rows_are_folded_in(self):
        st = _stage(max_tokens=3)
        gemm2 = _gemm2(4, 8)
        # Three staged rows; only the first is a real token this step.
        idx = torch.tensor([[0, 1], [2, 3], [2, 3]], dtype=torch.int64)
        weights = torch.ones(3, 2, dtype=torch.float32)
        ids = torch.tensor([[0, 1], [2, 3], [2, 3]], dtype=torch.int64)
        with _single_rank():
            st.stage(
                layer_idx=0,
                gemm2_out=gemm2,
                permuted_idx=idx,
                weights=weights,
                served_ids=ids,
            )
            st.combine(1)
        self.assertEqual(st.count[0, 0].item(), 1)
        self.assertEqual(st.count[0, 1].item(), 1)
        self.assertEqual(st.count[0, 2].item(), 0, "a padded row was measured")
        self.assertEqual(st.count[0, 3].item(), 0, "a padded row was measured")

    def test_a_batch_larger_than_the_buffer_is_refused_not_truncated(self):
        st = _stage(max_tokens=2)
        gemm2 = _gemm2(4, 8)
        idx = torch.zeros(4, 2, dtype=torch.int64)
        with _single_rank():
            st.stage(
                layer_idx=0,
                gemm2_out=gemm2,
                permuted_idx=idx,
                weights=torch.ones(4, 2),
                served_ids=torch.zeros(4, 2, dtype=torch.int64),
            )
        self.assertEqual(st.valid.sum().item(), 0)


class TestWindowAccounting(CustomTestCase):
    """Critical-path bookkeeping: a window's totals are taken and the
    accumulators reset. Red if the reset is dropped -- a monotonic float
    accumulator read by differencing stalls once the total outgrows the
    increment, which drives the busiest experts' delta to zero and inverts
    the ranking."""

    def test_take_window_returns_totals_and_zeroes(self):
        st = _stage()
        gemm2 = _gemm2(4, 8)
        with _single_rank():
            st.stage(
                layer_idx=0,
                gemm2_out=gemm2,
                permuted_idx=torch.tensor([[0, 1]], dtype=torch.int64),
                weights=torch.ones(1, 2),
                served_ids=torch.tensor([[1, 2]], dtype=torch.int64),
            )
            st.combine(1)
            num, count = st.take_window()
        self.assertGreater(num[0, 1].item(), 0.0)
        self.assertEqual(count[0, 1].item(), 1.0)
        self.assertEqual(st.num.sum().item(), 0.0)
        self.assertEqual(st.count.sum().item(), 0)

    def test_kt_half_is_folded_into_the_window(self):
        # kt scores the experts the GPU never runs -- every promotion
        # candidate. Its norms arrive per (token, slot) and are weighted and
        # attributed here, on the slots the GPU did NOT serve.
        st = _stage()
        gemm2 = _gemm2(4, 8)
        st.kt_norms[0, 0, 0] = 4.0  # slot 0 of token 0, which the GPU skipped
        with _single_rank():
            st.stage(
                layer_idx=0,
                gemm2_out=gemm2,
                permuted_idx=torch.tensor([[-1, 1]], dtype=torch.int64),
                weights=torch.tensor([[0.5, 0.5]], dtype=torch.float32),
                served_ids=torch.tensor([[2, 3]], dtype=torch.int64),
            )
            st.combine(1)
            num, count = st.take_window()
        self.assertAlmostEqual(num[0, 2].item(), 0.5 * 4.0, places=5)
        self.assertEqual(count[0, 2].item(), 1.0)

    def test_kt_does_not_double_count_a_slot_the_gpu_served(self):
        st = _stage()
        gemm2 = _gemm2(4, 8)
        st.kt_norms[0, 0, 1] = 99.0  # slot 1 IS served on the GPU
        with _single_rank():
            st.stage(
                layer_idx=0,
                gemm2_out=gemm2,
                permuted_idx=torch.tensor([[-1, 1]], dtype=torch.int64),
                weights=torch.tensor([[0.5, 0.5]], dtype=torch.float32),
                served_ids=torch.tensor([[2, 3]], dtype=torch.int64),
            )
            st.combine(1)
            num, count = st.take_window()
        self.assertEqual(count[0, 3].item(), 1.0)
        self.assertLess(num[0, 3].item(), 10.0)


if __name__ == "__main__":
    unittest.main()
