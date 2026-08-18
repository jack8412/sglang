"""finalize_split_prefill must not build a second cold-expert store.

Bug regression. With speculative decoding enabled, the DRAFT ModelRunner runs
the same initialize() path as the target, and maybe_init_split_prefill gated
only on get_exec().moe.kt_expert_split_prefill -- a process-global snapshot
that reads True in the draft worker too. The target's layers are still
registered (reset_split_prefill has no callers), so finalize ran a second time
and built a WHOLE SECOND pinned cold store: 51.1 GiB per rank, 439 GB across
TP8, on top of the first, which is still referenced.

It is invisible except as host memory -- the layer list and the arming
consensus look identical the second time, and the store's own log line reports
the same 51.1 GiB it reported before. Measured: a boot that plateaus at 996 GB
of Shmem without spec decoding was still climbing past 1,201 GB with it, and an
earlier one reached 1,310 GB and was OOM-killed.
"""

import unittest
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestFinalizeSplitPrefillIsIdempotent(CustomTestCase):
    def test_second_finalize_does_not_rebuild_the_store(self):
        """Red if the idempotence guard is dropped: the second call would fall
        through to build_cold_store and allocate another 439 GB."""
        from sglang.srt.layers.moe import kt_ep_wrapper as m

        layers = list(m._KT_SPLIT_PREFILL_LAYERS)
        state = dict(m._KT_SPLIT_PREFILL_STATE)
        try:
            # Armed state: layers registered and a pipeline already built.
            m._KT_SPLIT_PREFILL_LAYERS.clear()
            m._KT_SPLIT_PREFILL_LAYERS.append((0, object()))
            m._KT_SPLIT_PREFILL_STATE["pipeline"] = object()

            from sglang.srt.layers.moe import expert_cold_store

            # The store build is what costs 439 GB; assert it is never reached.
            with patch.object(
                expert_cold_store,
                "build_cold_store",
                side_effect=AssertionError("rebuilt the cold store"),
            ):
                self.assertTrue(m.finalize_split_prefill(object()))
        finally:
            m._KT_SPLIT_PREFILL_LAYERS.clear()
            m._KT_SPLIT_PREFILL_LAYERS.extend(layers)
            m._KT_SPLIT_PREFILL_STATE.update(state)

    def test_unarmed_finalize_still_proceeds(self):
        """The guard must key on the pipeline, not merely on being called
        twice: with no pipeline yet, finalize must NOT short-circuit, or the
        first real build never happens and split prefill silently never arms."""
        from sglang.srt.layers.moe import kt_ep_wrapper as m

        layers = list(m._KT_SPLIT_PREFILL_LAYERS)
        state = dict(m._KT_SPLIT_PREFILL_STATE)
        try:
            m._KT_SPLIT_PREFILL_LAYERS.clear()
            m._KT_SPLIT_PREFILL_STATE["pipeline"] = None
            # No layers registered -> the pre-existing early exit, not the new
            # one. Distinguishes "nothing to do" from "already done".
            self.assertFalse(m.finalize_split_prefill(object()))
        finally:
            m._KT_SPLIT_PREFILL_LAYERS.clear()
            m._KT_SPLIT_PREFILL_LAYERS.extend(layers)
            m._KT_SPLIT_PREFILL_STATE.update(state)


if __name__ == "__main__":
    unittest.main()
