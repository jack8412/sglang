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
    def test_second_finalize_does_not_rebuild_the_pipeline(self):
        """Red if the idempotence guard is dropped.

        The second call would fall through to the whole cold-source build.
        That used to mean 439 GB of pinned store; the store is gone, but the
        rebuild is still wrong -- it re-registers kt's arena and reallocates
        the pipeline's device buffers underneath a serving model. The guard is
        asserted by making the builder explode if it is reached at all.
        """
        from sglang.srt.layers.moe import kt_ep_wrapper as m

        layers = list(m._KT_SPLIT_PREFILL_LAYERS)
        state = dict(m._KT_SPLIT_PREFILL_STATE)
        try:
            # Armed state: layers registered and a pipeline already built.
            m._KT_SPLIT_PREFILL_LAYERS.clear()
            m._KT_SPLIT_PREFILL_LAYERS.append((0, object()))
            m._KT_SPLIT_PREFILL_STATE["pipeline"] = object()

            with patch.object(
                m,
                "_build_dynamic_swizzle_plan",
                side_effect=AssertionError("rebuilt the cold source"),
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


class TestDraftWorkerDoesNotBuildTheStore(CustomTestCase):
    """Bug regression: the draft ModelRunner must not run split-prefill finalize.

    maybe_init_split_prefill gated only on get_exec().moe.kt_expert_split_prefill,
    which is a PROCESS-GLOBAL snapshot -- so it read True inside the draft
    worker, which shares the process with the target and whose layers are still
    registered. The draft therefore built a second complete pinned cold store:
    51.1 GiB per rank, 439 GB across TP8, on top of the first.

    Tested here rather than on hardware because the trigger needs a draft model
    (a full DSpark checkpoint) that only one of our hosts has, while the defect
    itself is one branch."""

    def _runner(self, *, is_draft):
        from sglang.srt.model_executor.model_runner import ModelRunner

        class _Stub:
            is_draft_worker = is_draft
            server_args = object()

        return ModelRunner.maybe_init_split_prefill, _Stub()

    def test_draft_worker_skips_finalize(self):
        from sglang.srt.layers.moe import kt_ep_wrapper

        fn, stub = self._runner(is_draft=True)
        with patch.object(
            kt_ep_wrapper,
            "finalize_split_prefill",
            side_effect=AssertionError("draft built the cold store"),
        ):
            fn(stub)  # must return before touching finalize at all

    def test_target_worker_still_finalizes(self):
        """The guard must not disable split prefill outright -- red if someone
        'fixes' the draft path by skipping finalize for everyone, which boots
        fine and silently serves the margin-routed path instead."""
        from sglang.srt.layers.moe import kt_ep_wrapper
        from sglang.srt.model_executor import model_runner as mr

        called = []
        fn, stub = self._runner(is_draft=False)

        class _Moe:
            kt_expert_split_prefill = True

        class _Exec:
            moe = _Moe()

        with patch.object(mr, "get_exec", return_value=_Exec()), patch.object(
            kt_ep_wrapper, "finalize_split_prefill", side_effect=lambda sa: called.append(sa)
        ):
            fn(stub)
        self.assertEqual(len(called), 1)


if __name__ == "__main__":
    unittest.main()
