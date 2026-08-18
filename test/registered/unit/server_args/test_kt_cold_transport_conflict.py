# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""`--kt-cold-transport pinned-store` cannot be combined with cold-only residency.

The pinned store is a second copy of the cold set, built in RESIDENT layout out
of kt's CPU buffers. Under `--kt-cold-only-cpu-experts` only the cold experts
have CPU buffers at all, so there is nothing to build it from and the cold set
streams out of kt's arena instead.

The combination used to be accepted and silently ignored: the launch line named
a transport, every log line named a different one ("cold source kt-arena ... no
pinned store built"), and the flag sat in the campaign's serve scripts for
several runs doing nothing. A flag that quietly does nothing is worse than one
that refuses -- it survives review, gets copied forward, and is eventually
quoted as if it configured something.

    python -m pytest test/registered/unit/server_args/test_kt_cold_transport_conflict.py -v
"""

import unittest

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestKtColdTransportConflict(CustomTestCase):
    """Exercises the real _handle_kt, not a restatement of its condition."""

    def _run(self, transport, cold_only):
        args = ServerArgs.__new__(ServerArgs)
        # Only what _handle_kt reads before the cold-only block. kt_weight_path
        # must be set or the handler returns immediately.
        args.kt_weight_path = "/nonexistent"
        args.kt_gpu_experts_ratio = None
        args.kt_num_gpu_experts = 0
        args.kt_routing_margin = None
        args.kt_gpu_prefill_token_threshold = None
        args.kt_max_deferred_experts_per_token = None
        args.kt_method = "MXFP4"
        args.kt_cold_only_cpu_experts = cold_only
        args.kt_cold_transport = transport
        args._handle_kt()

    def test_pinned_store_with_cold_only_is_refused(self):
        with self.assertRaises(ValueError) as cm:
            self._run("pinned-store", True)
        self.assertIn("pinned-store", str(cm.exception))
        self.assertIn("cold-only", str(cm.exception))

    def test_other_transports_are_unaffected_by_cold_only(self):
        """The refusal must be specific to the combination, not to cold-only."""
        for transport in ("ring-export", "direct-dma"):
            try:
                self._run(transport, True)
            except ValueError as exc:
                self.assertNotIn(
                    "pinned-store", str(exc), f"{transport} hit the wrong guard"
                )
            except Exception:
                pass  # later kt checks (wheel/env) are not what this pins


if __name__ == "__main__":
    unittest.main()
