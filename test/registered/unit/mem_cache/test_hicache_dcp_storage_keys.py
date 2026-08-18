# SPDX-License-Identifier: Apache-2.0
"""Storage keys must name the shard, not just the model, once DCP is on.

Under plain TP an MLA KV cache is REPLICATED: every rank holds the same bytes,
so one rank writes and the key carries no rank at all. Under DCP that premise
inverts -- each rank owns a disjoint set of the tokens inside every page -- and
an unscoped key means eight ranks racing to overwrite one object with eight
different payloads. The reader then gets whichever rank wrote last, at the right
length, with no error anywhere.

The suffix also carries the SIZE, not just the rank. A shard is only meaningful
against the world size it was cut for, so a store written at --dcp-size 8 must
MISS on a --dcp-size 4 server rather than return a wrong-shaped tensor.

This mirrors what NSA context parallel already does one axis up (`_cp{rank}_{size}`).
"""

import os
import tempfile
import unittest

from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _config(**kw) -> HiCacheStorageConfig:
    base = dict(
        tp_rank=0,
        tp_size=8,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=True,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="kimi/k3",
    )
    base.update(kw)
    return HiCacheStorageConfig(**base)


class TestDcpStorageKeys(CustomTestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self._prev = os.environ.get("SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR")
        os.environ["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] = self._dir.name

        def _restore():
            if self._prev is None:
                os.environ.pop("SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR", None)
            else:
                os.environ["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] = self._prev

        self.addCleanup(_restore)

    def _suffix(self, **kw) -> str:
        return HiCacheFile(_config(**kw)).config_suffix

    def test_no_dcp_key_is_unchanged(self):
        """The whole non-DCP path must stay byte-identical."""
        self.assertEqual(self._suffix(), "_kimi-k3")
        self.assertEqual(self._suffix(dcp_size=1, dcp_rank=0), "_kimi-k3")

    def test_dcp_ranks_get_distinct_keys(self):
        suffixes = {r: self._suffix(dcp_rank=r, dcp_size=8) for r in range(8)}
        self.assertEqual(len(set(suffixes.values())), 8, suffixes)
        self.assertEqual(suffixes[0], "_kimi-k3_dcp0_8")
        self.assertEqual(suffixes[7], "_kimi-k3_dcp7_8")

    def test_a_different_world_size_is_a_different_key(self):
        """Cross-topology reuse must MISS, not silently return foreign bytes."""
        self.assertNotEqual(
            self._suffix(dcp_rank=0, dcp_size=8),
            self._suffix(dcp_rank=0, dcp_size=4),
        )

    def test_dcp_scope_is_independent_of_tp_rank(self):
        """At tp=8/dcp=4 there are two complete replica sets.

        They hold the same shard and should share one object, so the key must
        NOT pick up tp_rank -- otherwise the store doubles for no benefit and
        the two halves can never hit each other's writes.
        """
        self.assertEqual(
            self._suffix(tp_rank=0, dcp_rank=2, dcp_size=4),
            self._suffix(tp_rank=4, dcp_rank=2, dcp_size=4),
        )

    def test_non_mla_keeps_its_tp_scope_and_adds_dcp(self):
        suffix = self._suffix(is_mla_model=False, tp_rank=3, dcp_rank=5, dcp_size=8)
        self.assertIn("_3_8", suffix)
        self.assertIn("_dcp5_8", suffix)

    def test_every_rank_can_create_the_directory(self):
        """Backups now come from all ranks, so mkdir cannot be rank 0's alone."""
        target = os.path.join(self._dir.name, "nested")
        os.environ["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] = target
        for rank in (3, 5):
            HiCacheFile(_config(tp_rank=rank, dcp_rank=rank, dcp_size=8))
        self.assertTrue(os.path.isdir(target))


if __name__ == "__main__":
    unittest.main()
