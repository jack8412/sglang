"""HiCache under decode context parallelism (DCP): host-pool index math.

Under DCP the radix/controller layer works in a widened logical index space
(page_size * dcp_size wide pages, dcp_size * physical capacity), while each
rank's device and host buffers only materialize the owned 1/dcp_size token
shard (owner rule: index % dcp_size == dcp_rank, physical row = index //
dcp_size — the same rule the device-side KV write and page-table kernels
use). These tests cover the translation helper, the logical/physical host
pool sizing, and that the transfer entry points hand *physical* rows to the
kernels.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

DCP_SIZE = 8
PHYSICAL_PAGE = 64
WIDENED_PAGE = PHYSICAL_PAGE * DCP_SIZE


def _fake_mla_device_pool(size: int = 1024) -> SimpleNamespace:
    return SimpleNamespace(
        size=size,
        store_dtype=torch.float16,
        kv_lora_rank=8,
        qk_rope_head_dim=4,
        layer_num=2,
        start_layer=0,
        end_layer=1,
        device="cpu",
        layers_to_capture=None,
        layer_shard_enabled=False,
    )


def _make_host_pool(dcp_rank: int, device_size: int = 1024) -> MLATokenToKVPoolHost:
    return MLATokenToKVPoolHost(
        _fake_mla_device_pool(device_size),
        host_to_device_ratio=2.0,
        host_size=0,
        page_size=WIDENED_PAGE,
        layout="layer_first",
        pin_memory=False,
        device="cpu",
        dcp_size=DCP_SIZE,
        dcp_rank=dcp_rank,
    )


class TestDcpKernelIndices(CustomTestCase):
    def _bare_pool(self, dcp_size: int, dcp_rank: int) -> MLATokenToKVPoolHost:
        pool = MLATokenToKVPoolHost.__new__(MLATokenToKVPoolHost)
        pool.dcp_size = dcp_size
        pool.dcp_rank = dcp_rank
        return pool

    def test_identity_without_dcp(self):
        pool = self._bare_pool(1, 0)
        indices = torch.arange(37)
        self.assertIs(pool.dcp_kernel_indices(indices), indices)

    def test_aligned_page_translates_to_full_physical_page(self):
        # One widened page starting at logical 512 covers physical rows
        # 64..127 on every rank.
        indices = torch.arange(WIDENED_PAGE, 2 * WIDENED_PAGE)
        for rank in range(DCP_SIZE):
            pool = self._bare_pool(DCP_SIZE, rank)
            out = pool.dcp_kernel_indices(indices)
            torch.testing.assert_close(
                out, torch.arange(PHYSICAL_PAGE, 2 * PHYSICAL_PAGE)
            )

    def test_matches_owner_rule_on_merged_unordered_pages(self):
        # Concatenation of non-adjacent widened pages in arbitrary order, as
        # produced by merged CacheOperations after allocator churn.
        pages = [3, 0, 5]
        indices = torch.cat(
            [torch.arange(p * WIDENED_PAGE, (p + 1) * WIDENED_PAGE) for p in pages]
        )
        for rank in range(DCP_SIZE):
            pool = self._bare_pool(DCP_SIZE, rank)
            out = pool.dcp_kernel_indices(indices)
            expected = (
                indices[indices % DCP_SIZE == rank] // DCP_SIZE
            )  # owner rule, same as filter_dcp_local_kv_indices
            torch.testing.assert_close(out, expected)
            self.assertEqual(out.numel() * DCP_SIZE, indices.numel())

    def test_ragged_run_is_rejected(self):
        pool = self._bare_pool(DCP_SIZE, 0)
        with self.assertRaises(AssertionError):
            pool.dcp_kernel_indices(torch.arange(WIDENED_PAGE + 1))

    def test_positional_residue_pairing_survives_host_sort(self):
        # move_indices (direct/layer_first) sorts host indices and permutes
        # device indices to match. Independent residue filtering of both
        # tensors must keep the same token positions on every rank.
        g = torch.Generator().manual_seed(0)
        host_pages = [7, 2]
        device_pages = [1, 4]
        host = torch.cat(
            [torch.arange(p * WIDENED_PAGE, (p + 1) * WIDENED_PAGE) for p in host_pages]
        )
        device = torch.cat(
            [
                torch.arange(p * WIDENED_PAGE, (p + 1) * WIDENED_PAGE)
                for p in device_pages
            ]
        )
        # token identity: position i pairs host[i] <-> device[i]
        perm = torch.randperm(host.numel(), generator=g)
        # sort host as move_indices does, permuting device alongside
        host_sorted, order = host[perm].sort()
        device_matched = device[perm][order]
        for rank in range(DCP_SIZE):
            pool = self._bare_pool(DCP_SIZE, rank)
            host_mask = host_sorted % DCP_SIZE == rank
            device_mask = device_matched % DCP_SIZE == rank
            # same positions selected on both sides -> pairing preserved
            torch.testing.assert_close(host_mask, device_mask)
            self.assertEqual(
                pool.dcp_kernel_indices(host_sorted).numel(),
                host.numel() // DCP_SIZE,
            )


class TestHostPoolSizingUnderDcp(CustomTestCase):
    def test_logical_and_physical_sizing(self):
        pool = _make_host_pool(dcp_rank=3)
        # kernel-facing page is physical
        self.assertEqual(pool.page_size, PHYSICAL_PAGE)
        self.assertEqual(pool.logical_page_size, WIDENED_PAGE)
        # physical rows = ratio * device physical size, page aligned
        self.assertEqual(pool.size, pool.page_num * PHYSICAL_PAGE)
        self.assertEqual(pool.logical_size, pool.size * DCP_SIZE)
        # buffers materialize physical rows only
        self.assertEqual(pool.kv_buffer.shape[1], pool.size)
        # allocator surface is logical
        self.assertEqual(pool.free_slots.numel(), pool.logical_size)
        self.assertEqual(pool.mem_state.numel(), pool.logical_size)

    def test_alloc_is_widened_page_granular(self):
        pool = _make_host_pool(dcp_rank=0)
        out = pool.alloc(WIDENED_PAGE)
        self.assertEqual(out.numel(), WIDENED_PAGE)
        with self.assertRaises(AssertionError):
            pool.alloc(PHYSICAL_PAGE)  # not a multiple of the widened page

    def test_non_dcp_pool_unchanged(self):
        pool = MLATokenToKVPoolHost(
            _fake_mla_device_pool(),
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=PHYSICAL_PAGE,
            layout="layer_first",
            pin_memory=False,
            device="cpu",
        )
        self.assertEqual(pool.page_size, PHYSICAL_PAGE)
        self.assertEqual(pool.logical_size, pool.size)
        self.assertEqual(pool.logical_page_size, PHYSICAL_PAGE)


class TestTransferEntryPointsTranslate(CustomTestCase):
    def _run_backup(self, pool, host_indices, device_indices):
        device_pool = SimpleNamespace(
            data_ptrs=torch.zeros(2, dtype=torch.uint64),
            kv_buffer=[torch.zeros(1)] * 2,
        )
        # create=True: mla.py imports the kernel only under `if _is_cuda or
        # _is_hip`, so the name is absent on the CPU runner this test targets.
        with mock.patch(
            "sglang.srt.mem_cache.pool_host.mla.transfer_kv_all_layer_mla",
            create=True,
        ) as kernel:
            pool.can_use_jit = False
            pool.can_use_write_back_jit = False
            with mock.patch.object(
                MLATokenToKVPoolHost, "_is_device_layer_sharded", return_value=False
            ):
                pool.backup_from_device_all_layer(
                    device_pool, host_indices, device_indices, io_backend="kernel"
                )
            return kernel.call_args.kwargs

    def test_backup_receives_physical_rows(self):
        pool = _make_host_pool(dcp_rank=5)
        logical = torch.arange(2 * WIDENED_PAGE)
        kwargs = self._run_backup(pool, logical, logical.clone())
        expected = torch.arange(2 * PHYSICAL_PAGE)
        torch.testing.assert_close(kwargs["src_indices"], expected)
        torch.testing.assert_close(kwargs["dst_indices"], expected)


class TestL3PageTranslationUnderDcp(CustomTestCase):
    """The L3 storage entry points, which used to be refused outright.

    These three methods are where DCP does its damage, and the damage is
    asymmetric: the LENGTHS were always right (page_size is physical), only the
    OFFSETS were logical. That is exactly why a small smoke test passed while
    the bytes went to the wrong rows -- tensor slicing clamps, so a wrong index
    writes plausible-looking garbage and returns normally.
    """

    def test_data_page_offset_is_translated_to_a_physical_row(self):
        pool = _make_host_pool(dcp_rank=3)
        # Second widened page: logical WIDENED_PAGE -> physical PHYSICAL_PAGE.
        page = pool.get_data_page(WIDENED_PAGE, flat=False)
        expected = pool.kv_buffer[:, PHYSICAL_PAGE : 2 * PHYSICAL_PAGE, :, :]
        self.assertEqual(page.shape, expected.shape)
        self.assertEqual(page.data_ptr(), expected.data_ptr())

    def test_page_length_stays_physical(self):
        """The 8x bug: a page must be one RANK's shard, not the whole page."""
        pool = _make_host_pool(dcp_rank=0)
        flat = pool.get_data_page(0)
        self.assertEqual(
            flat.numel(), pool.layer_num * PHYSICAL_PAGE * pool.kv_cache_dim
        )
        self.assertEqual(flat.numel(), pool.get_dummy_flat_data_page().numel())

    def test_set_from_flat_data_page_round_trips_at_the_owned_row(self):
        pool = _make_host_pool(dcp_rank=6)
        pool.kv_buffer.zero_()
        payload = torch.arange(
            pool.layer_num * PHYSICAL_PAGE * pool.kv_cache_dim, dtype=pool.dtype
        )
        pool.set_from_flat_data_page(2 * WIDENED_PAGE, payload)
        torch.testing.assert_close(pool.get_data_page(2 * WIDENED_PAGE), payload)
        # and nothing outside that physical page moved
        self.assertEqual(
            int(pool.kv_buffer[:, : 2 * PHYSICAL_PAGE, :, :].abs().sum()), 0
        )

    def test_every_rank_reads_its_own_buffer_at_the_same_row(self):
        """Disjointness is across buffers, not within one.

        A widened page's slot s belongs to rank s % dcp_size, and each rank
        stores it at the SAME physical row -- the shards are disjoint because
        the buffers are per-rank, which is precisely why the storage KEYS have
        to carry dcp_rank or all eight ranks overwrite one object.
        """
        rows = {r: _make_host_pool(dcp_rank=r) for r in range(DCP_SIZE)}
        offsets = {
            r: pool.get_data_page(3 * WIDENED_PAGE, flat=False).data_ptr()
            - pool.kv_buffer.data_ptr()
            for r, pool in rows.items()
        }
        self.assertEqual(len(set(offsets.values())), 1, offsets)

    def test_unaligned_index_is_refused_rather_than_translated(self):
        pool = _make_host_pool(dcp_rank=1)
        with self.assertRaises(ValueError):
            pool.get_data_page(WIDENED_PAGE + 1)
        with self.assertRaises(ValueError):
            pool.set_from_flat_data_page(1, pool.get_dummy_flat_data_page())

    def test_without_dcp_offsets_are_untouched(self):
        """dcp_size == 1 must be bit-for-bit the old behaviour."""
        pool = MLATokenToKVPoolHost(
            _fake_mla_device_pool(1024),
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=PHYSICAL_PAGE,
            layout="layer_first",
            pin_memory=False,
            device="cpu",
        )
        self.assertEqual(pool.dcp_size, 1)
        for index in (0, 1, PHYSICAL_PAGE, PHYSICAL_PAGE + 7):
            self.assertEqual(pool.dcp_page_row(index), index)
        page = pool.get_data_page(PHYSICAL_PAGE, flat=False)
        expected = pool.kv_buffer[:, PHYSICAL_PAGE : 2 * PHYSICAL_PAGE, :, :]
        self.assertEqual(page.data_ptr(), expected.data_ptr())


if __name__ == "__main__":
    unittest.main()
