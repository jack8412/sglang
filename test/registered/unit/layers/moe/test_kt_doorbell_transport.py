"""Doorbell-transport tests (srt/layers/moe/kt_ep_wrapper).

Covers the CPU-provable requirements from SPEC-DOORBELL-TRANSPORT.md: the
arm/ring/wait constants, per-(layer, batch size) slot assignment, the
capture_bs gate that keeps the poller off buffers that get recycled, and the
config rails. The GPU-side gates (byte identity vs hostnode, nats, the
poller's own protocol under real graph replay) live in db1_protocol_gate.py
and Phase DB on the node.

The constants matter more than they look. A node captured in a CUDA graph
writes a CONSTANT, so the arm is not redundant bookkeeping: without it every
replay after the first finds the completion word already holding the value it
waits for, sails through, and reads a stale output -- the CPU experts appear
to work while contributing nothing but the previous step's data. Nothing about
that is visible in a smoke test, which is why it is pinned here.
"""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.moe import kt_ep_wrapper as ktw
from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_SUCCESS = "CUDA_SUCCESS"
_EQ = "CU_STREAM_WAIT_VALUE_EQ"


class _FakeDoorbellExt:
    """Stands in for kt_kernel_ext.doorbell with a recognisable address map."""

    RING = 0x1000
    COMP_BASE = 0x2000
    STRIDE = 128

    def __init__(self):
        self.calls = []

    def init(self, num_slots, num_pollers):
        self.calls.append(("init", num_slots, num_pollers))

    def start(self):
        self.calls.append(("start",))

    def ring_dev_addr(self):
        return self.RING

    def completion_dev_addr(self, slot):
        return self.COMP_BASE + slot * self.STRIDE

    def served(self):
        return 7

    def spins(self):
        return 11

    def unbound(self):
        return 0

    def work_ns_total(self):
        return 7000

    def work_ns_max(self):
        return 2500


class _FakeDriver:
    """Records the memops the transport records into the stream, in order."""

    def __init__(self):
        self.ops = []

    class CUresult:
        CUDA_SUCCESS = _SUCCESS

    class CUstreamWaitValue_flags:
        CU_STREAM_WAIT_VALUE_EQ = _EQ

    @staticmethod
    def CUdeviceptr(x):
        return x

    def cuStreamWriteValue64(self, stream, ptr, value, flags):
        self.ops.append(("write", stream, ptr, value))
        return (_SUCCESS,)

    def cuStreamWaitValue64(self, stream, ptr, value, flags):
        self.ops.append(("wait", stream, ptr, value, flags))
        return (_SUCCESS,)


def _install(ext, driver):
    """Intercept the real imports rather than rebinding module attributes.

    kt_ep_wrapper imports both lazily inside the helpers, so seeding
    sys.modules exercises the same import path production takes; rebinding
    ktw._kt_doorbell_ext would stop intercepting the moment that helper is
    inlined.
    """
    kt_pkg = types.ModuleType("kt_kernel")
    kt_ext = types.ModuleType("kt_kernel.kt_kernel_ext")
    kt_ext.doorbell = ext
    kt_pkg.kt_kernel_ext = kt_ext
    cuda_pkg = types.ModuleType("cuda")
    bindings = types.ModuleType("cuda.bindings")
    bindings.driver = driver
    cuda_pkg.bindings = bindings
    return patch.dict(
        sys.modules,
        {
            "kt_kernel": kt_pkg,
            "kt_kernel.kt_kernel_ext": kt_ext,
            "cuda": cuda_pkg,
            "cuda.bindings": bindings,
        },
    )


class _FakeWrapper:
    def __init__(self, capture_bs):
        self._capture_bs = capture_bs
        self.registered = []

    def get_capture_batch_sizes(self):
        return self._capture_bs

    def register_doorbell_slot(self, slot, staging_buffer, topk_ids):
        self.registered.append((slot, staging_buffer.shape[0]))


def _method(capture_bs=(1, 2, 4, 8)):
    m = MagicMock(spec=[])
    m._db_slots = {}
    m.wrapper = _FakeWrapper(list(capture_bs))
    return m


def _dispatch(batch_size, k=16):
    d = MagicMock(spec=["topk_output"])
    ids = torch.zeros(batch_size, k, dtype=torch.long)
    d.topk_output = (torch.zeros(batch_size, k), ids, None)
    return d


class _DoorbellTestCase(CustomTestCase):
    def setUp(self):
        super().setUp()
        self._saved = dict(ktw._KT_DOORBELL)
        ktw._KT_DOORBELL.update({"inited": False, "next_slot": 0})

    def tearDown(self):
        ktw._KT_DOORBELL.clear()
        ktw._KT_DOORBELL.update(self._saved)
        super().tearDown()


class TestProtocolConstants(_DoorbellTestCase):
    """The arm/ring/wait values, pinned. Red if the arm is dropped as
    'redundant', if the ring stops carrying slot+1 (the poller reads the slot
    index OUT of the ring word -- a bare flag would make it serve slot 0), or
    if the wait loosens from EQ in a way that lets a stale completion through."""

    def test_arm_writes_zero_to_this_slots_completion(self):
        ext, drv = _FakeDoorbellExt(), _FakeDriver()
        with _install(ext, drv):
            ktw.kt_doorbell_arm(3, stream=99)
        self.assertEqual(
            drv.ops, [("write", 99, ext.completion_dev_addr(3), 0)]
        )

    def test_ring_writes_slot_plus_one_to_the_global_ring(self):
        ext, drv = _FakeDoorbellExt(), _FakeDriver()
        with _install(ext, drv):
            ktw.kt_doorbell_ring(3, stream=99)
        self.assertEqual(drv.ops, [("write", 99, ext.RING, 4)])

    def test_wait_is_equality_on_slot_plus_one(self):
        ext, drv = _FakeDoorbellExt(), _FakeDriver()
        with _install(ext, drv):
            ktw.kt_doorbell_wait(3, stream=99)
        self.assertEqual(
            drv.ops, [("wait", 99, ext.completion_dev_addr(3), 4, _EQ)]
        )

    def test_arm_and_wait_address_the_same_word(self):
        # The arm retracts exactly what the wait tests. If these ever drift
        # apart the wait is satisfied by a word nobody clears.
        ext, drv = _FakeDoorbellExt(), _FakeDriver()
        with _install(ext, drv):
            ktw.kt_doorbell_arm(5, stream=1)
            ktw.kt_doorbell_wait(5, stream=1)
        self.assertEqual(drv.ops[0][2], drv.ops[1][2])

    def test_distinct_slots_use_distinct_completion_words(self):
        ext, drv = _FakeDoorbellExt(), _FakeDriver()
        with _install(ext, drv):
            ktw.kt_doorbell_arm(0, stream=1)
            ktw.kt_doorbell_arm(1, stream=1)
        self.assertNotEqual(drv.ops[0][2], drv.ops[1][2])

    def test_driver_error_is_raised_not_swallowed(self):
        ext, drv = _FakeDoorbellExt(), _FakeDriver()
        drv.cuStreamWriteValue64 = lambda *a: ("CUDA_ERROR_INVALID_VALUE",)
        with _install(ext, drv):
            with self.assertRaises(RuntimeError):
                ktw.kt_doorbell_ring(0, stream=1)


class TestSlotAssignment(_DoorbellTestCase):
    """One slot per (layer, BATCH SIZE). KExpertsCPUBuffer keys its rings by
    batch size, so a per-layer slot would point the poller at another tier's
    buffers for every size but the first -- silently, the shapes being
    identical. Red if slots collapse back to one per layer."""

    def _slot(self, method, bs):
        with _install(_FakeDoorbellExt(), _FakeDriver()):
            return KTEPWrapperMethod._kt_doorbell_slot(
                method, torch.zeros(bs, 8), _dispatch(bs)
            )

    def test_same_batch_size_reuses_its_slot(self):
        m = _method()
        self.assertEqual(self._slot(m, 4), self._slot(m, 4))
        self.assertEqual(len(m.wrapper.registered), 1)

    def test_different_batch_sizes_get_different_slots(self):
        m = _method()
        self.assertNotEqual(self._slot(m, 2), self._slot(m, 8))
        self.assertEqual(
            sorted(bs for _, bs in m.wrapper.registered), [2, 8]
        )

    def test_different_layers_get_different_slots(self):
        a, b = _method(), _method()
        self.assertNotEqual(self._slot(a, 4), self._slot(b, 4))

    def test_registration_carries_this_tiers_batch_size(self):
        m = _method()
        self._slot(m, 8)
        self.assertEqual(m.wrapper.registered[0][1], 8)

    def test_slot_exhaustion_raises(self):
        m = _method()
        ktw._KT_DOORBELL["next_slot"] = ktw._KT_DOORBELL_MAX_SLOTS
        with self.assertRaises(RuntimeError) as cm:
            self._slot(m, 4)
        self.assertIn("out of slots", str(cm.exception))


class TestCaptureBsGate(_DoorbellTestCase):
    """Only batch sizes kt CACHES may be bound. The poller holds raw pointers
    into a size's rings for the life of the process, but KExpertsCPUBuffer
    only keeps a tuple alive for sizes in capture_bs; every other size shares
    one temp_buffer that the next differently-sized forward replaces. Binding
    one would leave the poller reading freed memory. Red if the gate is
    dropped -- the symptom would be use-after-free under mixed prefill
    shapes, not a clean failure."""

    def _slot(self, method, bs):
        with _install(_FakeDoorbellExt(), _FakeDriver()):
            return KTEPWrapperMethod._kt_doorbell_slot(
                method, torch.zeros(bs, 8), _dispatch(bs)
            )

    def test_uncaptured_batch_size_returns_none(self):
        m = _method(capture_bs=(1, 2, 4))
        self.assertIsNone(self._slot(m, 4096))

    def test_uncaptured_batch_size_binds_nothing(self):
        m = _method(capture_bs=(1, 2, 4))
        self._slot(m, 4096)
        self.assertEqual(m.wrapper.registered, [])
        self.assertEqual(ktw._KT_DOORBELL["next_slot"], 0)

    def test_captured_batch_size_binds(self):
        m = _method(capture_bs=(1, 2, 4))
        self.assertEqual(self._slot(m, 2), 0)
        self.assertEqual(len(m.wrapper.registered), 1)


class TestDoorbellInit(_DoorbellTestCase):
    """init() must run once, before capture: cudaHostAlloc is illegal during
    capture and the graph bakes this page's addresses."""

    def test_init_is_idempotent(self):
        ext = _FakeDoorbellExt()
        with _install(ext, _FakeDriver()):
            ktw.kt_doorbell_init(1)
            ktw.kt_doorbell_init(1)
        self.assertEqual([c[0] for c in ext.calls], ["init", "start"])

    def test_init_sizes_for_every_layer_and_tier(self):
        ext = _FakeDoorbellExt()
        with _install(ext, _FakeDriver()):
            ktw.kt_doorbell_init(1)
        self.assertEqual(ext.calls[0][1], ktw._KT_DOORBELL_MAX_SLOTS)
        # 92 MoE layers x the captured decode tiers must fit with room over.
        self.assertGreaterEqual(ktw._KT_DOORBELL_MAX_SLOTS, 92 * 52)

    def test_stats_report_the_poller(self):
        with _install(_FakeDoorbellExt(), _FakeDriver()):
            st = ktw.kt_doorbell_stats()
        self.assertEqual(st["served"], 7)
        self.assertEqual(st["unbound"], 0)
        self.assertAlmostEqual(st["work_us_mean"], 1.0)
        self.assertAlmostEqual(st["work_us_max"], 2.5)


class TestTransportConfigRails(CustomTestCase):
    """Config-time rails. Each of these is a combination that fails silently
    or hangs at serving time, so it is refused where the operator can still
    read the reason."""

    def _args(self, **kw):
        from sglang.srt.server_args import ServerArgs

        base = dict(model_path="/dummy", kt_weight_path="/dummy", kt_method="MXFP4")
        base.update(kw)
        return ServerArgs(**base)

    def test_pdmux_is_refused(self):
        # One global ring word is only safe while at most one doorbell is
        # outstanding; pdmux runs prefill and decode concurrently, so a second
        # ring can overwrite an unconsumed one and its wait never completes.
        with self.assertRaises(ValueError) as cm:
            self._args(kt_transport="doorbell", enable_pdmux=True)
        self.assertIn("pdmux", str(cm.exception))

    def test_deferral_is_refused(self):
        # The deferred half lands in the successor layer's ring and is
        # collected by its sync, but the doorbell binds one closure with
        # incremental=False and never enqueues a second task.
        with self.assertRaises(ValueError) as cm:
            self._args(kt_transport="doorbell", kt_max_deferred_experts_per_token=2)
        self.assertIn("deferred", str(cm.exception))

    def test_non_mxfp4_is_refused(self):
        with self.assertRaises(ValueError) as cm:
            self._args(kt_transport="doorbell", kt_method="AMXINT4")
        self.assertIn("MXFP4", str(cm.exception))

    def test_hostnode_is_the_default(self):
        # The transport is opt-in until the node gates say otherwise.
        self.assertEqual(self._args().kt_transport, "hostnode")

    def test_doorbell_on_the_production_recipe_is_accepted(self):
        args = self._args(
            kt_transport="doorbell",
            kt_routing_margin=0.5,
            kt_expert_swap_transitions=5,
            kt_expert_swap_max=8,
        )
        self.assertEqual(args.kt_transport, "doorbell")


if __name__ == "__main__":
    unittest.main()
