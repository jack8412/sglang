"""Unit tests for the MXFP4 layerwise-prefill slot machinery in
srt/layers/moe/kt_ep_wrapper.py — no server, no CUDA.

Ported from kvcache-ai/sglang test/unit/test_mxfp4_layerwise_prefill.py
(ktfork #72/#73, commit b31c4dfeb1) onto this branch's layout:

- the module under test imports cleanly on CPU, so the fork's stub-loader is
  replaced by a plain import of ``sglang.srt.layers.moe.kt_ep_wrapper``;
- TP topology comes from ``get_parallel().override`` (runtime context), not
  the fork's patched ``get_tensor_model_parallel_world_size``;
- the fork's model_runner/memory_profiler reservation plumbing was replaced
  here by the KVCacheConfigurator hook; those assertions are rewritten
  against ``mxfp4_layerwise_prefill_reservation_gib``.

Function-level imports that need CUDA/triton at import time
(``v4_marlin_moe``) or runtime state (``expert_distribution``,
``token_dispatcher``) are stubbed per-test via ``mock.patch.dict``.
"""

import inspect
import sys
import types
import unittest
from collections import namedtuple
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.moe import kt_ep_wrapper as ktw
from sglang.srt.runtime_context import get_parallel, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

# Stand-in for the TP process-group descriptor returned by
# ``sglang.srt.distributed.get_tp_group`` (patched on the module under test
# in the consensus/transport cases; identity asserts pin the plane choice).
_TP_GROUP = SimpleNamespace(cpu_group=object(), device_group=object(), first_rank=0)


def _clear_mxfp4_registries():
    ktw._MXFP4_PREFILL_LAYER_REGISTRY.clear()
    ktw._MXFP4_LAYERWISE_MANAGERS.clear()
    ktw._MXFP4_LAYERWISE_DISABLED_REASONS.clear()


def _module(name, **attributes):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    return module


def _runtime_stubs(
    standard_combine_input=None,
    prepare_marlin=None,
    allocate_marlin=None,
    marlin_storage_nbytes=None,
):
    """sys.modules stubs for kt_ep_wrapper's function-level imports.

    ``v4_marlin_moe`` imports triton and ``sglang.kernels`` ops at module
    scope, so the real module cannot import on CPU CI.  Applied per-test via
    ``mock.patch.dict`` (never at module level).
    """
    stubs = {
        "sglang.srt.eplb.expert_distribution": _module(
            "sglang.srt.eplb.expert_distribution",
            get_global_expert_distribution_recorder=lambda: mock.Mock(),
        ),
        "sglang.srt.layers.quantization.v4_marlin_moe": _module(
            "sglang.srt.layers.quantization.v4_marlin_moe",
            prepare_v4_mxfp4_marlin=prepare_marlin or (lambda *_args, **_kwargs: None),
            allocate_v4_mxfp4_marlin=allocate_marlin or (lambda **_kwargs: object()),
            get_v4_mxfp4_marlin_storage_nbytes=(
                marlin_storage_nbytes or (lambda **_kwargs: 0)
            ),
        ),
    }
    if standard_combine_input is not None:
        stubs["sglang.srt.layers.moe.token_dispatcher"] = _module(
            "sglang.srt.layers.moe.token_dispatcher",
            StandardCombineInput=standard_combine_input,
        )
    return stubs


def _dummy_server_args(**fields):
    """Dummy-boundary ServerArgs: __post_init__ early-returns for
    model_path="dummy", so the defaulted kt_* fields stay assignable."""
    server_args = ServerArgs(model_path="dummy")
    for name, value in fields.items():
        setattr(server_args, name, value)
    return server_args


class _RecordingEvent:
    def __init__(self, name, log):
        self.name = name
        self.log = log

    def record(self, stream=None):
        self.log.append(("record", self.name, getattr(stream, "name", stream)))

    def synchronize(self):
        self.log.append(("synchronize", self.name))


class _RecordingStream:
    def __init__(self, name, log):
        self.name = name
        self.log = log

    def wait_event(self, event):
        self.log.append(("wait_event", self.name, event.name))

    def synchronize(self):
        self.log.append(("synchronize", self.name))


class _StateSlot:
    def __init__(self, index, log):
        self.index = index
        self.state = "EMPTY"
        self.layer_idx = None
        self.epoch = -1
        self.has_consumed_event = False
        self.ready_event = _RecordingEvent(f"ready{index}", log)
        self.consumed_event = _RecordingEvent(f"consumed{index}", log)

    def invalidate(self):
        self.state = "EMPTY"
        self.layer_idx = None
        self.epoch = -1


class _BoolValue:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _HostBuffer:
    def __init__(self, name, numel, element_size):
        self.name = name
        self._numel = numel
        self._element_size = element_size

    def numel(self):
        return self._numel

    def element_size(self):
        return self._element_size

    def __getitem__(self, slot):
        return (self.name, slot)


class _DestinationRow:
    def __init__(self, name, expert_id, log):
        self.name = name
        self.expert_id = expert_id
        self.log = log

    def copy_(self, source, non_blocking=False):
        self.log.append(("copy", self.name, self.expert_id, source, non_blocking))


class _Destination:
    def __init__(self, name, log):
        self.name = name
        self.log = log

    def __getitem__(self, expert_id):
        return _DestinationRow(self.name, expert_id, self.log)


@contextmanager
def _stream_context(stream, log):
    log.append(("enter_stream", stream.name))
    try:
        yield
    finally:
        log.append(("exit_stream", stream.name))


class TestMxfp4LayerwiseStateMachine(CustomTestCase):
    def setUp(self):
        _clear_mxfp4_registries()

    def tearDown(self):
        _clear_mxfp4_registries()
        reset_context()

    def test_sparse_successors_are_sorted_by_registered_layer_id(self):
        signature = ("cuda:0", "sparse")
        ktw._MXFP4_PREFILL_LAYER_REGISTRY[signature] = {
            41: (object(), object()),
            3: (object(), object()),
            17: (object(), object()),
        }
        manager = object.__new__(ktw._Mxfp4LayerwisePrefillManager)
        manager.signature = signature

        self.assertEqual(manager.layer_order, [3, 17, 41])
        self.assertEqual(manager.successor_layer_idx(3), 17)
        self.assertEqual(manager.successor_layer_idx(4), 17)
        self.assertEqual(manager.successor_layer_idx(17), 41)
        self.assertIsNone(manager.successor_layer_idx(41))

    def test_two_slots_alternate_and_previous_epoch_is_never_reused(self):
        signature = ("cuda:0", "epochs")
        log = []
        methods = {
            layer_idx: SimpleNamespace(
                kt_config=SimpleNamespace(layer_idx=layer_idx), tp_rank=1
            )
            for layer_idx in (3, 17, 41)
        }
        layers = {layer_idx: object() for layer_idx in methods}
        ktw._MXFP4_PREFILL_LAYER_REGISTRY[signature] = {
            layer_idx: (methods[layer_idx], layers[layer_idx]) for layer_idx in methods
        }

        gpu_method = SimpleNamespace(
            apply=lambda _layer, dispatch: (
                log.append(("compute", dispatch)) or f"result-{dispatch}"
            )
        )
        manager = object.__new__(ktw._Mxfp4LayerwisePrefillManager)
        manager.signature = signature
        manager.context = SimpleNamespace(gpu_method=gpu_method, gpu_layer=object())
        manager.slots = (_StateSlot(0, log), _StateSlot(1, log))
        manager.device = torch.device("cpu")
        manager.epoch = -1
        manager.last_layer_position = None
        manager.current_slot_index = None
        manager.round_active = False

        def load_slot(slot, layer_idx, _method, _layer):
            log.append(("load", layer_idx, slot.index, manager.epoch))
            slot.state = "READY"
            slot.layer_idx = layer_idx
            slot.epoch = manager.epoch

        current_stream = _RecordingStream("main", log)
        with (
            mock.patch.object(manager, "_load_slot", side_effect=load_slot),
            mock.patch.object(
                manager,
                "_bind_slot",
                side_effect=lambda slot: log.append(("bind", slot.index)),
            ),
            mock.patch.object(
                manager, "_record_prepared_backing_on_stream", return_value=None
            ),
            mock.patch.object(
                torch.cuda, "current_stream", return_value=current_stream
            ),
            mock.patch.object(torch.cuda, "synchronize") as global_synchronize,
        ):
            used_slots = []
            for layer_idx in (3, 17, 41):
                result = manager.apply(
                    methods[layer_idx], layers[layer_idx], f"layer-{layer_idx}"
                )
                self.assertEqual(result, f"result-layer-{layer_idx}")
                used_slots.append(manager.current_slot_index)

            self.assertEqual(used_slots, [0, 1, 0])
            self.assertEqual(
                [entry[:3] for entry in log if entry[0] == "load"],
                [("load", 3, 0), ("load", 17, 1), ("load", 41, 0)],
            )
            self.assertLess(
                log.index(("wait_event", "main", "ready0")),
                log.index(("bind", 0)),
            )
            self.assertLess(log.index(("bind", 0)), log.index(("compute", "layer-3")))
            self.assertLess(
                log.index(("compute", "layer-3")),
                log.index(("record", "consumed0", "main")),
            )
            self.assertLess(
                log.index(("compute", "layer-3")), log.index(("load", 17, 1, 0))
            )

            # A correct layer tag from the prior round is still stale.  Starting
            # again at the first sparse layer must advance the epoch and reload.
            manager.slots[1].state = "READY"
            manager.slots[1].layer_idx = 3
            manager.slots[1].epoch = 0
            manager.apply(methods[3], layers[3], "layer-3-round-2")
            self.assertEqual(manager.epoch, 1)
            self.assertIn(("load", 3, 1, 1), log)
            global_synchronize.assert_not_called()

    def test_manager_uses_dedicated_device_control_stream(self):
        manager_source = inspect.getsource(ktw._Mxfp4LayerwisePrefillManager)

        self.assertNotIn("torch.cuda.synchronize", manager_source)
        self.assertIn("self.control_stream", manager_source)
        self.assertIn("get_tp_group().device_group", manager_source)

    def test_lazy_manager_initialization_runs_once_per_signature(self):
        signature = ("cuda:0", "model-a")
        method = SimpleNamespace(_mxfp4_pipeline_signature=signature)
        layer = object()
        manager = object()

        def initialize(initialized_method, initialized_layer):
            self.assertIs(initialized_method, method)
            self.assertIs(initialized_layer, layer)
            ktw._MXFP4_LAYERWISE_MANAGERS[signature] = manager

        with mock.patch.object(
            ktw,
            "_initialize_mxfp4_layerwise_pipeline",
            side_effect=initialize,
        ) as initialize_mock:
            self.assertIs(
                ktw._get_or_initialize_mxfp4_layerwise_manager(method, layer),
                manager,
            )
            self.assertIs(
                ktw._get_or_initialize_mxfp4_layerwise_manager(method, layer),
                manager,
            )

        initialize_mock.assert_called_once_with(method, layer)

    def test_compute_exception_records_consumed_fence_before_consensus(self):
        log = []
        manager = object.__new__(ktw._Mxfp4LayerwisePrefillManager)
        manager.device = torch.device("cpu")
        manager.epoch = 4
        manager.context = SimpleNamespace(
            gpu_layer=object(),
            gpu_method=SimpleNamespace(
                apply=lambda *_args: (_ for _ in ()).throw(RuntimeError("launch"))
            ),
        )
        slot = SimpleNamespace(
            index=0,
            layer_idx=17,
            epoch=4,
            ready_event=_RecordingEvent("ready", log),
            consumed_event=_RecordingEvent("consumed", log),
            has_consumed_event=False,
            reuse_guard="ready",
            state="READY",
        )
        main_stream = _RecordingStream("main", log)
        method = SimpleNamespace(kt_config=SimpleNamespace(layer_idx=17), tp_rank=0)

        with (
            mock.patch.object(manager, "_acquire", return_value=(slot, True)),
            mock.patch.object(manager, "_bind_slot"),
            mock.patch.object(manager, "_record_prepared_backing_on_stream"),
            mock.patch.object(manager, "_prefetch_successor") as prefetch,
            mock.patch.object(torch.cuda, "current_stream", return_value=main_stream),
            mock.patch.object(ktw.dist, "is_initialized", return_value=False),
        ):
            with self.assertRaisesRegex(RuntimeError, "compute launch"):
                manager.apply(method, object(), object())

        self.assertIn(("record", "consumed", "main"), log)
        self.assertEqual(slot.reuse_guard, "consumed")
        prefetch.assert_not_called()


class TestMxfp4ReservationBudget(CustomTestCase):
    """KV-budget accounting for the lazily-allocated slots.

    On this branch the reservation is consumed by
    ``KVCacheConfigurator._profile_available_bytes`` through
    ``mxfp4_layerwise_prefill_reservation_gib`` (replaces the ktfork
    model_runner field + memory_profiler plumbing)."""

    def setUp(self):
        _clear_mxfp4_registries()

    def tearDown(self):
        _clear_mxfp4_registries()
        reset_context()

    # Per raw slot for (num_experts=2, hidden=64, intermediate=32):
    # 4096 + 2048 + 1024 + 512 bytes.  The pipeline has two raw slots plus
    # two prepared slots (123 bytes each via the stubbed storage helper).
    _EXPECTED_BYTES = 2 * (4096 + 2048 + 1024 + 512 + 123)

    def _register_layer(self, signature):
        # gpu_method carries the class name the layout classifier keys on
        # (marlin-prepared DSV4 path).
        method = SimpleNamespace(
            _full_init_args=(64, 32, torch.bfloat16),
            global_num_experts=2,
            gpu_method=DeepSeekMxfp4MoEMethod(None),
        )
        layer = object()
        ktw._MXFP4_PREFILL_LAYER_REGISTRY[signature] = {0: (method, layer)}

    def _reservation_patches(self):
        return (
            mock.patch.object(
                ktw, "_mxfp4_pipeline_runtime_supported", return_value=True
            ),
            mock.patch.dict(
                sys.modules,
                _runtime_stubs(marlin_storage_nbytes=lambda **_kwargs: 123),
            ),
        )

    def test_lazy_slots_reserve_kv_budget_without_creating_a_manager(self):
        signature = ("cuda:0", "model-a")
        self._register_layer(signature)

        supported, stubs = self._reservation_patches()
        with supported, stubs:
            self.assertEqual(
                ktw.get_mxfp4_layerwise_prefill_reservation_bytes(),
                self._EXPECTED_BYTES,
            )

        self.assertNotIn(signature, ktw._MXFP4_LAYERWISE_MANAGERS)

    def test_reservation_gib_gate(self):
        """Rewritten from ktfork's
        test_model_runner_defers_slot_allocation_until_threshold_request:
        the deferred-allocation budget now rides the KVCacheConfigurator
        hook.  Gate off -> 0.0 (either kt_method != MXFP4 or no positive
        threshold); gate on -> exactly bytes / 2**30, case-insensitively."""
        signature = ("cuda:0", "model-a")
        self._register_layer(signature)

        # Gate off: never touches the registry byte math.
        self.assertEqual(
            ktw.mxfp4_layerwise_prefill_reservation_gib(
                _dummy_server_args(kt_gpu_prefill_token_threshold=1024)
            ),
            0.0,
        )
        self.assertEqual(
            ktw.mxfp4_layerwise_prefill_reservation_gib(
                _dummy_server_args(kt_method="MXFP4")
            ),
            0.0,
        )
        self.assertEqual(
            ktw.mxfp4_layerwise_prefill_reservation_gib(
                _dummy_server_args(
                    kt_method="MXFP4", kt_gpu_prefill_token_threshold=0
                )
            ),
            0.0,
        )

        supported, stubs = self._reservation_patches()
        with supported, stubs:
            self.assertEqual(
                ktw.mxfp4_layerwise_prefill_reservation_gib(
                    _dummy_server_args(
                        kt_method="mxfp4", kt_gpu_prefill_token_threshold=1024
                    )
                ),
                self._EXPECTED_BYTES / 2**30,
            )

        self.assertNotIn(signature, ktw._MXFP4_LAYERWISE_MANAGERS)

    def test_reservation_gib_is_zero_with_empty_registry(self):
        # Gate on but nothing registered (e.g. non-KT model): the hook must
        # be a safe 0.0, not an error, since KVCacheConfigurator always runs.
        self.assertEqual(
            ktw.mxfp4_layerwise_prefill_reservation_gib(
                _dummy_server_args(
                    kt_method="MXFP4", kt_gpu_prefill_token_threshold=1024
                )
            ),
            0.0,
        )


class TestMxfp4HostTransport(CustomTestCase):
    def setUp(self):
        _clear_mxfp4_registries()

    def tearDown(self):
        _clear_mxfp4_registries()
        reset_context()

    def test_shm_uuid_broadcast_uses_tp_global_first_rank(self):
        source = inspect.getsource(ktw.SharedFullContext._create_cpu_buffers)

        self.assertIn("src=get_tp_group().first_rank", source)
        self.assertIn("ptr > 0", source)
        self.assertIn("int(register_result) != 0", source)

    def test_failed_shm_phase_aborts_every_rank_and_cleans_up(self):
        context = object.__new__(ktw.SharedFullContext)
        context._cleanup_cpu_buffers_after_failure = mock.Mock()

        with mock.patch.object(
            ktw, "_all_tp_ranks_succeeded", return_value=False
        ):
            with self.assertRaisesRegex(RuntimeError, "allocation failed"):
                context._commit_cpu_buffer_phase(None, "allocation")

        context._cleanup_cpu_buffers_after_failure.assert_called_once_with()

    def test_postprocess_exception_still_records_ready_fence(self):
        log = []
        manager = object.__new__(ktw._Mxfp4LayerwisePrefillManager)
        manager.postprocess_stream = _RecordingStream("postprocess", log)
        manager.prepared_layout = ktw._MXFP4_LAYOUT_MARLIN
        slot = SimpleNamespace(
            raw_ready_event=_RecordingEvent("raw_ready", log),
            ready_event=_RecordingEvent("ready", log),
            reuse_guard="raw",
            w13_weight=object(),
            w13_weight_scale_inv=object(),
            w2_weight=object(),
            w2_weight_scale_inv=object(),
            prepared=object(),
        )

        def fail_prepare(*_args, **_kwargs):
            raise RuntimeError("marlin prepare")

        with (
            mock.patch.dict(sys.modules, _runtime_stubs(prepare_marlin=fail_prepare)),
            mock.patch.object(
                torch.cuda,
                "stream",
                side_effect=lambda stream: _stream_context(stream, log),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "marlin prepare"):
                manager._postprocess_slot(slot, cpu_expert_ids=[])

        self.assertEqual(slot.reuse_guard, "ready")
        self.assertIn(("record", "ready", "postprocess"), log)

    def test_setup_exception_preserves_prior_consumed_fence(self):
        log = []
        manager, slot = self._make_manager_and_slot(log)
        manager.context.cpu_buffers = {}
        slot.reuse_guard = "consumed"
        method = SimpleNamespace(
            tp_rank=0,
            gpu_experts_mask=[_BoolValue(False) for _ in range(3)],
        )

        with (
            mock.patch.object(
                torch.cuda,
                "stream",
                side_effect=lambda stream: _stream_context(stream, log),
            ),
            mock.patch.object(ktw.dist, "is_initialized", return_value=False),
            mock.patch.object(ktw.os, "sched_getaffinity", return_value=set()),
        ):
            with self.assertRaisesRegex(RuntimeError, "transport setup"):
                manager._load_slot(slot, 17, method, object())

        wait_pos = log.index(("wait_event", "transfer", "consumed"))
        record_pos = log.index(("record", "raw_ready", "transfer"))
        self.assertLess(wait_pos, record_pos)
        self.assertEqual(slot.reuse_guard, "raw")

    def test_exception_fence_record_failure_synchronizes_current_generation(self):
        log = []
        manager, slot = self._make_manager_and_slot(log)
        manager.context.cpu_buffers = {}
        slot.reuse_guard = "ready"
        slot.raw_ready_event.record = mock.Mock(
            side_effect=RuntimeError("event record failed")
        )
        method = SimpleNamespace(
            tp_rank=0,
            gpu_experts_mask=[_BoolValue(False) for _ in range(3)],
        )

        with (
            mock.patch.object(
                torch.cuda,
                "stream",
                side_effect=lambda stream: _stream_context(stream, log),
            ),
            mock.patch.object(ktw.dist, "is_initialized", return_value=False),
            mock.patch.object(ktw.os, "sched_getaffinity", return_value=set()),
        ):
            with self.assertRaisesRegex(RuntimeError, "transport setup"):
                manager._load_slot(slot, 17, method, object())

        self.assertIn(("wait_event", "transfer", "ready"), log)
        self.assertIn(("synchronize", "transfer"), log)
        self.assertEqual(slot.reuse_guard, "synchronized")

    def test_peer_h2d_failure_keeps_local_host_slot_fenced(self):
        log = []
        manager, slot = self._make_manager_and_slot(log)
        slot.num_experts = 2
        # Simulate aborting an unconsumed prefetch, then reusing its slot.  A
        # failure in this new generation must not reuse the old "ready" guard.
        slot.reuse_guard = "ready"
        method = SimpleNamespace(
            tp_rank=1,
            wrapper=None,
            gpu_experts_mask=[_BoolValue(False), _BoolValue(False)],
            logical_to_gpu_index=None,
        )
        failing_destination = mock.MagicMock()
        failing_row = mock.Mock()
        failing_row.copy_.side_effect = RuntimeError("local H2D failure")
        failing_destination.__getitem__.return_value = failing_row
        setattr(
            slot,
            ktw._MXFP4_RAW_NAMES_BY_LAYOUT[ktw._MXFP4_LAYOUT_MARLIN][0],
            failing_destination,
        )

        def commit_device_phase(error, phase):
            log.append(("device_commit", phase, error))
            if error is not None:
                raise RuntimeError("peer H2D failure")

        with (
            mock.patch.object(
                manager,
                "_commit_tp_device_runtime_phase",
                side_effect=commit_device_phase,
            ),
            mock.patch.object(
                torch.cuda,
                "stream",
                side_effect=lambda stream: _stream_context(stream, log),
            ),
            mock.patch.object(ktw.os, "sched_getaffinity", return_value=set()),
            mock.patch.object(ktw.os, "sched_setaffinity"),
        ):
            with self.assertRaisesRegex(RuntimeError, "peer H2D failure"):
                manager._load_slot(slot, 17, method, object())

        record_pos = log.index(("record", "host0_free", "transfer"))
        consensus_pos = next(
            index
            for index, entry in enumerate(log)
            if entry[0] == "device_commit" and entry[2] is not None
        )
        self.assertLess(record_pos, consensus_pos)
        self.assertTrue(manager.host_slot_was_used[0])
        self.assertIn(("wait_event", "transfer", "ready"), log)
        self.assertIn(("record", "raw_ready", "transfer"), log)
        self.assertEqual(slot.reuse_guard, "raw")

    def test_bind_slot_selects_persistent_marlin_prepared_storage(self):
        prepared = object()
        layer = SimpleNamespace(_v4_tk_path=True)
        manager = object.__new__(ktw._Mxfp4LayerwisePrefillManager)
        manager.context = SimpleNamespace(gpu_layer=layer)
        manager.prepared_layout = ktw._MXFP4_LAYOUT_MARLIN

        manager._bind_slot(SimpleNamespace(prepared=prepared))

        self.assertIs(layer._v4_marlin_weights, prepared)
        self.assertTrue(layer._v4_marlin_path)
        self.assertFalse(layer._v4_tk_path)

    def _make_manager_and_slot(self, log):
        names = ktw._MXFP4_RAW_NAMES_BY_LAYOUT[ktw._MXFP4_LAYOUT_MARLIN]
        host_buffers = {
            names[0]: _HostBuffer(names[0], 8, 1),
            names[1]: _HostBuffer(names[1], 12, 2),
            names[2]: _HostBuffer(names[2], 4, 4),
            names[3]: _HostBuffer(names[3], 20, 2),
        }
        context = SimpleNamespace(
            cpu_buffers=host_buffers,
            all_rank_buffer_ptrs={
                name: [1000 + 100 * index, 2000 + 100 * index]
                for index, name in enumerate(names)
            },
        )
        manager = object.__new__(ktw._Mxfp4LayerwisePrefillManager)
        manager.context = context
        manager.prepared_layout = ktw._MXFP4_LAYOUT_MARLIN
        manager.raw_names = names
        manager.epoch = 7
        manager.transfer_stream = _RecordingStream("transfer", log)
        manager.postprocess_stream = _RecordingStream("postprocess", log)
        manager.host_slot_free_events = (
            _RecordingEvent("host0_free", log),
            _RecordingEvent("host1_free", log),
        )
        manager.host_slot_was_used = [False, False]

        slot = SimpleNamespace(
            index=0,
            state="EMPTY",
            layer_idx=None,
            epoch=-1,
            num_experts=3,
            has_consumed_event=True,
            consumed_event=_RecordingEvent("consumed", log),
            raw_ready_event=_RecordingEvent("raw_ready", log),
            ready_event=_RecordingEvent("ready", log),
            invalidate=lambda: None,
        )
        for name in names:
            setattr(slot, name, _Destination(name, log))
        slot.prepared = object()
        return manager, slot

    def _run_transport(self, tp_rank, reuse_guard="consumed"):
        log = []
        manager, slot = self._make_manager_and_slot(log)
        slot.reuse_guard = reuse_guard
        wrapper = mock.Mock()
        wrapper.submit_write_weight_scale_to_buffer.side_effect = lambda *_args: (
            log.append(("submit", _args[1]))
        )
        wrapper.sync_write_weight_scale_to_buffer.side_effect = lambda: log.append(
            ("sync_write",)
        )
        method = SimpleNamespace(
            tp_rank=tp_rank,
            wrapper=wrapper,
            gpu_experts_mask=[_BoolValue(False) for _ in range(3)],
            logical_to_gpu_index=None,
        )
        cpu_group = _TP_GROUP.cpu_group

        def all_reduce(_status, *, op, group):
            log.append(("all_reduce", op, group))

        def commit_device_phase(error, phase):
            log.append(("device_commit", phase, error))
            if error is not None:
                raise RuntimeError(phase) from error

        stream_ctor = mock.Mock()
        event_ctor = mock.Mock()
        with (
            mock.patch.dict(sys.modules, _runtime_stubs()),
            mock.patch.object(
                torch.cuda,
                "stream",
                side_effect=lambda stream: _stream_context(stream, log),
            ),
            mock.patch.object(torch.cuda, "Stream", stream_ctor),
            mock.patch.object(torch.cuda, "Event", event_ctor),
            mock.patch.object(torch.cuda, "synchronize") as global_synchronize,
            mock.patch.object(ktw.dist, "is_initialized", return_value=True),
            mock.patch.object(
                ktw.dist, "all_reduce", side_effect=all_reduce
            ) as dist_all_reduce,
            mock.patch.object(
                manager,
                "_commit_tp_device_runtime_phase",
                side_effect=commit_device_phase,
            ),
            mock.patch.object(ktw, "get_tp_group", return_value=_TP_GROUP),
            get_parallel().override(tp_rank=tp_rank, tp_size=2),
            mock.patch.object(ktw.os, "sched_getaffinity", return_value=set()),
            mock.patch.object(ktw.os, "sched_setaffinity"),
        ):
            manager._load_slot(slot, 17, method, object())

        self.assertEqual(slot.state, "READY")
        self.assertEqual(slot.layer_idx, 17)
        self.assertEqual(slot.epoch, 7)
        self.assertEqual(dist_all_reduce.call_count, 4)
        self.assertTrue(
            all(
                call.kwargs["group"] is cpu_group
                for call in dist_all_reduce.call_args_list
            )
        )
        self.assertEqual(len([entry for entry in log if entry[0] == "copy"]), 3 * 4)
        expected_guard = "ready" if reuse_guard == "ready" else "consumed"
        self.assertIn(("wait_event", "transfer", expected_guard), log)
        self.assertLess(
            log.index(("record", "raw_ready", "transfer")),
            log.index(("wait_event", "postprocess", "raw_ready")),
        )
        self.assertLess(
            log.index(("wait_event", "postprocess", "raw_ready")),
            log.index(("record", "ready", "postprocess")),
        )
        # Host slot 0 is reused by expert 2.  Its local DMA completion and the
        # all-rank free fence both precede TP0's overwrite.
        sync_position = log.index(("synchronize", "host0_free"))
        next_consensus = next(
            index
            for index, entry in enumerate(log[sync_position + 1 :], sync_position + 1)
            if entry[0] == "device_commit"
        )
        expert_two_submit = log.index(("submit", 2)) if tp_rank == 0 else len(log)
        self.assertLess(sync_position, next_consensus)
        self.assertLess(next_consensus, expert_two_submit)
        stream_ctor.assert_not_called()
        event_ctor.assert_not_called()
        global_synchronize.assert_not_called()
        return manager, wrapper, log

    def test_tp0_uses_cpu_group_and_host_dtype_pointer_offsets(self):
        manager, wrapper, _ = self._run_transport(tp_rank=0)

        self.assertEqual(wrapper.submit_write_weight_scale_to_buffer.call_count, 3)
        self.assertEqual(wrapper.sync_write_weight_scale_to_buffer.call_count, 3)
        second = wrapper.submit_write_weight_scale_to_buffer.call_args_list[1].args
        self.assertEqual(second[:2], (2, 1))
        expected_offsets = {
            name: buffer.numel() // 2 * buffer.element_size()
            for name, buffer in manager.context.cpu_buffers.items()
        }
        for position, name in enumerate(
            ktw._MXFP4_RAW_NAMES_BY_LAYOUT[ktw._MXFP4_LAYOUT_MARLIN], 2
        ):
            bases = manager.context.all_rank_buffer_ptrs[name]
            self.assertEqual(
                second[position], [base + expected_offsets[name] for base in bases]
            )

    def test_nonzero_tp_rank_only_fences_and_copies(self):
        _, wrapper, log = self._run_transport(tp_rank=1)

        wrapper.submit_write_weight_scale_to_buffer.assert_not_called()
        wrapper.sync_write_weight_scale_to_buffer.assert_not_called()
        self.assertEqual(len([entry for entry in log if entry[0] == "all_reduce"]), 4)
        self.assertEqual(
            len([entry for entry in log if entry[0] == "device_commit"]), 6
        )
        self.assertEqual(len([entry for entry in log if entry[0] == "copy"]), 12)

    def test_unconsumed_prefetch_waits_ready_before_slot_overwrite(self):
        _, _, log = self._run_transport(tp_rank=1, reuse_guard="ready")

        self.assertLess(
            log.index(("wait_event", "transfer", "ready")),
            next(index for index, entry in enumerate(log) if entry[0] == "copy"),
        )

    def test_tp_oom_consensus_uses_cpu_group(self):
        cpu_group = _TP_GROUP.cpu_group

        def remote_rank_failed(status, **_kwargs):
            status.zero_()

        with (
            mock.patch.object(ktw.dist, "is_initialized", return_value=True),
            mock.patch.object(
                ktw.dist, "all_reduce", side_effect=remote_rank_failed
            ) as all_reduce,
            mock.patch.object(ktw, "get_tp_group", return_value=_TP_GROUP),
            get_parallel().override(tp_rank=0, tp_size=2),
        ):
            self.assertFalse(ktw._all_tp_ranks_succeeded(True))

        self.assertIs(all_reduce.call_args.kwargs["group"], cpu_group)
        self.assertIs(all_reduce.call_args.kwargs["op"], ktw.dist.ReduceOp.MIN)

    def test_hot_transport_consensus_uses_device_group_and_control_stream(self):
        log = []
        manager = object.__new__(ktw._Mxfp4LayerwisePrefillManager)
        manager.control_stream = _RecordingStream("control", log)

        class Status:
            def __init__(self):
                self.value = 1

            def fill_(self, value):
                self.value = value
                log.append(("fill", value))

            def item(self):
                log.append(("item", self.value))
                return self.value

        manager.device_phase_status = Status()
        device_group = _TP_GROUP.device_group

        def remote_rank_failed(status, **_kwargs):
            log.append(("all_reduce", _kwargs["group"]))
            status.value = 0

        with (
            mock.patch.object(ktw.dist, "is_initialized", return_value=True),
            mock.patch.object(ktw, "get_tp_group", return_value=_TP_GROUP),
            get_parallel().override(tp_rank=0, tp_size=2),
            mock.patch.object(
                torch.cuda,
                "stream",
                side_effect=lambda stream: _stream_context(stream, log),
            ),
            mock.patch.object(
                ktw.dist, "all_reduce", side_effect=remote_rank_failed
            ) as all_reduce,
        ):
            self.assertFalse(manager._tp_device_phase_succeeded(True))

        self.assertIs(all_reduce.call_args.kwargs["group"], device_group)
        self.assertIn(("enter_stream", "control"), log)
        self.assertLess(log.index(("fill", 1)), log.index(("item", 0)))
        self.assertLess(log.index(("item", 0)), log.index(("exit_stream", "control")))


class _StandardCombineInput:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


_TopkOutput = namedtuple(
    "_TopkOutput", ["topk_weights", "topk_ids", "token_expert_indices"]
)
_DispatchOutput = namedtuple("_DispatchOutput", ["hidden_states", "topk_output"])


class DeepSeekMxfp4MoEMethod:
    """Test double named after the production wrapped-GPU method."""

    def __init__(self, output):
        self.output = output
        self.calls = []

    def apply(self, layer, dispatch_output):
        self.calls.append((layer, dispatch_output))
        return _StandardCombineInput(self.output)


class _SerialGpuMethod:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def apply(self, layer, dispatch_output):
        self.calls.append((layer, dispatch_output))
        return self.result


class TestMxfp4ApplyFallbacks(CustomTestCase):
    def setUp(self):
        _clear_mxfp4_registries()

    def tearDown(self):
        _clear_mxfp4_registries()
        reset_context()

    def _make_wrapper(self, threshold=4):
        wrapper = object.__new__(ktw.KTEPWrapperMethod)
        wrapper.tp_rank = 1
        wrapper.kt_config = SimpleNamespace(
            layer_idx=17,
            method="MXFP4",
            kt_enable_dynamic_expert_update=False,
        )
        wrapper.gpu_prefill_token_threshold = threshold
        wrapper._mxfp4_pipeline_signature = ("cuda:0", "apply")
        wrapper.gpu_experts_mask_cuda = torch.tensor([True])
        wrapper.logical_to_gpu_index_cuda = torch.tensor([0], dtype=torch.int32)
        wrapper.gpu_experts_mask = torch.tensor([True])
        wrapper.num_gpu_experts = 1
        wrapper._cpu_stream = None
        # This branch resolves the KT debug env vars once in __init__ into
        # instance flags; object.__new__ bypasses that, so pin the defaults.
        wrapper._kt_debug_timing = False
        wrapper._kt_debug_timing_deep = False
        wrapper._kt_no_cpu_stream = False
        wrapper._kt_bypass_gpu_moe = False
        wrapper.gpu_method = DeepSeekMxfp4MoEMethod(torch.full((8, 2), 9.0))
        return wrapper

    @staticmethod
    def _layer():
        return SimpleNamespace(
            w13_weight=object(),
            w13_weight_scale_inv=object(),
            w2_weight=object(),
            w2_weight_scale_inv=object(),
            _v4_tk_path=True,
        )

    @staticmethod
    def _dispatch(num_tokens):
        hidden = torch.zeros((num_tokens, 2))
        topk = _TopkOutput(
            topk_weights=torch.ones((num_tokens, 1)),
            topk_ids=torch.zeros((num_tokens, 1), dtype=torch.long),
            token_expert_indices=None,
        )
        return _DispatchOutput(hidden_states=hidden, topk_output=topk)

    def _apply(self, wrapper, layer, dispatch, runtime_supported):
        with (
            mock.patch.dict(
                sys.modules,
                _runtime_stubs(standard_combine_input=_StandardCombineInput),
            ),
            mock.patch.object(
                ktw,
                "_mxfp4_pipeline_runtime_supported",
                return_value=runtime_supported,
            ),
            mock.patch.object(
                ktw,
                "mask_and_remap_expert_ids",
                side_effect=lambda ids, *_args: ids,
            ),
            mock.patch.object(torch.cuda, "is_available", return_value=False),
        ):
            return wrapper.apply(layer, dispatch)

    def test_threshold_routes_pipeline_and_below_threshold_aborts_round(self):
        wrapper = self._make_wrapper()
        layer = self._layer()
        manager = mock.Mock()
        manager.apply.return_value = "pipeline-result"
        ktw._MXFP4_LAYERWISE_MANAGERS[wrapper._mxfp4_pipeline_signature] = manager

        result = self._apply(wrapper, layer, self._dispatch(4), True)
        self.assertEqual(result, "pipeline-result")
        manager.apply.assert_called_once()

        result = self._apply(wrapper, layer, self._dispatch(3), True)
        manager.abort_round.assert_called_once_with()
        self.assertTrue(torch.equal(result.hidden_states, torch.full((8, 2), 9.0)))

    def test_first_threshold_request_lazily_initializes_and_uses_manager(self):
        wrapper = self._make_wrapper()
        layer = self._layer()
        manager = mock.Mock()
        manager.apply.return_value = "pipeline-result"
        signature = wrapper._mxfp4_pipeline_signature

        def initialize(initialized_method, initialized_layer):
            self.assertIs(initialized_method, wrapper)
            self.assertIs(initialized_layer, layer)
            ktw._MXFP4_LAYERWISE_MANAGERS[signature] = manager

        with mock.patch.object(
            ktw,
            "_initialize_mxfp4_layerwise_pipeline",
            side_effect=initialize,
        ) as initialize_mock:
            first = self._apply(wrapper, layer, self._dispatch(4), True)
            second = self._apply(wrapper, layer, self._dispatch(4), True)

        self.assertEqual(first, "pipeline-result")
        self.assertEqual(second, "pipeline-result")
        initialize_mock.assert_called_once_with(wrapper, layer)
        self.assertEqual(manager.apply.call_count, 2)

    def test_below_threshold_request_does_not_initialize_slots(self):
        wrapper = self._make_wrapper()
        layer = self._layer()

        with mock.patch.object(
            ktw, "_initialize_mxfp4_layerwise_pipeline"
        ) as initialize_mock:
            result = self._apply(wrapper, layer, self._dispatch(3), True)

        initialize_mock.assert_not_called()
        self.assertTrue(torch.equal(result.hidden_states, torch.full((8, 2), 9.0)))

    def test_unsupported_backend_keeps_serial_full_gpu_fallback(self):
        wrapper = self._make_wrapper()
        layer = self._layer()
        serial_result = object()
        serial_gpu = _SerialGpuMethod(serial_result)
        context = SimpleNamespace(
            gpu_method=serial_gpu,
            gpu_layer=object(),
            _is_mxfp4_quant=True,
        )
        wrapper._build_full_context = mock.Mock(return_value=context)

        result = self._apply(wrapper, layer, self._dispatch(4), False)

        self.assertIs(result, serial_result)
        wrapper._build_full_context.assert_called_once_with(layer)
        self.assertEqual(len(serial_gpu.calls), 1)
        self.assertEqual(wrapper.gpu_method.calls, [])

    def test_slot_oom_disables_pipeline_and_falls_through_to_hybrid(self):
        wrapper = self._make_wrapper()
        layer = self._layer()
        signature = wrapper._mxfp4_pipeline_signature
        ktw._MXFP4_LAYERWISE_DISABLED_REASONS[signature] = "slot 1 OOM"
        wrapper._build_full_context = mock.Mock()

        result = self._apply(wrapper, layer, self._dispatch(4), True)

        wrapper._build_full_context.assert_not_called()
        self.assertTrue(torch.equal(result.hidden_states, torch.full((8, 2), 9.0)))
        self.assertEqual(len(wrapper.gpu_method.calls), 1)

    def test_lazy_slot_oom_falls_through_to_hybrid_for_triggering_request(self):
        wrapper = self._make_wrapper()
        layer = self._layer()
        signature = wrapper._mxfp4_pipeline_signature
        wrapper._build_full_context = mock.Mock()

        def initialize(_method, _layer):
            ktw._MXFP4_LAYERWISE_DISABLED_REASONS[signature] = "slot OOM"

        with mock.patch.object(
            ktw,
            "_initialize_mxfp4_layerwise_pipeline",
            side_effect=initialize,
        ) as initialize_mock:
            result = self._apply(wrapper, layer, self._dispatch(4), True)

        initialize_mock.assert_called_once_with(wrapper, layer)
        self.assertIn(signature, ktw._MXFP4_LAYERWISE_DISABLED_REASONS)
        wrapper._build_full_context.assert_not_called()
        self.assertTrue(torch.equal(result.hidden_states, torch.full((8, 2), 9.0)))
        self.assertEqual(len(wrapper.gpu_method.calls), 1)

    def test_pipeline_runtime_errors_fail_loud(self):
        wrapper = self._make_wrapper()
        layer = self._layer()
        manager = mock.Mock()
        manager.apply.side_effect = RuntimeError("transport failed")
        ktw._MXFP4_LAYERWISE_MANAGERS[wrapper._mxfp4_pipeline_signature] = manager

        with self.assertRaisesRegex(RuntimeError, "transport failed"):
            self._apply(wrapper, layer, self._dispatch(4), True)

    def test_second_slot_allocation_oom_is_persistently_disabled(self):
        wrapper = self._make_wrapper()
        wrapper._full_init_args = (2, 2, torch.float32)
        wrapper.global_num_experts = 1
        wrapper.moe_runner_config = object()
        layer = self._layer()
        raw = {
            name: torch.empty((1, 1, 1))
            for name in ktw._MXFP4_RAW_NAMES_BY_LAYOUT[ktw._MXFP4_LAYOUT_MARLIN]
        }
        gpu_layer = SimpleNamespace(
            **{name: SimpleNamespace(data=tensor) for name, tensor in raw.items()}
        )
        context = SimpleNamespace(
            gpu_layer=gpu_layer,
            _is_mxfp4_quant=True,
            mxfp4_prepared_layout=ktw._MXFP4_LAYOUT_MARLIN,
            initialize_cpu_buffers=mock.Mock(),
        )

        with (
            mock.patch.dict(sys.modules, _runtime_stubs()),
            mock.patch.object(
                ktw, "_mxfp4_pipeline_backend_supported", return_value=True
            ),
            mock.patch.object(
                ktw,
                "_mxfp4_pipeline_signature",
                return_value=wrapper._mxfp4_pipeline_signature,
            ),
            mock.patch.object(ktw, "SharedFullContext", return_value=context),
            mock.patch.object(
                torch, "empty_like", side_effect=torch.cuda.OutOfMemoryError("slot 1")
            ),
            mock.patch.object(ktw.dist, "is_initialized", return_value=False),
            mock.patch.object(torch.cuda, "is_available", return_value=False),
            mock.patch.object(ktw.gc, "collect"),
            # _disable_mxfp4_layerwise_pipeline logs on the TP0 rank read
            # from the runtime parallel context.
            get_parallel().override(tp_rank=0, tp_size=1),
        ):
            ktw._initialize_mxfp4_layerwise_pipeline(wrapper, layer)

        signature = wrapper._mxfp4_pipeline_signature
        self.assertIn(signature, ktw._MXFP4_LAYERWISE_DISABLED_REASONS)
        self.assertNotIn(signature, ktw._MXFP4_LAYERWISE_MANAGERS)
        context.initialize_cpu_buffers.assert_not_called()


if __name__ == "__main__":
    unittest.main()
