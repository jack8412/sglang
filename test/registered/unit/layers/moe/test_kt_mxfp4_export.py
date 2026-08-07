"""Unit tests for srt/layers/moe/kt_mxfp4_export.py and the F1 helpers it
feeds in srt/layers/moe/kt_ep_wrapper.py — no server, no CUDA, no flashinfer.

CPU stand-ins (semantics defined here; every permutation is an affine map
``i -> (mult * i + off) % n`` with ``mult`` coprime to ``n``, so bijectivity
and non-identity are provable by modular arithmetic instead of relying on a
seeded randperm):

- ``flashinfer.nvfp4_block_scale_interleave`` -> ``_mock_interleave``: the
  affine byte permutation ``i -> (5 i + 3) % numel`` on the flattened input,
  reshaped back.  Non-identity and not self-inverse, so a dropped, doubled,
  or misplaced interleave call cannot cancel out; inverted in-test via
  ``argsort``.
- ``mxfp4._get_flashinfer_mxfp4_device_permute_indices`` ->
  ``_RecordingPermuteProvider``: the affine row permutation
  ``i -> (5 i + 1) % rows``.  Keyed on the sample's row count only, so the
  weight and scale samples of one matrix receive the same row shuffle
  (matching the real provider's role); asymmetric w.r.t. the gate/up pair
  reorder, so composing in the wrong order cannot coincide.
- ``_reference_dequant``: independent OCP MXFP4 decode — low nibble = even
  (lower) column, E2M1 magnitude table (0, .5, 1, 1.5, 2, 3, 4, 6), sign bit
  0b1000, times ``2**(code - 127)`` per 32-column group.  Codes are kept in
  the normal range 1..254 where the kt and OCP conventions agree; the kt
  0/255 endpoints are pinned bit-exactly in the codec test instead.
"""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.moe import kt_ep_wrapper as ktw
from sglang.srt.layers.moe import kt_mxfp4_export as ktx
from sglang.srt.runtime_context import get_parallel, reset_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

# OCP MXFP4 (FP4 E2M1) magnitude table — an external-spec literal, not read
# from the module under test.
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_GROUP_SIZE = 32  # OCP MXFP4 scaling group


def _affine_perm(n: int, mult: int, off: int) -> torch.Tensor:
    """Bijection of range(n) whenever gcd(mult, n) == 1 (all uses below)."""
    return (mult * torch.arange(n, dtype=torch.long) + off) % n


def _inverse_perm(perm: torch.Tensor) -> torch.Tensor:
    return torch.argsort(perm)


def _unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """OCP nibble decode: low nibble -> even column, high nibble -> odd."""
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    nibbles = torch.stack((lo, hi), dim=-1).reshape(packed.shape[0], -1)
    nibbles = nibbles.to(torch.long)
    sign = 1.0 - 2.0 * (nibbles >> 3).to(torch.float32)
    magnitudes = torch.tensor(_E2M1_MAGNITUDES, dtype=torch.float32)
    return sign * magnitudes[nibbles & 0x7]


def _reference_dequant(packed: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """Independent fp32 decode (codes restricted to the normal range)."""
    values = _unpack_fp4(packed)
    scales = torch.ldexp(
        torch.ones(codes.shape, dtype=torch.float32),
        codes.to(torch.int64) - 127,
    )
    return values * scales.repeat_interleave(_GROUP_SIZE, dim=1)


def _mock_interleave(t: torch.Tensor) -> torch.Tensor:
    """Stand-in for flashinfer.nvfp4_block_scale_interleave (see module doc)."""
    perm = _affine_perm(t.numel(), 5, 3)
    return t.reshape(-1)[perm].reshape(t.shape)


def _mock_uninterleave(t: torch.Tensor) -> torch.Tensor:
    inv = _inverse_perm(_affine_perm(t.numel(), 5, 3))
    return t.reshape(-1)[inv].reshape(t.shape)


def _flashinfer_stub() -> dict:
    module = types.ModuleType("flashinfer")
    module.nvfp4_block_scale_interleave = _mock_interleave
    return {"flashinfer": module}


class TestE8m0Bf16Codec(CustomTestCase):
    def test_bf16_e8m0_roundtrip_exact(self):
        """Derived property + fail-loud contract.

        Pins the kt-kernel E8M0<->bf16 convention as a lossless bijection
        with code 0 -> +0.0 and 255 -> +inf (NOT the OCP NaN), and the
        recovery's hard rejection of non-E8M0 bf16 payloads.  Turns red if
        the codes are ever re-derived numerically (rounding), if 255 is
        switched to the OCP NaN semantics, or if the mantissa/sign guard is
        dropped (an fp32-scale wheel's export would then be silently
        rounded instead of failing loudly).
        """
        codes = torch.arange(256, dtype=torch.uint8)
        as_bf16 = ktx.e8m0_to_bf16(codes)
        self.assertEqual(as_bf16.dtype, torch.bfloat16)

        round_tripped = ktx.bf16_scales_to_e8m0(as_bf16)
        self.assertEqual(round_tripped.dtype, torch.uint8)
        self.assertTrue(torch.equal(round_tripped, codes))

        # kt endpoint conventions.
        self.assertEqual(as_bf16[0].item(), 0.0)
        self.assertFalse(bool(torch.signbit(as_bf16[0])))
        self.assertTrue(bool(torch.isinf(as_bf16[255])))
        self.assertGreater(as_bf16[255].item(), 0.0)

        # Interior codes decode to the mathematical 2**(code - 127),
        # computed independently in float64 (exact for exponents -126..127).
        interior = torch.arange(1, 255, dtype=torch.int64)
        expected = torch.ldexp(
            torch.ones(254, dtype=torch.float64), interior - 127
        )
        self.assertTrue(torch.equal(as_bf16[1:255].to(torch.float64), expected))

        # Fail-loud: any mantissa bit set (1.5 = 0x3FC0) or the sign bit set
        # (-1.0 = 0xBF80) is not an E8M0-resident export.
        with self.assertRaisesRegex(ValueError, "E8M0"):
            ktx.bf16_scales_to_e8m0(torch.tensor([1.5], dtype=torch.bfloat16))
        with self.assertRaisesRegex(ValueError, "E8M0"):
            ktx.bf16_scales_to_e8m0(torch.tensor([-1.0], dtype=torch.bfloat16))
        # Wrong dtype (fp32 scales, or raw uint8 codes) is a TypeError.
        with self.assertRaises(TypeError):
            ktx.bf16_scales_to_e8m0(torch.tensor([1.0], dtype=torch.float32))
        with self.assertRaises(TypeError):
            ktx.bf16_scales_to_e8m0(codes)


class TestDequantRefGroupBoundaries(CustomTestCase):
    def test_dequant_ref_group_boundaries(self):
        """Derived property: each E8M0 group scale applies to exactly its
        own 32 columns of the unpacked row, with the OCP low-nibble-first
        byte order.  Turns red on off-by-one group indexing, scaling on the
        wrong dim, a nibble-order swap, or any epsilon-introducing rewrite
        (all comparisons are exact fp32 on power-of-two scales).

        K3 shapes are exact multiples of 32, so only the exact-multiple
        case (k = 64, two groups, boundary at columns 31 -> 32) is pinned.
        """
        n, k = 8, 64
        gen = torch.Generator().manual_seed(5)
        packed = torch.randint(
            0, 256, (n, k // 2), dtype=torch.uint8, generator=gen
        )
        codes = torch.empty((n, 2), dtype=torch.uint8)
        codes[:, 0] = 120 + torch.arange(n)
        codes[:, 1] = 135 - torch.arange(n)  # adjacent groups always differ

        # Hand-pinned bytes spanning the group boundary on row 0:
        # byte 15 packs columns 30 (low nibble) and 31 (high nibble);
        # byte 16 packs columns 32 and 33.
        packed[0, 15] = 0xF7  # col30 = +6.0, col31 = -6.0 (sign bit)
        packed[0, 16] = 0x21  # col32 = +0.5, col33 = +1.0
        codes[0, 0] = 128  # group 0 (cols 0..31): x2
        codes[0, 1] = 126  # group 1 (cols 32..63): x0.5

        out = ktx.dequant_mxfp4_ref(packed, codes)
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(tuple(out.shape), (n, k))

        # Hand-computed expectations across the 31 -> 32 boundary.
        self.assertEqual(out[0, 30].item(), 12.0)
        self.assertEqual(out[0, 31].item(), -12.0)
        self.assertEqual(out[0, 32].item(), 0.25)
        self.assertEqual(out[0, 33].item(), 0.5)

        # Full-tensor agreement with the independent OCP decode.
        self.assertTrue(torch.equal(out, _reference_dequant(packed, codes)))

        # Bumping only group 1's code by one must exactly double columns
        # 32..63 in every row and leave columns 0..31 bit-identical.
        codes_bumped = codes.clone()
        codes_bumped[:, 1] += 1
        out_bumped = ktx.dequant_mxfp4_ref(packed, codes_bumped)
        self.assertTrue(torch.equal(out_bumped[:, :32], out[:, :32]))
        self.assertTrue(torch.equal(out_bumped[:, 32:], out[:, 32:] * 2))


class TestExpertBytesDigest(CustomTestCase):
    @staticmethod
    def _expert_bytes(seed=0):
        gen = torch.Generator().manual_seed(seed)

        def randbytes(shape):
            return torch.randint(0, 256, shape, dtype=torch.uint8, generator=gen)

        return ktx.Mxfp4ExpertBytes(
            w13=randbytes((8, 32)),
            w13_scale_e8m0=randbytes((8, 2)),
            w2=randbytes((6, 32)),
            w2_scale_e8m0=randbytes((6, 2)),
        )

    def test_extraction_digest_sensitivity(self):
        """Derived property (content-addressing invariant): the digest is a
        function of the exact bytes of ALL FOUR payload tensors — equal
        content gives an equal digest, and flipping one bit of any single
        tensor (weight bytes AND scale bytes) changes it.  Turns red if a
        future diff hashes only a subset of the fields, hashes tensor
        metadata/identity instead of content, or truncates a tensor before
        hashing.
        """
        baseline = self._expert_bytes()
        digest = ktx.expert_bytes_digest(baseline)

        fresh_clone = ktx.Mxfp4ExpertBytes(
            w13=baseline.w13.clone(),
            w13_scale_e8m0=baseline.w13_scale_e8m0.clone(),
            w2=baseline.w2.clone(),
            w2_scale_e8m0=baseline.w2_scale_e8m0.clone(),
        )
        self.assertEqual(ktx.expert_bytes_digest(fresh_clone), digest)

        perturbed_digests = []
        for field, position in (
            ("w13", (3, 17)),
            ("w13_scale_e8m0", (5, 1)),
            ("w2", (2, 29)),
            ("w2_scale_e8m0", (4, 0)),
        ):
            with self.subTest(field=field):
                perturbed = self._expert_bytes()
                tensor = getattr(perturbed, field)
                tensor[position] = tensor[position] ^ 1  # flip one bit
                perturbed_digest = ktx.expert_bytes_digest(perturbed)
                self.assertNotEqual(perturbed_digest, digest)
                perturbed_digests.append(perturbed_digest)
        # The four single-byte perturbations are mutually distinguishable.
        self.assertEqual(len(set(perturbed_digests + [digest])), 5)


class TestTrtllmSwizzle(CustomTestCase):
    """CPU-side proof of the trtllm-gen layout math (the GPU round-trip
    against the real kernels is checklist item G1)."""

    def test_swizzle_permutation_is_bijective_and_dequant_preserving(self):
        """Derived property: ``swizzle_trtllm_expert`` only MOVES bytes —
        weights are exactly ``input[perm]`` (no interleave), scales are the
        interleave of ``scale[perm]``, and applying the inverses recovers
        the source bytes, so dequantized values are preserved row-for-row.
        Turns red if the interleave call is dropped, applied to the weights,
        or composed before the row gather; if the gather stops being a pure
        permutation; or if the out_* path stops being copy_-only into
        caller-owned storage (F2's CUDA-graph-safety contract).
        """
        gen = torch.Generator().manual_seed(11)
        w13 = torch.randint(0, 256, (8, 32), dtype=torch.uint8, generator=gen)
        s13 = torch.randint(100, 150, (8, 2), dtype=torch.uint8, generator=gen)
        w2 = torch.randint(0, 256, (6, 32), dtype=torch.uint8, generator=gen)
        s2 = torch.randint(100, 150, (6, 2), dtype=torch.uint8, generator=gen)
        bytes_ = ktx.Mxfp4ExpertBytes(
            w13=w13, w13_scale_e8m0=s13, w2=w2, w2_scale_e8m0=s2
        )

        # Affine row permutations (provably bijective, non-identity); the
        # weight and scale rows of one matrix share the row permutation, as
        # in the real shuffled layout.
        p13 = _affine_perm(8, 3, 1)
        p2 = _affine_perm(6, 5, 2)
        indices = ktx.TrtllmPermuteIndices(
            w13_weight=p13,
            w13_scale=p13.clone(),
            w2_weight=p2,
            w2_scale=p2.clone(),
        )

        with mock.patch.dict(sys.modules, _flashinfer_stub()):
            out13, out13_s, out2, out2_s = ktx.swizzle_trtllm_expert(
                bytes_, indices
            )

        # Weights: exactly input[perm]; the inverse recovers the source.
        self.assertTrue(torch.equal(out13, w13[p13]))
        self.assertTrue(torch.equal(out2, w2[p2]))
        self.assertTrue(torch.equal(out13[_inverse_perm(p13)], w13))
        self.assertTrue(torch.equal(out2[_inverse_perm(p2)], w2))

        # Scales: the interleave is applied (a plain row gather would NOT
        # match, since the mock interleave is non-identity) ...
        self.assertFalse(torch.equal(out13_s, s13[p13]))
        self.assertTrue(torch.equal(out13_s, _mock_interleave(s13[p13])))
        # ... and un-interleave + inverse row gather recovers the codes.
        self.assertTrue(
            torch.equal(_mock_uninterleave(out13_s)[_inverse_perm(p13)], s13)
        )
        self.assertTrue(
            torch.equal(_mock_uninterleave(out2_s)[_inverse_perm(p2)], s2)
        )

        # Dequant preservation: every swizzled row decodes to its source
        # row's values under the matching (un-interleaved) scale row.
        self.assertTrue(
            torch.equal(
                _reference_dequant(out13, _mock_uninterleave(out13_s)),
                _reference_dequant(w13, s13)[p13],
            )
        )
        self.assertTrue(
            torch.equal(
                _reference_dequant(out2, _mock_uninterleave(out2_s)),
                _reference_dequant(w2, s2)[p2],
            )
        )

        # out_* path: same bytes, written copy_-only into stable storage.
        outs = tuple(
            torch.zeros_like(t) for t in (out13, out13_s, out2, out2_s)
        )
        pointers = tuple(t.data_ptr() for t in outs)
        with mock.patch.dict(sys.modules, _flashinfer_stub()):
            returned = ktx.swizzle_trtllm_expert(
                bytes_,
                indices,
                out_w13=outs[0],
                out_w13_scale=outs[1],
                out_w2=outs[2],
                out_w2_scale=outs[3],
            )
        for returned_t, out_t, pointer, expected_t in zip(
            returned, outs, pointers, (out13, out13_s, out2, out2_s)
        ):
            self.assertIs(returned_t, out_t)
            self.assertEqual(out_t.data_ptr(), pointer)
            self.assertTrue(torch.equal(out_t, expected_t))

    def test_w13_gate_up_halves_composition(self):
        """Derived property (part of case 4): with ``w13_gate_up_halves``,
        ``trtllm_permute_indices`` composes the (up_i, gate_i) pair reorder
        of mxfp4.py's non-interleaved branch INTO the provider's shuffle —
        the algebraic identity ``x[pair][shuffle] == x[composed]``, with the
        pair layout derived independently via ``stack((up, gate), dim=1)``.
        Turns red if the pair convention flips to (gate_i, up_i), if the
        composition order is swapped (the affine shuffle is asymmetric on
        purpose), if the flag leaks into the w2 indices, or if the
        ``num_elts_per_sf=16`` scale kwarg is dropped from the provider
        calls.
        """
        import sglang.srt.layers.quantization.mxfp4 as mxfp4_mod

        rows, half = 8, 4
        w13_sample = torch.zeros((rows, 16), dtype=torch.uint8)
        w13_scale_sample = torch.zeros((rows, 4), dtype=torch.uint8)
        w2_sample = torch.zeros((6, 16), dtype=torch.uint8)
        w2_scale_sample = torch.zeros((6, 4), dtype=torch.uint8)

        calls = []

        def provider(x, epilogue_tile_m, num_elts_per_sf=None):
            calls.append((tuple(x.shape), epilogue_tile_m, num_elts_per_sf))
            return _affine_perm(x.shape[0], 5, 1)

        with mock.patch.object(
            mxfp4_mod, "_get_flashinfer_mxfp4_device_permute_indices", provider
        ):
            composed = ktx.trtllm_permute_indices(
                w13_sample=w13_sample,
                w13_scale_sample=w13_scale_sample,
                w2_sample=w2_sample,
                w2_scale_sample=w2_scale_sample,
                w13_gate_up_halves=True,
            )
            plain = ktx.trtllm_permute_indices(
                w13_sample=w13_sample,
                w13_scale_sample=w13_scale_sample,
                w2_sample=w2_sample,
                w2_scale_sample=w2_scale_sample,
            )

        shuffle = _affine_perm(rows, 5, 1)  # what the provider returned

        # Still a bijection after composition.
        self.assertTrue(
            torch.equal(
                composed.w13_weight.sort().values, torch.arange(rows)
            )
        )
        # Weight and scale samples share a row count, hence a shuffle,
        # hence identical composed indices.
        self.assertTrue(torch.equal(composed.w13_weight, composed.w13_scale))

        # The algebraic identity on distinct-valued x: gathering with the
        # composed indices equals building the (up_i, gate_i) row pairs
        # first (independent stack construction, mxfp4.py L716-730
        # convention: non-interleaved [gate | up] halves, up first in each
        # pair) and then applying the shuffle.
        x = torch.arange(rows * 5, dtype=torch.float32).reshape(rows, 5)
        interleaved = torch.stack((x[half:], x[:half]), dim=1).reshape(rows, 5)
        self.assertTrue(torch.equal(x[composed.w13_weight], interleaved[shuffle]))

        # Without the flag the w13 indices are the raw shuffle ...
        self.assertTrue(torch.equal(plain.w13_weight, shuffle))
        self.assertFalse(torch.equal(composed.w13_weight, plain.w13_weight))
        # ... and w2 is never affected by the flag.
        self.assertTrue(torch.equal(composed.w2_weight, plain.w2_weight))
        self.assertTrue(torch.equal(composed.w2_scale, plain.w2_scale))

        # Provider contract: scale samples carry num_elts_per_sf=16,
        # weight samples do not; the epilogue tile is 128.
        sf_by_shape = {shape: sf for shape, tile, sf in calls}
        tiles = {tile for _shape, tile, _sf in calls}
        self.assertEqual(tiles, {128})
        self.assertEqual(
            sf_by_shape,
            {(8, 16): None, (8, 4): 16, (6, 16): None, (6, 4): 16},
        )


class TestTrtllmStorageSizing(CustomTestCase):
    K3_HIDDEN = 3584
    K3_INTERMEDIATE = 3072

    def test_trtllm_storage_sizing(self):
        """Critical-path bookkeeping (KV-budget byte accounting) + derived
        formula.  The prepared and raw slot byte counts are re-derived here
        independently from the export layout contract (w13 = [gate | up]
        nibbles [2I, H/2], per-32-column scales; prepared scales 1 B/elem,
        raw export scales bf16 = 2 B/elem).  Turns red if a slot dtype
        changes, if the scale-interleave padding stops being a no-op at
        %128 geometry, if the %128 guard is dropped, or if the raw slot
        stops sizing bf16 export scales.
        """
        hidden, inter = self.K3_HIDDEN, self.K3_INTERMEDIATE
        for num_experts in (1, 8, 384):
            with self.subTest(num_experts=num_experts):
                # Independent component derivation from the export layout.
                w13_nibble_bytes = num_experts * (2 * inter) * (hidden // 2)
                w13_scale_elems = num_experts * (2 * inter) * (hidden // 32)
                w2_nibble_bytes = num_experts * hidden * (inter // 2)
                w2_scale_elems = num_experts * hidden * (inter // 32)

                # Prepared trtllm slot: uint8 weights + 1 B/elem fp8-viewed
                # interleaved scales (interleave is exact at %128 geometry).
                expected_prepared = (
                    w13_nibble_bytes
                    + w13_scale_elems
                    + w2_nibble_bytes
                    + w2_scale_elems
                )
                # Cross-check the closed form: E * H * I * 51 / 32.
                self.assertEqual(
                    expected_prepared, num_experts * hidden * inter * 51 // 32
                )
                self.assertEqual(
                    ktx.get_trtllm_mxfp4_storage_nbytes(
                        num_experts=num_experts,
                        hidden_size=hidden,
                        intermediate_size=inter,
                    ),
                    expected_prepared,
                )

                # The prepared-slot geometry the byte count is built from.
                self.assertEqual(
                    ktx._trtllm_prepared_shapes(num_experts, hidden, inter),
                    (
                        (num_experts, 2 * inter, hidden // 2),
                        (num_experts, 2 * inter, hidden // 32),
                        (num_experts, hidden, inter // 2),
                        (num_experts, hidden, inter // 32),
                    ),
                )

                # Raw slot (kt_ep_wrapper): uint8 nibble weights + BF16
                # export scales (the SHM/H2D payload dtype), 2 B/elem.
                self.assertEqual(
                    ktw._mxfp4_trtllm_raw_slot_storage_nbytes(
                        num_experts=num_experts,
                        hidden_size=hidden,
                        intermediate_size=inter,
                    ),
                    w13_nibble_bytes
                    + w2_nibble_bytes
                    + 2 * (w13_scale_elems + w2_scale_elems),
                )

        # Non-%128 dims and non-positive expert counts fail loudly.
        with self.assertRaisesRegex(ValueError, "128"):
            ktx.get_trtllm_mxfp4_storage_nbytes(
                num_experts=4, hidden_size=hidden + 64, intermediate_size=inter
            )
        with self.assertRaisesRegex(ValueError, "128"):
            ktx.get_trtllm_mxfp4_storage_nbytes(
                num_experts=4, hidden_size=hidden, intermediate_size=inter - 64
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            ktx.get_trtllm_mxfp4_storage_nbytes(
                num_experts=0, hidden_size=hidden, intermediate_size=inter
            )


class _ClassifierLayer:
    """Carries exactly what the pipeline signature reads: one parameter
    whose device tags the signature."""

    def __init__(self):
        self._device_anchor = torch.zeros(1)

    def parameters(self):
        return iter([self._device_anchor])


class DeepSeekMxfp4MoEMethod:
    """Name-keyed test double (sibling-suite idiom): the classifier keys the
    DSV4 wrap chain on the class NAME; the real class lives in
    mxfp4_deepseek.py, which imports GPU-only sglang.kernels ops at module
    scope and therefore cannot import on CPU CI."""


def _real_trtllm_method(**overrides):
    """The REAL Mxfp4MoEMethod, uninitialized, with only the attrs the
    classifier reads — required because detection uses ``type(...) is
    Mxfp4MoEMethod``, so a type-named fake would not exercise it."""
    from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod

    attrs = {
        "use_deep_gemm": False,
        "use_marlin": False,
        "use_flashinfer": True,
        "_fi_kernel": "trtllm_sm100",
    }
    attrs.update(overrides)
    gpu_method = object.__new__(Mxfp4MoEMethod)
    for name, value in attrs.items():
        setattr(gpu_method, name, value)
    return gpu_method


def _same_name_subclass_method():
    """A subclass literally named Mxfp4MoEMethod: passes any name or
    isinstance check but must FAIL the precise ``type(...) is`` check."""
    from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod

    subclass = type("Mxfp4MoEMethod", (Mxfp4MoEMethod,), {})
    method = object.__new__(subclass)
    method.use_deep_gemm = False
    method.use_marlin = False
    method.use_flashinfer = True
    method._fi_kernel = "trtllm_sm100"
    return method


class TestPipelineLayoutDetection(CustomTestCase):
    _WEIGHT_PATH = "/nonexistent-dummy"
    _NUM_LAYERS = 4
    _NUM_EXPERTS = 8
    _FULL_INIT_ARGS = (256, 128, torch.bfloat16)

    def setUp(self):
        self._clear_registries()

    def tearDown(self):
        self._clear_registries()
        reset_context()

    @staticmethod
    def _clear_registries():
        ktw._MXFP4_PREFILL_LAYER_REGISTRY.clear()
        ktw._MXFP4_LAYERWISE_MANAGERS.clear()
        ktw._MXFP4_LAYERWISE_DISABLED_REASONS.clear()

    def _wrapper_method(self, gpu_method, activation="situ"):
        return SimpleNamespace(
            gpu_method=gpu_method,
            moe_runner_config=SimpleNamespace(activation=activation),
            gpu_prefill_token_threshold=1024,
            kt_config=SimpleNamespace(
                method="MXFP4",
                layer_idx=0,
                weight_path=self._WEIGHT_PATH,
                num_layers=self._NUM_LAYERS,
            ),
            global_num_experts=self._NUM_EXPERTS,
            _full_init_args=self._FULL_INIT_ARGS,
        )

    def _run_backend_supported(self, method, layer, tp_rank=0, tp_size=1):
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "empty_cache"),
            get_parallel().override(tp_rank=tp_rank, tp_size=tp_size),
        ):
            return ktw._mxfp4_pipeline_backend_supported(method, layer)

    def test_pipeline_layout_detection_and_disable_reasons(self):
        """Critical-path bookkeeping + negative-branch completeness for the
        layout classification table and the disable-reason registry.  Turns
        red if the precise ``type(...) is Mxfp4MoEMethod`` check is relaxed
        (the same-name subclass row), if the mode priority is reordered
        (deep_gemm before marlin), if a recognized-unsupported mode stops
        recording its reason (silently landing on the incompatible serial
        fallback instead of hybrid), if unknown layouts START recording
        reasons, or if the signature composition changes and breaks
        TP-consistent keying.
        """
        # The signature key is derived by hand: device str of the CPU
        # anchor parameter + the process-global config the classifier is
        # documented to be a pure function of.
        expected_signature = (
            "cpu",
            self._WEIGHT_PATH,
            self._NUM_LAYERS,
            self._NUM_EXPERTS,
            self._FULL_INIT_ARGS,
        )

        table = (
            # (name, method, expected_layout, expected_reason, substrings)
            (
                "flashinfer_trtllm",
                self._wrapper_method(_real_trtllm_method()),
                ktw._MXFP4_LAYOUT_TRTLLM,
                None,
                (),
            ),
            (
                "marlin_mode",
                self._wrapper_method(_real_trtllm_method(use_marlin=True)),
                None,
                "recorded",
                ("marlin mode", "SiTU"),
            ),
            (
                "deep_gemm_mode_takes_priority",
                self._wrapper_method(
                    _real_trtllm_method(use_deep_gemm=True, use_marlin=True)
                ),
                None,
                "recorded",
                ("deep_gemm mode",),
            ),
            (
                "non_sm100_fi_kernel",
                self._wrapper_method(
                    _real_trtllm_method(_fi_kernel="cutlass_sm90")
                ),
                None,
                "recorded",
                ("cutlass_sm90",),
            ),
            (
                "non_situ_activation",
                self._wrapper_method(_real_trtllm_method(), activation="silu"),
                None,
                "recorded",
                ("situ", "silu"),
            ),
            (
                "deepseek_marlin_unchanged",
                self._wrapper_method(DeepSeekMxfp4MoEMethod()),
                ktw._MXFP4_LAYOUT_MARLIN,
                None,
                (),
            ),
            (
                "unknown_method_plain_unsupported",
                self._wrapper_method(object()),
                None,
                None,
                (),
            ),
            (
                "same_name_subclass_not_vetted",
                self._wrapper_method(_same_name_subclass_method()),
                None,
                None,
                (),
            ),
        )

        for name, method, expected_layout, reason_kind, substrings in table:
            with self.subTest(case=name):
                self._clear_registries()
                layer = _ClassifierLayer()

                layout, reason = ktw._mxfp4_pipeline_layout_or_reason(method)
                self.assertEqual(layout, expected_layout)
                if reason_kind is None:
                    self.assertIsNone(reason)
                else:
                    for fragment in substrings:
                        self.assertIn(fragment, reason)

                if name == "flashinfer_trtllm":
                    self.assertTrue(self._run_backend_supported(method, layer))
                    self.assertEqual(ktw._MXFP4_LAYERWISE_DISABLED_REASONS, {})
                elif name == "deepseek_marlin_unchanged":
                    # The marlin backend gate needs a real CUDA device
                    # capability; only the layout classification is
                    # CPU-provable here.
                    pass
                elif reason_kind is None:
                    # Unknown layouts keep plain-unsupported semantics:
                    # no disable entry is ever recorded.
                    self.assertFalse(self._run_backend_supported(method, layer))
                    self.assertEqual(ktw._MXFP4_LAYERWISE_DISABLED_REASONS, {})
                else:
                    # Recognized-but-unsupported: the reason lands in the
                    # registry under the hand-derived TP-consistent key,
                    # identically on every rank, and only once.
                    per_rank = []
                    for tp_rank in (0, 1):
                        ktw._MXFP4_LAYERWISE_DISABLED_REASONS.clear()
                        self.assertFalse(
                            self._run_backend_supported(
                                method, layer, tp_rank=tp_rank, tp_size=2
                            )
                        )
                        per_rank.append(
                            dict(ktw._MXFP4_LAYERWISE_DISABLED_REASONS)
                        )
                    self.assertEqual(per_rank[0], per_rank[1])
                    self.assertEqual(
                        set(per_rank[0]), {expected_signature}
                    )
                    recorded = per_rank[0][expected_signature]
                    for fragment in substrings:
                        self.assertIn(fragment, recorded)
                    # A second qualifying call must not re-record.
                    before = dict(ktw._MXFP4_LAYERWISE_DISABLED_REASONS)
                    self.assertFalse(
                        self._run_backend_supported(
                            method, layer, tp_rank=1, tp_size=2
                        )
                    )
                    self.assertEqual(
                        ktw._MXFP4_LAYERWISE_DISABLED_REASONS, before
                    )


if __name__ == "__main__":
    unittest.main()
