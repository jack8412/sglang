"""mini-K3 CPU-only smoke for the k3-hybrid KT wiring (Phase-2/3 gate).

Constructs a miniature Kimi-K3 text backbone (4 KDA + 1 MLA layers, 64
experts, real *structure* with reduced dims), then drives KimiK3MoE.forward
end-to-end on CPU twice with identical dummy weights:

  1. monolithic     — KT disabled, all experts through the (unquantized) GPU
                      method's torch-native path;
  2. hybrid         — --kt-weight-path set, KTMoEWrapper mocked by a torch
                      reference implementation for the CPU-resident set;

and asserts the outputs match (weighted GPU+CPU merge == monolithic).

Run (CPU-only, confined on ai.v8.pro):
    python test/manual/kt_k3/mini_k3_kt_smoke.py

The attention layers are constructed (import/shape check) but not forwarded —
attention backends need a ModelRunner; the MoE region is what KT touches.
"""

import sys
import zlib
from unittest.mock import patch

import torch


def seed_params_by_name(module, salt=0):
    """Deterministic per-parameter init, independent of parameter iteration
    order and of other parameters' shapes (the KT and monolithic builds have
    different expert-table sizes, so a shared generator stream would
    diverge)."""
    for name, p in module.named_parameters():
        if not p.dtype.is_floating_point:
            continue
        g = torch.Generator().manual_seed(zlib.crc32(name.encode()) + salt)
        p.data.copy_(torch.randn(p.shape, generator=g) * 0.05)

from sglang.srt.configs.kimi_linear import KimiLinearConfig
from sglang.srt.layers.moe import kt_ep_wrapper as ktw
from sglang.srt.layers.moe.utils import initialize_moe_config
from sglang.srt.runtime_context import get_context, get_parallel, get_server_args
from sglang.srt.server_args import ServerArgs

NUM_EXPERTS = 64
TOP_K = 8
HIDDEN = 256
LATENT = 128
INTERMEDIATE = 96
NUM_TOKENS = 9


def build_config():
    return KimiLinearConfig(
        architectures=["KimiK3LinearForCausalLM"],
        vocab_size=512,
        hidden_size=HIDDEN,
        intermediate_size=192,
        num_hidden_layers=5,
        num_attention_heads=8,
        num_key_value_heads=8,
        hidden_act="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        num_experts=NUM_EXPERTS,
        num_experts_per_token=TOP_K,
        moe_intermediate_size=INTERMEDIATE,
        routed_expert_hidden_size=LATENT,
        latent_moe_use_norm=True,
        num_shared_experts=1,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        num_expert_group=1,
        topk_group=1,
        topk_method="noaux_tc",
        moe_renormalize=True,
        routed_scaling_factor=2.5,
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        v_head_dim=32,
        linear_attn_config=dict(
            kda_layers=[1, 2, 3, 4],
            full_attn_layers=[5],
            num_heads=4,
            head_dim=32,
            short_conv_kernel_size=4,
            use_full_rank_gate=True,
            gate_lower_bound=-5.0,
        ),
    )


class _RefKTMoEWrapper:
    """kt-kernel mock: torch reference for the CPU-resident expert set,
    returning the PRE-WEIGHTED contribution (the kt-kernel contract)."""

    weights = None  # dict expert_id -> (w13 [2I, L], w2 [L, I]); set by main
    gpu_mask = None
    ctor_log = []

    def __init__(self, **kwargs):
        allowed = ktw.KTMOE_WRAPPER_BASE_CTOR_PARAMS | ktw.KTMOE_WRAPPER_SITU_CTOR_PARAMS
        unknown = set(kwargs) - allowed
        assert not unknown, f"ctor params outside the wheel contract: {unknown}"
        type(self).ctor_log.append(kwargs)
        self._pending = None
        self._situ_beta = kwargs.get("situ_beta", 0.0)
        self._situ_linear_beta = kwargs.get("situ_linear_beta", 0.0)

    def _act(self, gate, up):
        if self._situ_beta:
            b, lb = self._situ_beta, self._situ_linear_beta
            up = lb * torch.tanh(up / lb) if lb else up
            return b * torch.tanh(gate / b) * torch.sigmoid(gate) * up
        return torch.nn.functional.silu(gate) * up

    def submit_forward(self, x, topk_ids, topk_weights, cuda_stream):
        self._pending = (x.clone(), topk_ids.clone(), topk_weights.clone())

    def sync_forward(self, ref, cuda_stream):
        x, ids, w = self._pending
        out = torch.zeros_like(x)
        for t in range(x.shape[0]):
            for k in range(ids.shape[1]):
                e = int(ids[t, k])
                if e < 0 or bool(type(self).gpu_mask[e]):
                    continue
                w13, w2 = type(self).weights[e]
                gu = x[t] @ w13.t()
                gate, up = gu.chunk(2)
                out[t] += w[t, k] * (self._act(gate, up) @ w2.t())
        return out.to(ref.dtype)

    def load_weights(self, physical_to_logical_map_cpu):
        pass


_CONFIG_DIR = None


def _config_dir():
    """Persist the mini-K3 config to disk once: KT mask generation resolves
    the HF config through ServerArgs.get_model_config(), which loads from
    model_path."""
    global _CONFIG_DIR
    if _CONFIG_DIR is None:
        import tempfile

        _CONFIG_DIR = tempfile.mkdtemp(prefix="mini-k3-")
        build_config().save_pretrained(_CONFIG_DIR)
    return _CONFIG_DIR


def publish_ctx(kt: bool):
    fields = dict(
        model_path=_config_dir(),
        chunked_prefill_size=64,
        disable_shared_experts_fusion=True,
    )
    if kt:
        fields.update(
            kt_weight_path="/dummy-kt-weights",
            kt_method="MXFP4",
            kt_cpuinfer=2,
            kt_threadpool_count=1,
            kt_num_gpu_experts=40,
            kt_expert_placement_strategy="uniform",
        )
    ctx = get_context().override_server_args(**fields)
    ctx.install()
    initialize_moe_config(get_server_args())
    return ctx


_DIST_READY = False


def init_single_process_distributed():
    """Real gloo world-size-1 groups: layer forwards call get_tp_group()
    even at tp==1 (pattern from test_dsa_layer_split_broadcast.py)."""
    global _DIST_READY
    if _DIST_READY:
        return
    import os

    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29723")
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="tcp://127.0.0.1:29723",
        backend="gloo",
    )
    initialize_model_parallel(tensor_model_parallel_size=1)
    _DIST_READY = True


def run(kt: bool, seed=0):
    torch.manual_seed(seed)
    ctx = publish_ctx(kt)
    init_single_process_distributed()
    try:
        with get_parallel().override(
            tp_rank=0,
            tp_size=1,
            moe_ep_rank=0,
            moe_ep_size=1,
            moe_tp_rank=0,
            moe_tp_size=1,
            attn_tp_rank=0,
            attn_tp_size=1,
        ):
            from sglang.srt.models.kimi_k3 import KimiK3MoE

            config = build_config()
            moe = KimiK3MoE(
                config=config, layer_idx=1, quant_config=None, prefix="model.layers.1.mlp"
            )
            moe.eval()

            seed_params_by_name(moe)

            if kt:
                # Hand the mock the SAME routed-expert weights FusedMoE holds
                # for the GPU set, plus the CPU set's (never-loaded) weights.
                w13 = moe.experts.w13_weight.data
                w2 = moe.experts.w2_weight.data
                method = moe.experts.quant_method
                assert type(method).__name__ == "KTEPWrapperMethod", type(method)
                full = {}
                cpu_gen = torch.Generator().manual_seed(seed + 2)
                for e in range(NUM_EXPERTS):
                    slot = int(method.logical_to_gpu_index[e])
                    if slot >= 0:
                        full[e] = (w13[slot].clone(), w2[slot].clone())
                    else:
                        full[e] = (
                            torch.randn(2 * INTERMEDIATE, LATENT, generator=cpu_gen) * 0.05,
                            torch.randn(LATENT, INTERMEDIATE, generator=cpu_gen) * 0.05,
                        )
                _RefKTMoEWrapper.weights = full
                _RefKTMoEWrapper.gpu_mask = method.gpu_experts_mask

            x = torch.randn(NUM_TOKENS, HIDDEN, generator=torch.Generator().manual_seed(99))
            with torch.no_grad():
                out = moe.forward(x.clone(), forward_batch=None)
            return out, moe
    finally:
        ctx.restore()


def main():
    # Monolithic pass needs all 64 experts' weights; run the hybrid FIRST so
    # its CPU-set weights exist, then rebuild monolithic with the union.
    with patch.multiple(
        ktw,
        KTRANSFORMERS_AVAILABLE=True,
        KTMoEWrapper=_RefKTMoEWrapper,
        KT_WHEEL_SUPPORTS_SITU=True,
    ), patch.object(ktw, "get_stream", lambda name: None), patch(
        "torch.cuda.Event", lambda *a, **k: None
    ):
        hybrid_out, hybrid_moe = run(kt=True)

    # situ params must have reached every wrapper ctor from the config.
    assert _RefKTMoEWrapper.ctor_log, "KTMoEWrapper was never constructed"
    for kw in _RefKTMoEWrapper.ctor_log:
        assert kw.get("situ_beta") == 4.0, kw
        assert kw.get("situ_linear_beta") == 25.0, kw
        assert kw.get("method") == "MXFP4", kw
        assert kw.get("hidden_size") == LATENT, kw
        assert kw.get("moe_intermediate_size") == INTERMEDIATE, kw
        assert kw.get("num_experts") == NUM_EXPERTS, kw

    # Monolithic reference: same weights, KT off. Rebuild the model and load
    # the union weight set into the full expert table.
    mono_out, mono_moe = None, None
    ctx = publish_ctx(False)
    init_single_process_distributed()
    try:
        with get_parallel().override(
            tp_rank=0,
            tp_size=1,
            moe_ep_rank=0,
            moe_ep_size=1,
            moe_tp_rank=0,
            moe_tp_size=1,
            attn_tp_rank=0,
            attn_tp_size=1,
        ):
            from sglang.srt.models.kimi_k3 import KimiK3MoE

            config = build_config()
            moe = KimiK3MoE(
                config=config, layer_idx=1, quant_config=None, prefix="model.layers.1.mlp"
            )
            moe.eval()
            seed_params_by_name(moe)  # same non-expert weights as the KT run
            for e, (w13, w2) in _RefKTMoEWrapper.weights.items():
                moe.experts.w13_weight.data[e] = w13
                moe.experts.w2_weight.data[e] = w2
            x = torch.randn(NUM_TOKENS, HIDDEN, generator=torch.Generator().manual_seed(99))
            with torch.no_grad():
                mono_out = moe.forward(x.clone(), forward_batch=None)
    finally:
        ctx.restore()

    torch.testing.assert_close(hybrid_out, mono_out, rtol=2e-3, atol=2e-3)
    print("mini-K3 KT hybrid == monolithic: PASS")

    # Full-model construction check (4 KDA + 1 MLA): imports + shapes only.
    ctx = publish_ctx(False)
    init_single_process_distributed()
    try:
        with get_parallel().override(
            tp_rank=0,
            tp_size=1,
            moe_ep_rank=0,
            moe_ep_size=1,
            moe_tp_rank=0,
            moe_tp_size=1,
            attn_tp_rank=0,
            attn_tp_size=1,
        ):
            from sglang.srt.models.kimi_k3 import KimiK3LinearForCausalLM

            model = KimiK3LinearForCausalLM(config=build_config(), quant_config=None)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"mini-K3 model constructed: {n_params/1e6:.1f}M params — PASS")
    finally:
        ctx.restore()


if __name__ == "__main__":
    sys.exit(main())
