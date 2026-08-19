#!/bin/bash
# usage: launch.sh <name> [extra server args...]
#        K3_PROFILE=prod|prod01|margin10|ceiling|bare   (default: prod)
#        K3_PORT=<n>      server port (default 30000, sglang's well-known one)
#
# The K3 server launcher. Lives in the repo so a bootstrapped node has it
# (k3.sh serve runs it from the node's sglang checkout); the previous copy
# lived only on the node and evaporated with the rental.
#
# ---------------------------------------------------------------------------
# PROFILES
# ---------------------------------------------------------------------------
# Margin routing is the product, not a variant: the point of the campaign is to
# make it faster, and everything else here exists to measure it. So `prod` is
# the default and carries the full shipping recipe. The instrument profiles are
# separate rather than "prod minus a flag" for a concrete reason -- see the
# store_true warning below.
#
#   prod      margin 0.5 + cold-only + swapping. Verified end to end by phase
#             CS: -0.0215 nats vs exact routing, greedy output byte-identical,
#             swapped-in experts bitwise-match the bulk loader on every NUMA
#             partition. This is what the numbers in HANDOFF describe.
#   prod01    the same stack at margin 0.1 -- the quality-first end of the
#             tradeoff. A smaller margin substitutes LESS, so more tokens reach
#             their CPU-resident first choice: closer to exact routing and
#             slower. The sweep that established the shape (phase C, uniform
#             placement, pre-optimisation build) read 31.8 / 42.0 / 47.7 tok/s
#             at margin 0.1 / 1.0 / 10. Its nats are UNMEASURED -- margin 0.0
#             is bit-exact (phase B) and 0.5 is -0.0215, so 0.1 sits between,
#             but gate it with logprobs before quoting it as a serving option.
#   margin10  margin 10. NOT a serving mode -- its nats are identical to full
#             override to four decimals (-0.5942), i.e. the same 31% of experts
#             made unreachable, 8 tok/s slower. It exists to measure the
#             sglang-side machinery with almost no CPU compute underneath.
#   ceiling   full override, no margin at all. The GPU-only floor. Builds no KT
#             wrapper and loads NO CPU expert weights (they are never computed),
#             so it starts in minutes rather than loading ~1.45 TB it never reads
#             -- which also stops it evicting the page cache for later rows.
#   bare      nothing routing-related. For A/B rows that set every knob
#             themselves, and the only safe base for a row that must NOT have
#             cold-only or swapping on.
#
# WARNING -- trailing args can override a VALUED flag (argparse keeps the last
# occurrence) but CANNOT unset a store_true one. `--kt-cold-only-cpu-experts`
# and `--kt-routing-full-override` cannot be turned off by anything you append.
# A row that needs them off must choose a profile that never sets them.
#
# ---------------------------------------------------------------------------
# Evidence for the non-obvious defaults
# ---------------------------------------------------------------------------
# - kt-transport doorbell: RB2 (margin 10) doorbell 75.3 tok/s vs packed
#   hostnode 57.5; DB (margin 0.5) 43.6 vs 39.0. The server default is still
#   hostnode, so the launcher sets doorbell explicitly. Full-override ceiling
#   rows ran hostnode (transport absent from their graph) -- pass
#   --kt-transport hostnode to reproduce those exactly.
# - cold-only requires --kt-routing-margin, which is why it lives in `prod`
#   and not in the shared defaults (CO1: outputs byte-identical, -0.0215 nats
#   unchanged, ~1 TB host RAM freed, weight load 110 s).
# - attention backends: leave UNSET -- the KimiK3 override resolves all three
#   to trtllm_mla on SM100/SM103 (verified in every old log). The fa2
#   UserWarning from the flashinfer prefill wrapper appeared in every old
#   campaign log too; it is noise, not a config error.
# - --kt-gpu-prefill-token-threshold deliberately absent: the full-GPU sweep is
#   5.2x slower than margin-routed prefill, costs 7.54 GiB/GPU, and is refused
#   at config time when a margin is set.
set -u

NAME=${1:?usage: launch.sh <name> [extra server args...]}; shift
WS=/workspace
mkdir -p $WS/runs/{status,probes,logs,meta}
LOG=$WS/runs/logs/$NAME.server.log
: > $LOG

# Prebuilt trtllm-gen MoE cubins. NOT optional for this config: with
# --moe-runner-backend flashinfer_mxfp4 on SM100 the server REFUSES to start
# without a valid pool (overrides.py raises; there is no pool-less JIT path).
# /opt is where the old image baked it in; /workspace is where k3.sh
# bootstrap installs it on images that ship without one.
POOL_VER=trtllm_gen_moe_cubin_pool_20260617_v0613rc1
for base in /opt/trtllm_gen_moe_cubin_pool /workspace/trtllm_gen_moe_cubin_pool; do
  if [ -d "$base/$POOL_VER" ]; then
    export SGLANG_TRTLLM_GEN_MOE_CUBIN_POOL=$base/$POOL_VER
    break
  fi
done
[ -n "${SGLANG_TRTLLM_GEN_MOE_CUBIN_POOL:-}" ] \
  || echo "WARNING: cubin pool missing ($POOL_VER) -- flashinfer_mxfp4 on SM100 will refuse to start; run k3.sh bootstrap" >> $LOG

source $WS/venv-k3/bin/activate

# Sized to the node, not hardcoded to a rental that no longer exists: one
# threadpool per NUMA node; cpuinfer ~85% of PHYSICAL cores (RUNBOOK step 3 --
# AMX is a per-core resource, SMT siblings add nothing). lscpu shows HOST
# topology and is cgroup-blind, so clamp by nproc (which honors the cpuset):
# on a container rental granted fewer CPUs than the host has, nproc wins.
#
# TWO CAVEATS, because this derivation is not identical to what was measured:
#  - the whole 2026-08-10/11 campaign ran --kt-cpuinfer 200. On that node this
#    formula yields 204 (nproc 240 clamps PHYS, 240*85/100), so a rerun is
#    close but not bit-identical to the archived rows. Pass --kt-cpuinfer 200
#    to reproduce one exactly.
#  - the nproc clamp conflates logical with physical: if the cpuset exposes SMT
#    siblings, PHYS becomes a LOGICAL count and 85% of it oversubscribes the
#    physical cores AMX actually runs on. Unverified either way on the rental;
#    check `lscpu -p=Core,Socket` inside the container before trusting it on a
#    new node.
PHYS=$(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)
[ "$PHYS" -gt 0 ] || PHYS=$(nproc)
[ "$PHYS" -gt "$(nproc)" ] && PHYS=$(nproc)
NUMAN=$(numactl --hardware 2>/dev/null | awk '/^available:/{print $2}')
[ -n "$NUMAN" ] || NUMAN=2

# kt PARTITIONS MUST EQUAL TP -- this is a correctness knob, not a tuning one.
# kt shards w2 by COLUMN per partition (its down buffer is a compacted
# [hidden, per_numa]), so a GPU rank's w2 slice is one contiguous block only
# when per_numa == per_gpu. Below that -- e.g. the NUMA node count, which is
# what this used to pass -- every expert's w2 becomes H strided
# (per_gpu/2)-byte reads, the DMA-efficiency trap that makes --kt-cold-transport
# direct-dma unusable. The NUMA node count is unrelated to the partition count:
# place the pools explicitly with --kt-numa-nodes (ai.v8.pro, GPUs on nodes
# 0,0,2,2,3,3,5,5: `--kt-numa-nodes 0 0 2 2 3 3 5 5`).
KTPOOLS=8

# CPUINFER, DERIVED. It used to be PHYS*0.85, which is a guess that happens to
# be right here and over-subscribes elsewhere. What actually constrains it:
#
#   * AMX computes on PHYSICAL cores -- two HT siblings share the tile
#     registers -- so the budget is physical cores, never `nproc`.
#   * kt splits cpuinfer evenly across its pools, and the pools pinned to one
#     NUMA node must fit that node, or the pin fails SILENTLY.
#   * The ranks whose GPUs sit on that node need cores too: one scheduler main
#     thread each, plus the doorbell pollers, plus driver/IO slack.
#
# so, per node:  threads_per_pool = (cores - ranks - pollers - 2) / pools
#
#   gpusrv    (96/2):  (48 - 4 - 2) / 4 = 10  ->  80
#   ai.v8.pro (144/6): (24 - 2 - 2) / 2 = 10  ->  80   (was 96 = 24 threads on a
#                      24-core node, i.e. the whole node with nothing left for
#                      the two schedulers pinned there)
#
# Both hosts land on 10 threads per pool, which is worth having deliberately:
# it removes one variable when comparing a measurement taken on one against
# the other.
#
# Erring low is free: the cpuinfer sweep at 48/81/96/144 is FLAT on AMX, where
# CPU expert compute stopped being the bottleneck. Cores left to the schedulers
# are worth more than cores added to a pool that is not the constraint.
POLLERS=2
CORES_PER_NODE=$(( PHYS / NUMAN ))
POOLS_PER_NODE=$(( (KTPOOLS + NUMAN - 1) / NUMAN ))
RESERVE=$(( POOLS_PER_NODE + POLLERS ))
PER_POOL=$(( (CORES_PER_NODE - RESERVE) / POOLS_PER_NODE ))
[ "$PER_POOL" -ge 1 ] || PER_POOL=1
CPUINF=$(( PER_POOL * KTPOOLS ))

# PLACEMENT PROFILES REMOVED (2026-08-14). The launcher no longer discovers or
# loads an expert-distribution dump, and no longer arms the recorder.
#
# Frequency placement did measure better than uniform (SPEC-MARGIN-ROUTING F2:
# exact-routing parity at 99.0% gsm8k), so this is not a claim the feature did
# nothing. It is a claim it does not pay for itself in production:
#
#  - Capturing a profile needs a DEDICATED run. The recorder cannot be turned
#    on later (unset, it is a Noop whose start_record() raises), and arming it
#    trips _disable_tc_piecewise_cudagraph_if_incompatible -- so the capture run
#    has a different cuda-graph configuration from the run it is meant to tune.
#  - A profile is workload-specific and goes stale as the served domain shifts,
#    which in production it does continuously. The cost is paid per capture; the
#    benefit decays from the moment it is taken.
#  - Expert SWAPPING (--kt-expert-swap-transitions) already adapts the resident set
#    at runtime from live insist/override counters. That is the same objective
#    pursued continuously instead of frozen at capture time, so a static profile
#    is redundant next to it, not merely stale.
#  - Under --kt-expert-split-prefill, placement does not affect prefill at all:
#    every one of the 896 experts is computed regardless of where it lives. Its
#    entire remaining influence was on decode.
#
# sglang's own --kt-expert-placement-strategy / --init-expert-location are
# untouched and can still be passed explicitly as trailing args; what is gone is
# this launcher discovering a *.pt and applying it behind your back. With no
# flag emitted, sglang uses its default (uniform) placement.

# The routing skeleton. Everything the campaign varies lives here and nowhere
# else, so a row's identity is one word rather than a flag list to diff.
PROFILE=${K3_PROFILE:-prod}
case "$PROFILE" in
  prod)     ROUTING=(--kt-routing-margin 0.5 --kt-cold-only-cpu-experts
                     --kt-expert-swap-transitions 5 --kt-expert-swap-max 8) ;;
  prod01)   ROUTING=(--kt-routing-margin 0.1 --kt-cold-only-cpu-experts
                     --kt-expert-swap-transitions 5 --kt-expert-swap-max 8) ;;
  margin10) ROUTING=(--kt-routing-margin 10) ;;
  ceiling)  ROUTING=(--kt-routing-full-override) ;;
  bare)     ROUTING=() ;;
  *) echo "FATAL: unknown K3_PROFILE '$PROFILE' (prod|prod01|margin10|ceiling|bare)" | tee -a $LOG >&2; exit 2 ;;
esac

echo "[launch] $NAME profile=$PROFILE placement=sglang-default(uniform)" >> $LOG

# Every flag below can be overridden by passing it again in the extra args:
# argparse keeps the last value (store_true flags excepted -- see the warning
# at the top of this file).
exec python -m sglang.launch_server \
  --model-path $WS/k3 --trust-remote-code --tp 8 --port ${K3_PORT:-30000} --host 127.0.0.1 \
  --kt-method MXFP4 --kt-weight-path $WS/k3 \
  --kt-num-gpu-experts 620 \
  --kt-threadpool-count $KTPOOLS --kt-cpuinfer $CPUINF \
  --kt-transport doorbell \
  "${ROUTING[@]}" \
  --moe-a2a-backend none --moe-runner-backend flashinfer_mxfp4 \
  "$@" >> $LOG 2>&1
