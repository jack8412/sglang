#!/bin/bash
# usage: launch.sh <name> [extra server args...]
# The K3 server launcher. Lives in the repo so a bootstrapped node has it
# (k3.sh serve runs it from the node's sglang checkout); the previous copy
# lived only on the node and evaporated with the rental.
#
# Env: only what sglang itself requires for K3 MXFP4 on Blackwell.
# NOTE: --kt-gpu-prefill-token-threshold is deliberately absent (and now
# defaults to unset): the full-GPU sweep is 5.2x slower than margin-routed
# prefill, costs 7.54 GiB/GPU, and is refused at config time when
# --kt-routing-margin is set.
NAME=${1:?usage: launch.sh <name> [extra server args...]}; shift
WS=/workspace
mkdir -p $WS/runs/{status,probes,logs,meta} $WS/runs/edr
LOG=$WS/runs/logs/$NAME.server.log
: > $LOG
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=$WS/runs/edr
# Prebuilt trtllm-gen MoE cubins. NOT optional for this config: with
# --moe-runner-backend flashinfer_mxfp4 on SM100 the server REFUSES to start
# without a valid pool (overrides.py raises; there is no pool-less JIT path).
# The guard exists so a missing pool fails with THIS line in the log instead
# of only the RuntimeError.
CUBIN_POOL=/opt/trtllm_gen_moe_cubin_pool/trtllm_gen_moe_cubin_pool_20260617_v0613rc1
if [ -d "$CUBIN_POOL" ]; then
  export SGLANG_TRTLLM_GEN_MOE_CUBIN_POOL=$CUBIN_POOL
else
  echo "WARNING: cubin pool missing at $CUBIN_POOL -- flashinfer_mxfp4 on SM100 will refuse to start" >> $LOG
fi
source $WS/venv-k3/bin/activate
# Sized to the node, not hardcoded to a rental that no longer exists: one
# threadpool per NUMA node; cpuinfer ~85% of PHYSICAL cores (RUNBOOK step 3 --
# AMX is a per-core resource, SMT siblings add nothing). lscpu shows HOST
# topology and is cgroup-blind, so clamp by nproc (which honors the cpuset):
# on a container rental granted fewer CPUs than the host has, nproc wins.
PHYS=$(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)
[ "$PHYS" -gt 0 ] || PHYS=$(nproc)
[ "$PHYS" -gt "$(nproc)" ] && PHYS=$(nproc)
NUMAN=$(numactl --hardware 2>/dev/null | awk '/^available:/{print $2}')
[ -n "$NUMAN" ] || NUMAN=2
# Placement: frequency from the newest recorded expert-distribution dump when
# one exists (the shipping recipe -- uniform placement was the entire quality
# gap, SPEC-MARGIN-ROUTING F2), else uniform. A measurement phase that must
# hold placement constant across rows sets K3_PLACE=off: overriding the
# strategy flag alone is NOT enough, because --init-expert-location would
# still be passed and it is a generic sglang arg, not a kt one.
PLACE=""
if [ "${K3_PLACE:-auto}" != "off" ]; then
  DUMP=$(ls -t $WS/runs/edr/*.pt 2>/dev/null | head -1)
  [ -n "$DUMP" ] && PLACE="--kt-expert-placement-strategy frequency --init-expert-location $DUMP"
fi
# --expert-distribution-recorder-mode stat is armed at LAUNCH because it
# cannot be turned on later: without it /start_expert_distribution_record
# raises in the scheduler request loop and KILLS the server (RUNBOOK G3),
# and regenerating a placement profile needs exactly that recording. Idle
# cost is zero until /start is called.
#
# Every flag below can be overridden by passing it again in the extra args:
# argparse keeps the last value.
exec python -m sglang.launch_server \
  --model-path $WS/k3 --trust-remote-code --tp 8 --port 31000 --host 127.0.0.1 \
  --kt-method MXFP4 --kt-weight-path $WS/k3 \
  --kt-num-gpu-experts 620 \
  --kt-threadpool-count $NUMAN --kt-cpuinfer $((PHYS * 85 / 100)) \
  --kt-expert-placement-strategy uniform $PLACE \
  --expert-distribution-recorder-mode stat \
  --moe-a2a-backend none --moe-runner-backend flashinfer_mxfp4 \
  --mem-fraction-static 0.90 --context-length 32768 \
  --chunked-prefill-size 16384 \
  "$@" >> $LOG 2>&1
