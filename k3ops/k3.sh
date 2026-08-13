#!/bin/bash
# k3.sh -- bring-up, deploy and health for the K3 hybrid build.
#
# Run from the VM; it drives the GPU node over ssh. Every subcommand is
# idempotent and verifies what it did, because each check below exists for a
# failure that actually happened and cost real time.
#
#   ./k3ops/k3.sh doctor      read-only: is the node sane, and if not, what fixes it
#   ./k3ops/k3.sh bootstrap [--weights]   freshly rented node: repos, venv, weights
#   ./k3ops/k3.sh provision   destroy and rebuild the venv
#   ./k3ops/k3.sh deploy      push local -> pull on node -> rebuild kt -> verify
#   ./k3ops/k3.sh env         repair the venv (torch/kt/flashinfer pinning)
#   ./k3ops/k3.sh serve NAME [args...]   launch a server and wait for health
#   ./k3ops/k3.sh stop        shut a server down without orphaning GPUs
#
# New rental:  bootstrap --weights   then  doctor  (rerun doctor to watch the
# weight download; bootstrap provisions the venv itself and detaches the
# download, so one command brings the node from nothing to ready).
#
# HOST defaults to gpusrv; pass K3_HOST=... to override.
#
# ---------------------------------------------------------------------------
# THE SEVEN FAILURES THIS ENCODES  (do not "simplify" these away)
# ---------------------------------------------------------------------------
# 1. kt-kernel MUST build with `--no-deps --no-build-isolation`. A plain
#    `pip install .` resolves its torch dependency from PyPI, uninstalls the
#    pinned torch 2.13.0+cu130 and installs 2.9.1+cu128 plus ~15 nvidia-*-cu12
#    packages. Every server then dies with
#    `RuntimeError: operator torchvision::nms does not exist`.
# 2. If that happens, do NOT `pip uninstall` the nvidia-*-cu12 packages. They
#    share files with the cu13 variants under the `nvidia/` namespace, so
#    removing them deletes libcusparseLt.so.0 / libnvshmem_host.so.3 that the
#    cu13 packages own, and torch then fails to import for a different reason.
#    `env` handles this correctly; `doctor` detects it.
# 3. NEVER rebuild kt while a server is running -- it swaps site-packages under
#    a live process. `deploy` refuses.
# 4. `pkill -f "sglang::scheduler"` inside an `ssh host '...'` one-liner matches
#    the remote shell's OWN command line and kills it, so the command dies
#    silently with no output. Every remote block here is fed over stdin
#    (`ssh host bash -s`), so the remote command line is just `bash -s`.
# 5. `set -e` plus a `pkill` that matched nothing (exit 1) aborts a script
#    silently. Nothing here runs under `set -e` with a pkill in it.
# 6. The documented bring-up order was circular: `bootstrap --weights` ran the
#    download with the venv's `hf`, but the venv was only created by
#    `provision`, which the order placed AFTER bootstrap -- so a fresh rental
#    died at the weights step. bootstrap now provisions the venv itself
#    (keeping a healthy one, so a rerun cannot destroy a live env) and runs
#    the download DETACHED in tmux with a completion sentinel: run in the ssh
#    session's foreground, hour N of a 1.6 TB download dies with the first
#    dropped connection.
# 7. kt-kernel does not build from a bare clone: its cmake add_subdirectory's
#    third_party/{pybind11,llama.cpp}, which are SUBMODULES of the
#    ktransformers repo. On the old node they were populated by hand and the
#    procedure was never recorded; the first deploy to a fresh node died with
#    "Unknown CMake command pybind11_add_module". bootstrap AND deploy run
#    `git submodule update --init --recursive` after every checkout/reset.
#
# Code reaches the node by git push + pull on the PRIVATE remotes, never scp:
# scp leaves the node's git state diverged from what was tested. Note the
# remote names differ -- VM: `jack` / `origin`(kt), node: `origin` for both.
# The server launcher is k3ops/launch.sh IN THE REPO for the same reason: the
# old copy lived only on the node and evaporated with the rental.
set -u

HOST=${K3_HOST:-gpusrv}
SGLANG_LOCAL=${K3_SGLANG:-/home/user/sglang}
KT_LOCAL=${K3_KT:-/home/user/ktransformers}
# Follow whatever branch this checkout is on, so starting a new work branch
# needs no edit here -- a stale pin silently deploys the OLD branch and the
# sha check then fails with the node looking mysteriously behind.
BRANCH_SGLANG=${K3_BRANCH:-$(git -C "$SGLANG_LOCAL" branch --show-current)}
BRANCH_KT=${K3_BRANCH_KT:-feat/mxfp4-kimi-k3}
REMOTE_WS=/workspace
VENV=$REMOTE_WS/venv-k3
TORCH_PIN=2.13.0

say(){ echo "[k3] $*"; }
die(){ echo "[k3] FATAL: $*" >&2; exit 1; }
[ -n "$BRANCH_SGLANG" ] || die "$SGLANG_LOCAL is on a detached HEAD -- \
check out a branch, or set K3_BRANCH to the one to deploy"
rsh(){ ssh -o ConnectTimeout=45 -o ServerAliveInterval=30 "$HOST" bash -s; }

# Prints ENV-PROBE:OK / ENV-PROBE:BAD. Used by bootstrap to decide between
# "keep" and "provision". The marker (not the exit code) carries the verdict:
# a dropped ssh during the probe must NOT read as "venv broken" -- that
# misread would rm -rf a healthy venv, and the probe is likeliest to stall
# exactly while the 1.6 TB download saturates the link.
env_state(){
  rsh <<EOS
VENV=$VENV; TORCH_PIN=$TORCH_PIN
source \$VENV/bin/activate 2>/dev/null || { echo ENV-PROBE:BAD; exit 0; }
python -c "import torch; assert torch.__version__.startswith('\$TORCH_PIN')" 2>/dev/null || { echo ENV-PROBE:BAD; exit 0; }
python -c "import torch, flashinfer, sgl_kernel, kt_kernel, sglang" 2>/dev/null || { echo ENV-PROBE:BAD; exit 0; }
echo ENV-PROBE:OK
EOS
}

# --------------------------------------------------------------------------
# doctor -- read-only. Names the failure AND the command that fixes it.
# --------------------------------------------------------------------------
cmd_doctor(){
  say "host=$HOST"
  rsh <<EOS
WS=$REMOTE_WS; VENV=$VENV; TORCH_PIN=$TORCH_PIN
echo "--- node"
hostname; uptime | sed 's/^ *//'
echo "--- GPUs (used MiB; anything nonzero means a server is holding them)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -8
echo "--- server"
pgrep -f "launch_serve[r]" >/dev/null && echo "RUNNING (deploy will refuse)" || echo "none"
echo "--- repos"
if [ -d \$WS/sglang/.git ]; then
  (cd \$WS/sglang && echo "sglang  \$(git log --oneline -1)  [\$(git rev-parse --abbrev-ref HEAD)]")
else echo "sglang MISSING -> run: k3.sh bootstrap"; fi
if [ -d \$WS/ktransformers/.git ]; then
  (cd \$WS/ktransformers && echo "kt      \$(git log --oneline -1)  [\$(git rev-parse --abbrev-ref HEAD)]")
else echo "kt MISSING -> run: k3.sh bootstrap"; fi
[ -d \$WS/runs/status ] || echo "runs/ tree MISSING -> run: k3.sh bootstrap"
POOL_VER=trtllm_gen_moe_cubin_pool_20260617_v0613rc1
[ -d /opt/trtllm_gen_moe_cubin_pool/\$POOL_VER ] || [ -d \$WS/trtllm_gen_moe_cubin_pool/\$POOL_VER ] \
  || echo "cubin pool MISSING (K3 mxfp4 on SM100 cannot start) -> run: k3.sh bootstrap"
echo "--- weights"
if [ -f \$WS/k3/.download-complete ]; then
  echo "k3 complete (\$(du -sh \$WS/k3 2>/dev/null | cut -f1))"
elif command -v tmux >/dev/null && tmux has-session -t k3dl 2>/dev/null; then
  echo "k3 DOWNLOADING: \$(du -sh \$WS/k3 2>/dev/null | cut -f1) so far (tmux k3dl; log \$WS/runs/logs/k3-download.log)"
elif [ -f \$WS/k3/config.json ]; then
  echo "k3 present but UNVERIFIED (\$(du -sh \$WS/k3 2>/dev/null | cut -f1), no completion sentinel)"
  echo "  -> k3.sh bootstrap --weights resumes/verifies (cheap when hf's own"
  echo "     .cache metadata is present; a tree copied without it re-reads ~1.6 TB)"
else
  echo "k3 ABSENT -> run: k3.sh bootstrap --weights (moonshotai/Kimi-K3, ~1.6 TB)"
fi
echo "--- env"
source \$VENV/bin/activate 2>/dev/null || { echo "NO VENV at \$VENV -> run: k3.sh bootstrap (fresh node) or k3.sh provision"; exit 0; }
TV=\$(python -c "import torch;print(torch.__version__)" 2>/dev/null)
if [ -z "\$TV" ]; then
  echo "torch DOES NOT IMPORT"
  python -c "import torch" 2>&1 | tail -2
  if pip list 2>/dev/null | grep -q "nvidia.*cu12"; then
    echo "  -> cu12 packages present alongside cu13; see failure #1/#2 -> run: k3.sh env"
  else
    echo "  -> shared nvidia namespace files may be missing -> run: k3.sh env"
  fi
  exit 0
fi
case "\$TV" in
  \$TORCH_PIN*) echo "torch \$TV  OK" ;;
  *) echo "torch \$TV  WRONG (pin \$TORCH_PIN) -> a bare 'pip install .' replaced it -> run: k3.sh env" ;;
esac
CU12=\$(pip list 2>/dev/null | grep -ci "nvidia.*cu12")
[ "\$CU12" -gt 1 ] && echo "  WARNING: \$CU12 nvidia-*-cu12 packages present; expected only nvidia-cutlass-dsl-libs-cu12"
python -c "import torch, flashinfer, sgl_kernel, kt_kernel, sglang; print('ENV-OK', torch.__version__, flashinfer.__version__, sglang.__version__)" 2>&1 | tail -2
python -c "from kt_kernel.experts_base import BaseMoEWrapper as B; print('kt packed-staging:', 'submit_forward_packed' in dir(B))" 2>&1 | tail -1
EOS
}

# --------------------------------------------------------------------------
# deploy -- the ONLY sanctioned way code reaches the node.
# --------------------------------------------------------------------------
cmd_deploy(){
  # Push first, and refuse to deploy something that only exists locally: the
  # node pulls from the private remotes, so an unpushed commit silently
  # deploys the PREVIOUS one and the run measures the wrong code.
  git -C "$SGLANG_LOCAL" push -q jack "$BRANCH_SGLANG" || die "sglang push failed"
  git -C "$KT_LOCAL" push -q origin "$BRANCH_KT" || die "kt push failed"
  # Full shas: `--short` picks its own length per repo, so comparing a local
  # short sha against the node's short sha reports a false mismatch.
  local s_sha k_sha
  s_sha=$(git -C "$SGLANG_LOCAL" rev-parse HEAD)
  k_sha=$(git -C "$KT_LOCAL" rev-parse HEAD)
  say "deploying sglang=$s_sha kt=$k_sha to $HOST"

  rsh <<EOS
WS=$REMOTE_WS; VENV=$VENV; TORCH_PIN=$TORCH_PIN
# Failure #3: a rebuild swaps site-packages under a live process.
if pgrep -f "launch_serve[r]" >/dev/null; then
  echo "REFUSING: a server is running. Stop it first (k3.sh stop)."
  exit 1
fi
cd \$WS/sglang && git fetch -q origin $BRANCH_SGLANG && git reset -q --hard FETCH_HEAD && echo "sglang \$(git log --oneline -1)"
cd \$WS/ktransformers && git fetch -q origin $BRANCH_KT && git reset -q --hard FETCH_HEAD && echo "kt     \$(git log --oneline -1)"
# Failure #7: the reset can move submodule pointers; empty submodules kill
# the kt build at cmake configure. Only the two the build consumes -- the
# other two (custom_flashinfer, a whole sglang fork) are dead weight here.
git submodule update --init --recursive third_party/pybind11 third_party/llama.cpp >/dev/null 2>&1 \
  || { echo "kt submodule update FAILED"; exit 1; }
mkdir -p \$WS/runs/logs
source \$VENV/bin/activate
# Failure #1: --no-deps --no-build-isolation is not optional.
cd \$WS/ktransformers/kt-kernel && pip install . --no-deps --no-build-isolation > \$WS/runs/logs/ktbuild.log 2>&1 \
  && echo "kt build OK" || { echo "kt build FAILED"; tail -25 \$WS/runs/logs/ktbuild.log; exit 1; }
python -c "import torch; assert torch.__version__.startswith('\$TORCH_PIN'), 'kt downgraded torch: '+torch.__version__; print('torch still', torch.__version__)" || exit 1
python -c "import torch, flashinfer, sgl_kernel, kt_kernel, sglang; print('ENV-OK', torch.__version__, flashinfer.__version__, sglang.__version__)"
EOS
  local rc=$?
  [ $rc -eq 0 ] || die "deploy failed (rc=$rc)"
  # The node must end up on exactly what we pushed -- BOTH repos, or a
  # measurement attributes itself to the wrong commit.
  local node_sha node_kt_sha
  node_sha=$(ssh -n -o ConnectTimeout=45 "$HOST" "cd $REMOTE_WS/sglang && git rev-parse HEAD" 2>/dev/null | tail -1)
  [ "$node_sha" = "$s_sha" ] || die "node sglang is at ${node_sha:0:12}, expected ${s_sha:0:12}"
  node_kt_sha=$(ssh -n -o ConnectTimeout=45 "$HOST" "cd $REMOTE_WS/ktransformers && git rev-parse HEAD" 2>/dev/null | tail -1)
  [ "$node_kt_sha" = "$k_sha" ] || die "node kt is at ${node_kt_sha:0:12}, expected ${k_sha:0:12}"
  say "deployed and verified: sglang=${s_sha:0:12} kt=${k_sha:0:12}"
}

# --------------------------------------------------------------------------
# env -- repair a broken venv, in the order that actually works.
# --------------------------------------------------------------------------
cmd_env(){
  say "repairing env on $HOST"
  rsh <<EOS
WS=$REMOTE_WS; VENV=$VENV; TORCH_PIN=$TORCH_PIN
if pgrep -f "launch_serve[r]" >/dev/null; then echo "REFUSING: server running"; exit 1; fi
export UV_CACHE_DIR=\$WS/.cache/uv PIP_CACHE_DIR=\$WS/.cache/pip UV_HTTP_TIMEOUT=600
mkdir -p \$WS/runs/logs
source \$VENV/bin/activate

# Failure #2: if a bad install already removed cu12 packages, the cu13 variants
# lost shared files under nvidia/. Restore every nvidia package's files first;
# --force-reinstall --no-deps rewrites them without touching torch.
if ! python -c "import torch" >/dev/null 2>&1; then
  echo "torch does not import; restoring nvidia namespace files"
  PKGS=\$(pip list 2>/dev/null | awk '/^nvidia-/ {print \$1"=="\$2}' | tr '\n' ' ')
  [ -n "\$PKGS" ] && pip install --force-reinstall --no-deps \$PKGS > \$WS/runs/logs/nvfix.log 2>&1
fi

# Reinstalling sglang editable pulls the pinned torch back from the sglang index.
uv pip install --prerelease=allow --index-strategy unsafe-best-match \
  --extra-index-url https://docs.sglang.ai/whl/cu130/ -e \$WS/sglang/python/ || exit 1
python -c "import torch; assert torch.__version__.startswith('\$TORCH_PIN'), torch.__version__; print('torch pinned:', torch.__version__)" || exit 1
cd \$WS/ktransformers/kt-kernel && pip install . --no-deps --no-build-isolation > \$WS/runs/logs/ktbuild.log 2>&1 \
  || { echo "kt build FAILED"; tail -25 \$WS/runs/logs/ktbuild.log; exit 1; }
python -c "import torch; assert torch.__version__.startswith('\$TORCH_PIN'), 'kt downgraded torch: '+torch.__version__"
python -c "import torch, flashinfer, sgl_kernel, kt_kernel, sglang; print('ENV-OK', torch.__version__, flashinfer.__version__, sglang.__version__)"
EOS
  [ $? -eq 0 ] || die "env repair failed"
  say "env repaired"
}

# --------------------------------------------------------------------------
# bootstrap -- a freshly rented node, from nothing, in ONE command.
#
# Order minimizes paid hours (RUNBOOK step 0): the 1.6 TB download starts
# EARLY -- right after the ~1 min prereq/repo step -- detached, and the
# venv build runs behind it.
#   1. prerequisites + repos + runs/ tree     (~1 min)
#   2. weight download, DETACHED in tmux      (hours; resumable; sentinel
#      $WS/k3/.download-complete written only on a clean `hf download` exit)
#   3. venv + packages                        (kept when already healthy;
#      rebuilt via provision otherwise)
#
# Weights are 1.6 TB and are NOT pulled unless --weights is passed, so a stray
# invocation cannot start a day-long download. Rerunning `bootstrap --weights`
# after any interruption resumes the download instead of restarting it.
#
# What this CANNOT do: the expert-distribution dump that frequency placement
# needs (runs/edr/*.pt, ~333 MB) is an OUTPUT of a profiling run, not an
# artifact to download. On a fresh node it must be regenerated --
# /start_expert_distribution_record, a representative workload, then dump --
# or copied from a previous node before that node is released. Copy it while
# you still can; regenerating costs a full run.
# --------------------------------------------------------------------------
cmd_bootstrap(){
  local want_weights=0
  [ "${1:-}" = "--weights" ] && want_weights=1
  say "bootstrapping $HOST (weights=$want_weights)"

  # ---- 1. prerequisites, repos, results tree (cheap, always) --------------
  rsh <<EOS
WS=$REMOTE_WS
WANTW=$want_weights

echo "--- prerequisites (fatal: serve and the download depend on them)"
missing=""
for t in git uv tmux numactl nvidia-smi curl; do
  command -v \$t >/dev/null || missing="\$missing \$t"
done
[ -n "\$missing" ] && { echo "MISSING:\$missing -- install these first"; exit 1; }
echo "ok"
echo "--- disk"
df -h \$WS | tail -1

echo "--- repos"
# Both remotes are anonymously readable over https, so no credentials are
# needed to CLONE. Pushing is a different matter and is never done from the
# node -- the node only ever pulls.
if [ -d \$WS/sglang/.git ]; then echo "sglang present"; else
  git clone -q https://github.com/jack8412/sglang.git \$WS/sglang || exit 1
  echo "sglang cloned"
fi
cd \$WS/sglang && git fetch -q origin $BRANCH_SGLANG && git checkout -q -B $BRANCH_SGLANG FETCH_HEAD && echo "sglang \$(git log --oneline -1)"

if [ -d \$WS/ktransformers/.git ]; then echo "ktransformers present"; else
  git clone -q https://github.com/jack8412/ktransformers.git \$WS/ktransformers || exit 1
  echo "ktransformers cloned"
fi
cd \$WS/ktransformers && git fetch -q origin $BRANCH_KT && git checkout -q -B $BRANCH_KT FETCH_HEAD && echo "kt \$(git log --oneline -1)"
# Failure #7: kt-kernel's cmake needs third_party/{pybind11,llama.cpp} --
# submodules a bare clone leaves empty. Only those two: custom_flashinfer
# and third_party/sglang (a whole fork) are not consumed by the kernel build.
git submodule update --init --recursive third_party/pybind11 third_party/llama.cpp >/dev/null 2>&1 \
  && echo "kt submodules ready" || { echo "kt submodule update FAILED"; exit 1; }

echo "--- results layout"
# The node mirrors this repo's runs/ layout, so pulling results is one rsync
# with an exclude list rather than a tar per category. Phase scripts write
# here; nothing downstream needs to know where a given kind of file lives.
mkdir -p \$WS/runs/{status,probes,logs,meta} \$WS/runs/edr
echo "runs/{status,probes,logs,meta,edr} ready at \$WS/runs"

echo "--- weights (state only; --weights acts on it below)"
if [ -f \$WS/k3/.download-complete ]; then
  echo "k3 complete (\$(du -sh \$WS/k3 2>/dev/null | cut -f1))"
elif tmux has-session -t k3dl 2>/dev/null; then
  echo "k3 DOWNLOADING (\$(du -sh \$WS/k3 2>/dev/null | cut -f1) so far, tmux k3dl)"
elif [ -f \$WS/k3/config.json ]; then
  echo "k3 present but unverified -- bootstrap --weights resumes/verifies"
elif [ \$WANTW -eq 1 ]; then
  echo "k3 absent -- the download starts below"
else
  echo "k3 ABSENT -- pass --weights to download (moonshotai/Kimi-K3, ~1.6 TB)"
fi

echo "--- trtllm-gen MoE cubin pool (K3 mxfp4 on SM100 REFUSES to start without it)"
# The old image shipped it under /opt; a fresh image does not, and there is
# no pool-less JIT path for flashinfer_mxfp4 on SM100 (overrides.py raises).
POOL_VER=trtllm_gen_moe_cubin_pool_20260617_v0613rc1
if [ -d /opt/trtllm_gen_moe_cubin_pool/\$POOL_VER ] || [ -d \$WS/trtllm_gen_moe_cubin_pool/\$POOL_VER ]; then
  echo "cubin pool present"
else
  echo "cubin pool ABSENT -- fetching from the sgl-project/whl release"
  mkdir -p \$WS/trtllm_gen_moe_cubin_pool
  curl -fsSL -o \$WS/trtllm_gen_moe_cubin_pool/\$POOL_VER.zip \
    https://github.com/sgl-project/whl/releases/download/trtllm_gen_moe_cubin_20260617/\$POOL_VER.zip \
    || { echo "cubin pool download FAILED"; exit 1; }
  python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
    \$WS/trtllm_gen_moe_cubin_pool/\$POOL_VER.zip \$WS/trtllm_gen_moe_cubin_pool \
    || { echo "cubin pool unzip FAILED"; exit 1; }
  rm -f \$WS/trtllm_gen_moe_cubin_pool/\$POOL_VER.zip
  [ -d \$WS/trtllm_gen_moe_cubin_pool/\$POOL_VER ] && echo "cubin pool installed at \$WS/trtllm_gen_moe_cubin_pool/\$POOL_VER" \
    || { echo "cubin pool layout unexpected after unzip"; exit 1; }
fi

echo "--- placement profile (frequency placement needs this)"
ls \$WS/runs/edr/*.pt >/dev/null 2>&1 \
  && echo "edr dump present: \$(ls -1 \$WS/runs/edr/*.pt | head -1)" \
  || echo "edr dump ABSENT -- it is a RUN OUTPUT, not a download. Copy it from the
   previous node before releasing it, or regenerate with
   /start_expert_distribution_record + a representative workload."
EOS
  [ $? -eq 0 ] || die "bootstrap failed (prerequisites/repos)"

  # ---- 2. weights, detached, BEFORE the venv build so the download runs ----
  # ---- behind it (failure #6)                                          ----
  if [ $want_weights -eq 1 ]; then
    rsh <<EOS
WS=$REMOTE_WS
export UV_CACHE_DIR=\$WS/.cache/uv
if [ -f \$WS/k3/.download-complete ]; then
  echo "weights already complete -- nothing to do"; exit 0
fi
if tmux has-session -t k3dl 2>/dev/null; then
  echo "download already running (\$(du -sh \$WS/k3 2>/dev/null | cut -f1) so far) -- not starting a second"
  exit 0
fi
# A download started by hand (outside tmux) must not get a competitor.
if pgrep -f "h[f] download" >/dev/null; then
  echo "an hf download is already running outside tmux -- not starting a second"
  exit 0
fi
# Sized from the repo manifest rather than guessed: moonshotai/Kimi-K3 is 118
# files totalling 1454 GiB (HF API with blobs=true, checked 2026-08-13). hf
# builds each file as a partial under the local dir and RENAMES it into place
# on the same filesystem, so peak usage is that total plus metadata -- it does
# not transiently double. 1500 keeps ~46 GiB of slack and still refuses a disk
# that genuinely cannot finish. The previous 1800 had no manifest behind it and
# refused a 1777 GiB B300 node with 300+ GiB to spare.
NEED_TOTAL=1500
# The gate must credit bytes already downloaded, or it refuses the very
# resume it advertises: a half-downloaded k3 on a right-sized disk has
# LESS than \$NEED_TOTAL GiB free precisely because the download made progress.
HAVE=\$(du -sBG \$WS/k3 2>/dev/null | cut -f1 | tr -dc '0-9'); HAVE=\${HAVE:-0}
FREE=\$(df -BG --output=avail \$WS | tail -1 | tr -dc '0-9')
[ -z "\$FREE" ] && { echo "REFUSING: cannot determine free space on \$WS"; exit 1; }
NEED=\$((NEED_TOTAL - HAVE)); [ \$NEED -lt 0 ] && NEED=0
[ "\$FREE" -lt "\$NEED" ] && { echo "REFUSING: \$FREE GiB free but ~\$NEED GiB still needed (have \$HAVE GiB of ~1.45 TB)"; exit 1; }
# Keep the previous attempt's tail: the rerun is the documented recovery
# step and must not erase the evidence of why the last attempt died.
[ -f \$WS/runs/logs/k3-download.log ] && mv \$WS/runs/logs/k3-download.log \$WS/runs/logs/k3-download.log.1
# 'uv tool run' provides hf without the venv (failure #6): the download must
# not wait on -- or be able to destroy -- the env build. hf skips finished
# files, so this same command resumes after any interruption.
# (NO BACKTICKS in comments inside unquoted heredocs -- they execute LOCALLY
# during expansion and splice their output into the remote script.)
tmux new-session -d -s k3dl "HF_HUB_ENABLE_HF_TRANSFER=1 UV_CACHE_DIR=\$WS/.cache/uv uv tool run --from 'huggingface_hub[hf_transfer]' hf download moonshotai/Kimi-K3 --local-dir \$WS/k3 > \$WS/runs/logs/k3-download.log 2>&1 && touch \$WS/k3/.download-complete"
sleep 5
if tmux has-session -t k3dl 2>/dev/null; then
  echo "download STARTED (detached, tmux k3dl) -> \$WS/k3"
  echo "watch: k3.sh doctor, or on the node: tail -f \$WS/runs/logs/k3-download.log"
elif [ -f \$WS/k3/.download-complete ]; then
  # A resume/verify over an already-complete tree can finish inside the 5 s
  # window; the vanished session is success, not failure.
  echo "download COMPLETE (verified in-place within 5 s)"
else
  echo "download FAILED at start:"; tail -5 \$WS/runs/logs/k3-download.log 2>/dev/null; exit 1
fi
EOS
    [ $? -eq 0 ] || die "bootstrap failed (weights)"
  fi

  # ---- 3. venv: keep a healthy one, otherwise provision -------------------
  # (kt staleness after a repo update is deploy's job, not bootstrap's)
  local envst
  envst=$(env_state)
  case "$envst" in
    *ENV-PROBE:OK*)  say "venv healthy -- keeping it (k3.sh deploy rebuilds kt when the repos move)" ;;
    *ENV-PROBE:BAD*) say "venv absent or broken -- provisioning"; cmd_provision ;;
    *) die "env probe returned nothing (ssh dropped?) -- NOT provisioning blind; rerun bootstrap" ;;
  esac
  say "bootstrap done -> k3.sh doctor (rerun it to watch the weight download)"
}

# --------------------------------------------------------------------------
# provision -- destroys and rebuilds the venv.
# --------------------------------------------------------------------------
cmd_provision(){
  say "provisioning a fresh venv on $HOST (destroys $VENV)"
  rsh <<EOS
WS=$REMOTE_WS; VENV=$VENV; TORCH_PIN=$TORCH_PIN
if pgrep -f "launch_serve[r]" >/dev/null; then echo "REFUSING: server running"; exit 1; fi
export UV_CACHE_DIR=\$WS/.cache/uv PIP_CACHE_DIR=\$WS/.cache/pip UV_HTTP_TIMEOUT=600
[ -d \$WS/sglang ] || { echo "FATAL: \$WS/sglang missing -- run: k3.sh bootstrap"; exit 1; }
[ -d \$WS/ktransformers ] || { echo "FATAL: \$WS/ktransformers missing -- run: k3.sh bootstrap"; exit 1; }
rm -rf \$VENV
uv venv \$VENV --python 3.12 --seed || exit 1
source \$VENV/bin/activate
[ "\$(command -v pip)" = "\$VENV/bin/pip" ] || { echo "FATAL pip outside venv"; exit 9; }
uv pip install --prerelease=allow --index-strategy unsafe-best-match \
  --extra-index-url https://docs.sglang.ai/whl/cu130/ -e \$WS/sglang/python/ || exit 1
python -c "import torch; assert torch.__version__.startswith('\$TORCH_PIN'), torch.__version__; print('STEP1 torch', torch.__version__, torch.version.cuda)" || exit 1
uv pip install --index-url https://flashinfer.ai/whl "flashinfer-cubin==0.6.15.post1" || exit 1
uv pip install --index-url https://flashinfer.ai/whl/cu130 "flashinfer-jit-cache==0.6.15.post1" || exit 1
python -c "import torch, flashinfer, sgl_kernel; print('STEP2 sgl_kernel+flashinfer OK')" || exit 1
cd \$WS/ktransformers/kt-kernel && pip install . --no-deps --no-build-isolation || exit 1
python -c "import torch; assert torch.__version__.startswith('\$TORCH_PIN'), 'kt downgraded torch: '+torch.__version__"
python -c "import torch, flashinfer, sgl_kernel, kt_kernel, sglang; print('ENV-OK', torch.__version__, flashinfer.__version__, sglang.__version__)"
EOS
  [ $? -eq 0 ] || die "provision failed"
  say "provisioned"
}

# --------------------------------------------------------------------------
# serve / stop -- the launcher is k3ops/launch.sh from the NODE's checkout,
# so it exists on any bootstrapped node and is versioned with the code it
# launches (the old copy lived only on the node and died with the rental).
# --------------------------------------------------------------------------
cmd_serve(){
  local name=${1:?usage: k3.sh serve NAME [extra server args...]}; shift
  say "launching $name on $HOST: $*"
  # launch.sh reads K3_PROFILE / K3_RECORD from its environment, and ssh does
  # not carry them, so forward them explicitly -- without this the caller's
  # `K3_PROFILE=bare k3.sh serve ...` is silently ignored and the launcher
  # refuses (or worse, serves a different placement than was asked for).
  # ssh reassembles the remote command into ONE string that the remote shell
  # re-splits, so an empty argument disappears and every later argument shifts
  # -- which silently made NAME the first server flag. Hence the sentinel.
  ssh -o ConnectTimeout=45 -o ServerAliveInterval=30 "$HOST" bash -s -- \
    "${K3_PROFILE:-__unset__}" "${K3_RECORD:-__unset__}" "$name" "$@" <<'EOS'
K3_PROFILE=$1; shift
K3_RECORD=$1; shift
NAME=$1; shift
[ "$K3_PROFILE" = __unset__ ] && K3_PROFILE=""
[ "$K3_RECORD" = __unset__ ] && K3_RECORD=""
WS=/workspace
LAUNCHER=$WS/sglang/k3ops/launch.sh
LOG=$WS/runs/logs/$NAME.server.log
[ -f "$LAUNCHER" ] || { echo "FATAL: $LAUNCHER missing -- node checkout predates it; run k3.sh deploy"; exit 1; }
# Serving against a half-downloaded checkpoint burns a full load cycle and
# dies with a cryptic shard/EOF traceback -- refuse while the download runs.
if tmux has-session -t k3dl 2>/dev/null; then
  echo "REFUSING: k3 weight download still running (tmux k3dl) -- watch k3.sh doctor"; exit 1
fi
[ -f $WS/k3/config.json ] || { echo "FATAL: no weights at $WS/k3 -- run k3.sh bootstrap --weights"; exit 1; }
[ -f $WS/k3/.download-complete ] || echo "WARNING: no completion sentinel at $WS/k3 (manually fetched tree?) -- proceeding"
tmux kill-session -t bench 2>/dev/null
pkill -INT -f "launch_serve[r]" 2>/dev/null; sleep 15
pkill -9 -f "sglang::sched[u]ler" 2>/dev/null; sleep 5
ENVP=""
[ -n "$K3_PROFILE" ] && ENVP="K3_PROFILE=$K3_PROFILE "
[ -n "$K3_RECORD" ] && ENVP="${ENVP}K3_RECORD=$K3_RECORD "
tmux new-session -d -s bench -n "$NAME" "${ENVP}bash $LAUNCHER $NAME $*"
echo "launched; waiting for /health_generate (log: $LOG)"
# Grace period: the first pgrep can beat launch.sh's exec of the server, and
# a false DIED here cost a full campaign (phase V rows all "died" at t+0.2s).
sleep 15
n=0
while [ $n -lt 120 ]; do
  c=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:${K3_PORT:-30000}/health_generate 2>/dev/null)
  [ "$c" = "200" ] && { echo "HEALTHY"; exit 0; }
  # A dead launcher must not be waited on for 40 minutes.
  pgrep -f "launch_serve[r]" >/dev/null || {
    echo "DIED: $(grep -E 'Error|Traceback|Segmentation' $LOG 2>/dev/null | tail -4 | tr '\n' ' ')"; exit 1; }
  sleep 20; n=$((n+1))
done
echo "TIMEOUT"; exit 1
EOS
}

cmd_stop(){
  say "stopping any server on $HOST"
  rsh <<'EOS'
tmux kill-session -t bench 2>/dev/null
pkill -INT -f "launch_serve[r]" 2>/dev/null; sleep 15
pkill -9 -f "sglang::sched[u]ler" 2>/dev/null; sleep 5
pgrep -f "launch_serve[r]" >/dev/null && echo "STILL RUNNING" || echo "stopped"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -3
EOS
}

case "${1:-doctor}" in
  doctor)    cmd_doctor ;;
  deploy)    cmd_deploy ;;
  env)       cmd_env ;;
  bootstrap) shift; cmd_bootstrap "${1:-}" ;;
  provision) cmd_provision ;;
  serve)     shift; cmd_serve "$@" ;;
  stop)      cmd_stop ;;
  *) die "unknown subcommand '${1}'. One of: doctor bootstrap provision deploy env serve stop" ;;
esac
