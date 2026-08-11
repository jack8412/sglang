#!/bin/bash
# k3.sh -- bring-up, deploy and health for the K3 hybrid build.
#
# Run from the VM; it drives the GPU node over ssh. Every subcommand is
# idempotent and verifies what it did, because each check below exists for a
# failure that actually happened and cost real time.
#
#   ./k3ops/k3.sh doctor      read-only: is the node sane, and if not, what fixes it
#   ./k3ops/k3.sh deploy      push local -> pull on node -> rebuild kt -> verify
#   ./k3ops/k3.sh env         repair the venv (torch/kt/flashinfer pinning)
#   ./k3ops/k3.sh provision   build the venv from scratch on a fresh node
#   ./k3ops/k3.sh serve NAME [args...]   launch a server and wait for health
#   ./k3ops/k3.sh stop        shut a server down without orphaning GPUs
#
# HOST defaults to gpusrv; pass K3_HOST=... to override.
#
# ---------------------------------------------------------------------------
# THE FIVE FAILURES THIS ENCODES  (do not "simplify" these away)
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
#
# Code reaches the node by git push + pull on the PRIVATE remotes, never scp:
# scp leaves the node's git state diverged from what was tested. Note the
# remote names differ -- VM: `jack` / `origin`(kt), node: `origin` for both.
set -u

HOST=${K3_HOST:-gpusrv}
SGLANG_LOCAL=${K3_SGLANG:-/home/user/sglang}
KT_LOCAL=${K3_KT:-/home/user/ktransformers}
BRANCH_SGLANG=k3-hybrid
BRANCH_KT=feat/mxfp4-kimi-k3
REMOTE_WS=/workspace
VENV=$REMOTE_WS/venv-k3
TORCH_PIN=2.13.0

say(){ echo "[k3] $*"; }
die(){ echo "[k3] FATAL: $*" >&2; exit 1; }
rsh(){ ssh -o ConnectTimeout=45 -o ServerAliveInterval=30 "$HOST" bash -s; }

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
cd \$WS/sglang 2>/dev/null && echo "sglang  \$(git log --oneline -1)  [\$(git rev-parse --abbrev-ref HEAD)]"
cd \$WS/ktransformers 2>/dev/null && echo "kt      \$(git log --oneline -1)  [\$(git rev-parse --abbrev-ref HEAD)]"
echo "--- env"
source \$VENV/bin/activate 2>/dev/null || { echo "NO VENV at \$VENV -> run: k3.sh provision"; exit 0; }
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
  local s_sha k_sha
  s_sha=$(git -C "$SGLANG_LOCAL" rev-parse --short HEAD)
  k_sha=$(git -C "$KT_LOCAL" rev-parse --short HEAD)
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
source \$VENV/bin/activate
# Failure #1: --no-deps --no-build-isolation is not optional.
cd \$WS/ktransformers/kt-kernel && pip install . --no-deps --no-build-isolation > \$WS/ktbuild.log 2>&1 \
  && echo "kt build OK" || { echo "kt build FAILED"; tail -25 \$WS/ktbuild.log; exit 1; }
python -c "import torch; assert torch.__version__.startswith('\$TORCH_PIN'), 'kt downgraded torch: '+torch.__version__; print('torch still', torch.__version__)" || exit 1
python -c "import torch, flashinfer, sgl_kernel, kt_kernel, sglang; print('ENV-OK', torch.__version__, flashinfer.__version__, sglang.__version__)"
EOS
  local rc=$?
  [ $rc -eq 0 ] || die "deploy failed (rc=$rc)"
  # The node must end up on exactly what we pushed, or a measurement attributes
  # itself to the wrong commit.
  local node_sha
  node_sha=$(ssh -n -o ConnectTimeout=45 "$HOST" "cd $REMOTE_WS/sglang && git rev-parse --short HEAD" 2>/dev/null | tail -1)
  [ "$node_sha" = "$s_sha" ] || die "node is at $node_sha, expected $s_sha"
  say "deployed and verified: sglang=$s_sha kt=$k_sha"
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
source \$VENV/bin/activate

# Failure #2: if a bad install already removed cu12 packages, the cu13 variants
# lost shared files under nvidia/. Restore every nvidia package's files first;
# --force-reinstall --no-deps rewrites them without touching torch.
if ! python -c "import torch" >/dev/null 2>&1; then
  echo "torch does not import; restoring nvidia namespace files"
  PKGS=\$(pip list 2>/dev/null | awk '/^nvidia-/ {print \$1"=="\$2}' | tr '\n' ' ')
  [ -n "\$PKGS" ] && pip install --force-reinstall --no-deps \$PKGS > \$WS/nvfix.log 2>&1
fi

# Reinstalling sglang editable pulls the pinned torch back from the sglang index.
uv pip install --prerelease=allow --index-strategy unsafe-best-match \
  --extra-index-url https://docs.sglang.ai/whl/cu130/ -e \$WS/sglang/python/ || exit 1
python -c "import torch; assert torch.__version__.startswith('\$TORCH_PIN'), torch.__version__; print('torch pinned:', torch.__version__)" || exit 1
cd \$WS/ktransformers/kt-kernel && pip install . --no-deps --no-build-isolation > \$WS/ktbuild.log 2>&1 \
  || { echo "kt build FAILED"; tail -25 \$WS/ktbuild.log; exit 1; }
python -c "import torch; assert torch.__version__.startswith('\$TORCH_PIN'), 'kt downgraded torch: '+torch.__version__"
python -c "import torch, flashinfer, sgl_kernel, kt_kernel, sglang; print('ENV-OK', torch.__version__, flashinfer.__version__, sglang.__version__)"
EOS
  [ $? -eq 0 ] || die "env repair failed"
  say "env repaired"
}

# --------------------------------------------------------------------------
# provision -- fresh node. Destroys and rebuilds the venv.
# --------------------------------------------------------------------------
cmd_provision(){
  say "provisioning a fresh venv on $HOST (destroys $VENV)"
  rsh <<EOS
WS=$REMOTE_WS; VENV=$VENV; TORCH_PIN=$TORCH_PIN
if pgrep -f "launch_serve[r]" >/dev/null; then echo "REFUSING: server running"; exit 1; fi
export UV_CACHE_DIR=\$WS/.cache/uv PIP_CACHE_DIR=\$WS/.cache/pip UV_HTTP_TIMEOUT=600
[ -d \$WS/sglang ] || { echo "FATAL: \$WS/sglang missing -- clone it first"; exit 1; }
[ -d \$WS/ktransformers ] || { echo "FATAL: \$WS/ktransformers missing -- clone it first"; exit 1; }
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
# serve / stop
# --------------------------------------------------------------------------
cmd_serve(){
  local name=${1:?usage: k3.sh serve NAME [extra server args...]}; shift
  say "launching $name on $HOST: $*"
  ssh -o ConnectTimeout=45 -o ServerAliveInterval=30 "$HOST" bash -s -- "$name" "$@" <<'EOS'
NAME=$1; shift
WS=/workspace; MB=$WS/margin-bench
tmux kill-session -t bench 2>/dev/null
pkill -INT -f "launch_serve[r]" 2>/dev/null; sleep 15
pkill -9 -f "sglang::sched[u]ler" 2>/dev/null; sleep 5
tmux new-session -d -s bench -n "$NAME" "bash $MB/launch.sh $NAME $*"
echo "launched; waiting for /health_generate"
n=0
while [ $n -lt 120 ]; do
  c=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:31000/health_generate 2>/dev/null)
  [ "$c" = "200" ] && { echo "HEALTHY"; exit 0; }
  # A dead launcher must not be waited on for 40 minutes.
  pgrep -f "launch_serve[r]" >/dev/null || {
    echo "DIED: $(grep -E 'Error|Traceback|Segmentation' $MB/$NAME.server.log | tail -4 | tr '\n' ' ')"; exit 1; }
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
  provision) cmd_provision ;;
  serve)     shift; cmd_serve "$@" ;;
  stop)      cmd_stop ;;
  *) die "unknown subcommand '${1}'. One of: doctor deploy env provision serve stop" ;;
esac
