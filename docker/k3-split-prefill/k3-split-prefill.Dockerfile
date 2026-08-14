# k3-hybrid serving image -- Kimi-K3 on 8xB200 with kt-kernel CPU experts.
#
# Built from THIS repo's working tree (the branch you are on), not a clone of
# upstream. That is the difference from docker/kimi_k3/kimi_k3_cu13.Dockerfile,
# which clones sgl-project/sglang main and contains no kt-kernel at all -- i.e.
# it cannot serve the hybrid CPU/GPU expert config this branch exists for.
#
# The build sequence below is `k3ops/k3.sh provision` transcribed, in the same
# order, with the same guards. That order is load-bearing: sglang's editable
# install is what pulls the pinned torch 2.13.0+cu130 from the sglang index, so
# it must precede kt-kernel, and kt-kernel must not be allowed to resolve its
# own torch dependency (see BUILD TRAP 1).
#
#   BR=$(git branch --show-current | tr '/' '-')
#   DOCKER_BUILDKIT=1 docker build -f docker/k3_hybrid/k3_hybrid.Dockerfile \
#     -t "k3:${BR}" \
#     --build-arg SGLANG_BUILD_COMMIT=$(git rev-parse HEAD) \
#     --build-arg SGLANG_BUILD_BRANCH="${BR}" .
#
# Tag from the BRANCH, not a fixed name: this tree moves between work branches
# (k3-hybrid -> k3-wip -> k3-split-prefill), and an image whose tag does not
# say which branch produced it cannot be told apart from the last one on a
# host that holds several. The directory and filename stay fixed on purpose --
# they name the image VARIANT (hybrid CPU/GPU expert serving), which outlives
# any one branch. `tr '/' '-'` because a slash in a branch name is legal in git
# and not in a docker tag.
#
# Run it (weights and results stay on the host; they outlive the image):
#
#   docker run --gpus all --ipc=host --shm-size=32g -p 30000:30000 \
#     -v /workspace/k3:/workspace/k3:ro \
#     -v /workspace/runs:/workspace/runs \
#     "k3:${BR}" bash /workspace/sglang/k3ops/launch.sh prod --host 0.0.0.0
#
# The trailing `--host 0.0.0.0` is REQUIRED to reach the server from outside
# the container: launch.sh hardcodes --host 127.0.0.1, so with -p alone the
# port forwards to a listener bound to the container's loopback and every
# request is refused. --host is a valued flag, so argparse keeps this last
# occurrence (store_true flags cannot be overridden this way -- see the warning
# at the top of launch.sh).
#
# Otherwise the image satisfies every path k3ops/launch.sh reaches for -- the
# venv at /workspace/venv-k3, the checkout at /workspace/sglang, the cubin pool
# under /opt, the runs/ tree -- so the launcher runs unmodified, and
# K3_PROFILE / K3_PORT behave exactly as they do on a bare node.
#
# --ipc=host (or a large --shm-size) is not decoration: tp 8 moves tensors
# through /dev/shm and the 64 MB default silently deadlocks NCCL init.
#
# NOT baked in, on purpose:
#   - the 1.45 TB checkpoint (mount at /workspace/k3)
#   - SGLANG_MAX_KV_CHUNK_CAPACITY=32768. It is the critical knob for 1M
#     context, but it is a per-run choice, not a property of the image; set it
#     on `docker run -e` for long-context runs.
#   - runs/edr/*.pt placement profiles. They are RUN OUTPUTS, and as of
#     2026-08-14 launch.sh no longer consumes them.
#
# ---------------------------------------------------------------------------
# DOWNLOAD BUDGET -- why BASE_IMAGE defaults to the fat sglang image
# ---------------------------------------------------------------------------
# Measured 2026-08-14 (compressed, i.e. actual bytes over the wire):
#
#   nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04     4.03 GB
#   lmsysorg/sglang:v0.5.16                       12.61 GB
#
# The fat image is 3x bigger yet the CHEAPER base, because it is already
# present on the build host (ai.v8.pro) -- so it costs 0 bytes -- and it
# already carries every build prerequisite this Dockerfile would otherwise
# apt-install or curl: nvcc 13.0, python3.12 + headers, g++ 13.3, make, ninja,
# cmake 3.31, git, unzip, lscpu, libnuma + headers, libgomp, numactl, rustc
# 1.97, uv. The apt / rustup / uv steps below all degrade to no-ops on it.
#
#   base + prereqs, nvidia base : ~4.6 GB
#   base + prereqs, sglang base : ~0   GB   <- default
#
# What is NOT avoidable either way, because torch 2.13.0 is pinned and NO
# locally-cached image has it (both sglang and kt-mxfp4:deps ship 2.11.0):
#
#   torch 2.13.0+cu130 + nvidia cu13 libs + sgl-kernel + deps  ~5-7  GB
#   flashinfer-cubin                                            0.57 GB
#   flashinfer-jit-cache  (skippable, see INSTALL_FLASHINFER_JIT_CACHE)
#                                                               1.48 GB
#   trtllm-gen cubin pool + ktransformers/llama.cpp clone       ~0.33 GB
#
# So: ~7.5-9.5 GB on the sglang base, ~12-14 GB on the nvidia base. Set
# INSTALL_FLASHINFER_JIT_CACHE=0 to shave another 1.48 GB at the cost of a
# slower first launch (FlashInfer JIT-compiles instead).
#
# REBUILDS are far cheaper than that: the pip/uv caches live in BuildKit cache
# mounts, not in image layers, so a second build re-downloads nothing that has
# not changed version. Keep BuildKit on (DOCKER_BUILDKIT=1, the default on
# Docker >= 23) or the --mount=type=cache lines are a hard error.
#
# ---------------------------------------------------------------------------
# BUILD TRAPS -- each of these has already cost real time on the node
# ---------------------------------------------------------------------------
# 1. kt-kernel MUST be installed with `--no-deps --no-build-isolation`. Its
#    pyproject.toml hard-pins `torch==2.9.1`, so a plain `pip install .`
#    uninstalls torch 2.13.0+cu130, drags in ~15 nvidia-*-cu12 wheels, and
#    every server then dies on `operator torchvision::nms does not exist`.
#    The RUN step below re-asserts the torch version afterwards so a
#    regression fails the BUILD rather than the first server launch.
#
# 2. kt-kernel does not build from a bare clone. Its CMake add_subdirectory's
#    third_party/{pybind11,llama.cpp}, which are submodules; an unpopulated
#    clone dies at configure with "Unknown CMake command pybind11_add_module".
#    Only those two are needed -- third_party/sglang is an entire fork and
#    custom_flashinfer is unused by the kernel build.
#
# 3. THE DOCKER-SPECIFIC ONE. kt-kernel's setup.py defaults
#    CPUINFER_CPU_INSTRUCT=NATIVE (-march=native) and cmake/DetectCPU.cmake
#    auto-detects AMX/AVX512 by reading /proc/cpuinfo. Inside a build that is
#    the BUILD HOST's cpuinfo, not the serving node's. Building on a machine
#    without AMX therefore yields a silently AMX-less kernel -- it imports and
#    serves, just far slower -- and -march=native bakes in the builder's ISA,
#    which SIGILLs on an older CPU. On a bare node build-host == run-host so
#    the default is safe and k3.sh sets nothing; in Docker it is not.
#
#    KT_CPU_VARIANT=all (the default) DISSOLVES this rather than working
#    around it: CPUINFER_BUILD_ALL_VARIANTS=1 compiles all six .so variants
#    (avx2, avx512_base, avx512_vnni, avx512_vbmi, avx512_bf16, amx) into one
#    wheel, and kt_kernel's python/_cpu_detect.py picks the best match at
#    IMPORT time on the SERVING host, with the fallback chain
#      amx -> avx512_bf16 -> avx512_vbmi -> avx512_vnni -> avx512_base -> avx2
#    and a KT_KERNEL_CPU_VARIANT env override to force one. This is also what
#    ktransformers' own docker/Dockerfile does.
#
#    That portability is not theoretical here: rentals change CPU vendor. The
#    2026-08-12 node is 2x Xeon 8559C with AMX; the node before it was EPYC
#    with none. An image pinned to `amx` serves the first and dies (or
#    silently crawls) on the second.
#
#    Cost: ~6x the kt-kernel compile, which is the slowest step in this build.
#    When you know the target CPU and want the build to finish sooner, pin a
#    single variant -- `--build-arg KT_CPU_VARIANT=amx` for the current node,
#    `avx512_bf16` for EPYC/Zen4. Single-variant builds compile with explicit
#    -DLLAMA_AVX512* flags rather than -march=native, so they are not
#    bit-identical to the node's own build -- they are the portable
#    equivalent, and they cross-build correctly from a host with no AVX512/AMX
#    of its own (the compiler emits instructions it need not execute).
#
# 4. NOTHING THAT LINKS libcuda.so.1 CAN BE IMPORTED DURING A BUILD. That is
#    the NVIDIA *driver* library; the container runtime injects it at
#    `docker run --gpus`, and `docker build` has no --gpus, so `import
#    sgl_kernel` fails with "libcuda.so.1: cannot open shared object file"
#    even when the install is perfectly good. Every build-time import check
#    below therefore runs with /usr/local/cuda/lib64/stubs/libcuda.so linked
#    under the driver's SONAME into a temp dir on LD_LIBRARY_PATH, and that
#    temp dir is deleted IN THE SAME RUN -- a stub left in the image would
#    shadow the real driver at serve time and every CUDA call would fail.
#    Consequence for reading build logs: these checks prove the extensions
#    LOAD, not that they can talk to a GPU. Only a `docker run --gpus` can
#    show that; see the post-build verification in the run instructions.

# Default to an image already cached on the build host. Override for a lean,
# self-contained build on a host that has neither:
#   --build-arg BASE_IMAGE=nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04
# Any override MUST provide nvcc: DeepGEMM and Triton JIT-compile at first
# launch, so a -runtime base saves a few GB and then fails to start.
ARG BASE_IMAGE=lmsysorg/sglang:v0.5.16
FROM ${BASE_IMAGE}

ARG KT_REPO=https://github.com/jack8412/ktransformers.git
ARG KT_REF=feat/mxfp4-kimi-k3
# Pinned so a rebuild is reproducible; pass KT_COMMIT= (empty) to track the
# branch tip instead.
#
# This is the sha the NODE actually runs (k3.sh doctor, 2026-08-14:
# "kt efb25f1 staging: let the host-node path dispatch from the packed buffer").
# It is deliberately NOT the sha in CLAUDE.md's Pins section
# (ab677cb43c9c2998694dc8d5e18fb6c34231b7b8) -- that records the Session-1
# wheel and PREDATES packed staging: `submit_forward_packed` does not appear
# anywhere in it. Building against it produces a kt_kernel that installs and
# imports cleanly and then cannot serve this branch's doorbell transport.
# The packed-staging assert in step 5 is what caught it; leave that assert in.
ARG KT_COMMIT=efb25f1bf4ffef3461a963e23dfb69c09e4987ba
ARG KT_CPU_VARIANT=all
ARG KT_CUDA_ARCHS=
ARG TORCH_PIN=2.13.0
ARG FLASHINFER_VERSION=0.6.15.post1
ARG INSTALL_FLASHINFER_JIT_CACHE=1
ARG RUST_VERSION=1.90.0
ARG BUILD_JOBS=

# UV_LINK_MODE=copy: the uv cache lives on a BuildKit cache mount, i.e. a
# different filesystem from the image layer, so uv cannot hardlink into the
# venv and warns on every install. Copying is what it falls back to anyway.
ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    WS=/workspace \
    VENV=/workspace/venv-k3 \
    PATH="/usr/local/cuda/bin:${PATH}" \
    UV_LINK_MODE=copy

# --- 0. system packages -----------------------------------------------------
# No-op on the default base, which already has all of these. numactl and
# util-linux are RUNTIME deps, not build deps: launch.sh sizes
# --kt-threadpool-count from `numactl --hardware` and --kt-cpuinfer from
# `lscpu -p=Core,Socket`. Without them it silently falls back to NUMAN=2.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-dev python3.12-venv \
      build-essential cmake ninja-build git \
      ca-certificates curl wget unzip \
      numactl libnuma-dev util-linux libgomp1 \
      tmux \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && apt-get clean

# uv provides the venv and the resolver k3.sh uses; it is not optional here
# because the sglang index needs --index-strategy unsafe-best-match, which
# plain pip has no equivalent for. Skipped when the base already ships it.
RUN command -v uv >/dev/null && { echo "uv present: $(uv --version)"; exit 0; }; \
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh \
    && uv --version

# sglang's python/pyproject.toml build-system requires setuptools-rust, and its
# setup.py shells out to `cargo metadata` on rust/Cargo.toml, so a WORKING
# cargo -- not merely a rustc on PATH -- is required, or the editable install
# dies inside the build backend.
#
# Do NOT set RUSTUP_HOME/CARGO_HOME here. The sglang base images ship rust at
# the rustup defaults (/root/.rustup + /root/.cargo) with both vars UNSET.
# Pointing them at /usr/local/* leaves the shims resolving an empty toolchain
# directory, and cargo fails with
#   error: rustup could not choose a version of cargo to run, because one
#   wasn't specified explicitly, and no default is configured
# while `command -v rustc` STILL SUCCEEDS -- so a presence check does not catch
# it. Install to the same default location the bases use, and probe by RUNNING
# cargo rather than by looking for a binary.
ENV PATH="/root/.cargo/bin:${PATH}"
RUN if cargo --version >/dev/null 2>&1; then \
      echo "cargo present: $(cargo --version)"; \
    elif command -v rustup >/dev/null 2>&1; then \
      echo "rustup present but no usable default toolchain; configuring one"; \
      rustup default "${RUST_VERSION}" || rustup default stable; \
    else \
      curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --no-modify-path --profile minimal --default-toolchain "${RUST_VERSION}"; \
    fi \
    && cargo --version

# --- 1. this branch's source ------------------------------------------------
# COPY, not git clone: the whole point is to serve the tree you are on. The
# editable install below binds to /workspace/sglang/python, so mounting a
# working tree over /workspace/sglang at run time keeps the install valid and
# picks up your edits -- that is the intended dev loop, no rebuild required.
WORKDIR ${WS}
COPY . ${WS}/sglang

# --- 2. venv + sglang editable (this is what pins torch 2.13.0+cu130) -------
# The venv is deliberately ISOLATED from the base image's site-packages: the
# sglang images ship torch 2.11.0, and letting 2.11 and the pinned 2.13 see
# each other is the exact class of mixed-install failure k3.sh exists to
# prevent. The base contributes its toolchain, not its Python packages.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv venv ${VENV} --python 3.12 --seed \
    && . ${VENV}/bin/activate \
    && [ "$(command -v pip)" = "${VENV}/bin/pip" ] || { echo "FATAL: pip outside venv"; exit 9; } \
    && uv pip install --prerelease=allow --index-strategy unsafe-best-match \
         --extra-index-url https://docs.sglang.ai/whl/cu130/ -e ${WS}/sglang/python/ \
    && python -c "import torch; assert torch.__version__.startswith('${TORCH_PIN}'), torch.__version__; print('STEP1 torch', torch.__version__, torch.version.cuda)"

# --- 3. flashinfer cubin + jit cache ---------------------------------------
# The trio (flashinfer-python from step 2, plus these) must be the same version
# or the import fails at runtime; the assert makes that a build error.
# jit-cache is 1.48 GB and purely a startup optimisation -- set
# INSTALL_FLASHINFER_JIT_CACHE=0 to trade it for runtime JIT compilation.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    . ${VENV}/bin/activate \
    && uv pip install --index-url https://flashinfer.ai/whl "flashinfer-cubin==${FLASHINFER_VERSION}" \
    && if [ "${INSTALL_FLASHINFER_JIT_CACHE}" = "1" ]; then \
         uv pip install --index-url https://flashinfer.ai/whl/cu130 "flashinfer-jit-cache==${FLASHINFER_VERSION}"; \
       else \
         echo "SKIPPING flashinfer-jit-cache (-1.48 GB); kernels will JIT-compile on first launch"; \
       fi \
    # BUILD TRAP 4: sgl_kernel links libcuda.so.1 -- the DRIVER library, which
    # the nvidia container runtime injects at `docker run --gpus` and which
    # therefore does not exist during `docker build` (there is no --gpus for
    # builds). Importing it here fails with
    #   ImportError: libcuda.so.1: cannot open shared object file
    # even though the install is perfectly good. The CUDA image ships a link
    # stub for exactly this case; point the loader at it under the driver's
    # SONAME just long enough to prove the extension loads, then delete it IN
    # THE SAME LAYER so it can never shadow the real driver at run time.
    && mkdir -p /tmp/cudastub \
    && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/cudastub/libcuda.so.1 \
    && LD_LIBRARY_PATH=/tmp/cudastub${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} \
       python -c "import torch, flashinfer, sgl_kernel; print('STEP2 sgl_kernel+flashinfer OK', flashinfer.__version__)" \
    && rm -rf /tmp/cudastub

# --- 4. ktransformers checkout (trap 2: submodules) ------------------------
# --depth is deliberately absent on the submodule update: kt pins submodule
# shas, and a shallow fetch of a non-tip sha fails on many git servers.
RUN git clone ${KT_REPO} ${WS}/ktransformers \
    && cd ${WS}/ktransformers \
    && if [ -n "${KT_COMMIT}" ]; then git checkout -q ${KT_COMMIT}; \
       else git checkout -q -B ${KT_REF} origin/${KT_REF}; fi \
    && git submodule update --init --recursive third_party/pybind11 third_party/llama.cpp \
    && test -f third_party/pybind11/CMakeLists.txt \
    && test -f third_party/llama.cpp/CMakeLists.txt \
    && echo "kt $(git log --oneline -1)"

# --- 5. kt-kernel (traps 1 and 3) ------------------------------------------
# The CPUINFER_* block is setup.py's variant preset, set explicitly so nothing
# is inferred from the build host. Keep these in sync with setup.py's
# `variants` list if it changes.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    . ${VENV}/bin/activate \
    && case "${KT_CPU_VARIANT}" in \
         all) export CPUINFER_BUILD_ALL_VARIANTS=1 ;; \
         amx) export CPUINFER_CPU_INSTRUCT=AVX512 CPUINFER_ENABLE_AVX512=ON \
                     CPUINFER_ENABLE_AVX512_VNNI=ON CPUINFER_ENABLE_AVX512_BF16=ON \
                     CPUINFER_ENABLE_AVX512_VBMI=ON CPUINFER_ENABLE_AMX=ON ;; \
         avx512_bf16) export CPUINFER_CPU_INSTRUCT=AVX512 CPUINFER_ENABLE_AVX512=ON \
                     CPUINFER_ENABLE_AVX512_VNNI=ON CPUINFER_ENABLE_AVX512_BF16=ON \
                     CPUINFER_ENABLE_AVX512_VBMI=ON CPUINFER_ENABLE_AMX=OFF ;; \
         avx512_vbmi) export CPUINFER_CPU_INSTRUCT=AVX512 CPUINFER_ENABLE_AVX512=ON \
                     CPUINFER_ENABLE_AVX512_VNNI=ON CPUINFER_ENABLE_AVX512_BF16=OFF \
                     CPUINFER_ENABLE_AVX512_VBMI=ON CPUINFER_ENABLE_AMX=OFF ;; \
         avx2) export CPUINFER_CPU_INSTRUCT=AVX2 CPUINFER_ENABLE_AVX512=OFF \
                     CPUINFER_ENABLE_AMX=OFF ;; \
         native) echo "WARNING: KT_CPU_VARIANT=native uses -march=native and auto-detects" \
                      "AMX from the BUILD host -- only correct when you build on the node" ;; \
         *) echo "FATAL: unknown KT_CPU_VARIANT '${KT_CPU_VARIANT}'" >&2; exit 2 ;; \
       esac \
    && export CPUINFER_USE_CUDA=1 \
    # KT_CUDA_ARCHS is deliberately EMPTY by default, which leaves kt-kernel's
    # own default of CPUINFER_CUDA_ARCHS="80;86;89;90" in force -- exactly what
    # `k3.sh provision` produces on the node, since it exports nothing either.
    # Note what that list does NOT contain: 100 (B200). The node has served on
    # this build regardless, so the CUDA half is evidently not on the hot path
    # for this config -- but the gap is real, and if you ever see a "no kernel
    # image is available for execution" out of kt_kernel, this is the first
    # thing to change (--build-arg KT_CUDA_ARCHS="80;86;89;90;100"). Do not
    # change it speculatively: it diverges the image from every measured run,
    # and kt's CUDA sources are unverified against sm_100.
    && if [ -n "${KT_CUDA_ARCHS}" ]; then export CPUINFER_CUDA_ARCHS="${KT_CUDA_ARCHS}"; fi \
    && if [ -n "${BUILD_JOBS}" ]; then export CPUINFER_PARALLEL=${BUILD_JOBS}; fi \
    && cd ${WS}/ktransformers/kt-kernel \
    # Trap 1: --no-deps --no-build-isolation is not optional.
    && pip install . --no-deps --no-build-isolation \
    # Driver stub for the build-time import checks below -- see BUILD TRAP 4.
    # Removed in this same RUN so it never reaches the final image.
    && mkdir -p /tmp/cudastub \
    && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/cudastub/libcuda.so.1 \
    && export LD_LIBRARY_PATH=/tmp/cudastub${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} \
    # ...and prove it did not happen anyway.
    && python -c "import torch; assert torch.__version__.startswith('${TORCH_PIN}'), 'kt downgraded torch: '+torch.__version__; print('torch still', torch.__version__)" \
    # packed-staging is the API this branch's doorbell transport needs, and the
    # one `k3.sh doctor` asserts. The old 0.6.1 wheel does NOT have it, so this
    # catches a stale kt getting installed instead of the branch build.
    && python -c "import kt_kernel; from kt_kernel.experts_base import BaseMoEWrapper as B; assert 'submit_forward_packed' in dir(B), 'kt lacks packed staging -- wrong kt build'; print('kt packed-staging: True')" \
    # With KT_CPU_VARIANT=all the wheel must carry all six ISA variants, or
    # runtime detection on a different CPU silently falls back to whatever
    # single .so exists -- exactly the failure this default exists to prevent.
    && if [ "${KT_CPU_VARIANT}" = "all" ]; then \
         n=$(find "$(python -c 'import kt_kernel,os;print(os.path.dirname(kt_kernel.__file__))')" \
               -name '_kt_kernel_ext_*.so' | wc -l); \
         echo "kt ISA variants built: $n"; \
         find "$(python -c 'import kt_kernel,os;print(os.path.dirname(kt_kernel.__file__))')" \
               -name '_kt_kernel_ext_*.so' -printf '  %f\n'; \
         [ "$n" -eq 6 ] || { echo "FATAL: expected 6 ISA variants, got $n" >&2; exit 3; }; \
       fi \
    && rm -rf /tmp/cudastub

# --- 6. trtllm-gen MoE cubin pool ------------------------------------------
# NOT optional for this config: with --moe-runner-backend flashinfer_mxfp4 on
# SM100 the server REFUSES to start without a valid pool (overrides.py raises;
# there is no pool-less JIT path). Nested under the version directory because
# that is the layout launch.sh probes for. Only ~30 MB compressed.
ARG POOL_VER=trtllm_gen_moe_cubin_pool_20260617_v0613rc1
ARG POOL_SHA256=4900501cbe782a76b08a5858f9f07152287b97cb68114466dac286366b66c192
RUN mkdir -p /opt/trtllm_gen_moe_cubin_pool \
    && curl -fsSL -o /tmp/pool.zip \
       "https://github.com/sgl-project/whl/releases/download/trtllm_gen_moe_cubin_20260617/${POOL_VER}.zip" \
    && echo "${POOL_SHA256}  /tmp/pool.zip" | sha256sum --check --strict - \
    && unzip -q /tmp/pool.zip -d /opt/trtllm_gen_moe_cubin_pool \
    && rm -f /tmp/pool.zip \
    && test -d "/opt/trtllm_gen_moe_cubin_pool/${POOL_VER}" \
    && test "$(find /opt/trtllm_gen_moe_cubin_pool/${POOL_VER} -type f -name '*.cubin' | wc -l)" -eq 1696 \
    && echo "cubin pool OK (1696 cubins)"

# --- 7. results tree + final gate ------------------------------------------
# Same five directories as the node, so the documented rsync mirrors a
# container run exactly as it mirrors a bare-node run.
RUN mkdir -p ${WS}/runs/status ${WS}/runs/probes ${WS}/runs/logs ${WS}/runs/meta ${WS}/runs/edr \
    && . ${VENV}/bin/activate \
    # The base image carries its OWN sglang (0.5.16, editable, exposed through
    # __editable__.sglang-0.5.16.pth in system dist-packages). It is invisible
    # here only because `uv venv --seed` builds an isolated venv and the base
    # leaves PYTHONPATH unset. Both are load-bearing and neither is obvious, so
    # assert rather than trust: if the venv ever gains system site-packages, the
    # server would silently run the BASE's 0.5.16 instead of this branch, and
    # every measurement would be attributed to the wrong code.
    && grep -q '^include-system-site-packages *= *false' ${VENV}/pyvenv.cfg \
       || { echo "FATAL: venv sees system site-packages -- base sglang can leak" >&2; exit 4; } \
    # BUILD TRAP 5: this RUN inherits cwd=/workspace (the WORKDIR set before the
    # COPY), and `python -c` puts cwd on sys.path -- so /workspace/sglang/, the
    # REPO DIRECTORY, is picked up as a NAMESPACE PACKAGE and shadows the
    # editable install. sglang.__file__ then comes back None and every submodule
    # "does not exist". This is the failure CLAUDE.md records as having silently
    # emptied a phase's benchmark rows; it reproduces inside the build. Verify
    # from / where nothing can shadow. (The image's final WORKDIR is
    # /workspace/sglang, which is safe -- there is no ./sglang beneath it.)
    && cd / \
    # Driver stub for the GPU-extension imports below -- see BUILD TRAP 4.
    && mkdir -p /tmp/cudastub \
    && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/cudastub/libcuda.so.1 \
    && export LD_LIBRARY_PATH=/tmp/cudastub${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} \
    && python -c "import sglang,os;assert sglang.__file__ is not None,'sglang imported as a NAMESPACE package -- a repo dir on sys.path shadowed the install (wrong cwd)';p=os.path.dirname(sglang.__file__);assert p.startswith('${WS}/sglang/python/'),'sglang resolved to '+p+' -- the base image 0.5.16 leaked into the venv';print('sglang from  :',p)" \
    && python -c "import kt_kernel,os;print('kt_kernel from:',os.path.dirname(kt_kernel.__file__))" \
    && python -c "import torch, flashinfer, sgl_kernel, kt_kernel, sglang; print('ENV-OK', torch.__version__, flashinfer.__version__, sglang.__version__)" \
    && rm -rf /tmp/cudastub /root/.cargo/registry

# Put the venv first so an interactive shell gets the right python without
# sourcing anything. launch.sh still activates it itself, which is harmless.
ENV PATH="/workspace/venv-k3/bin:${PATH}" \
    VIRTUAL_ENV="/workspace/venv-k3"

# The sglang base images set their own ENTRYPOINT; clear it so the documented
# `docker run ... bash .../launch.sh` and a bare shell both behave.
ENTRYPOINT []

# NEVER run `python -m sglang.*` with cwd /workspace: the repo dir
# /workspace/sglang/ shadows the editable install as a namespace package and
# every sglang submodule "does not exist". launch.sh is safe from anywhere, but
# an interactive shell is not, so land somewhere that cannot bite.
WORKDIR /workspace/sglang

EXPOSE 30000

# Provenance. .git is excluded from the build context, so setuptools_scm falls
# back to 0.0.0.dev0 and the tree carries no sha of its own -- pass it in:
#   --build-arg SGLANG_BUILD_COMMIT=$(git rev-parse HEAD) \
#   --build-arg SGLANG_BUILD_BRANCH=$(git branch --show-current)
# Read it back off a container with `docker inspect` or `echo $K3_BUILD_COMMIT`.
# Placed last so changing it does not invalidate any build layer.
ARG SGLANG_BUILD_COMMIT=unknown
ARG SGLANG_BUILD_BRANCH=unknown
ARG BASE_IMAGE
ENV K3_BUILD_COMMIT=${SGLANG_BUILD_COMMIT} \
    K3_BUILD_BRANCH=${SGLANG_BUILD_BRANCH}
LABEL ai.sglang.k3.sglang_commit="${SGLANG_BUILD_COMMIT}" \
      ai.sglang.k3.sglang_branch="${SGLANG_BUILD_BRANCH}" \
      ai.sglang.k3.base_image="${BASE_IMAGE}" \
      ai.sglang.k3.kt_ref="${KT_REF}" \
      ai.sglang.k3.kt_commit="${KT_COMMIT}" \
      ai.sglang.k3.kt_cpu_variant="${KT_CPU_VARIANT}" \
      ai.sglang.k3.torch_pin="${TORCH_PIN}"

CMD ["/bin/bash"]
