"""Launch the inference server."""

import asyncio
import os
import sys
import warnings


def _sweep_stale_torch_extension_locks():
    """Remove stale ninja locks under ~/.cache/torch_extensions before any
    torch.utils.cpp_extension build runs.

    torch's cpp_extension JIT builds take a ``lock`` / ``.ninja_lock`` file in
    the build dir and block while it is held. A run killed mid-build (SIGKILL,
    OOM, scheduler crash) leaves the lock on disk, and subsequent runs hang
    forever on the orphaned lock with zero CPU/GPU activity — indistinguishable
    from a deadlock. Sweeping locks older than SGLANG_STALE_LOCK_AGE_MINUTES
    (default 30m: never interrupts a live build, auto-recovers same-day reruns)
    eliminates this hang class at startup.
    """
    try:
        import time

        from sglang.srt.environ import envs

        cache_dir = os.path.expanduser(
            os.environ.get("TORCH_EXTENSIONS_DIR", "~/.cache/torch_extensions")
        )
        if not os.path.isdir(cache_dir):
            return
        max_age_min = envs.SGLANG_STALE_LOCK_AGE_MINUTES.get()
        if max_age_min <= 0:
            return
        cutoff = time.time() - max_age_min * 60
        swept = 0
        for root, _dirs, files in os.walk(cache_dir):
            for name in files:
                if name not in ("lock", ".ninja_lock"):
                    continue
                path = os.path.join(root, name)
                try:
                    if os.path.getmtime(path) < cutoff:
                        os.unlink(path)
                        swept += 1
                except OSError:
                    pass
        if swept:
            print(
                f"[sglang] swept {swept} stale ninja locks under {cache_dir} "
                f"(older than {max_age_min}m)",
                file=sys.stderr,
            )
    except Exception:
        # Best-effort cleanup; failures must not block startup.
        pass


_sweep_stale_torch_extension_locks()

from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.common import suppress_noisy_warnings

suppress_noisy_warnings()


def run_server(server_args):
    """Run the server based on the gRPC flags and server_args.encoder_only."""
    if server_args.encoder_only:
        # For encoder disaggregation
        if server_args.smg_grpc_mode or server_args.grpc_mode:
            from sglang.srt.disaggregation.encode_grpc_server import (
                serve_grpc_encoder,
            )

            asyncio.run(serve_grpc_encoder(server_args))
        else:
            from sglang.srt.disaggregation.encode_server import launch_server

            launch_server(server_args)
    elif server_args.smg_grpc_mode:
        # Legacy SMG gRPC server (--smg-grpc-mode, or the deprecated --grpc-mode
        # which __post_init__ folds into smg_grpc_mode). The native Rust gRPC
        # server is a separate path, enabled by --grpc-port, that starts
        # alongside the default HTTP server below.
        from sglang.srt.entrypoints.grpc_server import serve_grpc

        asyncio.run(serve_grpc(server_args))
    elif server_args.use_ray:
        # Ray mode: HTTP mode with Ray backend.
        try:
            from sglang.srt.ray.http_server import launch_server
        except ImportError:
            raise ImportError(
                "Ray is required for --use-ray mode. "
                "Install it with: pip install 'sglang[ray]'"
            )

        launch_server(server_args)
    else:
        # Default mode: HTTP mode.
        from sglang.srt.entrypoints.http_server import launch_server

        launch_server(server_args)


if __name__ == "__main__":
    warnings.warn(
        "'python -m sglang.launch_server' is still supported, but "
        "'sglang serve' is the recommended entrypoint.\n"
        "  Example: sglang serve --model-path <model> [options]",
        UserWarning,
        stacklevel=1,
    )

    from sglang.srt.plugins import load_plugins

    load_plugins()

    server_args = prepare_server_args(sys.argv[1:])

    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
