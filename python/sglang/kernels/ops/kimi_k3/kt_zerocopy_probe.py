"""JIT ops for the zero-copy data-path gate (measurement only).

Not wired into serving. These exist so the question "can kernels carry the
CPU-expert data path instead of Memcpy nodes?" is answered by measurement
before any of it is built.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_kt_zerocopy_module() -> Module:
    return load_jit(
        "kimi_k3_kt_zerocopy_probe",
        cuda_files=["kimi_k3/kt_zerocopy_probe.cuh"],
        cuda_wrappers=[
            ("alloc_mapped", "KtZeroCopy::alloc_mapped"),
            ("device_ptr", "KtZeroCopy::device_ptr"),
            ("free_mapped", "KtZeroCopy::free_mapped"),
            ("write", "KtZeroCopy::write"),
            ("merge", "KtZeroCopy::merge"),
            ("touch", "KtZeroCopy::touch"),
        ],
        extra_cuda_cflags=["-O3"],
    )


def kt_zc_write(
    src: torch.Tensor, dst_dev: int, flag: torch.Tensor, blocks: int = 8
) -> None:
    """Copy `src` (device) into mapped host memory addressed by `dst_dev`."""
    _jit_kt_zerocopy_module().write(src, dst_dev, flag, blocks)


def kt_zc_merge(
    src_dev: int, dst: torch.Tensor, flag: torch.Tensor, blocks: int = 8
) -> None:
    """`dst` += the bf16 block at mapped-host address `src_dev`."""
    _jit_kt_zerocopy_module().merge(src_dev, dst, flag, blocks)


def kt_zc_touch(flag: torch.Tensor, sink: torch.Tensor) -> None:
    """Read the predicate and return: the floor cost of a skipped layer."""
    _jit_kt_zerocopy_module().touch(flag, sink)
