"""Arena geometry and the CUDA copy shims the kt cold path is built from.

WHAT THIS IS NOT. It used to be ``kt_direct_dma.py`` and to carry a whole
cold transport -- ``--kt-cold-transport direct-dma``, an interval registrar
that pinned each rank's read-set per (layer, expert), and a per-expert copy
plan. That transport never armed on any node: it is gated
``and not cold_only_cpu_experts``, because its address plan is built at load
time and a swap hands a BufferB block to a different expert, so under
cold-only it would serve the previous occupant's bytes. Forbidding cold-only
made kt hold all 896 experts (1.35 TB of memfd arenas), which is what
exhausted pinnable memory every time it was tried. It also issued ~1,632
copies per layer where the source that replaced it issues 6.

What survives is the part that was never about that transport: the address
arithmetic over kt's memfd arenas (:class:`ArenaExpertRanges`) and the cudart
shims (:class:`CudaCopyLib`, :func:`cudart_register_fns`). Both are
PRODUCTION -- ``ArenaDmaColdSource`` and ``RankShardWriter`` are built from
them -- which is why the module was split rather than deleted, and renamed so
the name stops naming a transport that no longer exists.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
from typing import Callable, Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

PAGE = 4096
_CUDA_MEMCPY_HOST_TO_DEVICE = 1
_CUDA_MEMCPY_DEVICE_TO_HOST = 2

class CudaCopyLib:
    """ctypes handle to cudaMemcpyAsync / cudaMemcpy2DAsync.

    torch.cuda.cudart() exposes hostRegister but not the 2D copy, so the
    copies go straight to libcudart. Argtypes are declared once; the issue
    loop then pays only the bare foreign call (~1-2 us).
    """

    def __init__(self):
        self._lib = self._load()
        self._lib.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._lib.cudaMemcpyAsync.restype = ctypes.c_int
        self._lib.cudaMemcpy2DAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._lib.cudaMemcpy2DAsync.restype = ctypes.c_int

    @staticmethod
    def _load() -> ctypes.CDLL:
        # torch's own libcudart first -- it is the runtime every other CUDA
        # call in the process already goes through.
        import glob

        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        candidates = sorted(glob.glob(os.path.join(torch_lib, "libcudart*")))
        for lib_dir in (candidates or []):
            try:
                return ctypes.CDLL(lib_dir)
            except OSError:
                continue
        try:
            import nvidia.cuda_runtime as _cr  # packaged runtime

            for path in sorted(
                glob.glob(os.path.join(os.path.dirname(_cr.__file__), "lib", "libcudart*"))
            ):
                try:
                    return ctypes.CDLL(path)
                except OSError:
                    continue
        except ImportError:
            pass
        name = ctypes.util.find_library("cudart") or "libcudart.so"
        return ctypes.CDLL(name)

    def memcpy_h2d(self, dst: int, src: int, nbytes: int, stream: int) -> None:
        rc = self._lib.cudaMemcpyAsync(
            dst, src, nbytes, _CUDA_MEMCPY_HOST_TO_DEVICE, stream
        )
        if rc != 0:
            raise RuntimeError(f"cudaMemcpyAsync rc={rc}")

    def memcpy_d2h(self, dst: int, src: int, nbytes: int, stream: int) -> None:
        """Device -> registered host, the demotion direction."""
        rc = self._lib.cudaMemcpyAsync(
            dst, src, nbytes, _CUDA_MEMCPY_DEVICE_TO_HOST, stream
        )
        if rc != 0:
            raise RuntimeError(f"cudaMemcpyAsync D2H rc={rc}")

    def memcpy2d_d2h(
        self,
        dst: int,
        dpitch: int,
        src: int,
        spitch: int,
        width: int,
        height: int,
        stream: int,
    ) -> None:
        """Device -> registered host, pitched.

        This is what makes w2 free: the demoted expert's strips land at
        ``dpitch`` intervals inside kt's buffer, and the copy engine walks
        that stride itself. The host-side gather it replaces was the largest
        single component of demotion -- measured 0.44-1.14 ms per expert
        against 0.26 ms for the read-back that feeds it.
        """
        rc = self._lib.cudaMemcpy2DAsync(
            dst, dpitch, src, spitch, width, height,
            _CUDA_MEMCPY_DEVICE_TO_HOST, stream,
        )
        if rc != 0:
            raise RuntimeError(f"cudaMemcpy2DAsync D2H rc={rc}")

    def memcpy2d_h2d(
        self,
        dst: int,
        dpitch: int,
        src: int,
        spitch: int,
        width: int,
        height: int,
        stream: int,
    ) -> None:
        rc = self._lib.cudaMemcpy2DAsync(
            dst, dpitch, src, spitch, width, height,
            _CUDA_MEMCPY_HOST_TO_DEVICE, stream,
        )
        if rc != 0:
            raise RuntimeError(f"cudaMemcpy2DAsync rc={rc}")


_ERROR_CLEAR_LIB: Optional[ctypes.CDLL] = None


def _clear_sticky_cuda_error() -> None:
    """Swallow CUDA's per-thread sticky error after a failed runtime call.

    torch's cudart binding returns the raw rc WITHOUT clearing last-error, so
    a failed cudaHostRegister -- a case this transport treats as recoverable
    (arming falls back, a swap pair is skipped) -- would otherwise surface as
    a phantom 'CUDA error' at the next kernel-launch check on this thread,
    turning designed degradation into a crash mid-boot or a one-rank crash
    plus TP hang mid-window. cudaGetLastError is the only call that resets
    the state; go straight to libcudart for it.
    """
    global _ERROR_CLEAR_LIB
    try:
        if _ERROR_CLEAR_LIB is None:
            _ERROR_CLEAR_LIB = CudaCopyLib._load()
            _ERROR_CLEAR_LIB.cudaGetLastError.restype = ctypes.c_int
        _ERROR_CLEAR_LIB.cudaGetLastError()
    except Exception:
        logger.exception("[kt-dma] could not clear the sticky CUDA error")


def cudart_register_fns() -> Tuple[Callable[[int, int], int], Callable[[int], int]]:
    cudart = torch.cuda.cudart()

    def reg(ptr: int, nbytes: int) -> int:
        rc = int(cudart.cudaHostRegister(ptr, nbytes, 0))
        if rc != 0:
            _clear_sticky_cuda_error()
        return rc

    def unreg(ptr: int) -> int:
        rc = int(cudart.cudaHostUnregister(ptr))
        if rc != 0:
            _clear_sticky_cuda_error()
        return rc

    return reg, unreg


# -- per-expert range math ---------------------------------------------------



class ArenaExpertRanges:
    """Absolute source addresses of one rank's read-set, per (layer, expert).

    Wraps a ``KtArenaExpertSource`` (which already holds the mapped arenas,
    the per-(partition, expert) OFFSET rows and the geometry) and turns them
    into the address arithmetic the plans and the registrar consume:
    absolute address = this rank's mapping base + exported offset. The K3
    shape -- a rank's slice wholly inside ONE partition (per_numa a multiple
    of per_gpu) -- is asserted at construction; the general multi-piece case
    falls back to the export transport rather than guessing.
    """

    def __init__(self, source):
        from sglang.srt.layers.moe import kt_ram_source as krs

        self._src = source
        self.hidden = source.hidden
        self.per_numa = source.per_numa
        self.per_gpu = source.per_gpu
        self.group = source.group
        if self.per_numa % self.per_gpu:
            raise ValueError(
                f"per_numa {self.per_numa} not a multiple of per_gpu "
                f"{self.per_gpu}; direct-DMA needs the single-partition shape"
            )
        self.part = (source.tp_rank * self.per_gpu) // self.per_numa
        self.local_rank = (
            source.tp_rank - self.part * (self.per_numa // self.per_gpu)
        )
        # This rank only ever reads its own partition's arena.
        self._arena_base = int(source._arenas[self.part].data_ptr())
        # byte widths
        self.gu_w = self.per_gpu * self.hidden // 2
        self.gu_s = self.per_gpu * (self.hidden // self.group)
        self.w2_width = self.per_gpu // 2
        self.w2_pitch = self.per_numa // 2
        self.w2s_width = self.per_gpu // self.group
        self.w2s_pitch = self.per_numa // self.group
        self._kinds = (
            krs._GATE_B, krs._UP_B, krs._DOWN_B,
            krs._GATE_D, krs._UP_D, krs._DOWN_D,
        )

    def _addr(self, expert: int, kind_pos: int) -> int:
        src = self._src
        row = src._rows[self.part * src.experts + expert]
        off = row[self._kinds[kind_pos]]
        if off < 0:
            raise RuntimeError(f"expert {expert} absent from partition {self.part}")
        return self._arena_base + int(off)

    def op_addrs(self, expert: int) -> Dict[str, Tuple]:
        """Source addresses for the three op classes of one expert."""
        lr = self.local_rank
        gate_b = self._addr(expert, 0) + lr * self.gu_w
        up_b = self._addr(expert, 1) + lr * self.gu_w
        gate_d = self._addr(expert, 3) + lr * self.gu_s
        up_d = self._addr(expert, 4) + lr * self.gu_s
        down_b = self._addr(expert, 2)
        down_d = self._addr(expert, 5)
        return {
            "w13": ((gate_b, self.gu_w), (up_b, self.gu_w)),
            "w13_scale": ((gate_d, self.gu_s), (up_d, self.gu_s)),
            # pitched: (base + lr*width, spitch, width, height)
            "w2": (down_b + lr * self.w2_width, self.w2_pitch, self.w2_width, self.hidden),
            # whole block, contiguous: (base, nbytes); compact selects cols
            "w2_scale_block": (down_d, self.hidden * self.w2s_pitch),
        }
