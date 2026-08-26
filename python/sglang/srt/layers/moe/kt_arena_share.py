# SPDX-License-Identifier: Apache-2.0
"""Give every TP rank a read-only mapping of kt-kernel's expert buffers.

Under ``KT_BUFFER_B_MEMFD=1`` kt allocates each partition's resident BufferB
storage in one memfd-backed MAP_SHARED arena (``moe_base.hpp``) and
``expert_buffer_arenas()`` exports ``(fd, size, offsets)``. The fds are only
meaningful inside rank 0's process -- the kt engine lives there alone -- so
this module ships them to the other ranks over a unix socket with SCM_RIGHTS.
Plain fd passing is the one cross-process channel this node does NOT block
(``pidfd_getfd`` is seccomp-blocked and ``ptrace_scope=1`` kills
``process_vm_readv``; both were checked, not assumed).

Each rank then maps the fds READ-ONLY -- a bug in a consumer rank must not be
able to corrupt the weights every rank computes with -- closes the fds, and
builds a per-layer :class:`KtArenaExpertSource` for its own TP slice. The swap
window's promotion path reads those instead of the checkpoint, which is what
turns the measured ~22 s full-kt promotion window into a sub-second one.

Flow, per MoE layer, called from ``process_weights_after_loading`` on ALL
ranks (rank 0 right after ``wrapper.load_weights`` fills the buffers):

    rank 0                                 ranks 1..N-1
    ------                                 ------------
    listen on unix socket (once)
    broadcast socket path (once) --------> receive path (once)
    accept N-1 peers (once)      <-------- connect, with retry (once)
    export fds + offsets
    send_fds + pickled metadata ---------> recv, mmap read-only, build source
    build own source from bases

The ONE collective is the path broadcast, first call only, and every rank
reaches it unconditionally (the gate is an environment variable, identical
across ranks by construction). Everything after it is point-to-point, so a
rank that fails locally degrades to "no arena source on that rank" -- its
promotions fall back to the checkpoint path -- and cannot desynchronise the
group. That asymmetry-tolerance is a hard requirement here; per-rank-gated
collectives are what deadlocked M9/M11/M12.

Accounting note: the mapped pages are tmpfs/Shmem. They show up in
``buff/cache`` rather than any process's anon RSS, every mapping rank's RSS
grows by the pages it faults, and killing ranks does not free them while any
mapping survives. In-repo memory gates all read MemAvailable and are
unaffected (audited); external RSS-summing dashboards will over-count.
"""

from __future__ import annotations

import logging
import mmap
import os
import pickle
import socket
import struct
import tempfile
import time
import warnings
from typing import Dict, List

import numpy as np
import torch


logger = logging.getLogger(__name__)

# (layer_idx, n_fds, meta_len), little-endian int64s.
_HEADER = struct.Struct("<qqq")

# Generous on purpose: a peer's per-layer recv blocks while rank 0 loads that
# layer's 15+ GB of expert weights from disk, and the connect happens while
# rank 0 finishes its FIRST layer load. These waits are real work elsewhere,
# not hangs, and they exist with or without this channel (the ranks would
# otherwise sit at the post-load barrier).
_CONNECT_TIMEOUT_S = 3600.0
_IO_TIMEOUT_S = 3600.0

_STATE: Dict = {
    "decided": False,  # env looked at, path exchanged (or single-rank)
    "enabled": False,
    "failed": False,  # local transport/mapping failure: sources stay partial
    "listener": None,
    "conns": [],  # rank 0: one per peer, None where dead
    "sock": None,  # peers: connection to rank 0
    "mmaps": [],  # peers: keep mappings alive for the process lifetime
    "write_sources": {},  # layer_idx -> source for the rank-write path
}


def kt_arena_mode_requested() -> bool:
    """Mirror of the C++ gate in moe_base.hpp: set, non-empty, not \"0\"."""
    v = os.getenv("KT_BUFFER_B_MEMFD")
    return v is not None and v != "" and v != "0"


def arena_write_source_for(layer_idx: int):
    """This rank's arena source for the WRITE path, or None.

    THE ONLY REGISTRY. There used to be a second, read-path one beside it,
    for a promotion route that read arena bytes through load-time offsets.
    That route cannot be correct under cold-only residency -- swaps move
    BufferB ownership between expert ids, so those offsets go stale at the
    first swap and a reader would serve the previous occupant's bytes -- so
    the registry was never populated and the route never ran. The rank-write
    path has no such problem: it never reads through these sources, and it
    tracks every slot move in its own table.
    """
    return _STATE["write_sources"].get(layer_idx)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = []
    while n:
        b = sock.recv(n)
        if not b:
            raise ConnectionError("arena share peer closed mid-message")
        chunks.append(b)
        n -= len(b)
    return b"".join(chunks)


def _recv_header_and_fds(sock: socket.socket):
    """One header, with any fds riding its first bytes' ancillary data."""
    data, fds, _flags, _addr = socket.recv_fds(sock, _HEADER.size, 16)
    if not data:
        raise ConnectionError("arena share server closed the connection")
    if len(data) < _HEADER.size:
        data += _recv_exact(sock, _HEADER.size - len(data))
    return _HEADER.unpack(data), list(fds)


def _wrap_mapping(m) -> torch.Tensor:
    """uint8 view over a read-only mapping.

    torch.from_numpy warns that the array is not writable; that is the point
    of PROT_READ, not a problem, so the warning is silenced here and nowhere
    else.
    """
    arr = np.frombuffer(m, dtype=np.uint8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return torch.from_numpy(arr)


def _build_source_from_export(
    *, arenas: List[torch.Tensor], offsets, geometry, tp_rank: int, tp_size: int
):
    from sglang.srt.layers.moe.kt_ram_source import KtArenaExpertSource

    return KtArenaExpertSource(
        arenas=arenas,
        offsets=offsets,
        geometry=geometry,
        tp_rank=tp_rank,
        tp_size=tp_size,
    )


def _decide_once(*, tp_rank: int, tp_size: int) -> None:
    """First call: exchange the socket path and connect the mesh.

    The broadcast below is the module's only collective. Every rank reaches it
    on its first share call whenever the env gate is set -- the gate cannot
    differ across ranks of one launch -- so it is symmetric by construction.
    """
    import torch.distributed as dist

    from sglang.srt.distributed import get_tp_group

    _STATE["decided"] = True
    if not kt_arena_mode_requested():
        return
    _STATE["enabled"] = True
    if tp_size == 1 or not dist.is_initialized():
        return  # rank 0 builds locally; there is nobody to ship fds to

    # Rank 0's socket setup is fallible, and the broadcast below is a
    # collective: if rank 0 bailed out before broadcasting, the peers would
    # sit in it until the gloo timeout. So a rank-0 failure is broadcast as an
    # empty path instead of skipping the broadcast -- peers then disable
    # themselves immediately and rank 0 carries on serving itself locally.
    path = None
    listener = None
    if tp_rank == 0:
        try:
            path = os.path.join(
                tempfile.gettempdir(),
                f"kt_arena_{os.getpid()}_{time.time_ns()}.sock",
            )
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(path)
            listener.listen(tp_size)
            listener.settimeout(_CONNECT_TIMEOUT_S)
        except Exception:
            logger.exception(
                "[kt-arena] rank 0 could not open the fd-share socket; peers "
                "keep the checkpoint path"
            )
            path = ""
            listener = None

    holder = [path]
    dist.broadcast_object_list(
        holder, src=get_tp_group().first_rank, group=get_tp_group().cpu_group
    )
    path = holder[0]
    if not path:
        if tp_rank != 0:
            _STATE["failed"] = True
        return

    try:
        if tp_rank == 0:
            conns = []
            try:
                for _ in range(tp_size - 1):
                    conn, _ = listener.accept()
                    conn.settimeout(_IO_TIMEOUT_S)
                    conns.append(conn)
            except Exception:
                # Peers that never connected time out on their own; peers that
                # did connect get their sockets closed so their first recv
                # fails fast instead of waiting out the IO timeout.
                logger.exception(
                    "[kt-arena] rank 0 accept failed; serving itself only"
                )
                for c in conns:
                    c.close()
                conns = []
            _STATE["conns"] = conns
            listener.close()
            os.unlink(path)
        else:
            deadline = time.monotonic() + _CONNECT_TIMEOUT_S
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            while True:
                try:
                    sock.connect(path)
                    break
                except (FileNotFoundError, ConnectionRefusedError):
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(0.2)
            sock.settimeout(_IO_TIMEOUT_S)
            _STATE["sock"] = sock
    except Exception:
        logger.exception(
            "[kt-arena] rank %d could not join the fd-share mesh; promotions "
            "on this rank keep the checkpoint path",
            tp_rank,
        )
        _STATE["failed"] = True


def _serve_layer(*, method, layer_idx: int, tp_rank: int, tp_size: int) -> None:
    """Rank 0: export this layer's arenas, ship to peers, build own source."""
    from sglang.srt.layers.moe.kt_ram_source import export_kt_arenas, _wrap

    export = None
    try:
        export = export_kt_arenas(method)
    except Exception:
        logger.exception("[kt-arena] layer %d export failed", layer_idx)

    if export is None:
        fds: List[int] = []
        payload = b""
    else:
        fds, sizes, bases, offsets, geometry = export
        # No id map in the payload on purpose: every rank computes its own
        # _kt_physical_to_logical from process-global metadata, so the ranks
        # this channel cannot reach still translate checkpoint reads.
        payload = pickle.dumps(
            {"sizes": sizes, "offsets": offsets, "geometry": geometry}
        )

    header = _HEADER.pack(layer_idx, len(fds), len(payload))
    for i, conn in enumerate(_STATE["conns"]):
        if conn is None:
            continue
        try:
            if fds:
                socket.send_fds(conn, [header], fds)
            else:
                conn.sendall(header)
            if payload:
                conn.sendall(payload)
        except Exception:
            logger.exception(
                "[kt-arena] peer %d unreachable; it keeps the checkpoint path",
                i + 1,
            )
            try:
                conn.close()
            finally:
                _STATE["conns"][i] = None

    if export is None:
        if not _STATE.get("warned_no_export"):
            _STATE["warned_no_export"] = True
            logger.info(
                "[kt-arena] layer %d: no memfd arenas exported (kt build "
                "without expert_buffer_arenas, or memfd fell back to malloc)",
                layer_idx,
            )
        return
    arenas = [_wrap(base, size) for base, size in zip(bases, sizes)]
    source = _build_source_from_export(
        arenas=arenas,
        offsets=offsets,
        geometry=geometry,
        tp_rank=tp_rank,
        tp_size=tp_size,
    )
    _register_source(method, layer_idx, source)


def _register_source(method, layer_idx: int, source) -> None:
    """Publish a source to the READ registry, or the WRITE-only one.

    The read registry stays empty: kt holds only the cold set, so arena
    offsets go stale at the first swap and the promotion path would serve the
    previous occupant's bytes. The rank-write path tracks moves itself and
    never reads through these, so it gets its own registry instead.
    """
    _STATE["write_sources"][layer_idx] = source


def _receive_layer(*, method, layer_idx: int, tp_rank: int, tp_size: int) -> None:
    """Ranks 1..N-1: receive one layer's fds, map read-only, build the source."""
    sock = _STATE["sock"]
    (got_layer, n_fds, meta_len), fds = _recv_header_and_fds(sock)
    try:
        if got_layer != layer_idx:
            raise RuntimeError(
                f"arena share out of order: expected layer {layer_idx}, got "
                f"{got_layer} -- the per-layer hook order diverged across ranks"
            )
        if len(fds) != n_fds:
            raise RuntimeError(
                f"arena share fd loss: header says {n_fds}, kernel delivered "
                f"{len(fds)}"
            )
        if n_fds == 0:
            return  # this layer exports no arenas; nothing to map
        meta = pickle.loads(_recv_exact(sock, meta_len))
        # Rank-write needs a WRITABLE mapping: cudaHostRegister's page pinning
        # requires write access (attr cudaDevAttrHostRegisterReadOnlySupported
        # is 0 on this platform -- measured, Probe A session), so PROT_READ
        # mappings cannot be registered. The read-only containment ("a
        # consumer rank cannot corrupt the weights") is knowingly given up in
        # that mode and only in that mode; nothing ever writes by
        # construction. (The deleted direct-dma transport was the other reason
        # this could be writable; rank-write is now the only one.)
        prot = mmap.PROT_READ
        if method.kt_config.split_prefill:
            prot |= mmap.PROT_WRITE
        arenas = []
        for fd, size in zip(fds, meta["sizes"]):
            m = mmap.mmap(fd, size, prot=prot)
            _STATE["mmaps"].append(m)
            arenas.append(_wrap_mapping(m))
        source = _build_source_from_export(
            arenas=arenas,
            offsets=meta["offsets"],
            geometry=meta["geometry"],
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        _register_source(method, layer_idx, source)
    finally:
        for fd in fds:
            try:
                os.close(fd)
            except OSError:
                pass


def share_layer_arenas(*, method) -> None:
    """Per-layer entry point; call on EVERY rank, in the same layer order.

    Never raises: a failure here loses an optimization (arena-sourced
    promotion), not correctness, and it must not take the boot down with it.
    """
    from sglang.srt.runtime_context import get_parallel

    tp_rank = get_parallel().tp_rank
    tp_size = get_parallel().tp_size

    # Cold-only residency makes the arena's load-time offsets go stale the
    # moment a swap moves BufferB ownership between expert ids, so the READ
    # path (raw_shard -> promotions) would serve the previous occupant's
    # bytes. That is why sharing is refused here by default.
    #
    # The rank-write demotion path is the exception, and it is safe for the
    # opposite reason: it does not read through these sources at all. It
    # WRITES, through its own SlotOffsets table that replays every move --
    # plan data, identical on every rank -- so staleness is tracked rather
    # than assumed away. kt_config and the env gate are identical on every
    # rank, so this branch is symmetric and nobody enters the broadcast
    # alone.
    if not method.kt_config.split_prefill:
        if not _STATE.get("warned_cold_only"):
            _STATE["warned_cold_only"] = True
            logger.info(
                "[kt-arena] cold-only residency: arena offsets go stale "
                "across swap_expert_slot; promotions keep the checkpoint path"
            )
        return

    if not _STATE["decided"]:
        _decide_once(tp_rank=tp_rank, tp_size=tp_size)
    if not _STATE["enabled"] or _STATE["failed"]:
        return

    layer_idx = method.kt_config.layer_idx
    try:
        if tp_rank == 0 or tp_size == 1:
            _serve_layer(
                method=method, layer_idx=layer_idx, tp_rank=tp_rank, tp_size=tp_size
            )
        else:
            _receive_layer(
                method=method, layer_idx=layer_idx, tp_rank=tp_rank, tp_size=tp_size
            )
    except Exception:
        logger.exception(
            "[kt-arena] rank %d failed sharing layer %d; promotions on this "
            "rank keep the checkpoint path",
            tp_rank,
            layer_idx,
        )
        _STATE["failed"] = True
        # Tear the transport down on the way out so the OTHER side fails fast
        # on its next send/recv instead of waiting out the IO timeout: a
        # permanently failed rank 0 must not strand seven peers in hour-long
        # recvs, and a failed peer must not leave rank 0 writing into a full
        # buffer.
        sock = _STATE["sock"]
        if sock is not None:
            try:
                sock.close()
            finally:
                _STATE["sock"] = None
        for i, conn in enumerate(_STATE["conns"]):
            if conn is not None:
                try:
                    conn.close()
                finally:
                    _STATE["conns"][i] = None
