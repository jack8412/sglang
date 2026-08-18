# SPEC: host-RAM L2 cache for KV **and** mamba state

Status: PLAN. Nothing built yet. Every number below is either measured on this
node (marked MEASURED) or derived from those measurements (marked DERIVED).

## The opportunity

Deleting the pinned cold store freed 586 GB of host RAM (shmem 997 -> 411 GB).
The node has 1,996 GB total against a 1,916 GiB cgroup cap, and a running V12
sat at used 523 GB / shared 410 GB. That leaves roughly **900 GB unclaimed**.

Meanwhile a 947,900-token prefill costs **171.6 s** (MEASURED, V12). Any prefix
we can replay from host RAM instead of recomputing is worth ~3 minutes.

## What already exists -- this is mostly a configuration exercise

sglang's HiCache already implements exactly this, including the hybrid case:

* `--enable-hierarchical-cache` builds a `HostPoolGroup` carrying **both**
  `PoolName.KV` and `PoolName.MAMBA`, so one flag offloads the KDA recurrent
  state alongside the MLA KV pages.
  NOT via `HiMambaRadixCache` -- that class is **dead code**, constructed
  nowhere in `python/` (`registry.py:114-115` short-circuits every
  `is_hybrid_ssm` model to `_create_unified_radix_cache` before the
  hierarchical branch). The live path is `UnifiedRadixCache` +
  `ComponentType.MAMBA` (`registry.py:175-176,188-192`) -> `init_hicache`
  (`unified_radix_cache.py:321-357`) -> `_MambaStrategy`
  (`hybrid_pool_assembler.py:1191-1245`) -> `build_hybrid_mamba_stack`
  (`:612-700`), which builds the group at `:663-684`. Grep the log and write
  any assertion against THAT, never against `HiMambaRadixCache`.
* **DCP is supported for L1/L2**, and the device<->host path is DCP-aware
  (`dcp_size`/`dcp_rank` reach the host pool, which divides the widened page
  back down; `logical_size = size * dcp_size`).
  But `_resolve_hicache_dcp_compatibility` (`server_args.py:7517-7550`) refuses
  **five** things under `--dcp-size > 1`, not just L3:
  L3 storage backend (`:7520`), **speculative decoding** (`:7528`),
  `--enable-lmcache` (`:7534`), `--enable-hisparse` (`:7539`), and non-MLA
  (`:7544`). A separate refusal also exists for PD-decode + DCP + hicache
  (`arg_groups/pd_disaggregation_hook.py:49-52`).
* Our `page_size=64` is correct even though no k3ops file sets it: K3+DCP forces
  `tokenspeed_mla`, and `_mla_backend_page_constraints` snaps page_size to 64
  (`arg_groups/overrides.py:2087-2115`, running before `_page_size_default`).
  The host pool's `page_size % dcp_size == 0` assert fires on the ALREADY
  WIDENED page (512 % 8, via `kv_cache_configurator.py:1564-1566`), so it is
  satisfied by construction for any page size -- it can only trip if a caller
  forgets to widen.
* K3 is MLA, satisfying "HiCache + DCP is only wired for the MLA host pool".
* Keep `--hicache-mem-layout` at the default `page_first`: `MambaPoolHost`
  asserts `page_first`/`page_first_direct` only (`memory_pool_host.py:80-83`),
  and `--hicache-io-backend direct` would silently rewrite the layout
  (`server_args.py:7566-7573`).

So the first milestone is a boot with one extra flag, not a patch.

## Sizing (DERIVED from MEASURED device pools)

Per rank, V12: KV 602,176 tok / 7.75 GiB = **13,824 B/token**; mamba 40 slots /
1.11 GiB = **29.8 MB/slot**. Global KV = 602,176 x 8 = 4.82M tokens = 4.6x 1M.

`--hicache-size N` is **per rank, in GB** -- a hard limit, not a ratio -- and is
split between the KV and mamba host pools in proportion to their device-pool
bytes (87.5% / 12.5% here). Total host cost is **8N**, because each rank
allocates its own pool. It overrides `--hicache-ratio` entirely.

| N (GB/rank) | host total | KV tok/rank | KV tok global | vs device | ~1M contexts | mamba slots |
|---|---|---|---|---|---|---|
| 32 | 256 GB | 2.03M | 16.2M | 3.4x | 15 | 134 |
| **64** | **512 GB** | **4.05M** | **32.4M** | **6.7x** | **31** | **268** |
| 100 | 800 GB | 6.33M | 50.6M | 10.5x | 48 | 419 |
| 128 | 1,024 GB | 8.10M | 64.8M | 13.5x | 62 | 537 |

Recommend **64 GB/rank (512 GB = 477 GiB total; the flag is DECIMAL GB)**: it is 6.7x the device pool, and leaves
~400 GB of the ~900 GB headroom unspent against the cgroup cap.

## What this does and does not buy

**Does not:** raise concurrency. `max_running_requests` is set by the *device*
mamba slots and KV pool; an L2 pool adds no running slots. Concurrency 16 at 1M
remains a device-pool question, unchanged by this work.

**Does:** raise prefix-cache capacity by 6.7x, so repeated and shared prefixes
skip prefill. Ceiling on a full 948K-token hit (DERIVED): replay is an H2D copy
of 13,824 B x 948K = 13.1 GB total, 1.64 GB per rank, ~0.1 s at PCIe rates,
against **171.6 s** to recompute. That is the headline: ~100x on a repeat 1M
prefix, and proportional on a partial one.

Write cost is small and in the opposite direction: `write_through` (the default)
pushes 1.64 GB/rank D2H per 1M prefill, ~0.1 s against a 171.6 s prefill. It
does **not** contend with split-prefill's cold-expert stream, which is H2D --
PCIe is full duplex.

## The hybrid correctness gate

For a hybrid model a prefix hit must restore **both** the MLA KV pages and the
KDA recurrent state at that prefix position. KV alone is not enough and would be
silently wrong. `HiMambaRadixCache` exists precisely for this, and
`attach_hybrid_pool_to_mamba_cache` wires the MAMBA host pool next to the KV
one. **Gate S4 below is the test that this actually holds on K3.**

## Blockers and traps, in the order they will bite

1. **HiCache + speculative decoding is refused UNDER DCP** ("the draft-model
   host pool has no DCP index translation", `server_args.py:7528-7533`). So L2
   and DSpark are mutually exclusive **in our 1M config**, not in general: at
   `--dcp-size 1` the two are wired together on purpose
   (`speculative/base_spec_worker.py:234-266` builds a PACKED/SIDECAR draft plan
   when hierarchical cache is on). Any plan wanting both AT 1M needs that
   translation written first.
2. **The host-memory guard cannot save you.** Each rank checks
   `psutil.virtual_memory().available - 10 GiB` on its own, once per pool
   (`pool_host/base.py:140-151`, `memory_pool_host.py:123-131`), with no
   barrier -- all 8 see the same free memory, all 8 pass, 8 pools get allocated.
   Worse, psutil reads `/proc/meminfo`, so it is blind to the **1,916 GiB cgroup
   cap** that is the real limit here. Size deliberately and do the x8 by hand --
   this is the shape of the over-allocation that OOM-killed ai.v8.pro.
3. **`--hicache-storage-backend` must stay unset** (L3 is refused under DCP 8).
4. **Pinning cost at boot is UNMEASURED.** Both pools are pinned via
   `cudaHostRegister` (`pool_host/mla.py:62`, `memory_pool_host.py:72`,
   `pool_host/common.py:120-130`), so at N=64 that is the full **64 GB/rank**
   (56 KV + 8 mamba), not 56. Boot is already 17.5 min.
   The allocation is `mmap(MAP_SHARED|MAP_ANONYMOUS|MAP_POPULATE)` +
   `MADV_POPULATE_WRITE` (`storage/mmap/mmap_allocator.py:120-131`), so it lands
   in **Shmem** -- the same `free -g` bucket as the 410 GB kt arena -- and is
   fully resident the moment it is allocated. Watch `shared`, not `used`.
5. **Page-cache eviction.** A pinned pool displaces the checkpoint page cache.
   Harmless *now* only because the swap path no longer reads the checkpoint
   (rank-write demotion + arena promotion). Do not reintroduce a disk read.

Checked and clear: `--enable-int8-mamba-checkpoint` conflicts with hierarchical
cache (we do not use it); `--disable-radix-cache` conflicts (we do not use it);
and the L1/L2 path spawns **no new Python threads** in the scheduler process:
`HybridCacheController` starts none, and transfers ride CUDA streams
(`managers/cache_controller.py:305-306,696,806`). The threads belong to
`HiCacheController` and start for **any** `--hicache-storage-backend` value
(`cache_controller.py:367-389`, started at `:516`) -- not, as first written,
only for mooncake/umbp/nixl. With the backend unset none start, so the GIL tax
that made `assert_tables_consistent` cost 1.80 s does not apply.

## One boot, four gates

Boot is 17.5 min, so this is deliberately ONE restart that answers everything.
Nothing below needs a second boot, because the size is set directly in GB and
does not have to be swept.

Launch adds exactly two flags to the V13 line:

    --enable-hierarchical-cache --hicache-size 64

`--hicache-size` is a hard per-rank GB limit and **overrides** `--hicache-ratio`
(`if host_size > 0: size = host_size * 1e9 // size_per_token`, else the ratio
path). Sizing by GB is the right call: the ratio is defined against the device
pool, so it silently changes meaning whenever `--mem-fraction-static` or the KV
dtype moves, while a GB figure is the number the host-memory budget is actually
made of. Set the ratio never; set the GB once.

Then, in that single boot, in this order:

| gate | what to read | fails if |
|---|---|---|
| **G1 boot** | host-pool allocation lines; boot-time delta vs 17.5 min; `free -g` | total host lands near 8 x 64 = 512 GB, not 8x that (blocker 2) |
| **G2 no regression** | the standard prefill sweep + decode | prefill/decode differ materially from the V13 baseline |
| **G3 the hit** | same ~948K prompt twice; `cached_tokens` and TTFT | `cached_tokens` stays 0 -- the L2 path never engaged |
| **G4 hybrid correctness** | greedy output, cold prefill vs L2 hit | outputs differ -- the KDA state was not restored with the KV |

G4 is the one that cannot be skipped. For a hybrid model a prefix hit must
restore both the MLA KV pages and the KDA recurrent state; if only the KV came
back the failure is a silent quality regression, not a crash.

## Why 64 GB/rank, chosen once

The cap is the cgroup's 1,916 GiB against a running footprint of ~604 GB used
+ ~410 GB shared (the kt arena) = ~1,014 GB, leaving ~900 GB. At 64 GB/rank the
pool is 512 GB and ~390 GB stays free. At 96 GB/rank (768 GB) only ~130 GB
would remain, and because the pool is **pinned** it is unreclaimable -- an
overshoot is an OOM kill costing a 17.5-minute boot, not a slowdown. 64 is the
largest round figure that keeps a real margin, and it is already 6.7x the
device pool.
