# SPEC: Direct-DMA cold-expert transport for split prefill

Status: REV 2 — adversarial review (16 confirmed findings) and Probe A folded
in. Supersedes the per-layer batched export (`kt_export_source.py`) as the
*prefill* transport if adopted; the ring export survives only for swap-window
promotions (v1) and as the disarm fallback.

PROBE A RESULTS (2026-08-17, GPU7 beside E3, 40 GB memfd MAP_SHARED):
- PTE cost for a DENSE region: 82.0 MiB measured vs 80.0 predicted — the
  8 B/4K model is exact for dense VA. (Sparse-pattern cost still open: PTE
  pages allocate per 2 MB VA window, so a scattered cold set inside a
  contiguous arena can cost up to ~2-3x the dense model. Probe B measures the
  real pattern; even 3x fits headroom.)
- Registration: 0.117 us/page (40 GB in 1.22 s) — 30x faster than the prior
  JIT-pin model. Boot acquire of ~110 GB ≈ 3-4 s; window acquires ≈ 0.1 s;
  full unregister+re-register compaction of everything ≈ 4 s. Registration
  cost is a NON-ISSUE; design for simplicity, not registration avoidance.
- cudaHostUnregister: full VRAM returned; EXACT base of a prior register
  required (non-base → error 1). Overlapping register → error 712. Odd sizes
  round UP to page boundaries (next-page register → 712). Adjacent disjoint
  registrations both succeed. The same physical pages register fine from two
  processes.

## 1. Problem statement

Split prefill streams every cold expert's TP slice to its GPU each layer.
Today's export design moves each byte through DRAM **3 times** per layer
(export read + export write + DMA read ≈ 14.5 GB/layer), and DRAM is one
serial resource (~500 GB/s measured class): bus time is additive, so the
layer cadence floor is ~29 ms regardless of software overlap (measured live:
~42 ms). The M5 baseline (pinned store, 1 transit) ran ~25 ms/layer =
10.4–10.8k tok/s at chunk 24576.

Direct DMA deletes the prepare stage: each rank's copy engine reads kt's
weight memory **in place** (cudaHostRegister'd), 1 DRAM transit, bottleneck
moves to PCIe (~215 GB/s aggregate measured; 26.8 GB/s/GPU) → ~22.5 ms/layer
→ M5 parity without the 450 GB pinned store.

USER CONSTRAINTS (fixed):
- TP everywhere. EP rejected. The compute path, kernels, slot tables, swap
  machinery are unchanged; ONLY the transport that fills the raw landing
  buffers changes.
- Register **cold experts only** (not the full 1.45 TB arena set).
- GPU page tables for cold experts must be managed carefully **when the cold
  set changes** (swap windows).
- Confirmed config numbers frozen: chunk 24576, 624 GPU experts (272 cold),
  swap 50/8, margin 0.5, mem-fraction 0.89, mamba 40, dcp 8.

## 2. Hardware / driver facts this design rests on (all measured or audited)

| fact | value | consequence |
|---|---|---|
| aggregate H2D, 8 GPUs concurrent | ~215 GB/s (26.8 GB/s/GPU; half of Gen5 x16 line rate) | 604 MB/layer/rank ≈ 22.5 ms floor |
| node DRAM | ~500–550 GB/s class, 2 sockets | DMA read 4.8 GB/layer ≈ 43% occupancy, leaves headroom |
| GPU page tables for pinned sysmem | live in **VRAM** (PTE aperture FBMEM, 8 B / 4K page) | registered bytes cost VRAM: 8 B per 4 KB |
| 2 MB pages | impossible on this rental (shmem_enabled=never, sysfs RO, MADV_COLLAPSE EINVAL) | 4K granularity, full PTE price |
| cudaHostRegisterReadOnly (attr 113) | 0 = unsupported | arenas must be mapped RW (never written) |
| IOMMU | off | DMA at physical addresses; no IOVA overhead |
| driver / kernel | 595.71.05 (CUDA 13) / 6.17 | outside the 6.11–6.12 >2GiB pin-leak bug; cuMemcpyBatchAsync available |
| GPU↔socket affinity | GPUs 0–3 → socket 0, 4–7 → socket 1 | matches kt partition 0/1: all DMA socket-local |
| RLIMIT_MEMLOCK | not consumed by CUDA pinning | no ulimit issue |
| kernel pin accounting | shared pages pinned once per registration, N refs | no physical duplication; RSS inflation is cosmetic |
| GPU0 free VRAM (E-config, 624 experts) | ~2.1 GiB | PTE budget ceiling |
| anecdotal driver pinned-bytes cap | ~DRAM size, accounting unknown | keep nominal (per-rank sums) well under 2 TB |
| registration rate | ~0.7–4 µs per 4K page (prior JIT-pin analysis) | 110 GB ≈ 0.5–2 min boot; 3.2 GB/window ≈ 0.6–3 s |

## 3. kt memory layout (source of truth for what gets registered)

kt splits each expert by NUMA partition (cpu_tp=2): per (expert e, matrix m ∈
{gate, up, down}, partition p) one buffer `m_bb_[e]` with weights `->b` and
u8 E8M0 scales `->d`, bump-packed into the partition's arena. Addresses are
**stable for the process lifetime** (BufferB never freed, never reassigned;
under full-kt residency demotion is a CPU no-op and `swap_expert_slot` is not
called). Weights are immutable — arenas are read-only data.

Per-rank readable extents (rank r: partition p = r//4, local rank lr = r%4):

- **gate/up (w13)**: rank slice = contiguous `[lr·gu_w, (lr+1)·gu_w)` of
  `gate_bb_[e]->b` and `up_bb_[e]->b` (+ contiguous scale slices of `->d`).
  → registrable **slice-exact**, 4 ranges/expert.
- **down (w2)**: rank slice = H rows × `rank_w2_w` bytes at pitch
  `part_w2_pitch` — every 4K page holds all 4 co-partition ranks' bytes.
  → NOT slice-separable; register the **whole `down_bb_[e]` block** (weights
  + scales), shared ×4 ranks.

Registration accounting (272 cold × 92 layers, mixed granularity):

| per rank | bytes | GPU PTEs |
|---|---|---|
| gate/up slice-exact | ~37 GB | ~72 MB |
| w2 whole-block | ~73 GB | ~143 MB |
| **total** | **~110 GB** | **~215 MB** (vs ~2.1 GiB headroom) |

Nominal pinned across 8 ranks ≈ 880 GB (< 2 TB DRAM, clears the anecdotal
cap); unique pinned ≈ 440 GB. Whole-block-everything (simpler intervals)
would be 220 GB/rank → 1.76 TB nominal — rejected: brushes the cap and
doubles PTEs for no functional gain.

## 4. Design

### 4.1 kt side

1. **BufferB returns to memfd** (`KT_BUFFER_B_MEMFD=1`). REV 2, corrected
   against current kt code: the machinery is INTACT and already does most
   of what v1 of this section proposed —
   - arenas are per (layer, partition) → 184 separate memfd inodes by
     construction (inode "sharding" already exists);
   - `bb_arena_init` mmaps with **MAP_POPULATE, and it is load-bearing**
     (in-code measurement: 29 s/layer of serialized shmem faults without
     it vs 1.0 s/layer populated) plus MADV_HUGEPAGE (inert here);
   - the ONE genuinely new kt-side change is re-gating the handle-cycling
     fadvise (removed for anon in kt 2d5a0d2) **for memfd mode only**, so
     checkpoint page cache cannot force reclaim against 1.45 TB of shmem.
   Target boot: ≤ ~20 min (vs 17 min anon); populate replaces part of the
   load cost rather than adding to all of it.
2. **Address-table export API: ALREADY EXISTS** — `expert_buffer_arenas()`
   (fp4-moe.hpp) returns (fds, sizes, bases, per-(partition, expert)
   6-tuple offsets for gate_b/up_b/down_b/gate_d/up_d/down_d, geometry
   (numa, experts, H, I, group)), and `kt_arena_share.py` already ships it
   per layer over SCM_RIGHTS and builds per-rank sources. Slice extents
   (`gu_w`, `rank_w2_w`, pitches) are derived from the geometry exactly as
   `kt_ram_source.py` already derives them. AMX compute untouched.
3. **Mapping protection (review finding)**: `kt_arena_share` maps peers
   PROT_READ today; cudaHostRegister needs a WRITABLE mapping (attr 113=0,
   no read-only registration). Direct-DMA mode maps RW — the containment
   contract ("a consumer rank cannot corrupt the weights") is knowingly
   given up in this mode and documented; nothing ever writes by
   construction.

### 4.2 sglang side: `ArenaDirectSource` (new pipeline source)

Duck-compatible with `ExportColdSource` where `ColdExpertPipeline` touches a
source (`num_cold`, `layer_rows`-equivalent, `after_enqueue`, `reset`) — but
**nearly stateless**: weights are immutable, so there are NO ready/consumed
flags, no epochs, no WAR gates, no poison. The stage/slot structure of the
pipeline (2 device slots, prefetch/consume events, batched swizzle) is
unchanged; only "copy from ring stage" becomes "issue the layer's copy plan".

- **Mapping**: each rank maps the arena memfds once at boot (reuse
  `kt_arena_share` fd-passing; VA-only cost until registered).
- **Copy plan** per (layer, rank): for cold slot j → expert
  `_invert_cold_slot_table(...)[j]` (THE SAME table as routing and swaps —
  the design's one invariant). Three op classes (REV 2):
  - **w13 (contiguous class)**: 4 contiguous H2D copies per expert (gate
    slice, up slice, 2 scale slices) into the raw landing row `[gate; up]` —
    identical layout to today's ring rows, so `apply_batched_swizzle` is
    unchanged. Issued as ONE `cuMemcpyBatchAsync` call per layer (CUDA
    12.8+ LINEAR batch; driver here is CUDA 13), ctypes on libcuda;
    fallback: per-op `cudaMemcpyAsync` from a pre-built plan.
  - **w2 weights (pitched class)**: per expert width `rank_w2_w`, height H,
    spitch `part_w2_pitch`. `cuMemcpyBatchAsync` CANNOT express 2D
    (review finding): use `cuMemcpy3DBatchAsync` (also CUDA 12.8+) — one
    call per layer — falling back to 272 `cudaMemcpy2DAsync` calls.
  - **w2 scales (whole-block class)**: the naive pitched copy is **12 bytes
    wide** (`rank_w2_s = per_gpu/32`) — a DMA-efficiency trap the review
    flagged 4x. Instead copy each expert's WHOLE scale block contiguously
    (H x `part_w2s_pitch`, ~4x the bytes of the exact slice but tiny in
    absolute terms), then ONE on-GPU strided-view copy compacts this rank's
    columns into the landing layout the swizzle already expects (D2D,
    <1 ms/layer). Zero swizzle changes; the 12-byte pattern never exists.
  **Seam rule (review finding)**: every issued op must lie wholly inside ONE
  cudaHostRegister'd unit — CUDA does not guarantee a copy spanning two
  separately-registered ranges stays on the pinned path. The plan builder
  consults the registrar's unit map and splits ops at unit boundaries
  (rare after boot; the batch APIs absorb the extra ops).
  **Plan lifetime (review finding)**: plans are cached per (layer,
  table_version); the table version increments on EVERY logical_to_slot
  flip, and the pipeline checks it at pass start (reset/prime), not "at
  windows" — abort paths and skipped pairs can never leave a stale plan.
  NOT chosen: zero-copy gather kernel (reads registered memory from SMs) —
  rejected per the established handshake lesson: SM occupancy during
  compute-busy prefill; copy engines are free, SMs are not. Reconsider only
  if pitched-copy efficiency measures poorly in Probe B (see risks).

### 4.3 Page-table lifecycle (the careful part)

One component owns every cudaHostRegister/Unregister call:

**`IntervalRegistrar`** — page-granular *accounting*, registration-unit
*actions* (REV 2, per review + Probe A: cudaHostUnregister takes only the
exact base of a prior register call and frees that whole region — there is
no partial unregister).
- Durable objects are **immutable units**: the exact (ptr, size) of each
  cudaHostRegister call. Page refcounts decide *eligibility*; units are the
  only thing ever registered or unregistered.
- `acquire(ranges)` rounds to pages, computes the NOT-yet-registered gap
  intervals (cudaHostRegister errors with 712 on any overlap — adjacent
  experts share boundary pages in the bump-packed arena, so naive
  per-expert registration WILL collide; Probe A confirmed both the error
  and that adjacent disjoint registrations compose), and registers the gaps
  as units **split at expert-buffer boundaries** — never merged across
  experts even when VA-adjacent — so trim granularity stays useful and the
  seam rule's splits stay rare.
- `release(ranges)` decrements pages; a unit ALL of whose pages hit zero
  goes to the **trim list** — nothing is unregistered inline.
- **Compaction** (optional, quiesced windows only): unregister fragmented
  units and re-register the still-live subranges. Probe A prices this at
  0.117 us/page — full-set compaction ≈ 4 s, per-window touch-ups ≈ ms —
  so the ratchet scenario (one live expert pinning a large unit forever)
  has a cheap, always-available escape hatch.
- Invariants (enforced, not hoped; REV 2 orders the window protocol
  precisely — the review showed a per-pair ad-hoc consensus can either hang
  in the M9/M11 shape or serve wrong weights):
  1. **Register-before-routable, with one symmetric collective per layer.**
     Window protocol per layer: (a) every rank attempts `acquire` for ALL
     of the layer's proposed demotions — no table writes yet; (b) ONE
     fixed-shape collective (all_reduce of a per-pair success bitmask) that
     every rank executes unconditionally; (c) pairs with unanimous success
     flip `logical_to_slot`; failed pairs are skipped on EVERY rank
     (acquires for skipped pairs are released immediately) and logged
     loudly. The collective's shape depends only on the plan — identical on
     every rank — never on per-rank outcomes.
  2. **Unregister only in a quiesced window, only whole trim-list units,
     only after both streams synchronize.** Trim/release bookkeeping is
     keyed to the FLIPPED table, not the proposed plan (review finding: a
     skipped pair must not release ranges its still-cold expert needs).
     Trim runs when `registered_bytes > 1.25 x cold_set_bytes`, oldest
     non-cold first. A promoted expert's pages stay registered (harmless:
     pinned + PTEs, never DMA'd) until budget pressure — which also makes
     re-demotion of a recently promoted expert free.
  3. **Boot registration** = `acquire` of the initial cold set (from the
     same table), parallel across ranks, ~3–4 s (Probe A rate), before
     arming. TIMING (corrected by the implementation review): finalize runs
     inside ModelRunner.initialize, BEFORE the KV pool is carved — so the
     PTE VRAM (~0.2–0.6 GB) is absorbed into the pool sizing that follows
     (the pool measures free memory after registration), NOT taken from the
     serving margin. The free-VRAM floor check at arming is a sanity bound
     against grossly wrong projections; the post-pool margin is protected
     by the pool sizing order itself.
- Telemetry: registered bytes, PTE estimate, acquire/trim durations per
  window → server log (`kt-dma` prefix), so drift is visible in the
  existing log-mining flow.

Window cost budget: ≤ 8 pairs × 92 layers × ~4.4 MB ≈ 3.2 GB to acquire
≈ 0.6–3 s per window on top of today's 3.6 s. Trim adds similar only when
it actually runs.

### 4.4 Promotions and demotions

- **Promotion install (cold → resident rows)**: v1 KEEPS the ring export
  (`export_experts_sync`, ~1 ms/layer, proven). Rings shrink to promotion
  duty; they also remain the full fallback transport if direct-DMA disarms.
  v2 (optional): read the promoted expert's slices from the registered
  arena instead (it is cold ⇒ registered by construction), retire the
  export entirely.
- **Demotion**: CPU side is a no-op (full-kt residency). Transport side is
  exactly invariant 1 above. Nothing is ever written back to arenas.

### 4.5 Arming, fallback, config

- Launch flag `--kt-cold-transport {ring-export, direct-dma}` (default
  ring-export until direct-dma is proven; serve scripts change only when
  the user flips them). Arming follows the existing consensus idiom:
  capability probe → `_all_tp_ranks_succeeded` at every fallible collective
  boundary → unanimous arm or unanimous fallback to ring-export (which
  itself falls back to margin-routed CPU). A disarm after boot (registration
  failure at a window) is NOT supported in v1 — the failure mode is "skip
  that swap pair", never "switch transports mid-flight".

## 5. What gets deleted / simplified (accounting honestly for both sides)

- Per-layer export path: `_export_worker`, `_export_layer`, `_kick`, gpos
  epochs, WAR gates, `_publish_consumed`, poison — all prefill pacing.
  `export_experts_sync` + a 1-stage ring survive for promotions.
- DRAM traffic: 14.5 → 4.8 GB/layer; rank 0's CPU no longer burns 96
  threads × 24 ms per layer (also removes the NCCL-lockstep contention
  hypothesis from the board).
- ADDED complexity: IntervalRegistrar, copy plans, memfd boot path, the
  address-table API. Net: state machine shrinks (no cross-process pacing),
  bookkeeping grows (page intervals). The bookkeeping is testable in
  isolation; the pacing was not.

## 6. Probes before any code (approval gates, in order)

1. **Probe A** (runs beside E3, read-only for the server): on GPU7,
   cudaHostRegister a scratch ~7.5 GB memfd mapping; measure (a)
   cudaMemGetInfo delta → VRAM per registered GB, (b) registration µs/page,
   (c) unregister cost. Decides the PTE budget arithmetic with one number.
2. **Probe B** (needs the server down; next approved window): all 8 ranks
   register ~110 GB of a real memfd layout concurrently, then measure
   sustained 8-way H2D from registered-4K-scattered memory vs the pinned
   rings, including the w2 2D-copy pattern and cuMemcpyBatchAsync. Decides:
   driver pinned-cap behavior, DMA throughput from 4K-granular GMMU
   mappings, 2D-copy efficiency at width ~192 B. GO/NO-GO for the design.

## 7. Validation plan (after implementation)

1. **Unit**: IntervalRegistrar (overlap, boundary pages, refcounts, trim).
2. **Smoke** (`smoke_arena_pipeline.py` extension): direct source over fake
   small arenas; swap-window churn (register/flip/trim across windows);
   abort + re-prime passes; A/B bitwise: direct-DMA raw rows == ring-export
   raw rows for identical tables.
3. **Node boot gates**: bitwise row compare on first pass; prefill
   decomposition probe (copy ms/layer, cadence target ≤ ~25 ms); prefill
   ≥ ~10k tok/s @ 24576-token chunk; swap window ≤ ~6 s with registration
   telemetry; decode unchanged (~50–63 tok/s); 1M request regression;
   logprob gate vs the export build.

## 8. Risks

| risk | severity | mitigation / decider |
|---|---|---|
| ~~PTE VRAM larger than 8 B/4K model (dense)~~ | resolved | Probe A: 82.0 vs 80.0 MiB predicted — exact |
| PTE pages allocate per 2 MB VA window → scattered cold set costs up to ~2–3x dense model (review) | medium | Probe B registers the REAL scattered pattern and reads the delta; even 3x (~0.6 GB) fits headroom; arming free-VRAM floor guards the rest |
| driver pinned-bytes cap trips at ~880 GB nominal | kill | Probe B; mixed granularity keeps nominal minimal |
| DMA throughput from 4K-scattered registered pages < ring throughput | kill | Probe B measures; no mitigation on this rental (2 MB pages blocked) — rental-selection criterion recorded |
| copies spanning two registration units silently fall to the pageable path (review) | high | seam rule: plan builder splits ops at unit boundaries; Probe B seam sub-test verifies pinned-rate across an adjacent-registration seam |
| w2 weight pitched-copy efficiency (width `rank_w2_w`, ~hundreds of bytes) | high | Probe B measures `cuMemcpy3DBatchAsync`/`cudaMemcpy2DAsync` at the real geometry; fallbacks: SM gather for w2 only (1/3 of bytes), per-rank-contiguous shadow copy of cold w2 (~90 GB RAM, window-maintained) |
| ~~w2 SCALE 12-byte-wide pitched copies~~ | resolved by design | whole-block copy + on-GPU compact (§4.2); the pattern never exists |
| batch-API absence/limits (cuMemcpyBatchAsync / cuMemcpy3DBatchAsync) | medium | ctypes probes at arming; fallback per-op async copies from prebuilt plans (bounded: ~1.6k ops/layer ≈ 1–3 ms issue time) |
| memfd boot regression / reclaim collapse | medium | 184 inodes + MAP_POPULATE already in kt; fadvise re-gated for memfd; target ≤ 20 min |
| registration overlap/refcount/unit bugs | medium | IntervalRegistrar unit tests + smoke churn test; Probe A pinned the API semantics to build against |
| RW mapping of weights (no read-only registration) | low | never written by construction; documented blast radius |
| RSS-summing dashboards over-count; OOM badness | low | known (memory file); document in doctor |
| window cost grows | low | Probe A: acquires ≈ 0.1 s/window — noise against 3.6 s; telemetry + budget alarm anyway |

CONFIG NOTE (review): the numbers in §3 are computed for the frozen 624/272
config that takes effect at the next boot; the currently-serving E3 runs the
previous 620/276. All per-layer byte figures shift by ~1.5% between the two —
nothing decision-relevant changes, but probes and gates must quote which
config they measured.

## 9. Sequencing

1. Probe A (today, pending approval) → arithmetic locked.
2. Probe B at the next approved server-down window → GO/NO-GO.
3. kt: memfd re-enable + address-table API; sglang: IntervalRegistrar (+unit
   tests) — parallelizable, both behind the flag.
4. ArenaDirectSource + copy plans + window hook; smoke extension.
5. Adversarial review of the diff; deploy; boot behind
   `--kt-cold-transport direct-dma` on the E-config; run §7 gates.
6. Only after gates pass: user decides whether the flag default flips.
