<!-- P0 survey for making the L3 hicache storage backend work under --dcp-size > 1.
     Produced by a 7-agent read-only source survey over python/sglang/srt/mem_cache,
     managers/cache_controller.py and server_args.py, then cross-checked. Nothing in
     this document has been run: it is a source survey, and every "UNCLEAR" in it is
     load-bearing. Do not promote an UNCLEAR to a conclusion without measuring it. -->

# P0 — L3 HiCache under `--dcp-size > 1` (MLA): source survey

Tree: `/home/user/sglang`, branch `k3-split-prefill`, HEAD `e8132a02a0`.
Every line below was re-read at this HEAD (the six traces ran at `a7a3ff5e4a`; all cited line numbers still hold — I found no drift).

Units convention used throughout:
- **LOGICAL** = widened / global slot-or-token space, `page_size * dcp_size` wide, `size * dcp_size` deep. This is what the radix tree, `HiCacheController`, and `HostKVCache.alloc()` speak.
- **PHYSICAL** = this rank's rows, `page_size = widened // dcp_size`, `size` deep. This is what `kv_buffer`, the transfer kernels, and `mem_pool_host.page_size` speak.
- `MLATokenToKVPoolHost.get_size_per_token()` is **full per-token bytes** and belongs to *neither* — DCP shards tokens, not features, so one PHYSICAL row holds one whole token's MLA KV.

---

## 1. PREREQUISITE VERDICT

**PRESENT. Do not stop. No cherry-pick is required.** All six traces agree, and I verified it independently.

| Prereq | Verified location | Content |
|---|---|---|
| Class defaults | `python/sglang/srt/mem_cache/pool_host/base.py:82-83` | `dcp_size = 1` / `dcp_rank = 0` on `HostKVCache` |
| ctor kwargs + widening | `base.py:95-96, 102-110` | comment `# page_size arrives widened (x dcp_size); size/page_size/page_num are physical.`; `assert page_size % dcp_size == 0`; `self.page_size = page_size // dcp_size` |
| Logical accessors | `base.py:333-341` | `logical_size = self.size * self.dcp_size`, `logical_page_size = self.page_size * self.dcp_size` |
| Translation helper | `base.py:343-356` | `owned = indices[indices % self.dcp_size == self.dcp_rank] // self.dcp_size` + residue-balance assert |
| Free-list in LOGICAL slots | `base.py:305-316` (`mem_state`, `free_slots`, `slot_used` all `logical_size`), `base.py:358-362` (`assert need_size % self.logical_page_size == 0`) | radix/controller layer sees widened slots end-to-end |
| Page widening upstream | `python/sglang/srt/mem_cache/kv_cache_builder.py:212-223` | `page_size=(page_size if not get_parallel().dcp_enabled else token_to_kv_pool_allocator.page_size)` with the comment *"When dcp enabled, kv_pool_allocator.page_size is page_size * dcp_size."* |
| Construction sites | `python/sglang/srt/mem_cache/hiradix_cache.py:105-113` (`dcp_size=_parallel.attn_dcp_size, dcp_rank=_parallel.attn_dcp_rank`); `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py:95-100` (`assert use_mla` then the same two kwargs) | MHA branch (`hiradix_cache.py:86-93`) passes none — matches the `use_mla_backend()` gate |
| Owner rule matches device side | `python/sglang/srt/mem_cache/memory_pool.py:4032` (`valid_mask = loc % parallel.attn_dcp_size == parallel.attn_dcp_rank`); kernel `python/sglang/kernels/ops/kvcache/mla_buffer.py:39-41` (`is_valid = loc % DCP_WORLD_SIZE == DCP_RANK; safe_loc = safe_loc // DCP_WORLD_SIZE`) | host and device use the identical rule |
| Dedicated CPU test | `test/registered/unit/mem_cache/test_hicache_dcp_host_pool.py` (`DCP_SIZE=8`, `PHYSICAL_PAGE=64`, `WIDENED_PAGE=512` at `:25-27`) | translation, sizing, widened-page-granular alloc, `test_backup_receives_physical_rows` (`:195`), `test_l3_data_page_is_guarded` (`:203`) |

**Scope caveat that matters more than the verdict:** `dcp_kernel_indices` is invoked at exactly **two** call sites in the entire tree — `pool_host/mla.py:255-256` (`load_to_device_per_layer`) and `pool_host/mla.py:419-420` (`backup_from_device_all_layer`). Everything else that accepts an index on the host pool treats it as a PHYSICAL row. The prerequisite is complete *for L1↔L2 only*; it is a narrow bridge, not a general translation layer.

Two prerequisite gaps to note before building on it:
- `HostPoolGroup` (`python/sglang/srt/mem_cache/memory_pool_host.py:1540-1560`) — the object the storage backends actually see on the unified/hybrid path — forwards `layout`, `page_size` (PHYSICAL, `:1552`), `device`, `size`, `logical_size` (`:1555`) but **not** `logical_page_size`, `dcp_size`, `dcp_rank`, or `dcp_kernel_indices`. It is a plain class, not a `HostKVCache` subclass (`grep -c dcp memory_pool_host.py` → `0`). Any fix written as `host_pool.dcp_size` raises `AttributeError` on that path.
- No evidence in this tree that HiCache+DCP L1/L2 has ever run on hardware; the only coverage is the CPU unit test above. **UNCLEAR** — I could not check `runs/` (out of area, and no remote access permitted).

---

## 2. BLOCKERS

### B1 — `server_args.py::_resolve_hicache_dcp_compatibility()` — **DIFFERENT** (present, but five conditions, not one)

`python/sglang/srt/server_args.py:7517-7556`, called from `_handle_hicache` step 3 at `:7515`, itself called at `:3661`.

- `:7518-7519` guard: `if self.dcp_size <= 1 or not self.enable_hierarchical_cache: return`
- `:7520-7527` **the L3 raise** — *"under DCP each rank holds a distinct interleaved MLA KV shard, so the rank-0-only replicated-MLA backup and the storage keys must become dcp_rank-aware first. Run HiCache+DCP with L1/L2 only."*
- `:7528-7533` speculative decoding · `:7534-7538` `enable_lmcache` · `:7539-7544` `enable_hisparse` · `:7545-7550` `not use_mla_backend()`
- `:7551-7556` the "L1/L2 only" banner.

P1 relaxes **only** the first. The other four are separate unfinished paths.

Three additional facts about this gate that change the plan:

1. **`_handle_hicache` itself returns early** at `server_args.py:7502-7506` unless `enable_hierarchical_cache or disaggregation_decode_enable_offload_kvcache`.
2. **The LMCache condition is inert where LMCache is live.** `registry.py:117` returns the hierarchical cache first; the LMCache branch at `registry.py:131` and the FlexKV branch at `registry.py:144` are reached **only** when `enable_hierarchical_cache` is False — exactly the case in which `_resolve_hicache_dcp_compatibility` has already returned at `:7518`. Nothing forces `enable_hierarchical_cache` from either flag (`grep -n enable_lmcache\|enable_flexkv server_args.py` → `2899, 2911, 7534, 8260, 8407-8416`). So `--enable-lmcache --dcp-size 8` and `--enable-flexkv --dcp-size 8` both start today. **CONFIRMED.**
3. **A second, independent DCP+hicache rejection exists for PD decode**: `python/sglang/srt/arg_groups/pd_disaggregation_hook.py:37` (`decode` + `dcp_size > 1`) → `:49-52` `raise ValueError("PD decode DCP currently requires chunk cache; --enable-hierarchical-cache is not supported.")`, alongside transfer-backend (`:38-43`) and radix-cache (`:44-48`) rejections. Lifting B1 does **not** lift this. Note it does *not* mention `disaggregation_decode_enable_offload_kvcache`, and `python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py:70-76` builds an `MLATokenToKVPoolHost` with `self.page_size = server_args.page_size` (`:46`) and **no dcp kwargs** — so DCP + decode offload is uncovered by both gates.

### B2 — `cache_controller.py` `backup_skip` — **PRESENT**, but four copies plus a fifth write gate

`python/sglang/srt/managers/cache_controller.py:470-475`:
```
# for MLA models, only one rank needs to backup the KV cache
self.backup_skip = (
    self.storage_config.is_mla_model
    # todo: load balancing
    and self.storage_config.tp_rank != 0
)
```
Consumed at `:1247` (`if not self.backup_skip: self._page_backup(operation)`). `is_mla_model` is a misnomer: it is fed `is_rank_replicated = is_mla_model or is_compressed_mla_model` (`:615`) passed as `is_mla_model=is_rank_replicated` at `:640` with an in-tree `# TODO(hzh): Rename is_mla_model to is_rank_replicated.` at `:639`. `grep -c dcp cache_controller.py` → **0**.

Duplicates that must move together:

| # | Location | Text |
|---|---|---|
| a | `managers/cache_controller.py:471-475` | the above |
| b | `mem_cache/storage/nixl/hicache_nixl.py:106` | `self.backup_skip = self.is_mla_model and storage_config.tp_rank != 0` (used `:854`) |
| c | `mem_cache/storage/hf3fs/storage_hf3fs.py:221-224` | `if self.is_mla_model and self.rank != 0: self.skip_backup = True; self.rank = 0` (used `:438`); plus `rank_for_path = 0 if is_mla_model else rank` (`:361`) → all ranks on `prefix.0.bin` (`:365`) |
| d | `mem_cache/hybrid_cache/hybrid_cache_controller.py:776, 799-820` | `should_backup()` + overridden `backup_thread_func` |
| e | `mem_cache/storage/file/lru_file_evictor.py:83` | `self._is_storage_owner = (not is_mla_model) or (tp_rank == 0)` — non-owners get `reserve()` refused at `:178-183`, i.e. a **second write veto** whenever eviction is configured |
| f | `mem_cache/hicache_storage.py:390` | `if not os.path.exists(self.file_path) and tp_rank == 0 and attn_cp_rank == 0:` — directory creation only |

(d) is the useful precedent, not just a duplicate: `hybrid_cache_controller.py:799-806` already overrides `backup_skip` per pool — *"Kimi-K3 Mamba/KDA state is TP-sharded even when the primary MLA KV pool is replicated."*

### B3 — `hicache_storage.py` key/config suffix logic — **PRESENT**

`python/sglang/srt/mem_cache/hicache_storage.py:380-388` in `HiCacheFile.__init__` (class `:361`):
```
self.config_suffix = f"_{model_name}"
if not is_mla_model:  self.config_suffix += f"_{tp_rank}_{tp_size}"
if enable_pp:         self.config_suffix += f"_{pp_size}_{pp_rank}"
# Under NSA context parallel each CP rank holds a disjoint slice of every
# page, so give each rank its own file key to avoid a cross-rank write race.
if attn_cp_size > 1:  self.config_suffix += f"_cp{attn_cp_rank}_{attn_cp_size}"
```
Applied at `_get_suffixed_key` (`:435-436`) and `_get_component_key` (`:438-442`).

`HiCacheStorageConfig` (`:27-40`) has `tp_rank/tp_size/pp_rank/pp_size/attn_cp_rank/attn_cp_size/is_mla_model/...` and **no `dcp_rank`/`dcp_size`**. Populated at `cache_controller.py:586-648`; `attn_cp_rank/size` come from `get_attn_cp_rank_and_size()` (`cache_controller.py:322-330`), which reads the `attn_cp` ProcessGroup and returns `(0, 1)` when it is None. DCP does not populate it.

The hash itself carries nothing rank-scoped: `get_hash_str(token_ids, prior_hash, page_size)` → `get_native_hash` (`mem_cache/utils.py:106-112`). The per-backend suffix is the **only** seam where rank scoping can be added.

### B4 — `pool_host/mla.py` `assert dcp_size == 1` — **PRESENT, but it guards one of four L3 entry points**

`python/sglang/srt/mem_cache/pool_host/mla.py:524-529`, first statement of `get_data_page`:
```
assert self.dcp_size == 1, (
    "HiCache L3 storage paths are not yet DCP-aware (per-rank shards "
    "need dcp_rank-scoped keys); --hicache-storage-backend with "
    "--dcp-size > 1 should have been rejected at server start."
)
```

**ABSENT on all three siblings** — verified by reading the full bodies:

| Method | Line | Guard? | Reached from |
|---|---|---|---|
| `get_data_page` | `mla.py:524` | **yes** | `cache_controller.py:1134`, `:1191`; `storage_hf3fs.py:640`; `hicache_storage.py:669` |
| `get_dummy_flat_data_page` | `mla.py:543` | no | `cache_controller.py:976`, `:1204`; `storage_hf3fs.py:647` |
| `set_from_flat_data_page` | `mla.py:556` | no | `cache_controller.py:990-993`, `:1207-1209`; `storage_hf3fs.py:670-672`; `hicache_nixl.py:796-799` |
| `get_page_buffer_meta` | `mla.py:583` | no | `hicache_nixl.py:500`, `:675`; `hicache_simm.py:264`, `:277` |

**Correction to two traces:** `get_page_buffer_meta`'s only assert is `assert len(indices) % self.page_size == 0` (`mla.py:587`). Traces 1 and 3 claimed this would fire under DCP. It will **not**: a whole-widened-page run has `len = k · P_logical = k · dcp · P_phys`, which is divisible by `P_phys` for every `k`. The assert passes silently and offers zero protection. Only ragged runs would trip it.

---

## 3. THE SIZING MAP

Two live L3 shapes exist for the KV pool. Both start from the same LOGICAL host indices:
- **Generic path** (`file`, and any backend not in the zero-copy list): `page_get_func/page_set_func = _generic_page_get/_generic_page_set`, set at `cache_controller.py:499-500`.
- **Zero-copy v1 path** (`hf3fs`, `mooncake`, `eic`, `nixl`, `simm`, `mori`, and `dynamic` with `interface_v1`): `cache_controller.py:502-510`.

There is a **third** shape, `batch_set_v2`/`batch_get_v2` → `HiCacheFile._batch_io_v2` (`hicache_storage.py:672-696`), but see the *Disagreement D1* note under the tables: it is **not** on the KV path.

### 3a. L3 WRITE PATH (host → storage)

| # | Step | file:line | Byte-length / index expression | Unit | Needs change? |
|---|---|---|---|---|---|
| W1 | Node's host slots captured at write-through | `hiradix_cache.py:862` (`node.host_value = host_indices.clone()`); unified: `unified_radix_cache.py:1176-1180` | slots from `HostKVCache.alloc` → range `[0, size*dcp_size)` | **LOGICAL** | no |
| W2 | Hash keys computed | `hiradix_cache.py:1980`→`utils.py:126-135`→`utils.py:106-112` | one key per `self.page_size` tokens, `self.page_size` = widened | **LOGICAL** | **yes — key must gain a dcp term (B3)**; content hash cannot carry it |
| W3 | `write_backup_storage` → `write_storage` | `hiradix_cache.py:916-941`; `unified_radix_cache.py:1165-1191`; `cache_controller.py:1115-1129` | pass-through into `StorageOperation` | LOGICAL | no |
| W4 | Backup gate | `cache_controller.py:1245-1249`; hybrid override `hybrid_cache_controller.py:762-798` | `if not self.backup_skip` | n/a | **yes (B2)** |
| W5 | `_page_backup` batch slice | `cache_controller.py:1211-1218` | `host_indices[i*self.page_size : (i+n)*self.page_size]`, controller `page_size` | **LOGICAL** / LOGICAL — self-consistent | no |
| W6 | Completed-token accounting | `cache_controller.py:1235` | `+= self.page_size * len(batch_hashes)` | **LOGICAL** | no (but see M1) |
| **Generic (file) branch** | | | | | |
| W7 | `_generic_page_set` | `cache_controller.py:1132-1137` | `get_data_page(host_indices[i * self.page_size])` — **stride** LOGICAL & correct; **value** LOGICAL, consumed as PHYSICAL | **MIXED — BREAK** | **yes** |
| W8 | `get_data_page` payload | `pool_host/mla.py:524-541` | `kv_buffer[:, index : index+self.page_size, :, :]` (layer_first, `:531`) / `[index : index+ps, :, :, :]` (`:533`) / `real_index = index // self.page_size` (`:535-536`); `.flatten()` (`:540`) | length = `page_size_PHYS × size_per_token`, i.e. **PHYSICAL and already the right byte count**; offset **LOGICAL and wrong** | **yes — offset only, not length** |
| W9 | `HiCacheFile.set` dedup fast path | `hicache_storage.py:510-513` | `if self.exists(key): return True` | n/a | **yes** — with an unscoped key, ranks 1..7 skip their write and report success |
| W10 | `HiCacheFile.set` bytes | `hicache_storage.py:519` (`value_bytes = value.numel()*value.element_size()`), reserve `:521`, `tofile` `:529`, `os.replace` `:530` | inherited entirely from W8 | PHYSICAL | no — fixing W8 fixes this |
| W11 | `batch_set` | `hicache_storage.py:549-559` | plain loop, all-or-nothing bool | n/a | no |
| **Zero-copy v1 branch** | | | | | |
| W12 | `_page_set_zero_copy` | `cache_controller.py:1139-1142` | raw LOGICAL `host_indices` to `batch_set_v1` | **LOGICAL** | **yes** |
| W13 | `get_page_buffer_meta` pointers | `pool_host/mla.py:583-620` | layer_first `:591-597`: `base + indices[i]*kv_cache_dim*itemsize + layer_id*self.size*kv_cache_dim*itemsize`; page_first `:603-608`: `base + indices[i]*layer_num*kv_cache_dim*itemsize`; `element_size` `:599/:610-615` uses PHYSICAL `page_size` | **MIXED — length PHYSICAL/right, base pointer up to `dcp_size×` past the buffer** | **yes** |
| W14 | mooncake key/index ratio | `mooncake_store.py:1005-1007` | `assert len(keys) == len(host_indices) // self.mem_pool_host.page_size` — LOGICAL numerator / PHYSICAL denominator, ratio = `dcp_size` | MIXED | **yes — fires loudly** |
| W15 | simm / umbp same assert | `hicache_simm.py:290`; `umbp_store.py:971`, `:1076` | identical expression | MIXED | **yes — fires loudly** |
| W16 | hf3fs set preprocess | `storage_hf3fs.py:893` (`page_num = len(host_indices) // self.mem_pool_host.page_size`), `:898` (`host_indices[i * self.mem_pool_host.page_size]`) | LOGICAL / PHYSICAL → `page_num = dcp_size × len(keys)` | MIXED | **yes** (downstream failure shape **UNVERIFIED** — I did not trace the usrbio client) |
| W17 | nixl v1 length guard | `hicache_nixl.py:709-712` | `page_num = len(host_indices)//page_size`; `if len(keys) != page_num:` warn + `return [], []` | MIXED | **yes — silent all-False, not an exception** |
| W18 | `bytes_per_page` for hf3fs record slots | `storage/backend_factory.py:174-180` | `get_ksize_per_token() * mem_pool_host.page_size` (page_first) / `get_size_per_token() * page_size` (layer_first) | full-per-token × PHYSICAL page = **exactly one rank's shard of one widened page** | **NO — already correct. Do not double-divide.** |

### 3b. L3 READ PATH (storage → host)

| # | Step | file:line | Byte-length / index expression | Unit | Needs change? |
|---|---|---|---|---|---|
| R1 | Prefetch key alignment | `hiradix_cache.py:1778` (`prefetch_key = prefetch_key.page_aligned(self.page_size)`) | widened page | **LOGICAL** | no |
| R2 | Existence query | `cache_controller.py:1054-1076` — `get_hash_str(..., page_size=self.page_size)` `:1061`; `storage_query_count += hit_page_num * self.page_size` `:1070` | one key per widened page; LOGICAL token count | **LOGICAL** | **key: yes (B3)**; counts: no |
| R3 | Cross-rank hit MIN-reduce | `cache_controller.py:1096-1102` (`ReduceOp.MIN` over `prefetch_sync_groups`, built `:331-353`); mirrored `hiradix_cache.py:1500-1501` | scalar token count | LOGICAL | see risk R-2 |
| R4 | Host allocation for the hit | `hiradix_cache.py:633-658` — `alloc_len = operation.storage_hit_count` `:633`, `alloc(alloc_len)` `:634`, `hash_value[: alloc_len // self.page_size]` `:656-657` | `alloc` asserts `% logical_page_size == 0` (`base.py:360-362`) | **LOGICAL** | no |
| R5 | `_page_transfer` batch slice | `cache_controller.py:1001-1003` | `host_indices[i*self.page_size : (i+n)*self.page_size]`, controller page | **LOGICAL** / LOGICAL — self-consistent | no |
| R6 | Tail release | `cache_controller.py:1041`-ish (`operation.host_indices[completed_tokens:]`) | LOGICAL slice of LOGICAL array | LOGICAL | no |
| **Generic (file) branch** | | | | | |
| R7 | Destination buffers allocated | `cache_controller.py:975-978` → `pool_host/mla.py:543-554` | `zeros((layer_num, self.page_size, 1, kv_cache_dim)).flatten()` — PHYSICAL page | **PHYSICAL — correct length** | no (no guard though) |
| R8 | Backend read | `hicache_storage.py:463-486` | `expected = target_location.numel()*element_size()` `:472`; `f.readinto(buf) != expected → IOError("Short read")` `:474-476` | **PHYSICAL** | no — but note `readinto` under-fills only if the *file* is short, so a file holding a full widened page would be silently truncated to its first `1/dcp_size` |
| R9 | **Landing write** | `cache_controller.py:989-993` → `pool_host/mla.py:556-581` | `set_from_flat_data_page(host_indices[i * self.page_size], page)`; body: `kv_buffer[:, index : index+self.page_size, ...] = data_page.reshape(layer_num, self.page_size, 1, kv_cache_dim)` | **MIXED — BREAK.** LOGICAL index used as PHYSICAL row; no dcp assert | **yes — the core read bug** |
| R10 | Progress | `cache_controller.py:994-996` (`operation.increment(self.page_size)`) | LOGICAL | LOGICAL | no |
| **Zero-copy v1 branch** | | | | | |
| R11 | `_page_get_zero_copy` | `cache_controller.py:958-972` | raw LOGICAL `host_indices` to `batch_get_v1`; `inc += self.page_size` (LOGICAL) | **LOGICAL** | **yes (index)** |
| R12 | nixl v1 guard | `hicache_nixl.py:709-712` | as W17 → `([], [])` → `[False]*len(keys)` at `:830-832` | MIXED | **yes — silent prefetch miss** |
| R13 | nixl non-zero-copy landing | `hicache_nixl.py:780-799` | `page_num = len(host_indices)//page_size`; `set_from_flat_data_page(host_indices[i*page_size], self._bounce_get[i])` | MIXED | **yes** |
| R14 | hf3fs get pre/post | `storage_hf3fs.py:634` and `:657` (`page_num = len(host_indices) // self.mem_pool_host.page_size`); `:640`, `:671` (`host_indices[i * self.mem_pool_host.page_size]`) | LOGICAL / PHYSICAL, `page_num = dcp_size × len(keys)` | MIXED | **yes** |
| R15 | hf3fs slot stride / success check | `storage_hf3fs.py:394` (`page_index * self.bytes_per_page`), `:423` (`read_result == self.bytes_per_page`) | PHYSICAL | **PHYSICAL — already correct** | no |
| R16 | Landing back on device (L2) | `pool_host/mla.py:243-256` | `dcp_kernel_indices` on both host and device indices at `:255-256` | LOGICAL→**PHYSICAL** | **no — this leg is already correct** |

### 3c. Sidecar v2 path (`batch_set_v2` / `batch_get_v2`)

| # | Step | file:line | Expression | Unit | Needs change? |
|---|---|---|---|---|---|
| V1 | `_batch_io_v2` length check | `hicache_storage.py:672-696` | `page_size = getattr(host_pool, "page_size", 1) or 1` `:677`; `expected = len(keys) * page_size` `:678`; compared to `host_indices.numel()` `:681`; per-page offset `host_indices[i*page_size].item()` `:693` | **PHYSICAL** denominator against whatever the caller supplies | see D1 |
| V2 | Callers | `hybrid_cache_controller.py:758` (get), `:773` (set); `cache_controller.py:1165`, `:1176` (draft only) | sidecar `PoolTransfer`s only | — | — |

---

### Disagreements between traces (flagged, not averaged)

- **D1 — `_batch_io_v2` is NOT on the KV path.** Traces 2 and 3 present `hicache_storage.py:672-696` as "the v2 zero-copy *KV* write/read path" that "would fail the guard by `dcp_size`". I traced it and that is **wrong for the KV pool**. `unified_radix_cache.py:1176-1191` builds a `kv_xfer` but passes only `aux_xfers` (component + sidecar transfers) as `extra_pools`; the KV bytes go through `cache_controller.write_storage(spec.host_value, ...)` → base `_page_backup` → `page_set_func`. `hybrid_cache_controller.py:762-778` likewise sends only `should_backup`-filtered sidecar transfers to `batch_set_v2`, and calls `super()._page_backup(operation)` for KV. So the "loud v2 length mismatch" is not a safety net on the KV path, and `_batch_io_v2` under DCP matters only for sidecars (whose `page_size` is 1 for Mamba). **Do not rely on it as a guard.**
- **D2 — `get_page_buffer_meta`'s assert does not fire.** See §2/B4. Traces 1 and 3 said it would; the arithmetic says it always passes for whole widened-page runs. Corrected above.
- **D3 — failure *shape* of the zero-copy backends differs per backend, and traces generalised.** Verified individually: mooncake `:1007`, simm `:290`, umbp `:971` **raise** `AssertionError`; nixl `:709-712` **returns all-False silently**; hf3fs `:634/:657/:893` **computes a `dcp_size`-too-large page count with no check at all** (downstream shape UNVERIFIED). "It fails loudly" is true for three of six and false for the rest.
- **D4 — `append_host_mem_release` residue balance: UNCLEAR, traces conflict.** `cache_controller.py:951-956` does `pages = host_indices.split(self.mem_pool_host.page_size)` — LOGICAL indices split by the PHYSICAL page. Trace 1 says this can trip the residue-balance assert at `base.py:351-355` unless `dcp_size² | widened_page`; Trace 6 says chunks stay balanced because `page_size % dcp_size == 0`. The arithmetic supports **Trace 1**: a contiguous run of length `L` covers all residues evenly only if `dcp_size | L`, and here `L = P_widened / dcp_size`, so the requirement is `dcp_size² | P_widened`. **But** neither trace established the real trigger — `free()` (`base.py:382-394`) accepts any index set, and `alloc()` serves `free_slots[:need_size]` (`base.py:368-369`) from a possibly-arbitrary permutation after `_merge_release_slots` (`base.py:321-331`), so imbalance can arise regardless of the chunk size. This is a **pre-existing L1/L2+DCP hazard**, not L3-specific, and it is UNVERIFIED whether it can actually occur. Do not silently "fix" it as part of L3.
- **D5 — decode-offload coverage.** Only Trace 3 flagged `decode_kvcache_offload_manager.py:70-76`. Verified: it constructs `MLATokenToKVPoolHost` with `self.page_size = server_args.page_size` (`:46`) and no dcp kwargs, and neither `_resolve_hicache_dcp_compatibility` (returns at `server_args.py:7518` without `enable_hierarchical_cache`) nor `pd_disaggregation_hook.py:37-52` mentions `disaggregation_decode_enable_offload_kvcache`. **CONFIRMED gap.**

---

## 4. THE `attn_cp` PRECEDENT

**It fixes keys only. It fixes no lengths. It covers at most one third of the DCP work, and only in one backend.**

What it is, verbatim, `hicache_storage.py:385-388`:
```
# Under NSA context parallel each CP rank holds a disjoint slice of every
# page, so give each rank its own file key to avoid a cross-rank write race.
if attn_cp_size > 1:
    self.config_suffix += f"_cp{attn_cp_rank}_{attn_cp_size}"
```
Plus the two dataclass fields (`hicache_storage.py:32-33`), their population (`cache_controller.py:630, 637-638` from `get_attn_cp_rank_and_size()` at `:322-330`), and the mkdir guard (`hicache_storage.py:390`).

**Why key-only sufficed for `attn_cp`, and why that does not transfer:**
- `attn_cp` shards **layers**. The host pool already sizes itself per-rank through the *device* pool: `base.py:205-225` `_effective_host_layer_num()` returns `ceil(device_pool.layer_num / device_pool.layer_shard_size)`, which feeds `get_size_per_token` directly (`pool_host/mla.py:125-133`). Every CP rank ends up with an *identical* bytes-per-page (the `ceil` over-allocates uniformly) and only the *content* differs — so a key suffix closes the whole hole.
- `attn_cp` never widens the page. `kv_cache_builder.py:216-223` widens **only** `if get_parallel().dcp_enabled`. Under CP, `controller.page_size == host_pool.page_size`; LOGICAL and PHYSICAL coincide and every byte expression in §3 is trivially correct.
- DCP shards **tokens inside a page** — the same dimension the storage key indexes — and it *does* widen. Hence the whole MIXED column in §3, which `attn_cp` has no analogue for.

**And the precedent is itself unfinished:**
- It exists in **`HiCacheFile` only**. `mooncake_store.py:560-561` reads `attn_cp_rank/attn_cp_size` into `self` and never uses them in a key (its MLA suffix is `mla_suffix = f"{self.pp_rank}"` or `""`, `:573-577`). `hicache_nixl.py:110-113` has no cp term. hf3fs, eic, aibrix, umbp, simm, shm contain no `attn_cp` token at all.
- `_ATTN_CP` and `_DCP` are different groups (`parallel_state.py:2336-2351` builds `_DCP` as contiguous slices *inside* each TP group; `attn_tp_size = tp_size // attn_cp_size // attn_dp_size` at `:2360` has no dcp term). With `--dcp-size 8` and no NSA CP, `get_attn_cp_rank_and_size()` returns `(0, 1)` and the suffix is **omitted entirely**. Reusing the field would also break a future NSA-CP + DCP combination.
- It does **not** touch `backup_skip`. Under MLA without DP attention, `storage_config.tp_rank` is the global TP rank (`cache_controller.py:602-604`), so only rank 0 writes — under key `..._cp0_N` that ranks 1..N-1 can never hit. The `_cp` suffix is load-bearing only under DP attention (where `tp_rank = attn_tp_rank = 0` on every CP rank, `cache_controller.py:598-600`). Treat it as a **design precedent, not a battle-tested one**.
- Test coverage is `test/registered/unit/mem_cache/test_hicache_file_lru_unit.py:275-298` (`TestCPSuffix`) — three cases, all asserting on `config_suffix` **strings**. No test asserts a byte length under CP, which independently confirms the change was key-scope-only.
- **Do not model the plumbing on `CacheInitParams.attn_cp_rank/attn_cp_size`** (`mem_cache/cache_init_params.py:42-43`): those are **dead fields**, never assigned at either construction site and never read. The live channel is the ProcessGroup (`cache_init_params.py:26` → `cache_controller.py:239/251/322-330`). Note that `hiradix_cache.py:112-113` uses the *other* convention (`get_parallel().attn_dcp_rank`) for the host pool. Two conventions coexist; P1 must pick one deliberately.

---

## 5. MAMBA

**Recommendation: MAMBA needs a `tp_rank` suffix. It does NOT need a `dcp_rank` suffix. `dcp_rank` is the wrong axis for this pool.**

Grounds, all verified:

1. **Mamba is not DCP-sharded.** `grep -n dcp` over `MambaPool` (`memory_pool.py:329+`), `mem_cache/allocator/mamba.py`, `mamba_radix_cache.py`, `hi_mamba_radix_cache.py`, `memory_pool_host.py` → zero hits (`grep -c dcp memory_pool_host.py` → `0`). Mamba slots are per-sequence; every DCP rank holds a full state for the sequence.
2. **Mamba IS rank-distinct, by TP.** `configs/mamba_utils.py:219` `conv_state_shape = divide(conv_dim, tp_world_size), conv_kernel-1` and `:224` `temporal_state_shape = (divide(num_heads, tp_world_size), head_dim, state_size)`. `tp_world_size` is `attn_tp_size = tp_size // attn_cp_size // attn_dp_size` (`parallel_state.py:2360`) — **not** divided by `dcp_size`.
3. **`dcp_rank` is a strict function of `tp_rank`, and not injective.** `parallel_state.py:2336-2351` carves DCP groups as contiguous `dcp_size` slices of each TP group ⇒ `dcp_rank = (index within TP group) mod dcp_size`. With `--tp 8 --dcp-size 2`, TP ranks 0 and 2 share `dcp_rank 0` but hold **different** Mamba head-shards.
4. **The host pool is DCP-blind by construction and that is correct.** `MambaPoolHost.__init__` (`memory_pool_host.py:66-115`) never calls `super().__init__()`, hard-sets `self.page_size = 1` (`:78`), takes no dcp kwargs; both construction sites (`hybrid_pool_assembler.py:655`, `:759`) pass none. It inherits `dcp_size=1/dcp_rank=0` from `base.py:82-83`, so `dcp_kernel_indices` is a permanent no-op. Its sizing is PHYSICAL and per-rank throughout: `get_size_per_token` (`memory_pool_host.py:295-300`) sums TP-divided conv+temporal element counts; `get_dummy_flat_data_page` = `page_size(=1) * size_per_token` (`:532-538`). **Nothing here should gain a dcp factor.**
5. **Every TP rank already writes its own Mamba shard.** `hybrid_cache_controller.py:799-806`: `should_backup()` returns True for `PoolName.MAMBA` even when `backup_skip` is set, with the comment *"Kimi-K3 Mamba/KDA state is TP-sharded even when the primary MLA KV pool is replicated."*; `backup_thread_func` is overridden (`:821-836`) to always run `_page_backup`. The read/prefetch side has no rank gate.
6. **…into a key that has no `tp_rank` in it.** `hicache_storage.py:380-382` appends `_{tp_rank}_{tp_size}` **only** `if not is_mla_model`, and `is_mla_model` is computed from the full-attention pool alone (`cache_controller.py:256-261` unwraps `HybridLinearKVPool` → `.full_kv_pool`; `:611` `isinstance(..., MLATokenToKVPool)`), which is True for a K3-style hybrid. Same in `hicache_nixl.py:110-113`. Mamba keys are `<hash>.mamba<config_suffix>` (`hicache_storage.py:652` + `_get_component_key` `:438-442`).

**Failure mode of each wrong choice:**

| Choice | Failure |
|---|---|
| **No suffix (status quo)** | All 8 TP ranks write and read one key holding one rank's TP head-shard. Last writer wins; every other rank loads a foreign shard as its recurrent state. **Silent wrong output, no assert.** This is a **live bug today at `dcp_size == 1`**, independent of the DCP work, on `file`, `nixl`, and `hf3fs` (which additionally folds all MLA ranks onto rank 0, `storage_hf3fs.py:221-224`, `:361-365`). |
| **`dcp_rank` suffix only** | Still broken whenever `dcp_size < tp_size` (TP ranks 0 and 2 collide at `tp8/dcp2`). Looks perfectly correct at `--tp 8 --dcp-size 8` — which is exactly this project's proven 1M-context configuration — so the bug would not reproduce on the config P1 tests on. |
| **`tp_rank` + `dcp_rank`** | Correct but redundant: `dcp_rank` adds no information over `tp_rank` for this pool. Cost is that changing `--dcp-size` between runs needlessly invalidates on-disk Mamba entries that are still valid. |

**Therefore the suffix must be pool-aware, not global.** KV wants `dcp_rank`+`dcp_size` (replicated across TP, sharded across DCP); MAMBA wants `attn_tp_rank`+`attn_tp_size` unconditionally, ignoring `is_mla_model`. `mooncake` already does the Mamba half correctly — `mooncake_store.py:758-760` uses `mha_suffix` (which carries `local_rank` = `tp_rank`, `:573-577`) for MAMBA even on an MLA model.

Two more Mamba facts P1 must not miss:
- `MambaPoolHost` has **no** `assert self.dcp_size == 1` counterpart to `pool_host/mla.py:525`. Relaxing the `server_args` gate removes the only thing stopping this path.
- There are **two** implementations of the Mamba L3 key scheme: `hi_mamba_radix_cache.py:1772/2083/2091-2092` and `unified_cache/components/mamba_component.py:732/745`. Both use `keys=[node.hash_value[-1]]` with `PoolHitPolicy.TRAILING_PAGES`. Fixing one leaves the other divergent. **And note `registry.py:114-115` routes any `is_hybrid_ssm` model to `_create_unified_radix_cache` *before* the hierarchical branch at `:117`** — so a Kimi-K3-shaped model uses `unified_radix_cache.py` + `hybrid_cache_controller.py`, not `HiRadixCache`/`HiCacheController`. A P1 aimed only at `HiRadixCache` would miss the live code path for this model.

---

## 6. OTHER REPLICATION ASSUMPTIONS

| # | Assumption | Location | Verdict |
|---|---|---|---|
| 1 | `backup_skip` — MLA ⇒ only TP0 writes | `cache_controller.py:471-475` (used `:1247`) | **MUST CHANGE** |
| 2 | Same rule, nixl | `storage/nixl/hicache_nixl.py:106` (used `:854`) | **MUST CHANGE** |
| 3 | Same rule + identity rewrite, hf3fs | `storage/hf3fs/storage_hf3fs.py:221-224` (`self.rank = 0`), `:361-365` (`rank_for_path`), used `:438` | **MUST CHANGE** |
| 4 | Same rule, hybrid/unified controller | `hybrid_cache/hybrid_cache_controller.py:776, 799-820` | **MUST CHANGE** (this is the live path for K3) |
| 5 | LRU evictor write veto | `storage/file/lru_file_evictor.py:83` `_is_storage_owner = (not is_mla_model) or (tp_rank == 0)`; refusal at `:178-183`; cap enforced per-process `:278-296` | **MUST CHANGE** — a second veto that would defeat a `backup_skip` fix; also, once all ranks own, on-disk usage becomes `dcp_size ×` the configured cap |
| 6 | Directory mkdir on rank 0 only | `hicache_storage.py:390` | **MUST CHANGE** (or make idempotent) |
| 7 | `is_mla_model` overloaded as "rank replicated" | `cache_controller.py:611-615, 639-640` (with the in-tree rename TODO) | **MUST CHANGE** — one boolean read by 8 backends; DCP inverts its premise |
| 8 | Key suffix drops rank for MLA — file | `hicache_storage.py:380-382` | **MUST CHANGE** |
| 9 | …nixl | `hicache_nixl.py:110-113` (`_get_suffixed_key` `:186`) | **MUST CHANGE** |
| 10 | …mooncake | `mooncake_store.py:573-577` (`mla_suffix = f"{pp_rank}"` or `""`), applied `:785`, `:803` | **MUST CHANGE** |
| 11 | …simm / eic | `hicache_simm.py:207-211`; `eic_storage.py:316-321` | **MUST CHANGE** |
| 12 | Write dedup by existence — file | `hicache_storage.py:510-513` | **MUST CHANGE** — with an unscoped key this makes ranks 1..7 skip *and report success*; it survives a `backup_skip`-only fix |
| 13 | Write dedup by existence — hf3fs metadata | `storage/hf3fs/mini_3fs_metadata_server.py:50-53`; consumed `storage_hf3fs.py:458-460`, reported success `:493` | **MUST CHANGE** — same shape, arbitrated centrally so it also fails cross-node |
| 14 | `clear()` unlinks every file in the shared dir, from every rank, no suffix filter, no barrier | `hicache_storage.py:712-726`; called from `hiradix_cache.py:818-826` on every rank | **MUST CHANGE** (or add a barrier) |
| 15 | `MetadataCache` positive TTL cache keyed by suffixed key | `hicache_storage.py:329-360`; seeded `:450-462` | **rank-local** — follows the key fix automatically; no separate change |
| 16 | MIN all-reduce on storage hit count | `cache_controller.py:1096-1102`; `hiradix_cache.py:1500-1501` | **rank-local, and semantically right for DCP** — but combined with #1 it yields MIN = 0 forever. See risk R-2 |
| 17 | Other storage-control collectives (queue-size MIN `hiradix_cache.py:1560`; terminate MAX `:1610`; completed-token MIN `:1704`; barrier `~:2003`) | as listed | **rank-local** — executed unconditionally on every rank, so un-skipping backups does not desynchronise them |
| 18 | `metrics_reporter` host capacity | `scheduler_components/metrics_reporter.py:1016-1018` (`host_total = host_pool.logical_size`) | **rank-local, correct** — positive control for what a correct LOGICAL consumer looks like |
| 19 | `_transfer_num_bytes` | `cache_controller.py:759-764` — `len(op.device_indices)` (LOGICAL) × `size_per_token` (full) | **MUST CHANGE** — over-reports by `dcp_size`; **this is live on the currently-supported L1/L2+DCP path**, and it will make any L3 bandwidth number read 8× high |
| 20 | `append_host_mem_release` chunking | `cache_controller.py:951-956` | **MUST CHANGE (units)** — LOGICAL indices split by PHYSICAL page; correctness impact UNCLEAR, see D4 |
| 21 | Storage-metric label set has no dcp term; `backuped_tokens_total` is LOGICAL while bandwidth is PHYSICAL GB | `hiradix_cache.py:336-345`; token count from `cache_controller.py:1235`; GB from `backend_factory.py:174-180` / `mooncake_store.py:693` | **MUST CHANGE (decision, not mechanics)** — today non-zero on rank 0 only; once all ranks report, summing over-counts by `dcp_size` |
| 22 | No storage revoke on host eviction | `hiradix_cache.py:1333-1366` `evict_host()` never calls storage; `delete_keys` exists only in the hf3fs metadata client with no cache-layer caller | **ABSENT — and this simplifies P1**: there is no cross-rank revoke protocol to make consistent |
| 23 | `HostPoolGroup` exposes no dcp surface | `memory_pool_host.py:1540-1560`, `:1605-1607`, `:1629-1636` | **MUST CHANGE** if any fix touches the unified/hybrid path |
| 24 | `DSAIndexerPoolHost` / V4 pools bypass `HostKVCache.__init__` | `memory_pool_host.py` (`get_size_per_token` at `:880`, `:1300`; `get_page_buffer_meta` at `:554`, `:728`, `:1103`, `:1488`, `:2160`); `LogicalHostPool` at `:640` with `self.page_size = page_size` `:656` | **rank-local today** — `server_args.py:7539-7544` bars `enable_hisparse` under DCP; but DSA *is* an MLA backend, so DSA+DCP+L1/L2 passes the gate with these pools untranslated |
| 25 | PD-decode offload host pool, un-widened page, no dcp kwargs | `disaggregation/decode_kvcache_offload_manager.py:46, 70-76` | **MUST CHANGE (or gate)** — covered by neither `server_args.py:7517` nor `pd_disaggregation_hook.py:37-52` |
| 26 | LMCache / FlexKV escape the DCP gate | `server_args.py:7502-7506, 7518-7519` + `registry.py:117, 131, 144` | **MUST CHANGE (guard placement)** — the `enable_lmcache` raise at `:7534` fires only in the configuration where LMCache is inert; FlexKV has no DCP guard at all |

### Risks worth carrying into P1

- **R-1 (silent corruption ordering).** For small LOGICAL indices, `set_from_flat_data_page` (`mla.py:556-581`) writes the **wrong physical rows and returns normally**; only when the index exceeds `self.size` does the slice go empty and the reshape raise. A smoke test with a generous host pool and low slot numbers passes while producing garbage KV. Same shape on the write side: tensor slicing clamps, so a truncated or zero-byte page gets written and `exists()` reports a hit next request. *(Derived from shapes and standard slice semantics — this VM has no torch, so it is not a measured result.)*
- **R-2 (the "runs but does nothing" outcome).** Lift B1 + add per-rank keys but leave `backup_skip` ⇒ ranks 1..7 have nothing in storage ⇒ the MIN all-reduce at `cache_controller.py:1099` pins `storage_hit_count` at 0 forever ⇒ a healthy-looking 0 % L3 hit rate with no error anywhere.
- **R-3 (the guard is a false floor).** `mla.py:525` covers `get_data_page` only. Relaxing `server_args.py:7520` before extending the guard converts a clean startup error into out-of-bounds pointer arithmetic (`get_page_buffer_meta`) and wrong-row writes (`set_from_flat_data_page`) — and the unguarded read path is reached *before* the guarded write path in a prefetch-first workload.
- **R-4 (lengths are already right, which is what makes it look like it works).** `bytes_per_page` (`backend_factory.py:174-180`), `get_dummy_flat_data_page` numel (`mla.py:543-554`), hf3fs slot stride (`storage_hf3fs.py:394`, `:423`) and `HiCacheFile`'s `expected` (`hicache_storage.py:472`) are all correctly PHYSICAL. Reads return the right byte count and backends report success. The damage is entirely in the **offset** and the **key**. Do not "fix" the byte math — double-dividing is a plausible P1 regression.
- **R-5 (measurement trap, matches this branch's history).** Any L3 bandwidth number taken after P1 reads `dcp_size ×` high unless `_transfer_num_bytes` (`cache_controller.py:759-764`) is fixed first.
- **R-6 (collectives become couplings).** Once shards are disjoint, one rank's slow or failing backend turns five all-reduces plus a barrier into a whole-TP-group stall, which the watchdog will report as a scheduler-loop stall rather than the real cause.
- **R-7 (cross-topology poisoning).** A rank's shard is the strided subset `s ≡ dcp_rank (mod dcp_size)` of a widened page. A store written at `dcp_size=8` is meaningless to a `dcp_size=4` server even at the same rank index. The suffix must encode **both** rank and size — as `_cp{rank}_{size}` already does — and reusing a store across a `--dcp-size` change is silent garbage, not a miss.

---

## 7. OPEN QUESTIONS P1 MUST RESOLVE BEFORE CODE

1. **Storage granularity.** One key per widened page holding all `dcp_size` ranks' bytes (some rank gathers — reintroduces a collective that does not exist anywhere in `cache_controller.py`, and that the demotion design explicitly avoided), or `dcp_size` keys per page with a `_dcp{rank}_{size}` suffix (no collective, `dcp_size ×` the object count, and `bytes_per_page` at `backend_factory.py:174-180` stays as-is)? The B1 error text at `server_args.py:7521-7527` says *"the storage keys must become dcp_rank-aware"*, implying per-rank keys, but nothing in the tree commits to it. **This decision determines every other answer below.**
2. **Where the identity comes from.** A `get_dcp_rank_and_size()` modelled on `cache_controller.py:322-330` (real ProcessGroup, requires plumbing a dcp CPU group through `CacheInitParams`/`kv_cache_builder.py` like `attn_cp_cache_group` at `cache_init_params.py:26` / `kv_cache_builder.py:228`), or straight from `get_parallel().attn_dcp_rank/.attn_dcp_size` the way `hiradix_cache.py:112-113` already does? Both conventions are live in this tree and the hicache path deliberately uses the former. Do **not** model it on the dead `CacheInitParams.attn_cp_rank/attn_cp_size` fields (`cache_init_params.py:42-43`).
3. **New fields vs. reuse of `attn_cp_rank/attn_cp_size`.** Reuse gets the `HiCacheFile` suffix for free (the slot is empty under plain DCP since `get_attn_cp_rank_and_size()` returns `(0,1)`), but mislabels the key and breaks any future NSA-CP + DCP combination. New `dcp_rank/dcp_size` fields on `HiCacheStorageConfig` (`hicache_storage.py:27-40`) populated in `_generate_storage_config` (`cache_controller.py:586-648`) is more invasive but reaches all backends uniformly.
4. **Scope by `dcp_rank` only, or by `(tp_rank, dcp_rank)`?** At `tp=8/dcp=8` they coincide. At `tp=8/dcp=4` there are two full replica sets, and correctness requires scoping by `dcp_rank` **only** so the two groups share storage. `HiCacheStorageConfig` today cannot express "scope by dcp but not by tp".
5. **Pool-aware suffixes.** §5 argues KV wants `dcp_rank` and MAMBA wants `attn_tp_rank`. Is a per-pool suffix policy acceptable in `_get_component_key` (`hicache_storage.py:438-442`, `hicache_nixl.py:186-192`), and does the same apply to DRAFT / SWA / indexer sidecars? Note `hicache_storage.py:576-580` (`_collect_existing_component_keys`) calls `_get_component_key` directly while `get()/set()` go via `_log_key` (`:652`) + `_get_suffixed_key` — the two must stay byte-identical after any change or `exists()` and `get()` will disagree. (They agree today; I did not exercise it.)
6. **What replaces `backup_skip`?** If every dcp rank writes its own shard, does the flag become `False` outright, or `one writer per (tp_rank, dcp_rank)`? Whatever is chosen must land simultaneously on `hicache_nixl.py:106`, `storage_hf3fs.py:221-224`, `hybrid_cache_controller.py:799`, and `lru_file_evictor.py:83`, plus the mkdir at `hicache_storage.py:390`, plus the per-process disk cap.
7. **Where does the LOGICAL→PHYSICAL translation live?** Inside the pool (`get_data_page`/`set_from_flat_data_page`/`get_page_buffer_meta` accept logical indices and translate internally, mirroring `mla.py:255-256`) — one file, but the methods then accept logical indices while `self.page_size` stays physical, which is subtle. Or at the callers (`cache_controller.py:1134`, `:990`, `:1204-1209`, plus the seven backend `len(host_indices) // mem_pool_host.page_size` sites listed in §3) via an explicit helper alongside `dcp_kernel_indices`. **This answer decides whether the diff is 1 site or ~10.**
8. **Does `get_page_buffer_meta` (`mla.py:583-620`) get translated or refused in P1?** Translating it means the returned pointer/size list describes only this rank's `1/dcp_size` of the page, which changes the value layout the nixl/simm/mooncake backends see — directly coupled to question 1.
9. **Is the hybrid/unified path in scope for P1?** `registry.py:114-115` routes a Mamba-hybrid MLA model (i.e. K3) to `UnifiedRadixCache` + `HybridCacheController`, which have their own `_storage_hit_query`, `_page_backup` and `backup_skip`. A `HiRadixCache`-only first cut would not cover the model this project actually serves. Related: `HostPoolGroup` (`memory_pool_host.py:1540`) needs a dcp surface before any `host_pool.dcp_size` access is written.
10. **Guard placement for LMCache / FlexKV / decode-offload.** Should the DCP rejections move out of `_resolve_hicache_dcp_compatibility` (which is unreachable for both LMCache and FlexKV in their live configuration), and is FlexKV under DCP rejected or out of scope? Similarly, should `pd_disaggregation_hook.py:37-52` gain a `disaggregation_decode_enable_offload_kvcache` clause?
11. **Prefetch sync-group membership.** `prefetch_sync_groups` (`cache_controller.py:331-353`) is built from `attn_cp_group`/`attn_tp_group`, falling back to `tp_group`. With `tp=8/dcp=8` the DCP group *is* the TP group so coverage is incidental — is that intended, or should a dcp group join the set explicitly for `tp≠dcp` layouts? A partial hit on some dcp ranks leaves a page with holes.
12. **Metric units decision.** Once all ranks report, is `backuped_tokens_total` LOGICAL (aggregate must be divided by `dcp_size`) or PHYSICAL (per-rank series sum correctly but disagree with `req.storage_hit_length`, which is MIN-synced and LOGICAL at `hiradix_cache.py:1676-1679` → `scheduler.py:3238-3240`)? The collector mixes both today and has no dcp label.
13. **Should the two pre-existing L1/L2+DCP defects ship first, separately?** `_transfer_num_bytes` (`cache_controller.py:759-764`) and `append_host_mem_release` (`:951-956`) are observable today on a configuration users may already be running, and are unrelated to storage.
14. **Was the B4 assert's placement deliberate?** `test_l3_data_page_is_guarded` (`test_hicache_dcp_host_pool.py:203-206`) tests only `get_data_page`, so the suite does not answer whether the omission on the other three is reasoning or oversight. **UNCLEAR — leave it that way and just close all four.**
15. **Device-side `loc` convention — UNCLEAR, one trace raised it, I could not settle it.** `MLATokenToKVPool.set_kv_buffer` (`memory_pool.py:4030-4044`) masks with `loc % attn_dcp_size == attn_dcp_rank` and then indexes `self.kv_buffer[layer_id][loc]` with the surviving **widened** values (no `// dcp_size`), and bounds-checks against the un-widened `self.size + self.page_size` at `:4027`. `set_mla_kv_buffer` (`memory_pool.py:4095-4108`) documents *"loc is widened under DCP; the kernel divides by the world size itself"* and bounds-checks against `(size + page_size) * attn_dcp_size`, with the kernel dividing at `mla_buffer.py:39-41`. Either the two entry points take different `loc` conventions, or one of them is wrong. **P1 should determine which one defines the physical row the host pool must mirror before writing any translation.** Do not assume.
16. **UNVERIFIED / out of area:** whether HiCache+DCP L1/L2 has ever been exercised on hardware (only the CPU unit test exists in-tree); the downstream failure shape of the hf3fs `page_num = dcp_size × len(keys)` mismatch (I did not trace the usrbio client); and whether the extra-pool v2 read path's sidecar host indices (`hybrid_cache_controller.py:741-760`) live in the widened space at all — `hybrid_pool_assembler.py:93-100` passes dcp kwargs only to the KV pool.
