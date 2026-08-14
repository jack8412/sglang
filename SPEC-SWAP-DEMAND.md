# SPEC-SWAP-DEMAND.md — decouple the swap policy from margin routing

Status: **design, grounded in the source**. No code written. Follows
SPEC-MARGIN-ROUTING (margin ✓) and SPEC-COLD-ONLY-RESIDENCY (cold-only ✓);
this one removes a dependency between two of the shipped pillars rather than
adding a fourth.

## Claim

The expert-swap policy does not need margin routing, has never needed it, and
the coupling that exists today is **positional, not logical** — the counters it
reads happen to be computed inside the margin-override function, so they
inherited its gate.

## The proof is one line of set algebra

`_margin_override_topk_ids_impl` (`kt_ep_wrapper.py:3807`) computes

```python
cpu_routed     = ~gpu_experts_mask[safe_ids] & routed          # :3839
override_slots = cpu_routed & (lead < margin) & finite_alt     # :3858
insist_slots   = cpu_routed & ~override_slots                  # :3873
```

`insist_slots` and `override_slots` partition `cpu_routed`, so

```
insist + override == cpu_routed == routed & ~gpu_experts_mask[topk_ids]
```

`margin` appears only in how the partition falls, and cancels in the sum. The
same holds on the other side: `resident_hit == routed & gpu_experts_mask[...]`.

**All three counters are functions of `topk_ids` and `gpu_experts_mask` alone.**

And the swap driver consumes exactly that sum. `kt_expert_swap.py:13`:

> **demand** (`insist + override`) — the router chose a NON-resident expert.
> Whether we then paid the CPU (insist) or substituted a resident one
> (override) *is a serving decision*; either way the traffic wanted that expert.

`observe(demand_cum, hits_cum)` (`kt_expert_swap.py:174`) never sees the split.
The docstring already states the invariant; the code just doesn't exploit it.

## What the coupling costs today

| # | artifact | location |
|---|---|---|
| 1 | counters allocated only under `self._margin is not None` | `kt_ep_wrapper.py:4845` |
| 2 | swap driver bails on `method._margin_insist_count is None` | `kt_ep_wrapper.py:~6484` |
| 3 | config refuses `--kt-expert-swap-interval` without `--kt-routing-margin` | `server_args.py:7024` |
| 4 | three `scatter_add_` + four bitwise kernels per layer per step | `kt_ep_wrapper.py:5967`, priced in the comment at :4839 |

(3) is the user-visible one: **adaptive placement is unavailable to a
deployment that wants exact routing.** That is a legitimate and arguably
preferable production configuration — no substitution, no quality delta, but a
resident set that still follows the workload — and today it is rejected at
startup.

(4) is priced in the source at **~5.2% of decode GPU time**: 11,040 and 7,360
launches over 40 steps, i.e. 92 layers × 3 and 92 × 2. It is baked into the
captured decode graph, so it runs every step forever whether or not anything
reads it.

## The change

Compute the two signals the policy actually wants, directly from the router
output, before and independent of any override:

```python
# demand/hit accounting -- margin plays no part
routed       = topk_ids >= 0
is_resident  = gpu_experts_mask[topk_ids.clamp_min(0)]
demand_slots = routed & ~is_resident      # promotion signal
hit_slots    = routed &  is_resident      # demotion signal
```

Nothing new has to be plumbed: `_update_margin_counters`
(`kt_ep_wrapper.py:5922`) is *already* called with `topk_output.topk_ids`, the
ORIGINAL pre-override ids (`:5575`). The function currently derives its masks
from arguments the caller computed; it would derive them itself instead.

### Work items

1. **Rename and re-gate the buffers.** `_margin_insist_count` /
   `_margin_override_count` → a single `_expert_demand_count`;
   `_resident_hit_count` keeps its name. Allocate under
   `kt_config.expert_swap_interval > 0`, not `self._margin is not None`.
2. **Rewrite `_update_margin_counters`** as `_update_demand_counters(topk_ids)`
   computing the two masks above. Two `scatter_add_` instead of three.
3. **Re-gate the swap driver** (`maybe_run_expert_swap_window`) on the demand
   counter rather than the insist counter.
4. **Drop the validation** at `server_args.py:7024`. Swapping becomes legal with
   `--kt-routing-margin` unset.
5. **Keep insist/override as telemetry only.** They remain genuinely useful for
   *tuning margin* — `kt-margin` log lines report them per layer — and they are
   the one quantity here that is margin-dependent. Compute them only when margin
   is on **and** counters/telemetry are enabled, so the swap path never pays.

### What must not change

- **Counting stays on ORIGINAL ids.** Demand for expert *e* must be recorded
  against *e*, not against the resident expert that stood in for it. The current
  code is careful about this (`_orig_safe_ids` at `:5967`); the rewrite must be
  too, and it is easier to get right when the masks come from the same tensor.
- **In-place `scatter_add_` into persistent buffers.** This is what lets decode
  CUDA-graph replays keep counting. Any reallocation per step breaks capture.
- **The swap-window read-before-overwrite order** (SPEC-MARGIN-ROUTING): a
  promote and a demote share a row, and A must be read into staging before B is
  written. Untouched by this change, but it is the invariant most easily broken
  by editing nearby.

## Gates

1. **Algebraic equivalence, offline.** For random `topk_ids` / masks and a sweep
   of margins, assert `insist + override == demand` and
   `resident_hit == routed - demand` exactly. This is the whole claim; it is
   cheap and it is a unit test, not a server run.
2. **Behavioural equivalence.** Same workload, same seed, margin 0.5, swapping
   on: the sequence of promote/demote pairs must be **identical** before and
   after. The policy input is provably unchanged, so any difference is a bug in
   the rewrite.
3. **The new capability.** `--kt-expert-swap-interval 50` with no
   `--kt-routing-margin`: server starts (today it refuses), swaps fire, and
   greedy output is **bit-identical to exact routing** — no substitution is
   happening, so it must be.
4. **The cost claim.** Decode tok/s before vs after at fixed config. Expect a
   partial recovery of the 5.2%; report the measured figure rather than the
   predicted one.
5. **Swaps actually fire.** M4 ran with `--kt-expert-swap-interval 50` and
   logged **zero** `kt-swap` lines. That is unexplained and predates this
   change — it must be resolved first, or gate 2 has nothing to compare.

## Phase 2 — defer the histogram to the swap window

Items 1–5 remove a third of the per-step work. This removes most of the rest,
and it composes: do it after, not instead.

### CORRECTION: the premise below is largely obsolete — do not implement as written

This phase was drafted from the ~5.2% / ~11-kernels-per-layer figures in the
source comments. **Those describe the torch FALLBACK.** There is a fused CUDA
kernel — `sglang.kernels.ops.kimi_k3.kt_margin_counters` — that produces all
three counters in **one launch**, and it is active: zero
`fused demand counters unavailable` warnings in either the M4 or M5 server logs.

So the real per-step cost is ~92 launches (one per layer), not ~460. A ring
buffer would replace 92 fused-kernel launches with 92 buffer copies, which is
a wash. **Phase 2 is not worth doing at the fused kernel's cost.** It becomes
worth revisiting only if the fallback path is ever hit in production (watch for
that warning), or if the fused kernel is extended to write a ring instead.

The original reasoning is kept below because the *shape* of the argument still
applies to the fallback path, and because the equivalence property it
establishes (§ "It is exactly equivalent") is what any future batching of this
accounting would rest on.

### Why the per-step accounting is expensive at all (fallback path only)

A 896-bin `scatter_add_` with 128 updates is nothing computationally. The cost
is **per-kernel dispatch**, and the source counts it exactly (`:4839`): 11,040
and 7,360 launches over 40 steps = 92 layers × 3 and 92 × 2, so ~5 kernels per
layer per step, ~460 per step. At ~3 µs of dispatch each that is ~1.4 ms against
a ~21 ms decode step — which is the measured 5.2%.

Graph capture does **not** remove this. It removes the *CPU-side* launch cost;
the kernels still dispatch on device every replay. So the fix has to be fewer
kernels, not cheaper ones.

### The change

Stop reducing per step. Write the routing result into a ring buffer and build
the histogram **once per swap window**:

```python
# per layer, per step: one write, no atomics
ring.view(WINDOW, -1).index_copy_(0, slot_idx, topk_ids.reshape(1, -1))

# at the window, over the whole ring at once:
routed       = ring >= 0
is_resident  = gpu_experts_mask[ring.clamp_min(0)]
demand = bincount(ring[routed & ~is_resident], minlength=num_experts)
hits   = bincount(ring[routed &  is_resident], minlength=num_experts)
```

~460 kernels per step becomes **92** (one copy per layer), with two reductions
per window instead of 4,600.

### It is exactly equivalent, not approximately

`gpu_experts_mask` is mutated **only inside the swap window**, so it is constant
across every step the ring covers. Applying it once at the window therefore
yields bit-identical demand and hit counts to applying it per step. This is the
property that makes Phase 2 testable as a pure refactor (gate 2 extends to it
unchanged) rather than as a new heuristic.

### Buffer size — smaller than the premise

Buffering **router logits** would cost `[tokens, 896]` per layer per step:
8 × 896 × 92 × 50 × 4 B ≈ **132 MB**. But the demand signal needs only
`topk_ids`:

| | shape | 50-step window |
|---|---|---|
| decode, bs 8 | 8 × 16 × 92 | **2.4 MB** |
| decode, bs 8, window 200 | | 9.4 MB |

Logits are needed only for the insist/override *split*, which is telemetry
(item 5) and margin-dependent. The swap policy never needs them.

### Prefill keeps the eager path

Buffering a 24,576-token prefill forward would be 24576 × 16 × 92 × 4 B ≈
145 MB per forward, and prefill is not graph-captured in this config
(`cuda_graph_config.prefill.backend='disabled'`), so its dispatch cost is
amortised over thousands of tokens and is already negligible. Keep
`scatter_add_` there; ring only the decode path.

### The one hard part: the write index under graph capture

A captured graph writes to a **fixed** address, so the ring slot cannot be a
Python integer baked at capture time. It must be a device tensor the graph
reads — `slot_idx` above — advanced by the runner before each replay, the same
way `positions` and `seq_lens` are already refreshed. No in-graph increment, no
new capture semantics; but this is where a bug would land, and graph-capture
bugs present as silently stale counters rather than as errors.

Additional gate for Phase 2:

6. **Ring equals scatter.** Same seed, swapping on, run both accountings
   simultaneously for one window and assert the two histograms are equal
   element-wise. Then delete the scatter path. Do not infer equality from the
   swap decisions matching — a stale ring can still produce plausible swaps.

## Phase 3 — move the window off the prefill path

### Where swaps actually happen today

`maybe_run_expert_swap_window` is called from the first registered layer's
`apply()`, and its own docstring (`:6446`) states the constraint:

> It only runs eagerly (prefill), so Python is actually executing and a device
> sync is allowed — under a captured decode replay none of this code runs at all.

`_KT_SWAP_STATE["eager_forwards"]` therefore counts **prefill chunks**. Decode
steps replay a captured graph and cannot trigger a window at all.

So the arrangement is inverted on all three axes:

| | measured on | acted on | benefits |
|---|---|---|---|
| today | prefill forwards | prefill forwards | decode |

The quantity being optimised (which experts should be resident so decode stops
paying the CPU) is sampled and acted upon exclusively during the phase that
**does not consume it** — and, under `--kt-expert-split-prefill`, does not even
depend on it, because prefill computes all 896 experts wherever they live.

### Two consequences

**The stall lands on the throughput-critical path.** The window quiesces the
device (`torch.cuda.synchronize`) at a prefill chunk boundary. Prefill is the
phase this campaign spent its effort making fast; decode is the phase the swap
is for.

**Prefill routing is the most valuable signal available, and it is being spent
on the wrong action.** A prompt and its generation share a domain, so the
router's behaviour over the prompt is a *forecast* of what the decode about to
start will ask for — available before that decode issues a single token. Today
that forecast is consumed by windows that fire mid-prompt, re-cutting membership
several times while the prompt streams, and is stale by the time decode begins.

### The change: observe during prefill, act once at the boundary

- **Keep counting during prefill.** It is cheap, and under split prefill it is
  free of consequence — residency does not affect prefill's result, so the
  measurement perturbs nothing it measures.
- **Do not swap during prefill.** No mid-prompt re-cut; membership stays fixed
  for the whole prompt.
- **Swap once, at the prefill→decode transition**, from the union of two
  signals: the demand accumulated over *this* prompt (domain forecast) and the
  running EMA from prior decode (workload prior). The EMA already exists
  (`kt_expert_swap.py:168`); this only changes when it is consulted.
- **Then decode runs the whole generation on a set cut for its own domain**,
  with no further pauses unless demand genuinely shifts.

That is one pause per request instead of several per prompt, placed at the only
moment where information is maximal and nothing is mid-chunk.

Mechanically the trigger moves out of `apply()` and into the scheduler loop, at
the batch boundary where a request leaves prefill. The device-sync constraint is
unchanged and better satisfied there: the scheduler loop is Python between
forwards, so a quiesce is as legal as in `apply()` and strictly better placed.

**Rate-limiting is required, not optional.** With continuous batching and 8
concurrent requests, prefill→decode transitions arrive often; at 512 output
tokens and 25.1 tok/s per request a transition lands roughly every 2.5 s. An
unthrottled window there would cost far more than it returns. Bound it to one
window per T seconds of decode and let hysteresis suppress the rest — the sizing
of T against the swap budget is the calculation below.

### RESOLVED: this is why there were no swaps

Gate 5 asked why M4 logged **zero** `kt-swap` lines. It was neither of the
candidates originally listed (uniform workload, counter guard). It is
structural, and it is the reason this phase exists:

- `_split_prefill_apply` **returns at `kt_ep_wrapper.py:5350`**, before the
  margin block at 5568 that hosted the swap call.
- `maybe_run_expert_swap_window` runs **only eagerly** — its own docstring:
  *"under a captured decode replay none of this code runs at all."*
- Decode is fully captured in this config (bs=[1,2,4,8], `max_running_requests=8`).

Prefill returns before the call; decode never executes Python. **Under
`--kt-expert-split-prefill`, expert swapping was inert.** Every M4 and M5 run
advertised `--kt-expert-swap-interval 50 --kt-expert-swap-max 8` and could not
have swapped once. M5's sustained-decode probe agrees: 0 ITL outliers above 2x
the median over 512 tokens.

So Phase 3 is not an optimisation. It is the fix that makes the feature run at
all in the configuration this campaign ships.

Gate 5 is therefore replaced: after the move, assert `kt-swap` lines appear at
all, then check the demand distribution before concluding anything about
whether the *policy* is behaving — a uniform synthetic workload can still
legitimately swap nothing, and this campaign has already produced three
measurement artefacts of that kind.

## Sizing `--kt-expert-swap-max`

### Cost of one swap unit

`kt_expert_swap_max` is **per layer per window** (`server_args.py:3026`), so one
unit of budget is one 1:1 exchange in each of the 92 MoE layers.

```
per expert per layer per rank        2,193,408 B      (= 2.19 MB)
one 1:1 exchange  = promote H2D + demote D2H        4.39 MB
x 92 MoE layers                                    403.6 MB per rank
/ 27.9 GB/s measured H2D (8 ranks concurrent)     14.5 ms per unit
```

**14.5 ms per unit of `--kt-expert-swap-max`.** Halve it to ~7.2 ms if the
promote and demote are issued on separate streams — PCIe is full duplex, so the
H2D and D2H legs can overlap and today they need not be serialised. Treat 14.5
as the conservative figure and 7.2 as an available optimisation.

The 27.9 GB/s is the *measured* concurrent-rank figure from the cold pipeline
(21.70 ms for 577 MiB/layer/rank), not a spec sheet number. If
`probe_h2d_concurrent.py` shows rank contention is the cause and it can be
lifted toward 55.5 GB/s, every figure below halves.

### The current setting is 100x more conservative than it needs to be

At today's trigger (every 50 *eager forwards*, i.e. prefill chunks at ~2 s each
≈ one window per 100 s), the default budget costs:

```
8 units x 14.5 ms = 116 ms per window / 100 s  =  0.12% of serving time
```

The pause was sized as if it were expensive. It is not — it is two orders of
magnitude below where it would begin to matter.

### Recommended value

Under the Phase 3 trigger, the budget should be set against the decode phase it
serves, with the window rate-limited to one per T seconds:

| budget n | pause | 1% of serving needs T ≥ | 2% needs T ≥ |
|---|---|---|---|
| 8 (current) | 116 ms | 12 s | 6 s |
| 16 | 232 ms | 23 s | 12 s |
| **32** | **464 ms** | **46 s** | **23 s** |
| 64 | 928 ms | 93 s | 46 s |
| 276 (whole cold set) | 4.0 s | 400 s | 200 s |

**Recommendation: `--kt-expert-swap-max 32`, with the window rate-limited to
one per 30 s of decode.** That is 464 ms against 30 s = **1.5%**, a 4x larger
re-cut than today at a cost that is still small, and it is the point where a
single boundary swap can meaningfully re-shape residency for a new domain
rather than nibbling at it 8 experts at a time.

Two bounds keep it from going higher:

- **Diminishing candidates.** With 620 of 896 resident and top-16 routing, a
  domain concentrates demand on tens of experts, not hundreds. Past the top few
  dozen the demand differences fall inside `min_demand=1.0` and
  `hysteresis=2.0` and the policy correctly declines to act — so budget beyond
  the number of *qualifying* candidates is unusable, not merely expensive.
- **Copy-count overhead.** A unit is 92 separate 2.19 MB copies; at n=32 that
  is 2,944 copies per window. Each is large enough to be bandwidth-bound, but
  the launch overhead (~10 µs each) adds ~29 ms and grows linearly.

### Optional: scale the budget with the prefill just completed

A 1M-token prompt earns a large re-cut (464 ms against 152 s of prefill is
0.3%); a 24k prompt does not (464 ms against 2 s is 23%). Since the boundary
swap is triggered by a specific request finishing prefill, its size can follow
that request:

```
n = clamp(prefill_tokens // 8192, 4, 64)
```

24k → 4 units (58 ms), 262k → 32, 1M → 64 (928 ms, 0.6% of its prefill). This
spends bandwidth in proportion to both the evidence gathered and the decode
likely to follow. Recommended as a follow-on once the fixed value is measured,
not as part of the first patch.

### Gate

7. **Pause cost is linear and matches the model.** Time the window at n = 4, 8,
   16, 32 and check the slope against 14.5 ms/unit. A slope materially below it
   means promote and demote are already overlapping (good — re-derive from
   7.2); materially above means the per-copy overhead dominates and the copies
   need batching before the budget is raised.

## Interaction with per-request margin

This change is a **prerequisite** for making margin a per-request parameter, not
a complication.

The objection to per-request margin was that a batch mixing margins would blend
the insist/override counters and corrupt the swap signal. That objection is
void: the sum is margin-invariant, so mixed margins cannot affect demand or
hits at all. Only the telemetry split blends — and per item 5 above, telemetry
is where the split belongs.

Sequencing: land this first, then per-request margin becomes a change to one
broadcast comparison (`lead < margin` with `margin` a `[num_tokens, 1]` tensor)
plus its plumbing, with no swap-policy question attached.

## Why this and not more prefill work

Prefill is now copy-bound and understood (SPEC-SPLIT-PREFILL work,
`runs/status/phase-1m32k.status`). Decode is where the remaining serving cost
is — 47.6 tok/s single-stream, 11.0 tok/s at 1M context — and this touches
decode in two ways at once: it removes ~5% of its GPU time, and it unlocks the
one configuration (exact routing + adaptive placement) that improves decode
quality without paying for substitution.
