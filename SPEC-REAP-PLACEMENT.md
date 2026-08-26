# REAP expert placement

Rank experts for GPU residency by the damage their absence would cause, not by
how often they fire.

    S_k = (1/|X_k|) * SUM over x in X_k of  w_k(x) * ||f_k(x)||_2

`X_k` is the set of tokens routed to expert `k`, `w_k` its mixture weight,
`f_k` its output vector. Dividing by `|X_k|` is the point: an expert firing
5,000 times contributing 1.0 each scores below one firing 10 times contributing
40.0 each. Counting activations ranks them the other way round.

## Scope

Decode only. Split prefill computes all 896 experts on GPU, so nothing about a
prefill forward says which experts decode will need resident.

`select()` ranks on `S_k` alone. The activation-count EMAs (`demand_ema`,
`hits_ema`) and their device counters are deleted, not kept alongside.

Accepted: pure REAP ignores frequency, so a frequently-routed low-norm expert
may stay on CPU and cost decode throughput. User decision.

## The one thing that makes this hard

Both halves shard an expert on the **contraction** axis - GPU across TP ranks
(`intermediate_size_per_partition`), kt across NUMA partitions. A shard emits a
FULL-LENGTH but PARTIAL-VALUE vector, so

    f = SUM_p f_p        ||f||^2 = SUM_p ||f_p||^2 + 2 * SUM_{p<q} <f_p, f_q>

Summing per-shard squared norms drops the cross terms and is NOT `||f||`.
Squares add only across **disjoint coordinates**. So the vectors must be summed
before the norm is taken, on both sides.

## GPU side

`gemm2_out` is the per-expert output, unweighted; finalize applies `w`. K3's
latent MoE asserts `hidden_act == "situ"`, and the SITU branch in `mxfp4.py`
honours `flashinfer_trtllm_deferred_finalize_context` unconditionally, so the
seam exists on this path.

**Staged in the decode graph, per layer.** Buffers allocated at init - never on
first use, which under graph capture would take them from the capture pool:

    vec   [max_T, L, top_k, H]  bf16   gathered gemm2 rows, zero where not served
    ids   [max_T, L, top_k]     int32  served expert id
    w     [max_T, L, top_k]     fp32
    valid [max_T, L, top_k]     bool

`max_T` leads so `vec[:T]` is contiguous. At `max_T=39, L=92, top_k=16,
H=3584`: 411 MB, plus ~50 MB reduce-scatter output and ~1 MB of side tables.

**Combined outside the graph**, at the scheduler's decode hook - the same place
the swap window runs, and for the same reason: it needs Python and a device
sync, which a graph replay has neither of.

    N = T * L * top_k                      # N % 8 == 0 since 92*16 = 1472
    reduce_scatter_tensor: [N, H] -> [N/P, H]   each rank gets TRUE summed rows
    norm over H on the rank's own rows     -> [N/P]
    all_gather_into_tensor                 -> [N] identical on every rank
    contrib = where(valid, w * norm, 0)
    num[layer, id]   += contrib            # two scatter_adds, all layers at once
    count[layer, id] += valid

Reduce-scatter splits dim 0, so each rank owns a subset of ROWS at full width -
the sum is already complete before the norm. Exact, and it moves `(P-1)/P` of
the bytes rather than the `2(P-1)/P` an all-reduce would.

Every rank ends with identical `num`/`count`, so the swap window needs no
collective of its own.

## kt side

kt splits on `qlen > 1`, so a decode batch of T>1 tokens takes kt's *prefill*
loop -- but neither accumulation loop needs touching. Per-expert outputs already
sit at `m_local_down_output_ptr_[expert_id] + m_local_pos_[i][j] * hidden`, one
row per (token, slot) per NUMA partition, and `do_numa_job` returning is the
first point at which all partitions' rows are complete. So the seam is in
`TP_MOE_Common::forward`, right after that call and before `merge_results` --
NOT inside `merge_results`, which has neither `expert_ids`, `weights` nor `k`.

**kt emits norms, not scores.** For each (token, slot) it owns, kt sums the P
partition rows into fp32, takes the norm, and writes it to a `[qlen, k]` host
buffer. It does not weight, does not accumulate per expert, and keeps no
running total.

That division of labour drops three problems at once:

- kt is handed the PADDED decode batch and has no way to know which rows are
  real. Python does -- it slices `[:real_tokens]` exactly as on the GPU side.
- a per-expert float total inside kt would inherit the same saturation hazard
  the Python side avoids by zeroing per window. A per-step buffer cannot.
- the ids and weights already exist in Python, so shipping them into kt to
  produce a score that Python then has to agree with is duplicated state.

The transfer is `qlen * k` floats per layer per step -- about 2.5 KB at
`qlen=39, k=16`, against the ~4.6 MB of expert rows it summarises.

Parallelise the merge over (token, slot) with the TP-level pool rather than
inside one NUMA node's job: the reads are cross-node whichever way it is
written, and one node's threads doing all of them serialises the work.

A slot kt did not compute reads back 0, which contributes nothing.

## Policy

Accumulators are **zeroed after each window snapshot**, so what the policy reads
is already that window's delta. No `_prev_*` baselines, and no unbounded fp32
accumulator - a monotonic one stalls once `total/increment > 2^24`, which sends
the busiest experts' delta to zero and inverts the ranking.

Recency without breaking the mean: decay the sum and the count together.

    sum_k   <- decay * sum_k   + window_sum_k
    count_k <- decay * count_k + window_count_k
    S_k      = sum_k / count_k          (0 where count_k == 0)

This is a pooled mean over a decayed window, so a window with one activation
carries the weight of one activation. Averaging per-window MEANS instead would
let a single sample move the score as far as five thousand.

`select()`: promote highest `S_k` among non-resident, demote lowest among
resident, greedy and disjoint, `S_promote > S_demote * hysteresis`.
`count_k == 0` scores 0, which is correct in both directions - an unmeasured
resident is the safest thing to demote, and an unmeasured non-resident has no
evidence to justify promoting.

## Invariants

- Every rank reaches every collective, with identical shapes. Nothing on the
  staging path may consult a rank-0-only object (`self.wrapper` is one).
- Only real tokens are measured. Captured graphs pad to the capture size and
  padding rows carry the previous replay's ids; counting them corrupts `|X_k|`.
- Buffers allocated at init, not on first use.
- Prefill is never measured, including chunks below the split-prefill threshold.
