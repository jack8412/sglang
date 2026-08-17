# SPDX-License-Identifier: Apache-2.0
"""Swap policy for KT hybrid MoE expert membership.

Which experts sit on GPU is a cache-residency problem: the router asks for
whatever the traffic needs, and a set chosen once at launch goes stale. This
module decides *what to swap*, from counters the forward pass already keeps;
performing the swap (weight movement, mask/index updates) lives in
``kt_ep_wrapper``.

The two inputs come from the margin-routing counters and are on the same
scale, both counted per routed slot on the ORIGINAL (pre-override) expert id:

* **demand** (``insist + override``) — the router chose a NON-resident expert.
  Whether we then paid the CPU (insist) or substituted a resident one
  (override) is a serving decision; either way the traffic wanted that expert,
  so both count toward promoting it.
* **resident hits** — the router chose an expert that was already GPU-resident
  and it was served there. Its inverse ranks demotion victims.

Counters are cumulative since launch, so the policy differences them per
evaluation and folds the deltas into an EMA: swaps should track *recent*
traffic, not the whole history, or the set freezes once early traffic
dominates the totals.
"""

import logging
from typing import Callable, List, NamedTuple, Optional

import torch

logger = logging.getLogger(__name__)


class ExpertSwap(NamedTuple):
    """One 1:1 in-layer exchange: ``promote`` takes ``demote``'s GPU row."""

    promote: int  # logical expert id, currently CPU-resident
    demote: int  # logical expert id, currently GPU-resident
    demand: float  # EMA demand of the promoted expert
    hits: float  # EMA resident hits of the demoted expert


class SwapTables(NamedTuple):
    """The four tables that must agree about where an expert lives.

    Kept together because a swap has to update them as one unit: any window in
    which they disagree is a window in which a token is computed with the
    wrong weights, on whichever side read the stale table.
    """

    gpu_experts_mask: torch.Tensor  # bool [num_experts], CPU
    gpu_experts_mask_cuda: torch.Tensor  # bool [num_experts], device
    logical_to_gpu_index: torch.Tensor  # int32 [num_experts], -1 = not resident
    logical_to_gpu_index_cuda: torch.Tensor  # int32 [num_experts], device
    gpu_index_to_logical: torch.Tensor  # int32 [num_gpu_experts]
    pinned_mask: Optional[torch.Tensor]  # uint8/bool, pointer held by kt C++
    # Split prefill's ONE slot space (residents [0, num_gpu), cold above it).
    # Optional because only split-prefill-capable methods build it -- but when
    # present it MUST flip with the rest: it went stale across swaps before,
    # which silently misrouted the swapped pair on the next split prefill.
    logical_to_slot: Optional[torch.Tensor] = None  # int32 [num_experts], CPU
    logical_to_slot_cuda: Optional[torch.Tensor] = None  # device


def apply_swaps_to_tables(tables: SwapTables, swaps: List[ExpertSwap]) -> List[int]:
    """Point every membership table at the new occupants. Returns the rows used.

    Every write is in place (``copy_``/index assignment, never rebinding),
    because decode CUDA graphs captured these tensors' addresses and the
    kt-kernel C++ side holds a raw pointer to ``pinned_mask``. Rebinding any of
    them would leave the graph and the C++ half reading freed memory -- the
    failure the existing dynamic-update path already guards against.

    Row assignment is preserved rather than re-densified: the promoted expert
    takes exactly the demoted expert's row, so no other expert's row moves and
    no other resident weight has to be touched.
    """
    rows: List[int] = []
    for s in swaps:
        row = int(tables.logical_to_gpu_index[s.demote].item())
        if row < 0:
            raise ValueError(
                f"demote target {s.demote} is not GPU-resident (row {row})"
            )
        if int(tables.logical_to_gpu_index[s.promote].item()) >= 0:
            raise ValueError(f"promote target {s.promote} is already resident")

        tables.gpu_experts_mask[s.promote] = True
        tables.gpu_experts_mask[s.demote] = False
        tables.logical_to_gpu_index[s.promote] = row
        tables.logical_to_gpu_index[s.demote] = -1
        tables.gpu_index_to_logical[row] = s.promote
        if tables.logical_to_slot is not None:
            # The pair EXCHANGE slots: the promoted expert takes the demoted
            # one's resident slot (== its row) and the demoted expert takes
            # the promoted one's cold slot, whose staging row the cold source
            # will fill with the demoted expert's bytes on the next prefill.
            # No other entry moves, mirroring the row assignment above.
            p_slot = int(tables.logical_to_slot[s.promote].item())
            d_slot = int(tables.logical_to_slot[s.demote].item())
            tables.logical_to_slot[s.promote] = d_slot
            tables.logical_to_slot[s.demote] = p_slot
        rows.append(row)

    tables.gpu_experts_mask_cuda.copy_(tables.gpu_experts_mask, non_blocking=True)
    tables.logical_to_gpu_index_cuda.copy_(
        tables.logical_to_gpu_index, non_blocking=True
    )
    if tables.logical_to_slot_cuda is not None:
        tables.logical_to_slot_cuda.copy_(tables.logical_to_slot, non_blocking=True)
    if tables.pinned_mask is not None:
        # kt-kernel reads this every forward with no lock; it must be written
        # last, after the GPU rows already hold the new weights, so the CPU
        # half never disclaims an expert whose weights are not yet on GPU.
        tables.pinned_mask.copy_(tables.gpu_experts_mask)
    return rows


def assert_tables_consistent(tables: SwapTables, num_gpu_experts: int) -> None:
    """Post-window invariant: every expert resident in exactly one place.

    Cheap enough to run after every swap window, and worth it: the failure
    mode of a desynced table is a silently wrong answer, not a crash.
    """
    mask = tables.gpu_experts_mask
    l2g = tables.logical_to_gpu_index
    g2l = tables.gpu_index_to_logical

    resident = torch.nonzero(mask, as_tuple=False).flatten().tolist()
    if len(resident) != num_gpu_experts:
        raise AssertionError(
            f"mask says {len(resident)} residents, expected {num_gpu_experts}"
        )
    rows = sorted(int(l2g[e].item()) for e in resident)
    if rows != list(range(num_gpu_experts)):
        raise AssertionError("resident rows are not a permutation of 0..N-1")
    for e in resident:
        row = int(l2g[e].item())
        if int(g2l[row].item()) != e:
            raise AssertionError(
                f"round trip broken: expert {e} -> row {row} -> "
                f"{int(g2l[row].item())}"
            )
    non_resident = (~mask).nonzero(as_tuple=False).flatten()
    if non_resident.numel() and int(l2g[non_resident].max().item()) >= 0:
        raise AssertionError("a non-resident expert still maps to a GPU row")


class ExpertSwapPolicy:
    """Per-layer EMA bookkeeping plus swap selection.

    Args:
        num_experts: Logical expert count for the layer.
        ema_alpha: Weight of the newest observation, in (0, 1]. 1.0 uses only
            the latest interval.
        hysteresis: A promotion must beat the demotion victim by this factor
            before the swap is taken. >1 creates a dead band so two experts of
            similar weight cannot trade places every interval (thrashing);
            each swap costs a weight transfer, so an unprofitable one is
            strictly worse than doing nothing.
        max_swaps: Per-evaluation budget, bounding transfer traffic and
            keeping any single decision's blast radius small.
        min_demand: Absolute floor on EMA demand; below it a promotion is
            noise rather than signal.
    """

    def __init__(
        self,
        num_experts: int,
        *,
        ema_alpha: float = 0.3,
        hysteresis: float = 2.0,
        max_swaps: int = 4,
        min_demand: float = 1.0,
    ):
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha}")
        if hysteresis < 1.0:
            raise ValueError(f"hysteresis must be >= 1.0, got {hysteresis}")
        self.num_experts = num_experts
        self.ema_alpha = ema_alpha
        self.hysteresis = hysteresis
        self.max_swaps = max_swaps
        self.min_demand = min_demand

        self.demand_ema = torch.zeros(num_experts, dtype=torch.float64)
        self.hits_ema = torch.zeros(num_experts, dtype=torch.float64)
        self._prev_demand = torch.zeros(num_experts, dtype=torch.float64)
        self._prev_hits = torch.zeros(num_experts, dtype=torch.float64)
        self._observed = False

    def observe(self, demand_cum: torch.Tensor, hits_cum: torch.Tensor) -> None:
        """Fold one interval's counter deltas into the EMAs.

        ``demand_cum`` / ``hits_cum`` are cumulative-since-launch counts on
        CPU. The first call only establishes the baseline: its "delta" would
        be the entire history, which is exactly the stale signal the EMA
        exists to avoid.
        """
        demand_cum = demand_cum.to(torch.float64).cpu()
        hits_cum = hits_cum.to(torch.float64).cpu()
        if demand_cum.shape != (self.num_experts,) or hits_cum.shape != (
            self.num_experts,
        ):
            raise ValueError(
                f"counter shape mismatch: expected ({self.num_experts},), got "
                f"{tuple(demand_cum.shape)} and {tuple(hits_cum.shape)}"
            )

        # Counters only ever grow; a decrease means they were reset (or the
        # layer was rebuilt), so treat the new value as the delta rather than
        # producing a negative one.
        d_delta = torch.clamp(demand_cum - self._prev_demand, min=0.0)
        h_delta = torch.clamp(hits_cum - self._prev_hits, min=0.0)
        self._prev_demand = demand_cum.clone()
        self._prev_hits = hits_cum.clone()

        if not self._observed:
            self._observed = True
            return

        a = self.ema_alpha
        self.demand_ema = (1.0 - a) * self.demand_ema + a * d_delta
        self.hits_ema = (1.0 - a) * self.hits_ema + a * h_delta

    def select(self, gpu_experts_mask: torch.Tensor) -> List[ExpertSwap]:
        """Pick up to ``max_swaps`` profitable 1:1 exchanges.

        Greedy and disjoint: the highest-demand non-resident expert is paired
        with the least-used resident one, then both are removed from
        consideration, so one evaluation never promotes or demotes the same
        expert twice.
        """
        if self.max_swaps <= 0 or not self._observed:
            return []
        mask = gpu_experts_mask.to(torch.bool).cpu()
        if mask.shape != (self.num_experts,):
            raise ValueError(
                f"mask shape {tuple(mask.shape)} != ({self.num_experts},)"
            )

        # A non-resident expert's demand is meaningful only if it cleared the
        # floor; a resident expert is a candidate victim regardless of hits
        # (zero hits is the strongest case for demoting it).
        cand_promote = [
            i
            for i in range(self.num_experts)
            if not mask[i] and self.demand_ema[i].item() >= self.min_demand
        ]
        cand_demote = [i for i in range(self.num_experts) if mask[i]]
        if not cand_promote or not cand_demote:
            return []

        cand_promote.sort(key=lambda i: -self.demand_ema[i].item())
        cand_demote.sort(key=lambda i: self.hits_ema[i].item())

        swaps: List[ExpertSwap] = []
        for promote, demote in zip(cand_promote, cand_demote):
            if len(swaps) >= self.max_swaps:
                break
            demand = self.demand_ema[promote].item()
            hits = self.hits_ema[demote].item()
            # Dead band: demand must beat the incumbent by the hysteresis
            # factor. With hits == 0 any demand above the floor wins, which is
            # the intended behaviour for an unused resident.
            if demand <= hits * self.hysteresis:
                break  # sorted, so no later pair can qualify either
            swaps.append(
                ExpertSwap(promote=promote, demote=demote, demand=demand, hits=hits)
            )
        return swaps

    def note_swapped(self, swaps: List[ExpertSwap]) -> None:
        """Reset EMAs for experts that just changed side.

        After a swap the promoted expert's demand history describes a state
        that no longer exists (it is resident now and will accumulate hits
        instead), and the demoted expert's hit history likewise. Leaving the
        stale values in place would let the same pair immediately qualify to
        swap back.
        """
        for s in swaps:
            self.demand_ema[s.promote] = 0.0
            self.hits_ema[s.promote] = 0.0
            self.demand_ema[s.demote] = 0.0
            self.hits_ema[s.demote] = 0.0

    def snapshot_counters(
        self,
        insist: torch.Tensor,
        override: torch.Tensor,
        resident_hits: torch.Tensor,
    ) -> None:
        """Fold one interval from the three device counters.

        ``insist`` and ``override`` both mean "the router asked for a
        non-resident expert" and are summed into demand; whether we paid the
        CPU or substituted is a serving decision, not a statement about what
        the traffic wanted.
        """
        self.observe(
            (insist.to(torch.int64) + override.to(torch.int64)).cpu(),
            resident_hits.to(torch.int64).cpu(),
        )

    def state_dict(self) -> dict:
        """Serialisable state, for persisting across restarts as a seed."""
        return {
            "num_experts": self.num_experts,
            "demand_ema": self.demand_ema.tolist(),
            "hits_ema": self.hits_ema.tolist(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("num_experts") != self.num_experts:
            raise ValueError(
                f"swap state is for {state.get('num_experts')} experts, "
                f"layer has {self.num_experts}"
            )
        self.demand_ema = torch.tensor(state["demand_ema"], dtype=torch.float64)
        self.hits_ema = torch.tensor(state["hits_ema"], dtype=torch.float64)
        self._observed = True


# ---------------------------------------------------------------------------
# Swap window driver
# ---------------------------------------------------------------------------

# Moves one expert's weights into a resident GPU row:
#   move_weights(layer, dst_row, logical_expert_id) -> None
# Isolated behind this alias deliberately. Everything else in a swap window is
# bookkeeping that can be asserted; this is the one step that physically
# rewrites weights, so it is the one step worth testing on its own (bitwise,
# against a known-good full-set copy for the same expert) before it is trusted.
# (layer, dst_row, promote_logical_id, demote_logical_id). The demoted id is
# passed rather than looked up from gpu_index_to_logical[dst_row] because a
# mover that also maintains a cold-side store needs to know which expert is
# leaving, and the tables still describe the PRE-swap placement at this point
# -- a reverse lookup would be correct today and silently wrong the moment
# this call moved after apply_swaps_to_tables.
MoveWeightsFn = Callable[[object, int, int, int], None]


class SwapInstallError(RuntimeError):
    """A CPU-side expert install failed.

    Distinguished from every other swap failure because it must NOT be
    absorbed by the per-layer skip. The install runs only on the rank that
    owns the CPU path, so skipping the layer there while the other ranks apply
    the same swap leaves them advertising different expert placements -- far
    worse than a failed cache-tuning operation. Raising is the lesser harm.
    """


class SwapWindowResult(NamedTuple):
    swaps_applied: int
    layers_touched: int
    skipped_layers: int


def run_swap_window(
    layers: List[dict],
    *,
    move_weights: MoveWeightsFn,
    install_cpu_expert: Optional[Callable[[dict, int, int], None]] = None,
    begin_layer: Optional[Callable[[dict, list, list], None]] = None,
    finish_layer: Optional[Callable[[], None]] = None,
    after_flip: Optional[Callable[[dict, list], None]] = None,
    on_layer_abort: Optional[Callable[[dict], None]] = None,
    quiesce: Optional[Callable[[], None]] = None,
) -> SwapWindowResult:
    """Apply pending swaps for every layer, at an already-paused point.

    ``layers`` is a list of dicts with keys ``policy``, ``tables``, ``layer``,
    ``num_gpu_experts`` and ``layer_idx``.

    The caller is responsible for having quiesced the pipeline: this function
    rewrites resident weights and flips the membership tables that both the
    GPU MoE and the kt-kernel CPU half read, so nothing may be in flight. The
    optional ``quiesce`` callback is invoked once before any mutation as a
    last-line barrier.

    Ordering within a layer is deliberate and load-bearing:
      1. write the promoted expert's weights into the demoted expert's row
      2. only then flip the tables
    Doing it the other way round would, for the window between the two,
    advertise an expert as GPU-resident while its row still held the previous
    occupant's weights -- every token routed there would silently compute with
    the wrong expert. Nothing crashes; the answers are just wrong.

    ``begin_layer(entry, swaps, rows)`` runs once per layer, before the first
    move, with the layer's whole plan in hand. Two properties follow, and both
    are load-bearing for a mover that reads demoted weights back off the GPU:
    every row it names still holds its DEMOTED occupant (no move has run yet),
    and any collective it issues is keyed to the plan rather than to per-swap
    conditions. A read-back that instead decided per swap -- skipping when an
    expert had no cold-store slot, or when its row had already been written --
    made ranks disagree on how many collectives to run, and the window
    deadlocked in NCCL rather than falling back.

    ``begin_layer`` may RETURN a filtered ``(swaps, rows)`` pair: the
    direct-DMA transport must register a demoted expert's pages on every
    rank BEFORE that expert becomes routable, so pairs any rank could not
    register are dropped everywhere (the filter must be consensus-derived --
    identical on all ranks -- for exactly the reason above). Returning None
    keeps the plan unchanged.

    ``after_flip(entry, swaps)`` runs once per layer AFTER the tables
    flipped, with the applied plan. This is where bookkeeping keyed to the
    NEW table belongs (the direct-DMA transport releases the promoted
    experts' page pins here -- releasing on the proposed plan instead would
    drop pins a skipped pair's still-cold expert needs). Failures are
    logged, never raised: the swap itself completed.

    ``on_layer_abort(entry)`` runs when a layer fails AFTER begin_layer --
    tables not flipped, layer skipped (or the window re-raising). Whatever
    begin_layer acquired for this layer's plan must be undone here: without
    it, a demotion acquire whose expert stays RESIDENT is never released by
    any future window (no promotion of a resident expert exists), so its
    pages are live forever and re-attempts stack unreleasable refcounts.
    Best-effort: failures logged, never raised.

    ``finish_layer`` exists for movers that BATCH their writes: they treat
    ``move_weights`` as a record step and apply a layer's copies in bulk, so
    without a hook they would flush lazily on the next layer's first move --
    that is, AFTER this layer's tables had already flipped, which is exactly
    the inversion described above. It is called after the layer's last move and
    before the flip, inside the same try, so a failure leaves the tables
    untouched and is attributed to the layer that actually failed.

    Eager demotion: the demoted expert stays computable on the CPU side, so at
    no point is an expert resident nowhere. This is why the tables can be
    asserted consistent immediately on return, instead of "eventually".
    """
    if quiesce is not None:
        quiesce()

    applied = 0
    touched = 0
    skipped = 0
    for entry in layers:
        policy: ExpertSwapPolicy = entry["policy"]
        tables: SwapTables = entry["tables"]
        swaps = policy.select(tables.gpu_experts_mask)
        if not swaps:
            continue
        try:
            rows = [
                int(tables.logical_to_gpu_index[s.demote].item()) for s in swaps
            ]
            if begin_layer is not None:
                filtered = begin_layer(entry, swaps, rows)
                if filtered is not None:
                    swaps, rows = filtered
                    if not swaps:
                        skipped += 1
                        continue
            for s, row in zip(swaps, rows):
                move_weights(entry["layer"], row, s.promote, s.demote)
                if install_cpu_expert is not None:
                    # BEFORE the tables flip: the demoted expert must not be
                    # routable on the CPU until its weights are actually
                    # there. Under cold-only residency it holds no buffer at
                    # all until this runs, so flipping first would point the
                    # forward at null.
                    try:
                        install_cpu_expert(entry, s.promote, s.demote)
                    except Exception as exc:
                        raise SwapInstallError(
                            f"CPU install failed for demote={s.demote} "
                            f"promote={s.promote} on layer "
                            f"{entry.get('layer_idx')}"
                        ) from exc
            if finish_layer is not None:
                finish_layer()
            apply_swaps_to_tables(tables, swaps)
            assert_tables_consistent(tables, entry["num_gpu_experts"])
            if after_flip is not None:
                try:
                    after_flip(entry, swaps)
                except Exception:
                    logger.exception(
                        "[kt-swap] after_flip hook failed on layer %s "
                        "(bookkeeping only; the swap itself completed)",
                        entry.get("layer_idx"),
                    )
        except SwapInstallError:
            # Never absorbed: see SwapInstallError. Skipping here would leave
            # this rank's placement disagreeing with every other rank's.
            if on_layer_abort is not None:
                try:
                    on_layer_abort(entry)
                except Exception:
                    logger.exception("[kt-swap] on_layer_abort failed")
            raise
        except Exception:
            # A layer that fails mid-window is left as it was found: weights
            # may have been written but the tables were not flipped, so the
            # rewritten row is still advertised as its previous occupant and
            # nothing routes to it. Skipping is therefore safe, while raising
            # would take down a live server for a cache-tuning operation.
            logger.exception(
                "[kt-swap] layer %s: swap window failed, layer left unchanged",
                entry.get("layer_idx"),
            )
            if on_layer_abort is not None:
                try:
                    on_layer_abort(entry)
                except Exception:
                    logger.exception("[kt-swap] on_layer_abort failed")
            skipped += 1
            continue
        policy.note_swapped(swaps)
        applied += len(swaps)
        touched += 1
        logger.info(
            "[kt-swap] layer=%s applied %d swap(s): %s",
            entry.get("layer_idx"),
            len(swaps),
            ", ".join(
                f"{s.promote}(d={s.demand:.1f})<-row{r}-{s.demote}(h={s.hits:.1f})"
                for s, r in zip(swaps, rows)
            ),
        )
    return SwapWindowResult(
        swaps_applied=applied, layers_touched=touched, skipped_layers=skipped
    )
