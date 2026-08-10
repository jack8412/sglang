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

from typing import List, NamedTuple, Optional

import torch


class ExpertSwap(NamedTuple):
    """One 1:1 in-layer exchange: ``promote`` takes ``demote``'s GPU row."""

    promote: int  # logical expert id, currently CPU-resident
    demote: int  # logical expert id, currently GPU-resident
    demand: float  # EMA demand of the promoted expert
    hits: float  # EMA resident hits of the demoted expert


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
