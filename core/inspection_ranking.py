"""
GasSetu — Inspection ranking of delivery agents (Layer 5, v2)
==============================================================
Answers the enforcement question: with a few inspections available, which delivery agents
(and so which distributors) should be visited first?

Why not judge households one by one: a single home's bookings are weak evidence, and a wrong
accusation hurts a family. Why not the allocated-minus-confirmed gap
(core/distributor_reconciliation.py): since the delivery OTP (DAC) became mandatory, a diverted
cylinder usually has a confirmation, so the gap stays at zero.

What works instead is counting. A corrupt agent leaves the same weak trace on many connections of
one beat. Each signal below is a count out of the agent's own deliveries or households, compared
with what is typical for all agents (the median rate), and expressed as a binomial z-score:
how many standard deviations above typical. Counts add up, so 40 weak traces on one beat stand
out even when none of them would justify reviewing a household.

Signals (any subset can be supplied):
    unusual_households   homes on the beat whose booking history is unusual for similar homes
                         (consumption risk score in the top 2%)
    far_confirmations    deliveries whose OTP was entered far from the registered address
    fast_confirmations   deliveries whose OTP was entered within minutes of dispatch

The output is a ranking with reasons for an inspector. It is not a finding of guilt.
"""

from dataclasses import dataclass

import numpy as np

SIGNAL_TEXT = {
    "unusual_households": "{x:.0f} of {n:.0f} households on the beat have an unusual booking history ({rate:.1%}; typical {typical:.1%})",
    "far_confirmations":  "{x:.0f} of {n:.0f} deliveries were confirmed far from the registered address ({rate:.1%}; typical {typical:.1%})",
    "fast_confirmations": "{x:.0f} of {n:.0f} deliveries were confirmed within minutes of dispatch ({rate:.1%}; typical {typical:.1%})",
}
REASON_Z = 3.0          # a signal is listed as a reason when it is this many standard deviations above typical


@dataclass
class AgentRanking:
    score:   np.ndarray      # one value per agent; higher = inspect earlier
    order:   np.ndarray      # agent indices, most suspicious first
    z:       dict            # signal name → z-score per agent
    typical: dict            # signal name → the median rate used as "typical"
    counts:  dict            # the input, kept for the reasons

    def reasons(self, agent: int) -> list:
        out = []
        for name, (x, n) in self.counts.items():
            if self.z[name][agent] >= REASON_Z:
                out.append(SIGNAL_TEXT.get(name, name + ": {x:.0f} of {n:.0f}").format(
                    x=x[agent], n=n[agent], rate=x[agent] / max(n[agent], 1), typical=self.typical[name]))
        return out


def rank_agents(counts: dict) -> AgentRanking:
    """
    counts: signal name → (x, n), two arrays with one value per agent: x events out of n chances.

    For each signal, "typical" is the median agent's rate (a minority of corrupt agents cannot move
    it). z = (x − n·typical) / sqrt(n·typical·(1 − typical)). An agent's score is the sum of its
    positive z-scores, so being better than typical on one signal does not hide another.
    """
    z, typical = {}, {}
    for name, (x, n) in counts.items():
        x, n = np.asarray(x, float), np.maximum(np.asarray(n, float), 1.0)
        p0 = float(np.clip(np.median(x / n), 1e-4, 0.5))
        typical[name] = p0
        z[name] = (x - n * p0) / np.sqrt(n * p0 * (1 - p0))
    score = sum(np.maximum(v, 0.0) for v in z.values())
    return AgentRanking(score, np.argsort(-score, kind="stable"), z, typical,
                        {k: (np.asarray(x, float), np.asarray(n, float)) for k, (x, n) in counts.items()})


def counts_from_population(pop, unusual_household: np.ndarray, with_delivery_signals: bool = True) -> dict:
    """Per-agent counts for a simulated population (core/synthetic_population.py), last 12 months of deliveries."""
    agent = pop.household["agent"]
    n_agents = len(pop.agent["is_corrupt"])
    per_agent = lambda values: np.bincount(agent, weights=np.asarray(values, float), minlength=n_agents)
    counts = {"unusual_households": (per_agent(unusual_household), per_agent(np.ones(len(agent))))}
    if with_delivery_signals:
        recent = ~np.isnan(pop.refill_day) & (pop.refill_day > pop.scenario.days - 365)
        deliveries = per_agent(recent.sum(1))
        counts["far_confirmations"] = (per_agent((pop.dac_far & recent).sum(1)), deliveries)
        counts["fast_confirmations"] = (per_agent((pop.dac_fast & recent).sum(1)), deliveries)
    return counts
