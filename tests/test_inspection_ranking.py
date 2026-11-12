"""Layer 5 v2: ranking delivery agents for inspection by counting weak traces across a beat."""

import numpy as np
import pytest

from core.anomaly_detection import ConsumptionRiskModel
from core.distributor_reconciliation import AlertLevel, DistributorReconciliation
from core.features import population_features
from core.inspection_ranking import counts_from_population, rank_agents
from core.synthetic_population import Scenario, simulate


def test_an_agent_with_many_weak_traces_is_ranked_first_with_readable_reasons():
    n = np.full(6, 400.0)
    counts = {"unusual_households": (np.array([8, 9, 7, 31, 8, 10.0]), n),
              "far_confirmations":  (np.array([70, 65, 72, 400, 68, 75.0]), n * 6)}
    ranking = rank_agents(counts)
    assert ranking.order[0] == 3
    reasons = ranking.reasons(3)
    assert len(reasons) == 2 and "31 of 400 households" in reasons[0] and "typical" in reasons[0]
    assert ranking.reasons(0) == []


def test_being_better_than_typical_on_one_signal_does_not_hide_another():
    n = np.full(5, 400.0)
    hides = rank_agents({"unusual_households": (np.array([8, 8, 8, 8, 40.0]), n),
                         "far_confirmations":  (np.array([70, 70, 70, 70, 0.0]), n * 6)})
    assert hides.order[0] == 4 and hides.score[4] > 0


def test_the_median_is_not_moved_by_a_corrupt_minority():
    x = np.array([8.0] * 9 + [200.0])
    assert rank_agents({"unusual_households": (x, np.full(10, 400.0))}).typical["unusual_households"] == pytest.approx(0.02)


def test_d23_ranking_finds_corrupt_agents_that_the_confirmation_gap_cannot_see():
    """Every diverted cylinder in the simulation has an OTP, so the old gap is zero for every agent."""
    pop = simulate(Scenario(n_distributors=10, agents_per_distributor=10, households_per_agent=400, seed=11))
    corrupt = pop.agent["is_corrupt"].astype(bool)
    f = population_features(pop)
    unusual = ConsumptionRiskModel.load().percentile(f) >= 98

    ranking = rank_agents(counts_from_population(pop, unusual))
    top = ranking.order[:corrupt.sum()]
    assert corrupt[top].mean() >= 0.75, "most of the top-ranked agents are the corrupt ones"
    assert ranking.reasons(int(ranking.order[0])), "the top agent comes with reasons"

    bookings_only = rank_agents(counts_from_population(pop, unusual, with_delivery_signals=False))
    assert corrupt[bookings_only.order[:corrupt.sum()]].mean() >= 0.4

    recon = DistributorReconciliation("D", "All OTPs entered")
    recon.allocate(50)
    for i in range(50):
        recon.assign_cylinder(f"C{i}", f"Q{i}", "A1", f"K{i}", "9")
        recon.mark_delivered(f"C{i}")
        recon.confirm_otp(f"C{i}", "000000")
    assert recon.reconcile().alert_level == AlertLevel.CLEAN
