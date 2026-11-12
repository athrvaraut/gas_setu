"""
Layer 5 evaluation: which delivery agents to inspect first. Run from the repo root:

    python experiments/agent_ranking_eval.py            # 10 seeds, writes experiments/results/
    python experiments/agent_ranking_eval.py --quick    # 3 seeds

Each seed draws a fresh world (Scenario.random). Inspectors can visit 5% of agents. Reported:
precision (share of visited agents that are corrupt), recall (share of corrupt agents visited) and
ROC-AUC, as mean and range over seeds. The consumption model used for "unusual households" is the
shipped one (core/consumption_model.json), fitted on a different world.

The second table is a detectability curve: how small can a corrupt agent's operation be before the
ranking stops finding it? That depends mostly on counting statistics (beat size and base rates),
less on the simulator's behavioural assumptions.

Simulated data: these numbers describe the method under stated assumptions, not real fraud.
"""
import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from core.anomaly_detection import ConsumptionRiskModel
from core.features import population_features
from core.inspection_ranking import counts_from_population, rank_agents
from core.synthetic_population import Scenario, simulate

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
BUDGET = 0.05


def metrics(name, score, corrupt, rng):
    k = max(1, int(round(BUDGET * len(score))))
    top = np.lexsort((rng.random(len(score)), -score))[:k]          # ties broken at random
    auc = roc_auc_score(corrupt, score) if len(set(score)) > 1 else 0.5
    return {"method": name, "precision": corrupt[top].mean(), "recall": corrupt[top].sum() / corrupt.sum(), "roc_auc": auc}


def one_world(scenario, model, rng):
    pop = simulate(scenario)
    f = population_features(pop)
    corrupt = pop.agent["is_corrupt"]
    agent = pop.household["agent"]
    mean_by_agent = lambda v: np.bincount(agent, weights=np.asarray(v, float)) / np.bincount(agent)
    unusual = model.percentile(f) >= 98
    return [
        # the old reconciliation gap: every diverted cylinder has an OTP, so every agent's gap is zero
        metrics("Confirmation gap (old formula)", np.zeros(len(corrupt)), corrupt, rng),
        metrics("Mean refills per household", mean_by_agent(f["refills_12m"]), corrupt, rng),
        metrics("Mean household risk score", mean_by_agent(model.raw_score(f)), corrupt, rng),
        metrics("GasSetu ranking, bookings only",
                rank_agents(counts_from_population(pop, unusual, with_delivery_signals=False)).score, corrupt, rng),
        metrics("GasSetu ranking, bookings + delivery confirmations",
                rank_agents(counts_from_population(pop, unusual)).score, corrupt, rng),
    ]


def summarise(df, by):
    out = df.groupby(by, sort=False)[["precision", "recall", "roc_auc"]].agg(["mean", "min", "max"])
    return pd.DataFrame({c: [f"{m:.3f} ({lo:.3f}–{hi:.3f})" for m, lo, hi in
                             zip(out[c]["mean"], out[c]["min"], out[c]["max"])] for c in ["precision", "recall", "roc_auc"]},
                        index=out.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    seeds = range(3) if ap.parse_args().quick else range(10)
    model = ConsumptionRiskModel.load()
    rows, curve = [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        rows += [{**r, "seed": seed} for r in one_world(Scenario.random(3000 + seed), model, rng)]
        for share in (0.005, 0.01, 0.02, 0.04, 0.08):               # share of the beat a corrupt agent captures
            scn = Scenario.random(4000 + seed, capture_share_low=share, capture_share_high=share)
            curve += [{**r, "seed": seed, "captured_share_of_beat": share,
                       "captured_connections": round(share * scn.households_per_agent)} for r in one_world(scn, model, rng)]
    df, cv = pd.DataFrame(rows), pd.DataFrame(curve)
    table = summarise(df, "method")
    keep = cv["method"].isin(["GasSetu ranking, bookings only", "GasSetu ranking, bookings + delivery confirmations"])
    curve_table = summarise(cv[keep], ["method", "captured_connections"])

    pd.set_option("display.width", 200)
    print(f"\nAgent-level ranking, {len(list(seeds))} seeds, 300 agents of 400 households, inspection budget {BUDGET:.0%} of agents\n")
    print(table.to_string())
    print("\nDetectability: connections captured per corrupt agent (out of a beat of 400)\n")
    print(curve_table.to_string())
    if len(list(seeds)) >= 10:
        os.makedirs(RESULTS, exist_ok=True)
        df.to_csv(os.path.join(RESULTS, "agent_ranking_by_seed.csv"), index=False)
        table.to_csv(os.path.join(RESULTS, "agent_ranking_summary.csv"))
        cv.to_csv(os.path.join(RESULTS, "agent_detectability_by_seed.csv"), index=False)
        curve_table.to_csv(os.path.join(RESULTS, "agent_detectability_summary.csv"))
        print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
