"""
Layer 3 evaluation on the v2 synthetic population. Run from the repo root:

    python experiments/consumption_eval.py            # full run: 10 seeds, writes experiments/results/ and the model file
    python experiments/consumption_eval.py --quick    # 2 seeds, smaller populations, no model file

Protocol
  - Every seed draws TWO different worlds with Scenario.random(): methods that need labels are
    trained on one and tested on the other, so nothing is graded on the assumptions it was fitted to.
  - Unsupervised methods see only the test world (no labels).
  - Budget: reviewers can look at 0.5% of connections. precision@budget is the share of those that
    are real diversion; recall@budget is the share of all diversion found.
  - Every number is reported as mean and range over the seeds.

What the numbers mean: behaviour under the simulator's assumptions and under shifts of them
(core/synthetic_population.py lists which parameters are published and which are assumed).
They are not a measurement of real fraud.
"""
import argparse
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from core.anomaly_detection import ConsumptionRiskModel, fit_peer_stats, peer_group, risk_inputs
from core.features import FEATURE_NAMES, population_features
from core.synthetic_population import FRAUD_NAMES, Scenario, simulate

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
BUDGET = 0.005
SUBGROUPS = {
    "PMUY":           lambda f: f["is_pmuy"] == 1,
    "non-PMUY":       lambda f: f["is_pmuy"] == 0,
    "rural":          lambda f: f["is_rural"] == 1,
    "urban":          lambda f: f["is_rural"] == 0,
    "1-2 people":     lambda f: f["household_size"] <= 2,
    "7+ people":      lambda f: f["household_size"] >= 7,
}


def raw_matrix(f):
    return np.column_stack([np.asarray(f[k], float) for k in FEATURE_NAMES])


def top_k(score, budget, rng):
    """Indices of the top `budget` share. Ties are broken at random, so coarse scores get no free ordering."""
    k = max(1, int(round(budget * len(score))))
    return np.lexsort((rng.random(len(score)), -score))[:k]


def evaluate(name, score, y, f, fraud_type, rng, flagged=None):
    """Ranking metrics for a score. `flagged` overrides the top-k set (used for the peer-percentile tier)."""
    flagged = top_k(score, BUDGET, rng) if flagged is None else flagged
    hit = np.zeros(len(y), bool)
    hit[flagged] = True
    genuine = y == 0
    row = {"method": name, "pr_auc": average_precision_score(y, score), "roc_auc": roc_auc_score(y, score),
           "share_reviewed": hit.mean(), "precision_at_budget": y[hit].mean() if hit.any() else 0.0,
           "recall_at_budget": y[hit].sum() / y.sum()}
    for label, pick in SUBGROUPS.items():
        row[f"fpr_{label}"] = hit[genuine & pick(f)].mean()
    rates = [row[f"fpr_{label}"] for label in SUBGROUPS]
    row["fpr_gap_pp"] = (max(rates) - min(rates)) * 100          # 0 = every group carries the same false-alarm rate
    for code, label in FRAUD_NAMES.items():
        if code:
            row[f"recall_{label}"] = hit[fraud_type == code].mean()
    return row


def one_seed(seed, n_distributors, slow_sample):
    rng = np.random.default_rng(seed)
    train = simulate(Scenario.random(1000 + seed, n_distributors=n_distributors))
    test = simulate(Scenario.random(2000 + seed, n_distributors=n_distributors))
    f_tr, f_te = population_features(train), population_features(test)
    y_tr, y_te = train.household["is_fraud"], test.household["is_fraud"]
    kind = test.household["fraud_type"]
    rows = []
    add = lambda name, score, **kw: rows.append({**evaluate(name, np.asarray(score, float), y_te, f_te, kind, rng, **kw),
                                                 "seed": seed, "prevalence": y_te.mean()})

    X_tr_raw, X_te_raw = raw_matrix(f_tr), raw_matrix(f_te)
    X_tr_peer = risk_inputs(f_tr, fit_peer_stats(f_tr))
    X_te_peer = risk_inputs(f_te, fit_peer_stats(f_te))           # peer statistics of the test world itself: no labels used

    # ── rules ──
    add("Rule: refill count", f_te["refills_12m"])
    add("Rule: excess over peers (z)", X_te_peer[:, 0])

    # ── unsupervised (test world only) ──
    scaled = StandardScaler().fit_transform(X_te_raw)
    add("Isolation Forest, raw features",
        -IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1).fit(scaled).score_samples(scaled))
    add("Isolation Forest, peer-relative inputs",
        -IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1).fit(X_te_peer).score_samples(X_te_peer))
    sample = rng.choice(len(y_te), min(slow_sample, len(y_te)), replace=False)   # LOF and OC-SVM do not scale
    for name, model in [("Local Outlier Factor (sample)", LocalOutlierFactor(n_neighbors=35, novelty=True)),
                        ("One-Class SVM (sample)", OneClassSVM(nu=0.02, gamma="scale"))]:
        model.fit(scaled[sample])
        sub = {k: np.asarray(v)[sample] for k, v in f_te.items()}
        rows.append({**evaluate(name, -model.score_samples(scaled[sample]), y_te[sample], sub, kind[sample], rng),
                     "seed": seed, "prevalence": y_te[sample].mean()})

    # ── supervised: trained on the other world ──
    both_tr, both_te = np.hstack([X_tr_raw, X_tr_peer]), np.hstack([X_te_raw, X_te_peer])
    sc = StandardScaler().fit(both_tr)
    add("Logistic regression, all features",
        LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(both_tr), y_tr).decision_function(sc.transform(both_te)))
    add("Random forest",
        RandomForestClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=seed).fit(both_tr, y_tr).predict_proba(both_te)[:, 1])
    add("Gradient boosting (HistGB)",
        HistGradientBoostingClassifier(random_state=seed).fit(both_tr, y_tr).predict_proba(both_te)[:, 1])

    # ── GasSetu v2: five readable inputs, weights from the training world, percentile within peers ──
    model = ConsumptionRiskModel.fit(f_tr, y_tr, reference_features=f_te, version="eval")
    score = model.raw_score(f_te)
    add("GasSetu v2 score", score)
    # the alternative we tried and did not adopt: the same budget spent group by group
    group = peer_group(f_te["is_pmuy"], f_te["is_rural"], f_te["household_size"])
    per_group = np.concatenate([np.flatnonzero(group == g)[top_k(score[group == g], BUDGET, rng)] for g in np.unique(group)])
    add("GasSetu v2 score, separate cut in each peer group", score, flagged=per_group)
    return rows


def stealth_sweep(seeds, n_distributors):
    """Cheaters who spread their bookings out to hide. The models are trained on worlds with stealth 0–0.5."""
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        train = simulate(Scenario.random(1000 + seed, n_distributors=n_distributors))
        f_tr, y_tr = population_features(train), train.household["is_fraud"]
        for stealth in (0.0, 0.25, 0.5, 0.75, 1.0):
            test = simulate(Scenario.random(2000 + seed, n_distributors=n_distributors, stealth=stealth))
            f_te, y_te, kind = population_features(test), test.household["is_fraud"], test.household["fraud_type"]
            model = ConsumptionRiskModel.fit(f_tr, y_tr, reference_features=f_te, version="eval")
            for name, score in [("Rule: refill count", np.asarray(f_te["refills_12m"], float)),
                                ("GasSetu v2 score", model.raw_score(f_te))]:
                rows.append({**evaluate(name, score, y_te, f_te, kind, rng), "seed": seed, "stealth": stealth})
    return pd.DataFrame(rows)


def summarise(df, by):
    """Mean and range over seeds for the headline columns."""
    cols = ["pr_auc", "roc_auc", "precision_at_budget", "recall_at_budget", "fpr_gap_pp"]
    out = df.groupby(by, sort=False)[cols].agg(["mean", "min", "max"])
    return pd.DataFrame({c: [f"{m:.3f} ({lo:.3f}–{hi:.3f})" for m, lo, hi in
                             zip(out[c]["mean"], out[c]["min"], out[c]["max"])] for c in cols}, index=out.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    seeds = range(2) if args.quick else range(10)
    n_dist, slow_sample = (6, 6000) if args.quick else (20, 15000)
    os.makedirs(RESULTS, exist_ok=True)

    t0 = time.time()
    rows = []
    for seed in seeds:
        rows += one_seed(seed, n_dist, slow_sample)
        print(f"seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    table = summarise(df, "method")
    sweep = stealth_sweep(list(seeds)[:5], n_dist)
    sweep_table = summarise(sweep, ["method", "stealth"])

    fpr_cols = [c for c in df.columns if c.startswith("fpr_") and c != "fpr_gap_pp"]
    recall_cols = [c for c in df.columns if c.startswith("recall_") and c != "recall_at_budget"]
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(f"\nHousehold-level detection, {len(list(seeds))} seeds, fraud prevalence {df['prevalence'].mean() * 100:.2f}% "
          f"({df['prevalence'].min() * 100:.2f}–{df['prevalence'].max() * 100:.2f}%), review budget {BUDGET * 100:.1f}%\n")
    print(table.to_string())
    print("\nFalse alarms: share of GENUINE homes in each group that are sent for review (mean over seeds)\n")
    print((df.groupby("method", sort=False)[fpr_cols].mean() * 100).round(3).to_string())
    print("\nShare of each diversion type found within the budget (mean over seeds)\n")
    print(df.groupby("method", sort=False)[recall_cols].mean().round(3).to_string())
    print("\nCheaters who hide (stealth 0 = book the day the lock ends, 1 = spread out)\n")
    print(sweep_table.to_string())

    if args.quick:
        return
    df.to_csv(os.path.join(RESULTS, "household_models_by_seed.csv"), index=False)
    table.to_csv(os.path.join(RESULTS, "household_models_summary.csv"))
    (df.groupby("method", sort=False)[fpr_cols + recall_cols].mean()).to_csv(os.path.join(RESULTS, "household_fairness_and_recall.csv"))
    sweep.to_csv(os.path.join(RESULTS, "stealth_sweep_by_seed.csv"), index=False)
    sweep_table.to_csv(os.path.join(RESULTS, "stealth_sweep_summary.csv"))

    # ── the model file the API loads ──
    train = simulate(Scenario.random(101))
    reference = simulate(Scenario(seed=7))
    g2 = df[df["method"] == "GasSetu v2 score"]
    model = ConsumptionRiskModel.fit(
        population_features(train), train.household["is_fraud"], population_features(reference), version="2.0-sim",
        notes={
            "data": "SIMULATED (core/synthetic_population.py). Re-fit on real data before any real use.",
            "weights_fitted_on": "Scenario.random(101), 120,000 households",
            "reference_population": "Scenario(seed=7), 120,000 households",
            "evaluation": f"{len(g2)} seeds, train and test worlds drawn separately, review budget {BUDGET * 100:.1f}%",
            "precision_at_budget_mean": round(float(g2["precision_at_budget"].mean()), 3),
            "recall_at_budget_mean": round(float(g2["recall_at_budget"].mean()), 3),
            "pr_auc_mean": round(float(g2["pr_auc"].mean()), 3),
        })
    model.save()
    print(f"\nwrote {RESULTS} and core/consumption_model.json  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
