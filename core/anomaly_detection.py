"""
GasSetu — Consumption Risk Model (Layer 3, v2)
===============================================
Scores one LPG connection from its booking history. Replaces the v1 Isolation Forest, which
flagged whatever was rare: it rated genuine low-usage homes as CRITICAL and missed organised
diversion (build_plan.md §13.2, D12/D16/D24).

How v2 works:
  1. PEER GROUPS. A home is compared only with similar homes: PMUY or not, rural or urban,
     household-size band. Twelve refills a year is ordinary for a joint family of nine and
     unusual for a two-person PMUY home.
  2. A SMALL, READABLE SCORE. Five inputs (below) are combined with weights fitted by logistic
     regression. Every input can be shown to a reviewer as a sentence, and no weight is negative:
     an input can only add concern, never hide it.
  3. PERCENTILE. The score is turned into "higher than x% of connections" using a reference
     population, and the tiers are cut on that percentile (HIGH = top 0.5%). Because the main input
     is measured against similar homes, large families and low-usage homes are not flagged for
     being what they are. experiments/consumption_eval.py reports the false-alarm rate per group,
     and also what happens with a separate cut in every group (precision halves, parity is no better).

The model never blocks a booking. Its strongest tier (HIGH) means "a person should look".

IMPORTANT: the weights and the reference population in consumption_model.json come from the
simulator (core/synthetic_population.py), because no labelled LPG data is public. On real data both
must be re-fitted: the peer statistics from the oil company's own population, the weights from
inspection outcomes. Regenerate the file with:  python experiments/consumption_eval.py

Author: Athrva — GasSetu Project
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from core.features import FEATURE_NAMES, LOCK_DAYS, rate_jump

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "consumption_model.json")

INPUT_NAMES = ["z_refills", "jump", "woke_up", "mobile_changes", "mobile_with_jump"]
PERCENTILE_GRID = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 92, 94, 96, 98, 98.5, 99, 99.5, 99.8, 99.9, 100]
TIERS = (("HIGH", 99.5), ("MEDIUM", 98.0), ("LOW", 90.0))        # percentile in the reference population
MIN_GROUP = 300                                                    # smaller groups borrow from a wider one

RECOMMENDATION = {
    "HIGH":   "Send to the human review queue. The refill proceeds.",
    "MEDIUM": "No action on this refill. Keep the connection on the watch list.",
    "LOW":    "No action. Slightly unusual.",
    "NORMAL": "No action. Ordinary pattern for similar households.",
}


# ─── PEER GROUPS ──────────────────────────────────────────────────────────────

_BAND_EDGES, _BAND_LABELS = (2, 5, 8), ("1–2 people", "3–5 people", "6–8 people", "9 or more people")


def peer_group(is_pmuy, is_rural, household_size) -> np.ndarray:
    """Group id 0–15: PMUY × rural × household-size band."""
    band = np.searchsorted(_BAND_EDGES, np.asarray(household_size), side="left")
    return np.asarray(is_pmuy).astype(int) * 8 + np.asarray(is_rural).astype(int) * 4 + band


def peer_label(group: int) -> str:
    return f"{'PMUY' if group & 8 else 'non-PMUY'}, {'rural' if group & 4 else 'urban'}, {_BAND_LABELS[group & 3]}"


def _group_masks(group: np.ndarray):
    """
    For each of the 16 groups, the households that stand in for it. A group with fewer than MIN_GROUP
    members borrows from a wider one, keeping the household-size band as long as possible because
    consumption depends on it most: same PMUY + band → same band → everyone.
    """
    for g in range(16):
        for mask in (group == g, (group & 11) == (g & 11), (group & 3) == (g & 3), np.ones(len(group), bool)):
            if mask.sum() >= MIN_GROUP:
                break
        yield g, mask


def fit_peer_stats(features: dict) -> dict:
    """Mean, spread and median of refills a year for every peer group."""
    group = peer_group(features["is_pmuy"], features["is_rural"], features["household_size"])
    refills = np.asarray(features["refills_12m"], float)
    return {g: {"n": int(mask.sum()), "refills_mean": float(refills[mask].mean()),
                "refills_sd": float(refills[mask].std()), "refills_median": float(np.median(refills[mask]))}
            for g, mask in _group_masks(group)}


def risk_inputs(features: dict, peer_stats: dict) -> np.ndarray:
    """
    The five model inputs, one row per connection.

    z_refills         refills in 12 months, in standard deviations above similar households
    jump              last 6 months against the connection's own earlier habit (square-root scale,
                      so a home going from 0 to 1 refill does not look like a tenfold jump)
    woke_up           silent for 8 months or more, then active again
    mobile_changes    registered mobile number changed (capped at 2)
    mobile_with_jump  ...and the refill rate jumped as well
    """
    f = {k: np.atleast_1d(np.asarray(features[k], float)) for k in FEATURE_NAMES}
    group = peer_group(f["is_pmuy"], f["is_rural"], f["household_size"])
    col = lambda key: np.array([peer_stats[int(g)][key] for g in group])

    z_refills = (f["refills_12m"] - col("refills_mean")) / np.maximum(col("refills_sd"), 1.0)
    jump = 2 * (np.sqrt(f["refills_6m"] + 0.375) - np.sqrt(f["refills_prev_18m"] / 3 + 0.375))
    woke_up = (f["dormant_days"] >= 240) * np.minimum(f["refills_6m"], 6) / 6.0
    mobile = np.minimum(f["mobile_changes_12m"], 2.0)
    return np.column_stack([z_refills, jump, woke_up, mobile, mobile * (jump > 1.0)])


# ─── RESULT STRUCTURE ─────────────────────────────────────────────────────────

@dataclass
class ConsumptionRisk:
    consumer_id:      str
    risk_level:       str            # NORMAL | LOW | MEDIUM | HIGH
    percentile:       float          # score is higher than this % of connections in the reference population
    peer_group:       str
    refills_12m:      int
    peer_median:      float          # refills a year a typical similar household takes
    reasons:          list           # [{"code": ..., "detail": ...}], readable by a reviewer
    recommendation:   str
    model_version:    str
    checked_at:       str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return dict(self.__dict__)


# ─── MODEL ────────────────────────────────────────────────────────────────────

class ConsumptionRiskModel:
    """
    Usage:
        model = ConsumptionRiskModel.load()
        risk = model.check("CON-001", refills_12m=13, refills_6m=7, refills_prev_18m=6, earliest_day_refills=8,
                           dormant_days=40, household_size=2, is_pmuy=1, is_rural=1, mobile_changes_12m=1)
    """

    def __init__(self, spec: dict):
        self.spec = spec
        self.version = spec["version"]
        self.weights = np.array([spec["weights"][k] for k in INPUT_NAMES])
        self.intercept = float(spec["intercept"])
        self.peer_stats = {int(g): s for g, s in spec["peer_stats"].items()}
        self.quantiles = np.array(spec["score_quantiles"])
        self.trained = True

    # ── fitting (used by experiments/consumption_eval.py) ────────────────────

    @classmethod
    def fit(cls, train_features: dict, train_labels, reference_features: dict, version: str, notes: Optional[dict] = None):
        """Weights from a labelled population; peer statistics and percentile tables from a reference population."""
        from sklearn.linear_model import LogisticRegression

        # Fit, drop any input whose weight comes out negative, and fit again: in a score a reviewer reads,
        # an input may only add concern. (Booking on the first allowed day was dropped for this reason: genuine
        # heavy users do it as often as diverters, so it carried no evidence beyond the refill count.)
        X, y = risk_inputs(train_features, fit_peer_stats(train_features)), np.asarray(train_labels)
        keep = np.ones(len(INPUT_NAMES), bool)
        while True:
            lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(X[:, keep], y)
            if (lr.coef_[0] >= 0).all():
                break
            keep[np.flatnonzero(keep)[lr.coef_[0] < 0]] = False
        weights = np.zeros(len(INPUT_NAMES))
        weights[keep] = lr.coef_[0]

        peer_stats = fit_peer_stats(reference_features)
        spec = {
            "version":     version,
            "created":     datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "lock_days":   LOCK_DAYS,
            "inputs":      INPUT_NAMES,
            "weights":     {k: round(float(w), 4) for k, w in zip(INPUT_NAMES, weights)},
            "intercept":   round(float(lr.intercept_[0]), 4),
            "tiers":       {name: cut for name, cut in TIERS},
            "percentile_grid": PERCENTILE_GRID,
            "peer_stats":  {str(g): {k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()}
                            for g, s in peer_stats.items()},
            "score_quantiles": [0.0] * len(PERCENTILE_GRID),
            "notes":       notes or {},
        }
        score = cls(spec).raw_score(reference_features)
        spec["score_quantiles"] = [round(float(q), 4) for q in np.percentile(score, PERCENTILE_GRID)]
        return cls(spec)

    def save(self, path: str = MODEL_PATH):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.spec, fh, indent=1)

    @classmethod
    def load(cls, path: str = MODEL_PATH):
        with open(path, encoding="utf-8") as fh:
            return cls(json.load(fh))

    # ── scoring ──────────────────────────────────────────────────────────────

    def raw_score(self, features: dict) -> np.ndarray:
        """Weighted sum of the five inputs (a logit). Higher = more like the diversion patterns."""
        return risk_inputs(features, self.peer_stats) @ self.weights + self.intercept

    def percentile(self, features: dict) -> np.ndarray:
        """0–100: share of reference connections with a lower score (step function on the stored grid)."""
        idx = np.searchsorted(self.quantiles, self.raw_score(features), side="right") - 1
        return np.array(PERCENTILE_GRID, float)[np.clip(idx, 0, len(PERCENTILE_GRID) - 1)]

    @staticmethod
    def tier(percentile: float) -> str:
        for name, cut in TIERS:
            if percentile >= cut:
                return name
        return "NORMAL"

    def check(self, consumer_id: str, refills_12m: float, refills_6m: float, refills_prev_18m: float,
              earliest_day_refills: float, dormant_days: float, household_size: float, is_pmuy: int,
              is_rural: int, mobile_changes_12m: float = 0) -> ConsumptionRisk:
        """Score one connection. The history numbers come from core/features.py, never from the consumer."""
        features = dict(refills_12m=refills_12m, refills_6m=refills_6m, refills_prev_18m=refills_prev_18m,
                        earliest_day_refills=earliest_day_refills, dormant_days=dormant_days,
                        household_size=household_size, is_pmuy=is_pmuy, is_rural=is_rural,
                        mobile_changes_12m=mobile_changes_12m)
        features = {k: np.array([v], float) for k, v in features.items()}
        group = int(peer_group(features["is_pmuy"], features["is_rural"], features["household_size"])[0])
        stats = self.peer_stats[group]
        percentile = float(self.percentile(features)[0])
        level = self.tier(percentile)
        z_refills, jump, woke_up, mobile, _ = risk_inputs(features, self.peer_stats)[0]
        ratio = float(rate_jump(refills_6m, refills_prev_18m))

        reasons = []
        if z_refills >= 2.0:
            reasons.append({"code": "EXCESS_VS_PEERS", "detail":
                            f"{refills_12m:.0f} refills in 12 months; similar households ({peer_label(group)}) "
                            f"take about {stats['refills_median']:.0f}."})
        if earliest_day_refills >= 6:
            reasons.append({"code": "EARLIEST_DAY_BOOKING", "detail":
                            f"{earliest_day_refills:.0f} of the last {refills_12m:.0f} refills were booked within "
                            f"2 days of the {LOCK_DAYS}-day lock ending."})
        if jump >= 1.5 and refills_6m >= 4:
            reasons.append({"code": "SUDDEN_INCREASE", "detail":
                            f"{refills_6m:.0f} refills in the last 6 months, {ratio:.1f} times this connection's earlier rate."})
        if woke_up > 0 and refills_6m >= 3:
            reasons.append({"code": "DORMANT_THEN_ACTIVE", "detail":
                            f"No refill for {dormant_days:.0f} days, then {refills_6m:.0f} refills in 6 months."})
        if mobile >= 1 and reasons:
            reasons.append({"code": "MOBILE_CHANGED", "detail":
                            f"Registered mobile number changed {mobile_changes_12m:.0f} time(s) in 12 months."})

        return ConsumptionRisk(
            consumer_id     = consumer_id,
            risk_level      = level,
            percentile      = percentile,
            peer_group      = peer_label(group),
            refills_12m     = int(refills_12m),
            peer_median     = stats["refills_median"],
            reasons         = reasons if level != "NORMAL" else [],
            recommendation  = RECOMMENDATION[level],
            model_version   = self.version,
        )
