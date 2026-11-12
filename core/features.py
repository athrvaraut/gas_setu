"""
GasSetu — Booking-history features
===================================
Turns a connection's refill history (the last 24 months) into the few numbers the consumption
risk model uses. The same code runs on a whole simulated population (experiments) and on one
consumer (the API), so training and serving cannot drift apart.

The history must come from the system of record (the oil company's booking database), not from
the consumer. A consumer cannot type their own risk score.
"""

from datetime import date
from typing import Iterable, Optional
import numpy as np

HISTORY_DAYS = 730
LOCK_DAYS = 25             # minimum days between refills (uniform rule since Sep 2026)
EARLY_SLACK_DAYS = 2       # "booked as soon as the lock ended" = within this many days of it

# Computed from the refill dates
HISTORY_FEATURES = ["refills_12m", "refills_6m", "refills_prev_18m", "earliest_day_refills", "dormant_days"]
# Taken from the connection's record
RECORD_FEATURES = ["household_size", "is_pmuy", "is_rural", "mobile_changes_12m"]
FEATURE_NAMES = HISTORY_FEATURES + RECORD_FEATURES


def history_features(refill_day: np.ndarray, today: float = HISTORY_DAYS, lock_days: int = LOCK_DAYS) -> dict:
    """
    refill_day: (connections, K) day number of each refill, ascending along each row, NaN-padded.
                Day `today - 730` is the start of the history window.

    refills_12m           refills in the last 12 months
    refills_6m            refills in the last 6 months
    refills_prev_18m      refills in the 18 months before that (the connection's own earlier habit)
    earliest_day_refills  refills in the last 12 months booked within 2 days of the lock ending
    dormant_days          longest silence that ended with a refill in the last 12 months
    """
    day = np.atleast_2d(np.asarray(refill_day, dtype=float))
    start = today - HISTORY_DAYS
    valid = ~np.isnan(day) & (day > start) & (day <= today)
    last12 = valid & (day > today - 365)
    last6 = valid & (day > today - 182.5)
    earlier18 = valid & ~last6

    previous = np.concatenate([np.full((len(day), 1), np.nan), day[:, :-1]], axis=1)
    gap = day - previous                                        # NaN for the first refill in the window
    silence = np.where(np.isnan(gap), day - start, gap)         # first refill: silent since the window opened

    return {
        "refills_12m":          last12.sum(1),
        "refills_6m":           last6.sum(1),
        "refills_prev_18m":     earlier18.sum(1),
        "earliest_day_refills": (last12 & (gap <= lock_days + EARLY_SLACK_DAYS)).sum(1),
        "dormant_days":         np.round(np.where(last12, silence, 0.0).max(1), 0),
    }


def rate_jump(refills_6m, refills_prev_18m):
    """Last 6 months compared with an average 6 months of the 18 before (smoothed so 0 → 1 is not infinite)."""
    return (np.asarray(refills_6m, float) + 0.5) / (np.asarray(refills_prev_18m, float) / 3 + 0.5)


def features_from_refill_dates(refill_dates: Iterable[date], today: Optional[date] = None,
                               lock_days: int = LOCK_DAYS) -> dict:
    """History features for one connection from its refill dates (any order; older than 24 months is ignored)."""
    today = today or date.today()
    days = sorted(HISTORY_DAYS - (today - d).days for d in refill_dates)
    days = [d for d in days if 0 < d <= HISTORY_DAYS]
    row = np.array([days or [np.nan]], dtype=float)
    return {k: float(v[0]) for k, v in history_features(row, HISTORY_DAYS, lock_days).items()}


def population_features(pop) -> dict:
    """All model features for a simulated population (core/synthetic_population.py)."""
    feats = history_features(pop.refill_day, pop.scenario.days, pop.scenario.lock_days)
    feats.update({k: pop.household[k] for k in RECORD_FEATURES})
    return feats
