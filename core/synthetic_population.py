"""
GasSetu — Synthetic LPG population (v2)
========================================
A simulator of distributors, delivery agents, households and their refills over 24 months.
It replaces the v1 generator (experiments/legacy_v1.py), whose classes one threshold could separate
and whose "normal" home booked 12 cylinders a year.

What is taken from published figures (build_plan.md §13.3):
  - PMUY (Ujjwala) share of connections: 10.33 of 32.94 crore                      → 0.31
  - PMUY homes are mostly rural (over 80%)
  - Refills per year: PMUY 4.71 (FY 2025-26); other homes 6–7
  - Average household size 4.4 (NFHS-5)
  - Booking rule: 25 days between refills for everyone (since Sep 2026), so at most ~14 a year
  - Delivery needs the consumer's DAC (OTP), so diverted cylinders also show a confirmation

What is ASSUMED (no public data exists; change it in `Scenario` and re-run the experiments):
  - the spread of consumption between similar homes, and seasonality
  - how often genuine homes go quiet and come back, grow, or change their mobile number
  - how each kind of diversion looks in the booking record, and how common it is

So results on this data say how a method behaves under these assumptions and under shifts of
them. They are not a measurement of real fraud.

Three kinds of diversion are modelled (label `fraud_type`):
  1 SELLER      a household books far more than it uses and sells the surplus
  2 CAPTURED    a corrupt delivery agent books on a low-usage connection and keeps the cylinders
                (the OTP is obtained by trick or collusion); clustered under corrupt agents
  3 COMMERCIAL  an eatery runs on several domestic connections held in individuals' names
"""

from dataclasses import dataclass, replace
import numpy as np

SELLER, CAPTURED, COMMERCIAL = 1, 2, 3
FRAUD_NAMES = {0: "genuine", SELLER: "seller", CAPTURED: "captured", COMMERCIAL: "commercial"}

# Household size 1..12, mean 4.45 (NFHS-5 reports 4.4)
_SIZE_P = np.array([0.05, 0.11, 0.16, 0.24, 0.19, 0.11, 0.065, 0.035, 0.018, 0.012, 0.006, 0.004])
_TODAY = np.datetime64("2026-11-12")


@dataclass(frozen=True)
class Scenario:
    # ── size ──
    n_distributors:         int   = 30
    agents_per_distributor: int   = 10
    households_per_agent:   int   = 400
    days:                   int   = 730       # 24 months of history; "today" is day `days`
    lock_days:              int   = 25        # minimum days between two refills
    # ── genuine behaviour ──
    need_pmuy:              float = 4.7       # mean refills a year, all PMUY homes (published)
    need_other_rural:       float = 6.0       # published range for non-PMUY is 6–7; the split is assumed
    need_other_urban:       float = 7.0
    need_sigma:             float = 0.40      # spread between similar homes (assumed)
    pmuy_inactive_share:    float = 0.07      # PMUY homes that take almost no refills (assumed)
    winter_boost:           float = 0.12      # extra use November–February (assumed)
    migrant_share:          float = 0.03      # homes locked up for 8–16 months, then back (assumed)
    growth_share:           float = 0.03      # homes whose need grows 1.4–1.8× (assumed)
    mobile_change_rate:     float = 0.06      # genuine mobile-number changes per home per year (assumed)
    # ── diversion (all assumed) ──
    corrupt_agent_share:    float = 0.08
    capture_share_low:      float = 0.04      # share of a corrupt agent's beat that is captured
    capture_share_high:     float = 0.12
    seller_share:           float = 0.006
    careful_seller_share:   float = 0.40      # sellers who stay at 9–11 refills a year
    commercial_share:       float = 0.003
    stealth:                float = 0.0       # 0 = book the day the lock ends; 1 = spread bookings out
    seed:                   int   = 0

    @classmethod
    def random(cls, seed: int, **fixed) -> "Scenario":
        """A scenario whose assumed parameters are drawn from ranges: used to train on one world and test on another."""
        r = np.random.default_rng(seed)
        low = r.uniform(0.02, 0.06)
        drawn = dict(
            need_pmuy=r.uniform(4.4, 5.0), need_other_rural=r.uniform(5.5, 6.5), need_other_urban=r.uniform(6.5, 7.5),
            need_sigma=r.uniform(0.30, 0.50), pmuy_inactive_share=r.uniform(0.04, 0.10),
            winter_boost=r.uniform(0.05, 0.20), migrant_share=r.uniform(0.02, 0.05),
            growth_share=r.uniform(0.02, 0.05), mobile_change_rate=r.uniform(0.04, 0.09),
            corrupt_agent_share=r.uniform(0.05, 0.12), capture_share_low=low,
            capture_share_high=low + r.uniform(0.04, 0.10), seller_share=r.uniform(0.003, 0.010),
            careful_seller_share=r.uniform(0.2, 0.6), commercial_share=r.uniform(0.001, 0.005),
            stealth=r.uniform(0.0, 0.5), seed=seed,
        )
        return cls(**{**drawn, **fixed})

    def with_(self, **changes) -> "Scenario":
        return replace(self, **changes)


@dataclass
class Population:
    scenario:    Scenario
    household:   dict          # name → array with one value per household
    agent:       dict          # name → array with one value per delivery agent
    refill_day:  np.ndarray    # (households, K) day of each recorded refill, ascending, NaN-padded
    diverted:    np.ndarray    # (households, K) this refill did not go to the registered home
    dac_fast:    np.ndarray    # (households, K) OTP entered within minutes of dispatch
    dac_far:     np.ndarray    # (households, K) OTP entered far from the registered address

    @property
    def n(self) -> int:
        return len(self.household["agent"])


def _is_winter(day: np.ndarray, days: int) -> np.ndarray:
    """Day 0 is `days` days before today. November–February count as winter."""
    month = ((_TODAY - days + np.floor(day).astype("timedelta64[D]")).astype("datetime64[M]").astype(int) % 12) + 1
    return (month >= 11) | (month <= 2)


def simulate(s: Scenario = Scenario()) -> Population:
    rng = np.random.default_rng(s.seed)
    n_agents = s.n_distributors * s.agents_per_distributor
    n = n_agents * s.households_per_agent
    agent = np.arange(n) // s.households_per_agent
    distributor = agent // s.agents_per_distributor

    # ── who lives there ───────────────────────────────────────────────────────
    # Distributors are urban, mixed or rural, so PMUY and rural homes cluster as they do in reality.
    rural_p = rng.choice([0.05, 0.50, 0.95], size=s.n_distributors)[distributor]
    is_rural = rng.random(n) < rural_p
    is_pmuy = rng.random(n) < np.where(is_rural, 0.50, 0.12)        # → 31% PMUY, 81% of them rural
    size = rng.choice(np.arange(1, 13), size=n, p=_SIZE_P)

    size_factor = np.sqrt(size / 4.45)
    size_factor /= (np.sqrt(np.arange(1, 13) / 4.45) * _SIZE_P).sum()
    spread = rng.lognormal(-s.need_sigma ** 2 / 2, s.need_sigma, n)
    active_pmuy = (s.need_pmuy - s.pmuy_inactive_share * 0.6) / (1 - s.pmuy_inactive_share)
    need = np.where(is_pmuy, active_pmuy, np.where(is_rural, s.need_other_rural, s.need_other_urban)) * size_factor * spread
    inactive = is_pmuy & (rng.random(n) < s.pmuy_inactive_share)
    need = np.where(inactive, rng.uniform(0.0, 1.2, n), need)

    migrant = rng.random(n) < s.migrant_share
    away_from = np.where(migrant, rng.uniform(-100, s.days - 300, n), np.inf)
    away_until = away_from + rng.uniform(240, 480, n)
    grows = rng.random(n) < s.growth_share
    grow_day = np.where(grows, rng.uniform(0, s.days, n), np.inf)
    grow_by = rng.uniform(1.4, 1.8, n)

    # ── diversion ─────────────────────────────────────────────────────────────
    fraud = np.zeros(n, dtype=int)
    regime_start = np.full(n, np.inf)             # day the connection switches to near-maximum booking
    extra_mean = np.zeros(n)                      # mean days waited after the lock ends
    careful_rate = np.zeros(n)                    # > 0: a seller aiming at this many refills a year

    corrupt = np.zeros(n_agents, dtype=bool)
    corrupt[rng.choice(n_agents, max(1, round(s.corrupt_agent_share * n_agents)), replace=False)] = True
    for a in np.flatnonzero(corrupt):
        beat = np.flatnonzero(agent == a)
        k = max(1, round(rng.uniform(s.capture_share_low, s.capture_share_high) * len(beat)))
        weight = 1.0 / (need[beat] + 1.0) ** 2    # low-usage connections are the easy targets
        taken = rng.choice(beat, size=k, replace=False, p=weight / weight.sum())
        fraud[taken] = CAPTURED
        regime_start[taken] = rng.uniform(-200, s.days - 120, k)
        extra_mean[taken] = 3 + 25 * s.stealth

    for _ in range(round(s.commercial_share * n / 4.5)):
        a = rng.integers(n_agents)
        first = a * s.households_per_agent + rng.integers(0, s.households_per_agent - 6)
        members = np.arange(first, first + rng.integers(3, 7))
        members = members[fraud[members] == 0]
        fraud[members] = COMMERCIAL
        regime_start[members] = -np.inf
        extra_mean[members] = 3 + 20 * s.stealth
        size[members] = rng.integers(1, 4, len(members))      # recorded as small households

    sellers = (fraud == 0) & (rng.random(n) < s.seller_share)
    fraud[sellers] = SELLER
    regime_start[sellers] = np.where(rng.random(sellers.sum()) < 0.5, -np.inf, rng.uniform(0, s.days - 120, sellers.sum()))
    extra_mean[sellers] = 6 + 30 * s.stealth
    careful = sellers & (rng.random(n) < s.careful_seller_share)
    careful_rate[careful] = rng.uniform(9, 11, careful.sum())

    # ── refills: one renewal process per connection ───────────────────────────
    K = s.days // s.lock_days + 6
    day = np.full((n, K), np.nan)
    t = -rng.random(n) * np.minimum(365 / np.maximum(need, 0.05), s.days)
    for k in range(K):
        in_regime = t >= regime_start
        need_now = need * np.where(t >= grow_day, grow_by, 1.0) * (1 + s.winter_boost * _is_winter(np.clip(t, 0, s.days - 1), s.days))
        # a home that runs out before the lock ends books a few days after it, like anybody would: without this,
        # "booked on the first allowed day" would separate genuine heavy users from diverters for no real reason
        own_gap = np.maximum(s.lock_days + rng.exponential(3.0, n), rng.gamma(4.0, 365 / np.maximum(need_now, 0.05) / 4.0))
        fast_gap = np.where(careful_rate > 0,
                            np.maximum(s.lock_days, rng.gamma(6.0, 365 / np.maximum(careful_rate, 1) / 6.0)),
                            s.lock_days + rng.exponential(np.maximum(extra_mean, 0.01)))
        nxt = t + np.where(in_regime, fast_gap, own_gap)
        # the regime begins before this home's own next refill would have come
        crossing = ~in_regime & (nxt > regime_start)
        nxt = np.where(crossing, np.maximum(regime_start + rng.uniform(0, 5, n), t + s.lock_days), nxt)
        t = nxt
        seen = (t >= 0) & (t < s.days) & ~((t >= away_from) & (t < away_until) & (fraud == 0))
        day[seen, k] = t[seen]
    day.sort(axis=1)                                            # NaN goes last

    # ── delivery side: the OTP is always entered; how and where differs ───────
    valid = ~np.isnan(day)
    after = valid & (day >= regime_start[:, None])
    kept_by_home = rng.random(day.shape) < np.clip(need / 13.0, 0, 1)[:, None]
    diverted = after & (((fraud == CAPTURED)[:, None] & ~kept_by_home) | (fraud == COMMERCIAL)[:, None])
    u1, u2 = rng.random(day.shape), rng.random(day.shape)
    captured_div = diverted & (fraud == CAPTURED)[:, None]
    commercial_div = diverted & (fraud == COMMERCIAL)[:, None]
    dac_fast = valid & (u1 < np.where(captured_div, 0.70, np.where(commercial_div, 0.05, 0.02)))
    dac_far = valid & (u2 < np.where(captured_div, 0.80, np.where(commercial_div, 0.50, 0.03)))

    mobile_changes = rng.poisson(s.mobile_change_rate, n)
    swapped = (fraud == CAPTURED) & (regime_start > s.days - 365 - 45) & (rng.random(n) < 0.6)
    mobile_changes = mobile_changes + swapped

    household = dict(distributor=distributor, agent=agent, is_pmuy=is_pmuy.astype(int), is_rural=is_rural.astype(int),
                     household_size=size, need=need, fraud_type=fraud, is_fraud=(fraud > 0).astype(int),
                     regime_start=regime_start, mobile_changes_12m=mobile_changes,
                     migrant=migrant.astype(int), grows=grows.astype(int))
    agents = dict(distributor=np.arange(n_agents) // s.agents_per_distributor, is_corrupt=corrupt.astype(int))
    return Population(s, household, agents, day, diverted, dac_fast, dac_far)
