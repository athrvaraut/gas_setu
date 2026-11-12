"""Layer 3: booking-history features, the synthetic population and the consumption risk model (v2)."""

from datetime import date, timedelta

import numpy as np
import pytest

from core.anomaly_detection import ConsumptionRiskModel, fit_peer_stats, peer_group, peer_label
from core.features import FEATURE_NAMES, features_from_refill_dates, history_features, population_features
from core.synthetic_population import CAPTURED, COMMERCIAL, Scenario, simulate

TODAY = date(2026, 11, 12)
SMALL = Scenario(n_distributors=6, agents_per_distributor=5, households_per_agent=300, seed=3)


def dates(gap, count, last=9):
    return [TODAY - timedelta(days=last + i * gap) for i in range(count)]


@pytest.fixture(scope="module")
def model():
    return ConsumptionRiskModel.load()


@pytest.fixture(scope="module")
def pop():
    return simulate(SMALL)


def check(model, refill_dates, **record):
    record = {"household_size": 4, "is_pmuy": 0, "is_rural": 0, "mobile_changes_12m": 0, **record}
    return model.check("C", **features_from_refill_dates(refill_dates, TODAY), **record)


# ─── features ─────────────────────────────────────────────────────────────────

def test_history_features_for_a_steady_home():
    f = features_from_refill_dates(dates(52, 14), TODAY)
    assert f == {"refills_12m": 7, "refills_6m": 4, "refills_prev_18m": 10, "earliest_day_refills": 0, "dormant_days": 52}


def test_history_features_see_a_wake_up_and_earliest_day_booking():
    f = features_from_refill_dates(dates(26, 8, last=4) + dates(121, 4, last=330), TODAY)
    assert f["refills_6m"] == 7 and f["refills_prev_18m"] == 5
    assert f["earliest_day_refills"] == 7 and f["dormant_days"] >= 140


def test_no_history_and_old_history_give_zeros():
    assert features_from_refill_dates([], TODAY)["refills_12m"] == 0
    assert features_from_refill_dates([TODAY - timedelta(days=900)], TODAY)["refills_12m"] == 0


def test_single_history_and_population_code_agree(pop):
    f = population_features(pop)
    for i in (0, 17, 400):
        days = pop.refill_day[i][~np.isnan(pop.refill_day[i])]
        one = history_features(days[None, :], pop.scenario.days, pop.scenario.lock_days)
        assert all(one[k][0] == f[k][i] for k in one)


# ─── synthetic population ─────────────────────────────────────────────────────

def test_population_matches_the_published_figures_it_is_built_on(pop):
    h, f = pop.household, population_features(pop)
    genuine = h["is_fraud"] == 0
    assert 0.25 < h["is_pmuy"].mean() < 0.40
    assert 4.2 < f["refills_12m"][genuine & (h["is_pmuy"] == 1)].mean() < 5.3          # published: 4.71
    assert 5.6 < f["refills_12m"][genuine & (h["is_pmuy"] == 0)].mean() < 7.4          # published: 6–7
    assert 4.2 < h["household_size"].mean() < 4.7                                       # NFHS-5: 4.4


def test_population_obeys_the_booking_lock(pop):
    gaps = np.diff(pop.refill_day, axis=1)
    assert np.nanmin(gaps) >= pop.scenario.lock_days - 1e-6
    assert population_features(pop)["refills_12m"].max() <= 15


def test_diversion_is_rare_and_genuine_heavy_users_exist(pop):
    h, f = pop.household, population_features(pop)
    assert 0.005 < h["is_fraud"].mean() < 0.04
    assert (f["refills_12m"][h["is_fraud"] == 0] >= 12).sum() > 0, "the classes must overlap"


def test_same_seed_same_population():
    again = simulate(SMALL)
    assert np.array_equal(again.refill_day, simulate(SMALL).refill_day, equal_nan=True)


# ─── model ────────────────────────────────────────────────────────────────────

def test_model_file_states_that_it_was_fitted_on_simulated_data(model):
    assert "SIMULATED" in model.spec["notes"]["data"]
    assert set(model.spec["weights"]) == {"z_refills", "jump", "woke_up", "mobile_changes", "mobile_with_jump"}
    assert min(model.spec["weights"].values()) >= 0, "an input may only add concern"
    assert len(model.peer_stats) == 16


def test_d16_genuine_low_usage_homes_are_normal(model):
    assert check(model, dates(88, 8), household_size=5, is_pmuy=1, is_rural=1).risk_level == "NORMAL"      # 4 a year
    assert check(model, dates(120, 6), household_size=4, is_pmuy=1, is_rural=1).risk_level == "NORMAL"     # 3 a year
    assert check(model, dates(60, 12), household_size=2).risk_level == "NORMAL"                             # elderly couple
    assert check(model, [], household_size=4, is_pmuy=1, is_rural=1).risk_level == "NORMAL"                 # no refills


def test_d12_ordinary_and_large_families_are_not_sent_for_review(model):
    assert check(model, dates(52, 14)).risk_level == "NORMAL"
    assert check(model, dates(31, 23), household_size=9).risk_level in ("NORMAL", "LOW", "MEDIUM")


def test_d12_diversion_patterns_rank_above_genuine_homes(model):
    captured = check(model, dates(26, 8, last=4) + dates(121, 4, last=330), household_size=2, is_pmuy=1, is_rural=1,
                     mobile_changes_12m=1)
    max_rate = check(model, dates(26, 28, last=3), household_size=2)
    rural_low = check(model, dates(88, 8), household_size=5, is_pmuy=1, is_rural=1)
    assert captured.risk_level == "HIGH" and max_rate.risk_level == "HIGH"
    assert rural_low.percentile < captured.percentile
    codes = {r["code"] for r in captured.reasons}
    assert {"SUDDEN_INCREASE", "EARLIEST_DAY_BOOKING", "MOBILE_CHANGED"} <= codes
    assert "EXCESS_VS_PEERS" in {r["code"] for r in max_rate.reasons}


def test_same_bookings_are_judged_against_the_right_peers(model):
    """Thirteen refills a year: unusual for two people, ordinary for nine."""
    small = check(model, dates(28, 26), household_size=2)
    large = check(model, dates(28, 26), household_size=9)
    assert small.percentile > large.percentile
    assert small.peer_median < large.peer_median


def test_review_burden_is_small_and_no_group_is_singled_out(model, pop):
    """About 0.5% of connections are HIGH, and no large peer group carries many times that (on unseen data)."""
    f = population_features(pop)
    high = model.percentile(f) >= 99.5
    assert 0.001 < high.mean() < 0.02
    group = peer_group(f["is_pmuy"], f["is_rural"], f["household_size"])
    genuine = pop.household["is_fraud"] == 0
    for g in np.unique(group):
        if (group == g).sum() >= 500:
            assert high[(group == g) & genuine].mean() < 0.02, peer_label(int(g))


def test_model_ranks_captured_and_commercial_connections_high(pop):
    f = population_features(pop)
    fitted = ConsumptionRiskModel.fit(f, pop.household["is_fraud"], f, version="test")
    score = fitted.raw_score(f)
    cut = np.percentile(score, 99)
    organised = np.isin(pop.household["fraud_type"], (CAPTURED, COMMERCIAL))
    assert (score[organised] >= cut).mean() > 0.3
    assert (score[pop.household["is_fraud"] == 0] >= cut).mean() < 0.01


def test_fit_and_reload_round_trip(tmp_path, pop):
    f = population_features(pop)
    fitted = ConsumptionRiskModel.fit(f, pop.household["is_fraud"], f, version="test")
    fitted.save(tmp_path / "m.json")
    loaded = ConsumptionRiskModel.load(tmp_path / "m.json")
    assert np.allclose(loaded.raw_score(f), fitted.raw_score(f))
    assert set(FEATURE_NAMES) <= set(f) and len(fit_peer_stats(f)) == 16
