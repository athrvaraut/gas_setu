"""Layer 3: consumption anomaly detection (Isolation Forest on synthetic data)."""

import pytest
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from core.anomaly_detection import ConsumptionAnomalyDetector, generate_synthetic_data

NORMAL = dict(days_between_bookings=26, household_size=4, booking_frequency_change=0.05,
              address_change_count=0, days_since_last_booking=28, agent_change_rate=0.1,
              bookings_last_12_months=13, complaint_rate=0.0)
RESELLER = dict(days_between_bookings=15.2, household_size=1.5, booking_frequency_change=0.88,
                address_change_count=0, days_since_last_booking=15, agent_change_rate=0.75,
                bookings_last_12_months=24, complaint_rate=0.01)


@pytest.fixture(scope="module")
def detector():
    d = ConsumptionAnomalyDetector()
    d.train(verbose=False)
    return d


def test_untrained_model_refuses_to_score():
    with pytest.raises(RuntimeError):
        ConsumptionAnomalyDetector().check(consumer_id="X", **NORMAL)


RURAL_GENUINE = dict(days_between_bookings=38, household_size=1.0, booking_frequency_change=0.05,
                     address_change_count=0, days_since_last_booking=40, agent_change_rate=0.05,
                     bookings_last_12_months=9, complaint_rate=0.0)

D12 = ("D12: risk tiers come from an uncalibrated score mapping (normal family 0.59 = MEDIUM, reseller "
       "0.64 = MEDIUM, genuine rural household 0.72 = HIGH). Fix with the ML re-evaluation in ROADMAP "
       "step 22, then remove this marker.")


def test_reseller_pattern_gets_rule_based_reasons(detector):
    r = detector.check(consumer_id="R", **RESELLER)
    assert any(f.startswith("BOOKING_RATE_MAX") for f in r.flags)
    assert any(f.startswith("SUDDEN_FREQUENCY_CHANGE") for f in r.flags)


@pytest.mark.xfail(strict=True, reason=D12)
def test_d12_normal_household_is_low_risk(detector):
    r = detector.check(consumer_id="N", **NORMAL)
    assert r.risk_level in ("NORMAL", "LOW")


@pytest.mark.xfail(strict=True, reason=D12)
def test_d12_reseller_pattern_is_high_risk(detector):
    r = detector.check(consumer_id="R", **RESELLER)
    assert r.risk_level in ("HIGH", "CRITICAL") and r.anomaly_detected


@pytest.mark.xfail(strict=True, reason=D12)
def test_d12_genuine_rural_household_scores_below_reseller(detector):
    rural = detector.check(consumer_id="RU", **RURAL_GENUINE).anomaly_score
    reseller = detector.check(consumer_id="R", **RESELLER).anomaly_score
    assert rural < reseller


def test_score_is_bounded(detector):
    for features in (NORMAL, RESELLER):
        assert 0.0 <= detector.check(consumer_id="B", **features).anomaly_score <= 1.0


def test_known_limitation_v1_synthetic_data_is_trivially_separable():
    """Documents research gap G3: one threshold on one feature separates the v1 synthetic classes,
    so scores measured on this data say nothing about real fraud. Replace this test when the
    independent synthetic generator v2 lands (ROADMAP step 22)."""
    X, y = generate_synthetic_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    stump = DecisionTreeClassifier(max_depth=1).fit(Xtr, ytr)
    assert stump.score(Xte, yte) == 1.0
