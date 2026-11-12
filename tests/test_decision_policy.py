"""
Decision policy: the rule table, and the one rule it must never break:
a household's refill is never refused or held by an automatic check.
"""

import itertools
from types import SimpleNamespace

import pytest

from core import gstn_check as g
from core.decision_policy import (CLEAR, CONVERT_TO_COMMERCIAL, HOLD_FOR_REVIEW, MONITOR, REVIEW, decide)
from core.gstn_check import GSTNCheckResult

ALL_OUTCOMES = [g.NO_GSTIN, g.INVALID_GSTIN, g.NOT_REGISTERED, g.INACTIVE, g.NAME_MISMATCH, g.NAME_UNCERTAIN,
                g.VERIFIED, g.UNAVAILABLE]


def registry(outcome, premises=g.UNKNOWN, heavy=False, cross_state=False):
    return GSTNCheckResult(outcome, "evidence text", gstin="27AAPFU0939F1ZV", legal_name="Some Business",
                           premises=premises, lpg_heavy_business=heavy, jurisdiction_mismatch=cross_state)


def consumption(level):
    return SimpleNamespace(risk_level=level, percentile=99.8, peer_group="non-PMUY, urban, 3–5 people",
                           reasons=[{"code": "EXCESS_VS_PEERS", "detail": "13 refills; peers take about 6."}])


def test_the_invariant_holds_for_every_combination_of_inputs():
    """Domestic bookings: only verified registry evidence of an LPG-heavy business at the address stops a refill."""
    combos = itertools.product(ALL_OUTCOMES, [g.SAME_ADDRESS, g.SAME_PINCODE, g.DIFFERENT, g.UNKNOWN],
                               [True, False], [True, False], [None, "NORMAL", "LOW", "MEDIUM", "HIGH"],
                               ["Meena Devi", "Hotel Sai Palace"])
    for outcome, premises, heavy, cross, level, name in combos:
        d = decide("domestic", name, registry(outcome, premises, heavy, cross), consumption(level) if level else None)
        assert d.decision != HOLD_FOR_REVIEW
        if not d.proceed:
            assert d.decision == CONVERT_TO_COMMERCIAL
            assert (outcome, premises, heavy) == (g.VERIFIED, g.SAME_ADDRESS, True)


def test_the_consumption_model_alone_can_ask_for_review_at_most():
    for kind in ("domestic", "commercial"):
        assert decide(kind, "Meena Devi", registry(g.VERIFIED, g.DIFFERENT), consumption("HIGH")).decision == REVIEW
    assert decide("domestic", "Meena Devi", registry(g.NO_GSTIN), consumption("MEDIUM")).decision == MONITOR
    assert decide("domestic", "Meena Devi", registry(g.NO_GSTIN), consumption("LOW")).decision == CLEAR
    assert decide("domestic", "Meena Devi", registry(g.NO_GSTIN), None).decision == CLEAR


@pytest.mark.parametrize("outcome", [g.INVALID_GSTIN, g.NOT_REGISTERED, g.INACTIVE, g.NAME_MISMATCH])
def test_identity_problems_hold_a_commercial_booking_but_only_review_a_household(outcome):
    business = decide("commercial", "Rahul Verma", registry(outcome))
    assert business.decision == HOLD_FOR_REVIEW and not business.proceed and business.review_within_hours == 24
    home = decide("domestic", "Rahul Verma", registry(outcome))
    assert home.decision == REVIEW and home.proceed


def test_d6_registry_down_fails_open_with_a_note_to_verify_again():
    for kind in ("domestic", "commercial"):
        d = decide(kind, "Sharma Restaurant", registry(g.UNAVAILABLE))
        assert d.decision == MONITOR and d.proceed and d.reasons[0].code == "VERIFICATION_UNAVAILABLE"


def test_d19_a_gst_registration_alone_is_not_treated_as_misuse():
    assert decide("domestic", "Priya Nair", registry(g.VERIFIED, g.DIFFERENT, heavy=True)).decision == CLEAR
    home_office = decide("domestic", "Priya Nair", registry(g.VERIFIED, g.SAME_ADDRESS, heavy=False))
    assert home_office.decision == MONITOR and home_office.proceed
    nearby = decide("domestic", "Annapurna Restaurant", registry(g.VERIFIED, g.SAME_PINCODE, heavy=True))
    assert nearby.decision == REVIEW and nearby.proceed


def test_restaurant_at_the_delivery_address_is_offered_a_commercial_cylinder():
    d = decide("domestic", "Annapurna Restaurant", registry(g.VERIFIED, g.SAME_ADDRESS, heavy=True))
    assert d.decision == CONVERT_TO_COMMERCIAL and not d.proceed
    assert "commercial" in d.recommended_action.lower() and "appeal" in d.recommended_action.lower()
    assert decide("commercial", "Annapurna Restaurant", registry(g.VERIFIED, g.SAME_ADDRESS, heavy=True)).decision == CLEAR


def test_blank_gstin_with_an_establishment_name_goes_to_review():
    d = decide("domestic", "Sharma Restaurant", registry(g.NO_GSTIN))
    assert d.decision == REVIEW and d.proceed and d.reasons[0].code == "ESTABLISHMENT_NAME"


def test_strongest_reason_comes_first_and_all_reasons_are_kept():
    d = decide("commercial", "Rahul Verma", registry(g.NAME_MISMATCH), None)
    assert d.reasons[0].asks == HOLD_FOR_REVIEW and d.primary_reason == d.reasons[0].detail
    both = decide("domestic", "Meena Devi", registry(g.NAME_UNCERTAIN), consumption("HIGH"))
    assert {r.code for r in both.reasons} == {"GSTIN_NAME_UNCERTAIN", "CONSUMPTION_HIGH"}
