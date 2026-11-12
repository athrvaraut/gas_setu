"""End-to-end tests of the FastAPI app. GSTINs come from the fictional demo registry; nothing calls the live API."""

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

import core.gstn_check as g
from api.booking_validate import app

client = TestClient(app)
RESTAURANT = g.DEMO_GSTINS["Annapurna Family Restaurant"]
SHOP = "Shop 4, Sector 17 Market, Vashi, Navi Mumbai"


def dates(gap, count, last=9):
    return [(date.today() - timedelta(days=last + i * gap)).isoformat() for i in range(count)]


def booking(name="Priya Sharma", **fields):
    body = {"consumer_id": "CON-T-1", "consumer_name": name, "address": "Flat 12B, Kandivali West, Mumbai",
            "pincode": "400067", "state_code": "27", "cylinder_type": "domestic", "company": "indane", **fields}
    response = client.post("/booking/validate", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def test_health():
    data = client.get("/health").json()
    assert data["status"] == "running" and data["model_trained"] is True
    assert "SIMULATED" in data["model_data"]


def test_root_serves_dashboard_and_index_moved_to_api():
    assert "text/html" in client.get("/").headers["content-type"]
    assert client.get("/api").json()["validate"] == "POST /booking/validate"


def test_normal_household_is_clear():
    r = booking(refill_dates=dates(52, 14), household_size=4)
    assert r["decision"] == "CLEAR" and r["proceed"] and r["risk_level"] == "NORMAL"


@pytest.mark.parametrize("gap, count", [(88, 8), (120, 6)])
def test_d16_low_usage_ujjwala_home_is_never_stopped(gap, count):
    r = booking("Sunita Devi", refill_dates=dates(gap, count), household_size=5, is_pmuy=True, is_rural=True)
    assert r["decision"] == "CLEAR" and r["proceed"]


def test_d16_no_household_input_can_stop_a_refill_without_registry_evidence():
    """Sweep extreme histories: the strongest outcome is REVIEW and the refill always proceeds."""
    for r12, r6, prev, early, dormant, size, pmuy in [(15, 8, 0, 14, 400, 1, True), (15, 8, 22, 14, 25, 1, False),
                                                      (0, 0, 0, 0, 0, 12, True), (15, 15, 0, 15, 700, 30, False)]:
        r = booking("Meena Devi", refills_12m=r12, refills_6m=r6, refills_prev_18m=prev, earliest_day_refills=early,
                    dormant_days=dormant, household_size=size, is_pmuy=pmuy, mobile_changes_12m=3)
        assert r["proceed"] and r["decision"] in ("CLEAR", "MONITOR", "REVIEW")


def test_captured_connection_goes_to_review_with_reasons():
    r = booking("Kamalabai Jadhav", refill_dates=dates(26, 8, last=4) + dates(121, 4, last=330), household_size=2,
                is_pmuy=True, is_rural=True, mobile_changes_12m=1)
    assert r["decision"] == "REVIEW" and r["proceed"] and r["risk_level"] == "HIGH"
    assert r["review_within_hours"] == 72 and r["reasons"][0]["code"] == "CONSUMPTION_HIGH"
    assert "compared with: PMUY, rural" in r["primary_reason"]


def test_history_can_be_sent_as_dates_or_as_derived_numbers():
    by_dates = booking("Meena Devi", refill_dates=dates(26, 28, last=3), household_size=2)
    by_numbers = booking("Meena Devi", refills_12m=14, refills_6m=7, refills_prev_18m=14, earliest_day_refills=14,
                         dormant_days=26, household_size=2)
    assert by_dates["risk_level"] == by_numbers["risk_level"] == "HIGH"


def test_without_history_the_consumption_check_is_not_scored():
    r = booking()
    assert r["risk_level"] == "NOT_SCORED" and r["decision"] == "CLEAR"
    assert [c["outcome"] for c in r["checks_run"]] == ["NO_GSTIN", "NOT_SCORED"]


def test_restaurant_at_its_registered_address_is_offered_a_commercial_cylinder():
    r = booking("Annapurna Restaurant", address=SHOP, pincode="400703", consumer_gstin=RESTAURANT)
    assert r["decision"] == "CONVERT_TO_COMMERCIAL" and not r["proceed"]
    assert "commercial" in r["recommended_action"].lower()
    assert "fictional demo record" in r["checks_run"][0]["detail"]
    ok = booking("Annapurna Restaurant", address=SHOP, pincode="400703", consumer_gstin=RESTAURANT, cylinder_type="commercial")
    assert ok["decision"] == "CLEAR" and ok["proceed"]


def test_someone_elses_gstin_holds_a_commercial_booking_for_review():
    r = booking("Rahul Verma", consumer_gstin=RESTAURANT, cylinder_type="commercial")
    assert r["decision"] == "HOLD_FOR_REVIEW" and not r["proceed"] and r["review_within_hours"] == 24
    assert r["reasons"][0]["code"] == "GSTIN_NAME_MISMATCH" and r["recommended_action"].startswith("Verify who is booking")


def test_d3_trade_name_match_is_accepted(monkeypatch):
    monkeypatch.setattr(g, "lookup_gstin", lambda number: {
        "status": "found", "legal_name": "UrbanClap Technologies India Pvt Ltd", "trade_name": "Urban Company",
        "gst_status": "Active", "pincode": "411001", "address": ""})
    r = booking("Urban Company", consumer_gstin="27AAPFU0939F1ZV", cylinder_type="commercial")
    assert r["decision"] == "CLEAR" and r["checks_run"][0]["outcome"] == "VERIFIED"


def test_d6_registry_down_does_not_fail_the_request(monkeypatch):
    monkeypatch.setattr(g, "lookup_gstin", lambda number: {"status": "unavailable", "reason": "GSTN provider timed out"})
    r = booking("Sharma Restaurant", consumer_gstin="27AAPFU0939F1ZV", cylinder_type="commercial")
    assert r["decision"] == "MONITOR" and r["proceed"] and r["reasons"][0]["code"] == "VERIFICATION_UNAVAILABLE"


def test_bad_input_is_rejected_with_422():
    base = dict(consumer_id="C", consumer_name="N", address="A", pincode="400067")
    assert client.post("/booking/validate", json={**base, "pincode": "4000"}).status_code == 422
    assert client.post("/booking/validate", json={**base, "cylinder_type": "industrial"}).status_code == 422
    assert client.post("/booking/validate", json={**base, "household_size": 0}).status_code == 422


def test_d21_endpoint_runs_in_the_threadpool():
    """A blocking registry call inside an async endpoint would stall every other request."""
    import inspect
    from api.booking_validate import validate_booking
    assert not inspect.iscoroutinefunction(validate_booking)


def test_demo_profiles_give_the_outcomes_the_dashboard_promises():
    profiles = {p["key"]: p for p in client.get("/demo/profiles").json()}
    expected = {"normal_household": "CLEAR", "ujjwala_low_usage": "CLEAR", "captured_connection": "REVIEW",
                "max_rate_booker": "REVIEW", "restaurant_domestic": "CONVERT_TO_COMMERCIAL",
                "restaurant_commercial": "CLEAR", "borrowed_gstin": "HOLD_FOR_REVIEW", "home_office": "MONITOR",
                "suspended_gstin": "HOLD_FOR_REVIEW"}
    for key, decision in expected.items():
        p = profiles[key]
        fields = {k: p[k] for k in ("address", "pincode", "cylinder_type", "household_size", "is_pmuy", "is_rural",
                                    "mobile_changes_12m")}
        r = booking(p["consumer_name"], consumer_gstin=p["consumer_gstin"] or None, **fields, **(p["history"] or {}))
        assert r["decision"] == decision, key
    joint = profiles["joint_family"]
    r = booking(joint["consumer_name"], household_size=9, **joint["history"])
    assert r["proceed"] and r["decision"] in ("CLEAR", "MONITOR")
