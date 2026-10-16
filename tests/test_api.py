"""End-to-end tests of the FastAPI app, with the live GSTN lookup mocked."""

import pytest
from fastapi.testclient import TestClient

import core.gstn_check as g
from api.booking_validate import app

client = TestClient(app)

NORMAL_HISTORY = dict(days_between_bookings=27, household_size=3, booking_frequency_change=0.05,
                      address_change_count=0, days_since_last_booking=28, agent_change_rate=0.1,
                      bookings_last_12_months=13, complaint_rate=0.0)


def booking(name, gstin=None, cylinder="domestic", **history):
    body = dict(consumer_id="CON-T-1", consumer_name=name, address="Shop 4, Andheri Market, Mumbai",
                pincode="400053", state_code="27", cylinder_type=cylinder, company="indane",
                consumer_gstin=gstin, **{**NORMAL_HISTORY, **history})
    return client.post("/booking/validate", json=body).json()


@pytest.fixture
def gst(monkeypatch):
    def install(legal, trade="", status="Active"):
        monkeypatch.setattr(g, "lookup_gstin", lambda gstin: {
            "found": True, "gstin": gstin, "legal_name": legal, "trade_name": trade, "status": status,
            "business_type": "", "taxpayer_type": "Regular", "pincode": "400053", "state": "27", "nature": []})
    return install


def test_health():
    data = client.get("/health").json()
    assert data["status"] == "running" and data["model_trained"] is True


def test_root_serves_dashboard_and_index_moved_to_api():
    assert "text/html" in client.get("/").headers["content-type"]
    assert client.get("/api").json()["validate"] == "POST /booking/validate"


def test_normal_household_is_approved():
    r = booking("Priya Sharma")
    assert r["decision"] == "APPROVED"


def test_restaurant_on_domestic_cylinder_is_blocked_with_commercial_action(gst):
    gst("Sharma Restaurant Pvt Ltd")
    r = booking("Sharma Restaurant", gstin="27AAPFU0939F1ZV")
    assert r["decision"] == "BLOCKED" and r["risk_level"] == "CRITICAL"
    assert "commercial cylinder" in r["recommended_action"]


def test_stolen_gstin_gets_identity_verification_action(gst):
    gst("Mahajan Foods LLP")
    r = booking("Rahul Verma", gstin="27AAPFU0939F1ZV")
    assert r["decision"] == "BLOCKED" and r["primary_reason"].startswith("STOLEN GSTIN")
    assert r["recommended_action"].startswith("Verify consumer identity")


def test_d3_trade_name_match_is_not_blocked_by_the_api(gst):
    gst("UrbanClap Technologies India Pvt Ltd", trade="Urban Company")
    r = booking("Urban Company", gstin="27AAPFU0939F1ZV", cylinder="commercial")
    assert r["decision"] == "APPROVED"
    identity = [c for c in r["checks_run"] if c["check_name"] == "GSTIN_IDENTITY_CHECK"]
    assert identity and identity[0]["passed"]


@pytest.mark.xfail(strict=True, reason="D12: uncalibrated ML risk tiers score this reseller MEDIUM. "
                                       "Fix in ROADMAP step 22, then remove this marker.")
def test_reseller_pattern_without_gstin_is_flagged_or_blocked():
    r = booking("Ramesh Kumar", days_between_bookings=15.2, household_size=1.5, booking_frequency_change=0.88,
                days_since_last_booking=15, agent_change_rate=0.75, bookings_last_12_months=24,
                complaint_rate=0.01)
    assert r["risk_level"] in ("HIGH", "CRITICAL")
