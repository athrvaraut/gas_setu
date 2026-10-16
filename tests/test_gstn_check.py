"""
GSTN check (Layer 1 and the name match), with the Appyflow API mocked so tests run offline
and use no credits.
"""

import pytest

import core.gstn_check as g


def gst_record(legal="Sharma Restaurant Pvt Ltd", trade="", status="Active"):
    return {"found": True, "gstin": "27AAPFU0939F1ZV", "legal_name": legal, "trade_name": trade,
            "status": status, "business_type": "", "taxpayer_type": "Regular", "pincode": "400053",
            "state": "Maharashtra", "nature": []}


@pytest.fixture
def mock_lookup(monkeypatch):
    def install(result=None, forbid=False):
        def fake(gstin):
            if forbid:
                raise AssertionError("lookup_gstin must not be called for this input")
            return result
        monkeypatch.setattr(g, "lookup_gstin", fake)
    return install


def validate(name="Sharma Restaurant", gstin="27AAPFU0939F1ZV", booking_type="domestic", pincode="400053"):
    return g.validate_booking_address(consumer_name=name, address_line="Shop 4, Andheri", pincode=pincode,
                                      state_code="27", booking_type=booking_type, consumer_gstin=gstin)


def test_no_gstin_allows_domestic(mock_lookup):
    mock_lookup(forbid=True)
    r = validate(gstin=None)
    assert not r.is_blocked() and r.recommendation == "ALLOW_DOMESTIC"


def test_bad_format_is_blocked_without_an_api_call(mock_lookup):
    mock_lookup(forbid=True)
    r = validate(gstin="27ABCDE1234F1Z")          # 14 characters
    assert r.is_blocked() and r.reason.startswith("FAKE GSTIN")


def test_suspended_gstin_is_blocked(mock_lookup):
    mock_lookup(gst_record(status="Suspended"))
    r = validate()
    assert r.is_blocked() and r.reason.startswith("INACTIVE GSTIN")


def test_name_mismatch_is_blocked_as_stolen(mock_lookup):
    mock_lookup(gst_record(legal="Mahajan Foods LLP"))
    r = validate(name="Rahul Verma")
    assert r.is_blocked() and r.reason.startswith("STOLEN GSTIN")


def test_trade_name_match_allows_commercial(mock_lookup):
    mock_lookup(gst_record(legal="UrbanClap Technologies India Pvt Ltd", trade="Urban Company"))
    r = validate(name="Urban Company", booking_type="commercial")
    assert not r.is_blocked() and r.recommendation == "ALLOW_COMMERCIAL"


def test_active_gstin_on_domestic_booking_is_blocked(mock_lookup):
    mock_lookup(gst_record())
    r = validate()
    assert r.is_blocked() and r.reason.startswith("FRAUD")


def test_jurisdiction_mismatch_is_flagged(mock_lookup):
    mock_lookup(gst_record(legal="Dishant Mahajan"))
    r = validate(name="Dishant Mahajan", booking_type="commercial", pincode="110001")   # Delhi pincode
    assert not r.is_blocked() and "JURISDICTION FLAG" in r.reason


@pytest.mark.xfail(strict=True, reason="D6: an API timeout is treated like 'not registered' (fails open). "
                                       "Fix in ROADMAP step 19, then remove this marker.")
def test_d6_api_unavailable_must_not_silently_allow(mock_lookup):
    mock_lookup({"found": False, "reason": "API timeout — will retry"})
    assert validate().recommendation != "ALLOW_DOMESTIC"


@pytest.mark.xfail(strict=True, reason="D7: the GSTIN check digit is not verified yet. "
                                       "Fix in ROADMAP step 19, then remove this marker.")
def test_d7_wrong_check_digit_is_rejected():
    assert g.is_valid_gstin_format("27AAPFU0939F1ZV")        # real check digit V
    assert not g.is_valid_gstin_format("27AAPFU0939F1ZA")    # same GSTIN, wrong check digit
