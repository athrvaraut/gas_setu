"""
Registry verification (Layers 1 and 2). The GSTN lookup is mocked or served from the fictional
demo registry, so the tests run offline and use no API credits.
"""

import pytest

import core.gstn_check as g

REAL_SHAPE = "27AAPFU0939F1ZV"                                   # a GSTIN with a correct check digit
RESTAURANT = g.DEMO_GSTINS["Annapurna Family Restaurant"]
SHOP = "Shop 4, Sector 17 Market, Vashi, Navi Mumbai"


def gstin(state: str, body: str = "AAPFU0939F1") -> str:
    """A well-formed GSTIN for any state (check digit computed)."""
    first_14 = state + body + "Z"
    return first_14 + g.gstin_check_digit(first_14)


def record(legal="Sharma Restaurant Pvt Ltd", trade="", gst_status="Active", pincode="400053", address=""):
    return {"status": "found", "legal_name": legal, "trade_name": trade, "gst_status": gst_status,
            "taxpayer_type": "Regular", "pincode": pincode, "address": address}


@pytest.fixture
def mock_lookup(monkeypatch):
    def install(result=None, forbid=False):
        def fake(_gstin):
            if forbid:
                raise AssertionError("lookup_gstin must not be called for this input")
            return result
        monkeypatch.setattr(g, "lookup_gstin", fake)
    return install


def validate(name="Sharma Restaurant", number=REAL_SHAPE, booking_type="domestic", pincode="400053", address="Shop 4, Andheri"):
    return g.validate_booking_address(consumer_name=name, address_line=address, pincode=pincode, state_code="27",
                                      booking_type=booking_type, consumer_gstin=number)


# ─── format and check digit (D7) ──────────────────────────────────────────────

def test_check_digit_accepts_the_right_digit_and_rejects_every_other():
    assert g.gstin_check_digit(REAL_SHAPE[:14]) == "V"
    assert g.is_valid_gstin_format(REAL_SHAPE)
    for wrong in "0123456789ABCDEFGHIJKLMNOPQRSTUWXYZ":            # every character except V
        assert not g.is_valid_gstin_format(REAL_SHAPE[:14] + wrong)


def test_a_single_wrong_character_anywhere_is_rejected():
    for i in (2, 5, 9, 11):
        changed = REAL_SHAPE[:i] + ("B" if REAL_SHAPE[i] != "B" else "C") + REAL_SHAPE[i + 1:]
        assert not g.is_valid_gstin_format(changed)


def test_invalid_gstin_makes_no_api_call(mock_lookup):
    mock_lookup(forbid=True)
    assert validate(number="27ABCDE1234F1Z").outcome == g.INVALID_GSTIN          # 14 characters
    assert validate(number=REAL_SHAPE[:14] + "A").outcome == g.INVALID_GSTIN     # wrong check digit


def test_no_gstin_makes_no_api_call(mock_lookup):
    mock_lookup(forbid=True)
    assert validate(number=None).outcome == g.NO_GSTIN
    assert validate(number="   ").outcome == g.NO_GSTIN


def test_demo_gstins_are_well_formed_and_cannot_be_real():
    for number in g.DEMO_REGISTRY:
        assert g.is_valid_gstin_format(number)
        assert number[5] == "X", "fourth PAN letter X is not a PAN holder type"


# ─── registry answers (D6) ────────────────────────────────────────────────────

def test_d6_registry_unavailable_is_reported_as_unavailable(mock_lookup):
    mock_lookup({"status": "unavailable", "found": False, "reason": "GSTN provider timed out"})
    r = validate()
    assert r.outcome == g.UNAVAILABLE and "could not be verified" in r.reason


def test_no_provider_key_means_unavailable_not_an_error(monkeypatch):
    monkeypatch.setattr(g, "APPYFLOW_KEY", None)
    g.GSTIN_CACHE.clear()
    assert g.lookup_gstin(REAL_SHAPE)["status"] == "unavailable"


def test_not_registered(mock_lookup):
    mock_lookup({"status": "not_found", "found": False})
    assert validate().outcome == g.NOT_REGISTERED


def test_suspended_gstin_is_inactive(mock_lookup):
    mock_lookup(record(gst_status="Suspended"))
    assert validate().outcome == g.INACTIVE


def test_cache_entries_expire(monkeypatch):
    g.GSTIN_CACHE.clear()
    g._cache_put("X", {"status": "found"}, ttl=100)
    assert g._cache_get("X") == {"status": "found"}
    monkeypatch.setattr(g.time, "time", lambda: 10 ** 12)
    assert g._cache_get("X") is None and "X" not in g.GSTIN_CACHE


# ─── name match (D3, D18) ─────────────────────────────────────────────────────

@pytest.mark.parametrize("booking, legal, trade, level", [
    ("Sharma Restaurant",   "Sharma Restaurant Pvt Ltd",            "",              "MATCH"),
    ("Urban Company",       "UrbanClap Technologies India Pvt Ltd", "Urban Company", "MATCH"),     # D3: trade name
    ("Urban Clap",          "UrbanClap Technologies",               "",              "PARTIAL"),
    ("Laxmi Sweets",        "Lakshmi Sweets",                       "",              "MATCH"),     # spelling variant
    ("M/s Shree Ganesh Traders", "Ganesh Traders",                  "",              "MATCH"),
    ("Rahul Sharma",        "Sharma Restaurant Pvt Ltd",            "",              "PARTIAL"),   # D18: one shared word
    ("Patel Kirana Stores", "Patel Chemicals Ltd",                  "",              "PARTIAL"),
    ("New Shree Caterers",  "Shree Cement Limited",                 "",              "MISMATCH"),
    ("Rahul Verma",         "Mahajan Foods LLP",                    "",              "MISMATCH"),
    ("Mr Kumar",            "Tata Motors Limited",                  "",              "MISMATCH"),
    ("A",                   "Reliance Industries Ltd",              "",              "INCONCLUSIVE"),
    ("",                    "Reliance Industries Ltd",              "",              "INCONCLUSIVE"),
])
def test_name_match_levels(booking, legal, trade, level):
    result = g.check_name_gstin_match(booking, legal, trade)
    assert result["level"] == level
    assert result["match"] is (level == "MATCH")


def test_name_outcomes(mock_lookup):
    mock_lookup(record(legal="Mahajan Foods LLP"))
    assert validate(name="Rahul Verma").outcome == g.NAME_MISMATCH
    mock_lookup(record(legal="Sharma Restaurant Pvt Ltd"))
    assert validate(name="Rahul Sharma").outcome == g.NAME_UNCERTAIN
    assert validate(name="Sharma Restaurant").outcome == g.VERIFIED


# ─── jurisdiction (D17) ───────────────────────────────────────────────────────

HOME_PINCODES = {       # GST state code → a pincode inside that state or UT
    "01": "190001", "02": "171001", "03": "141001", "04": "160017", "05": "248001", "06": "122001",
    "07": "110001", "08": "302001", "09": "226001", "10": "800001", "11": "737101", "12": "791111",
    "13": "797001", "14": "795001", "15": "796001", "16": "799001", "17": "793001", "18": "781001",
    "19": "700001", "20": "834001", "21": "751001", "22": "492001", "23": "462001", "24": "380001",
    "26": "396210", "27": "400053", "29": "560001", "30": "403001", "31": "682555", "32": "695001",
    "33": "600001", "34": "605001", "35": "744101", "36": "500001", "37": "520001", "38": "194101",
}


@pytest.mark.parametrize("state, pincode", sorted(HOME_PINCODES.items()))
def test_d17_no_state_is_flagged_against_its_own_pincode(state, pincode):
    assert not g.jurisdiction_mismatch(gstin(state), pincode)


@pytest.mark.parametrize("state, pincode", [("27", "110001"), ("07", "400053"), ("33", "560001"),
                                            ("10", "751001"), ("37", "500001"), ("09", "302001")])
def test_real_cross_state_pairs_are_flagged(state, pincode):
    assert g.jurisdiction_mismatch(gstin(state), pincode)


def test_unknown_pincode_prefix_never_flags():
    assert not g.jurisdiction_mismatch(gstin("27"), "990001")
    assert not g.jurisdiction_mismatch(gstin("27"), "")


def test_jurisdiction_note_is_in_the_reason(mock_lookup):
    mock_lookup(record(legal="Dishant Mahajan"))
    r = validate(name="Dishant Mahajan", booking_type="commercial", pincode="110001")
    assert r.outcome == g.VERIFIED and r.jurisdiction_mismatch and "JURISDICTION FLAG" in r.reason


# ─── business type and premises (D19) ─────────────────────────────────────────

def test_lpg_heavy_keywords():
    assert g.looks_lpg_heavy("Annapurna Family Restaurant")
    assert g.looks_lpg_heavy("", "Konkan Sweets")
    assert not g.looks_lpg_heavy("Sahyadri IT Consultants", "Priya Nair")


def test_premises_comparison():
    assert g.compare_premises(SHOP, "400703", "Shop 4, Sector 17 Market, Vashi, Navi Mumbai", "400703") == g.SAME_ADDRESS
    assert g.compare_premises("Shop no. 4, Sec 17 Mkt, Vashi", "400703", SHOP, "400703") == g.SAME_ADDRESS
    assert g.compare_premises("Flat 9, Sector 2, Vashi", "400703", SHOP, "400703") == g.SAME_PINCODE
    assert g.compare_premises(SHOP, "400703", SHOP, "411001") == g.DIFFERENT
    assert g.compare_premises(SHOP, "400703", "", "") == g.UNKNOWN


def test_d19_verified_result_describes_the_business_instead_of_calling_it_fraud():
    r = validate(name="Annapurna Restaurant", number=RESTAURANT, pincode="400703", address=SHOP)
    assert r.outcome == g.VERIFIED and r.premises == g.SAME_ADDRESS and r.lpg_heavy_business and r.is_demo_record
    assert "FRAUD" not in r.reason.upper()

    home = validate(name="Priya Nair", number=g.DEMO_GSTINS["Priya Nair"], pincode="410210",
                    address="Flat 12B, Sea Breeze CHS, Sector 9, Kharghar, Navi Mumbai")
    assert home.outcome == g.VERIFIED and home.premises == g.SAME_ADDRESS and not home.lpg_heavy_business
