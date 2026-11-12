"""
GasSetu — Registry Verification (Layers 1 and 2)
=================================================
Answers three questions about a GSTIN that arrives with a booking:

  1. Is the number itself valid?            format + check digit, no API call
  2. Is the business real and active?       live GSTN lookup through a provider
  3. Does it belong to this booking?        name match, registered address vs delivery address,
                                            GST state vs delivery pincode

This module only reports what it found (an OUTCOME plus evidence). It never decides what happens
to the booking; that is the job of core/decision_policy.py.

Changes in v4 (review round 2, build_plan.md §13):
  D6   a registry that cannot be reached is reported as UNAVAILABLE, not as "not registered"
  D7   the GSTIN check digit is verified
  D17  jurisdiction check compares GST state codes with the states a pincode prefix can belong to
  D18  name match scores the whole name; one shared word is no longer a match
  D19  a GST registration is not called fraud; the result says where the business is and what it does
  D21  cache entries expire

Author: Athrva Raut — GasSetu Project
"""

import os
import re
import time
import requests
from collections import OrderedDict
from datetime import datetime, timezone
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
APPYFLOW_KEY = os.getenv("APPYFLOW_KEY")


# ─── OUTCOMES ─────────────────────────────────────────────────────────────────

NO_GSTIN        = "NO_GSTIN"          # nothing was declared
INVALID_GSTIN   = "INVALID_GSTIN"     # fails the format or the check digit (no API call made)
NOT_REGISTERED  = "NOT_REGISTERED"    # the registry says this GSTIN does not exist
INACTIVE        = "INACTIVE"          # suspended or cancelled
NAME_MISMATCH   = "NAME_MISMATCH"     # booking name matches neither the legal nor the trade name
NAME_UNCERTAIN  = "NAME_UNCERTAIN"    # partial match: a person has to look
VERIFIED        = "VERIFIED"          # active, and the name matches
UNAVAILABLE     = "UNAVAILABLE"       # the registry could not be reached; nothing was verified

# Where the registered place of business is, compared with the delivery address
SAME_ADDRESS, SAME_PINCODE, DIFFERENT, UNKNOWN = "SAME_ADDRESS", "SAME_PINCODE", "DIFFERENT", "UNKNOWN"


# ─── GSTIN FORMAT AND CHECK DIGIT ─────────────────────────────────────────────

# 2-digit state + 10-character PAN + entity number + 'Z' + check digit
GSTIN_PATTERN = re.compile(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$")
_BASE36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def gstin_check_digit(first_14: str) -> str:
    """Check digit for the first 14 characters of a GSTIN (Luhn mod 36, weights 1,2,1,2…)."""
    total = 0
    for i, ch in enumerate(first_14.upper()):
        product = _BASE36.index(ch) * (1 if i % 2 == 0 else 2)
        total += product // 36 + product % 36
    return _BASE36[(36 - total % 36) % 36]


def is_valid_gstin_format(gstin: str) -> bool:
    """True when the GSTIN has the right shape AND the right check digit. No API call."""
    if not gstin or len(gstin) != 15:
        return False
    gstin = gstin.upper()
    if not GSTIN_PATTERN.match(gstin):
        return False
    return gstin_check_digit(gstin[:14]) == gstin[14]


# ─── JURISDICTION: GST STATE CODE vs DELIVERY PINCODE ─────────────────────────

GSTIN_STATE_CODES = {
    "01": "Jammu & Kashmir",  "02": "Himachal Pradesh",  "03": "Punjab",
    "04": "Chandigarh",       "05": "Uttarakhand",       "06": "Haryana",
    "07": "Delhi",            "08": "Rajasthan",         "09": "Uttar Pradesh",
    "10": "Bihar",            "11": "Sikkim",            "12": "Arunachal Pradesh",
    "13": "Nagaland",         "14": "Manipur",           "15": "Mizoram",
    "16": "Tripura",          "17": "Meghalaya",         "18": "Assam",
    "19": "West Bengal",      "20": "Jharkhand",         "21": "Odisha",
    "22": "Chhattisgarh",     "23": "Madhya Pradesh",    "24": "Gujarat",
    "25": "Daman & Diu",      "26": "Dadra & Nagar Haveli and Daman & Diu",
    "27": "Maharashtra",      "28": "Andhra Pradesh (old code)",
    "29": "Karnataka",        "30": "Goa",               "31": "Lakshadweep",
    "32": "Kerala",           "33": "Tamil Nadu",        "34": "Puducherry",
    "35": "Andaman & Nicobar Islands",                   "36": "Telangana",
    "37": "Andhra Pradesh",   "38": "Ladakh",
}


def _prefixes(first: int, last: int, *codes: str) -> dict:
    return {f"{p:02d}": frozenset(codes) for p in range(first, last + 1)}


# First two digits of a pincode → every GST state code that prefix can belong to.
# A prefix often covers more than one state or UT (403 is Goa inside "40", 248 is Uttarakhand
# inside "24", 737 is Sikkim inside "73"), so each entry is a SET. A mismatch is flagged only when
# the GSTIN's state is not in the set, which means a correct pair is never flagged.
PINCODE_PREFIX_STATES = {
    **_prefixes(11, 11, "07"),                               # Delhi
    **_prefixes(12, 13, "06"),                               # Haryana
    **_prefixes(14, 15, "03"),                               # Punjab
    **_prefixes(16, 16, "03", "04"),                         # Punjab, Chandigarh
    **_prefixes(17, 17, "02"),                               # Himachal Pradesh
    **_prefixes(18, 19, "01", "38"),                         # Jammu & Kashmir, Ladakh
    **_prefixes(20, 28, "09", "05"),                         # Uttar Pradesh, Uttarakhand
    **_prefixes(30, 34, "08"),                               # Rajasthan
    **_prefixes(36, 39, "24", "25", "26"),                   # Gujarat, Daman & Diu, Dadra & Nagar Haveli
    **_prefixes(40, 40, "27", "30"),                         # Maharashtra, Goa
    **_prefixes(41, 44, "27"),                               # Maharashtra
    **_prefixes(45, 48, "23"),                               # Madhya Pradesh
    **_prefixes(49, 49, "22"),                               # Chhattisgarh
    **_prefixes(50, 50, "36", "28"),                         # Telangana
    **_prefixes(51, 53, "37", "28", "34"),                   # Andhra Pradesh, Yanam (Puducherry)
    **_prefixes(56, 59, "29"),                               # Karnataka
    **_prefixes(60, 64, "33", "34"),                         # Tamil Nadu, Puducherry, Karaikal
    **_prefixes(67, 69, "32", "31", "34"),                   # Kerala, Lakshadweep, Mahe (Puducherry)
    **_prefixes(70, 74, "19", "11", "35"),                   # West Bengal, Sikkim, Andaman & Nicobar
    **_prefixes(75, 77, "21"),                               # Odisha
    **_prefixes(78, 78, "18"),                               # Assam
    **_prefixes(79, 79, "12", "13", "14", "15", "16", "17"), # North-eastern states
    **_prefixes(80, 85, "10", "20"),                         # Bihar, Jharkhand
}


def get_state_from_gstin(gstin: str) -> str:
    """State name from the first two digits of a GSTIN."""
    if gstin and len(gstin) >= 2:
        return GSTIN_STATE_CODES.get(gstin[:2], "Unknown")
    return "Unknown"


def jurisdiction_mismatch(gstin: str, pincode: str) -> bool:
    """True only when the GSTIN's state cannot contain the delivery pincode. Unknown prefixes never flag."""
    if not gstin or not pincode or len(pincode) < 2:
        return False
    possible = PINCODE_PREFIX_STATES.get(pincode[:2])
    return possible is not None and gstin[:2] in GSTIN_STATE_CODES and gstin[:2] not in possible


# ─── NAME MATCH ───────────────────────────────────────────────────────────────

# Dropped before matching: legal forms and honorifics say nothing about who the entity is
DROPPED_WORDS = {
    "pvt", "ltd", "private", "limited", "llp", "opc", "inc", "ms", "messrs",
    "shri", "shree", "sri", "smt", "mr", "mrs", "dr", "km", "kumari",
}
# Kept but worth little: sharing only these words is not evidence of the same entity
GENERIC_WORDS = {
    "and", "the", "new", "co", "company", "corporation", "enterprises", "enterprise", "traders",
    "trading", "services", "service", "solutions", "industries", "industry", "group", "india",
    "international", "global", "associates", "agency", "agencies", "stores", "store", "centre",
    "center", "works", "house", "sons", "brothers", "bros",
}
MATCH_THRESHOLD, PARTIAL_THRESHOLD = 0.80, 0.45

_PHONETIC_RULES = (("ksh", "ks"), ("x", "ks"), ("chh", "c"), ("sh", "s"), ("ch", "c"), ("ph", "f"),
                   ("th", "t"), ("dh", "d"), ("bh", "b"), ("gh", "g"), ("kh", "k"), ("jh", "j"),
                   ("ee", "i"), ("oo", "u"), ("ou", "u"), ("aa", "a"), ("w", "v"), ("y", "i"),
                   ("z", "j"), ("q", "k"))


def _name_tokens(name: str) -> list:
    words = re.sub(r"[^a-z0-9 ]", " ", (name or "").lower().replace("m/s", " ")).split()
    return [w for w in words if w not in DROPPED_WORDS and len(w) > 1]


def _phonetic_key(token: str) -> str:
    """Rough key for common English spellings of Indian names (laxmi = lakshmi, choudhary ~ chaudhari)."""
    for old, new in _PHONETIC_RULES:
        token = token.replace(old, new)
    return re.sub(r"(.)\1+", r"\1", token)


def _same_word(a: str, b: str) -> bool:
    if a == b:
        return True
    if len(a) >= 4 and len(b) >= 4 and _phonetic_key(a) == _phonetic_key(b):
        return True
    return len(a) >= 6 and len(b) >= 6 and SequenceMatcher(None, a, b).ratio() >= 0.87


def _fuse_split_words(tokens: list, other: list) -> list:
    """Join two neighbouring words when the other name writes them as one ("urban clap" vs "urbanclap")."""
    out, i = [], 0
    while i < len(tokens):
        if i + 1 < len(tokens) and tokens[i] + tokens[i + 1] in other:
            out.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


def name_similarity(a: str, b: str) -> float:
    """
    0.0–1.0. Weighted share of words the two names have in common (Dice), where generic words
    ("enterprises", "traders") count for a quarter. Spelling variants of the same word count as equal.
    """
    ta, tb = _name_tokens(a), _name_tokens(b)
    ta, tb = _fuse_split_words(ta, tb), _fuse_split_words(tb, ta)
    if not ta or not tb:
        return 0.0
    weight = lambda w: 0.25 if w in GENERIC_WORDS else 1.0
    unused, shared = list(tb), 0.0
    for w in ta:
        hit = next((x for x in unused if _same_word(w, x)), None)
        if hit is not None:
            unused.remove(hit)
            shared += min(weight(w), weight(hit))
    return round(2 * shared / (sum(map(weight, ta)) + sum(map(weight, tb))), 3)


def check_name_gstin_match(booking_name: str, legal_name: str, trade_name: str = "") -> dict:
    """
    Compares the booking name with the registry's legal name and trade name.

    Returns {"match": bool, "level": MATCH | PARTIAL | MISMATCH | INCONCLUSIVE,
             "score": float, "matched_on": "legal_name" | "trade_name" | "none"}

    MATCH needs most of the name to agree. One shared word ("Sharma") gives PARTIAL, which the
    decision policy sends to a person instead of treating as verified. A booking name with no
    usable words is INCONCLUSIVE, never a match.

    The thresholds are engineering defaults. They still have to be measured on labelled name pairs
    (ROADMAP step 21).
    """
    if not _name_tokens(booking_name):
        return {"match": False, "level": "INCONCLUSIVE", "score": 0.0, "matched_on": "none"}

    legal_score = name_similarity(booking_name, legal_name)
    trade_score = name_similarity(booking_name, trade_name) if trade_name else 0.0
    score, matched_on = max((legal_score, "legal_name"), (trade_score, "trade_name"))

    if score >= MATCH_THRESHOLD:
        level = "MATCH"
    elif score >= PARTIAL_THRESHOLD:
        level = "PARTIAL"
    else:
        level, matched_on = "MISMATCH", "none"
    return {"match": level == "MATCH", "level": level, "score": score, "matched_on": matched_on}


# ─── WHAT KIND OF BUSINESS, AND WHERE ─────────────────────────────────────────

# Trades that burn a lot of LPG. A keyword hit in the legal or trade name is a hint, not proof.
LPG_HEAVY_KEYWORDS = {
    "restaurant", "restaurants", "hotel", "hotels", "dhaba", "caterer", "caterers", "catering",
    "canteen", "mess", "sweets", "sweet", "bakery", "bakers", "foods", "food", "kitchen", "kitchens",
    "cafe", "bhojnalaya", "bhojanalay", "tiffin", "tiffins", "snacks", "biryani", "dosa", "pizza",
    "lodge", "resort", "banquet", "halwai", "mithai", "namkeen", "farsan", "eatery", "bar",
    "chinese", "tandoor", "juice", "tea", "chai",
}


def looks_lpg_heavy(*names: str) -> bool:
    """True when a name contains a word typical of a food or hospitality business."""
    return any(w in LPG_HEAVY_KEYWORDS for name in names for w in _name_tokens(name))


_ADDRESS_FILLER = {"near", "nr", "opp", "opposite", "behind", "beside", "the", "and", "of", "no", "number", "at", "post"}
_ADDRESS_SHORT = {"rd": "road", "st": "street", "sec": "sector", "bldg": "building", "flr": "floor",
                  "apt": "apartment", "apts": "apartment", "apartments": "apartment", "soc": "society",
                  "mkt": "market", "ngr": "nagar", "chsl": "chs", "ind": "industrial", "est": "estate"}


def _address_tokens(address: str) -> set:
    words = re.sub(r"[^a-z0-9 ]", " ", (address or "").lower()).split()
    return {_ADDRESS_SHORT.get(w, w) for w in words if w not in _ADDRESS_FILLER}


def address_similarity(a: str, b: str) -> float:
    """Share of address words in common (Dice) after expanding abbreviations and dropping filler."""
    ta, tb = _address_tokens(a), _address_tokens(b)
    if not ta or not tb:
        return 0.0
    return round(2 * len(ta & tb) / (len(ta) + len(tb)), 3)


def compare_premises(delivery_address: str, delivery_pincode: str,
                     registered_address: str, registered_pincode: str) -> str:
    """Is the registered place of business the delivery address? Threshold to be measured (ROADMAP step 20)."""
    if not registered_pincode:
        return UNKNOWN
    if str(registered_pincode).strip() != str(delivery_pincode).strip():
        return DIFFERENT
    if registered_address and address_similarity(delivery_address, registered_address) >= 0.6:
        return SAME_ADDRESS
    return SAME_PINCODE


# ─── REGISTRY LOOKUP ──────────────────────────────────────────────────────────

# Fictional records for the dashboard, the tests and offline demos. The fourth PAN letter is 'X',
# which is not a PAN holder type, so none of these numbers can belong to a real taxpayer.
def _demo(state_pan_entity: str, **record) -> tuple:
    gstin = state_pan_entity + "Z" + gstin_check_digit(state_pan_entity + "Z")
    return gstin, {"status": "found", "found": True, "gstin": gstin, "demo": True, "business_type": "",
                   "taxpayer_type": "Regular", "state": "Maharashtra", "nature": [], **record}


DEMO_REGISTRY = dict([
    _demo("27DEMXR0001A1", legal_name="Annapurna Family Restaurant", trade_name="Annapurna Restaurant",
          gst_status="Active", pincode="400703", address="Shop 4, Sector 17 Market, Vashi, Navi Mumbai"),
    _demo("27DEMXC0002B1", legal_name="Priya Nair", trade_name="Sahyadri IT Consultants",
          gst_status="Active", pincode="410210", address="Flat 12B, Sea Breeze CHS, Sector 9, Kharghar, Navi Mumbai"),
    _demo("27DEMXS0003C1", legal_name="Konkan Sweets and Farsan", trade_name="Konkan Sweets",
          gst_status="Suspended", pincode="400706", address="Shop 11, Station Road, Nerul, Navi Mumbai"),
])
DEMO_GSTINS = {record["legal_name"]: gstin for gstin, record in DEMO_REGISTRY.items()}

CACHE_TTL_FOUND, CACHE_TTL_NOT_FOUND, CACHE_MAX = 24 * 3600, 6 * 3600, 5000
GSTIN_CACHE: "OrderedDict[str, tuple]" = OrderedDict()      # gstin → (expires_at, result)


def _cache_get(gstin: str) -> Optional[dict]:
    hit = GSTIN_CACHE.get(gstin)
    if hit and hit[0] > time.time():
        return hit[1]
    GSTIN_CACHE.pop(gstin, None)
    return None


def _cache_put(gstin: str, result: dict, ttl: int):
    GSTIN_CACHE[gstin] = (time.time() + ttl, result)
    while len(GSTIN_CACHE) > CACHE_MAX:
        GSTIN_CACHE.popitem(last=False)


def _unavailable(reason: str) -> dict:
    return {"status": "unavailable", "found": False, "reason": reason}


def lookup_gstin(gstin: str) -> dict:
    """
    Look a GSTIN up. Returns a dict whose "status" is one of
        "found"        the registry has this GSTIN (see the other keys)
        "not_found"    the registry answered, and the GSTIN does not exist
        "unavailable"  no answer (no key, timeout, network or server error): NOTHING was verified

    Order: fictional demo records → cache → Appyflow (a wrapper over the public GSTN search).
    """
    if gstin in DEMO_REGISTRY:
        return DEMO_REGISTRY[gstin]
    cached = _cache_get(gstin)
    if cached:
        return cached
    if not APPYFLOW_KEY:
        return _unavailable("no GSTN provider configured (APPYFLOW_KEY is not set)")

    try:
        response = requests.get("https://appyflow.in/api/verifyGST",
                                params={"gstNo": gstin, "key_secret": APPYFLOW_KEY}, timeout=10)
    except requests.exceptions.Timeout:
        return _unavailable("GSTN provider timed out")
    except requests.exceptions.RequestException as e:
        return _unavailable(f"network error: {type(e).__name__}")

    if response.status_code == 404:
        result = {"status": "not_found", "found": False, "reason": "GSTIN not registered in GSTN"}
        _cache_put(gstin, result, CACHE_TTL_NOT_FOUND)
        return result
    if response.status_code != 200:
        return _unavailable(f"GSTN provider returned HTTP {response.status_code}")

    try:
        data = response.json()
    except ValueError:
        return _unavailable("GSTN provider returned an unreadable answer")

    if data.get("error"):
        message = str(data.get("message", ""))
        if re.search(r"invalid|not found|no record|does not exist", message, re.I):
            result = {"status": "not_found", "found": False, "reason": f"GSTN: {message}"}
            _cache_put(gstin, result, CACHE_TTL_NOT_FOUND)
            return result
        return _unavailable(f"GSTN provider error: {message or 'unknown'}")

    info = data.get("taxpayerInfo") or {}
    addr = (info.get("pradr") or {}).get("addr") or {}
    result = {
        "status":         "found",
        "found":          True,
        "gstin":          gstin,
        "legal_name":     info.get("lgnm", ""),
        "trade_name":     info.get("tradeNam", ""),
        "gst_status":     info.get("sts", "Unknown"),
        "business_type":  info.get("ctb", ""),
        "taxpayer_type":  info.get("dty", "Unknown"),
        "pincode":        str(addr.get("pncd", "") or ""),
        "state":          addr.get("stcd", ""),
        "address":        ", ".join(str(addr[k]) for k in ("bno", "flno", "bnm", "st", "loc", "dst") if addr.get(k)),
        "nature":         info.get("nba", []),
    }
    _cache_put(gstin, result, CACHE_TTL_FOUND)
    return result


# ─── RESULT STRUCTURE ─────────────────────────────────────────────────────────

@dataclass
class GSTNCheckResult:
    outcome:               str                    # one of the OUTCOMES at the top of this file
    reason:                str                    # one sentence a reviewer can read
    gstin:                 Optional[str] = None
    legal_name:            str = ""
    trade_name:            str = ""
    taxpayer_type:         str = "Unknown"
    name_match:            Optional[dict] = None
    premises:              str = UNKNOWN          # SAME_ADDRESS | SAME_PINCODE | DIFFERENT | UNKNOWN
    lpg_heavy_business:    bool = False
    jurisdiction_mismatch: bool = False
    is_demo_record:        bool = False
    checked_at:            str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return dict(self.__dict__)


# ─── MAIN CHECK ───────────────────────────────────────────────────────────────

def validate_booking_address(
    consumer_name:   str,
    address_line:    str,
    pincode:         str,
    state_code:      str,
    booking_type:    str,
    consumer_gstin:  Optional[str] = None,
) -> GSTNCheckResult:
    """
    Registry check for one booking. Reports an outcome; decides nothing.
    ┌────────────────────────────────────────────────────────────────────────┐
    │ No GSTIN given                      → NO_GSTIN                         │
    │ Bad format or check digit           → INVALID_GSTIN  (no API call)     │
    │ Registry unreachable                → UNAVAILABLE    (nothing verified)│
    │ Registry: no such GSTIN             → NOT_REGISTERED                   │
    │ Suspended / cancelled               → INACTIVE                         │
    │ Name matches neither legal nor trade→ NAME_MISMATCH                    │
    │ Name partly matches                 → NAME_UNCERTAIN                   │
    │ Active and name matches             → VERIFIED (+ premises, business   │
    │                                       type, jurisdiction evidence)     │
    └────────────────────────────────────────────────────────────────────────┘
    `state_code` and `booking_type` are part of the booking record; the checks here do not need them.
    """
    if not (consumer_gstin and consumer_gstin.strip()):
        return GSTNCheckResult(NO_GSTIN, "No GSTIN was declared with this booking.")

    gstin = consumer_gstin.strip().upper()
    if not is_valid_gstin_format(gstin):
        return GSTNCheckResult(
            INVALID_GSTIN, gstin=gstin,
            reason=f"'{gstin}' is not a valid GSTIN (wrong format or check digit). No registry call was made.")

    record = lookup_gstin(gstin)
    status = record.get("status")
    if status not in ("found", "not_found", "unavailable"):
        status = "found" if record.get("found") else "not_found"

    if status == "unavailable":
        return GSTNCheckResult(
            UNAVAILABLE, gstin=gstin,
            reason=f"GSTIN {gstin} could not be verified: {record.get('reason', 'registry unreachable')}. "
                   f"Verify again when the registry answers.")
    if status != "found":
        return GSTNCheckResult(
            NOT_REGISTERED, gstin=gstin,
            reason=f"GSTIN {gstin} has a valid format but the registry has no such registration.")

    legal, trade = record.get("legal_name", ""), record.get("trade_name", "")
    common = dict(
        gstin                 = gstin,
        legal_name            = legal,
        trade_name            = trade,
        taxpayer_type         = record.get("taxpayer_type", "Unknown"),
        is_demo_record        = bool(record.get("demo")),
        lpg_heavy_business    = looks_lpg_heavy(legal, trade),
        jurisdiction_mismatch = jurisdiction_mismatch(gstin, pincode),
        premises              = compare_premises(address_line, pincode,
                                                 record.get("address", ""), record.get("pincode", "")),
    )

    gst_status = str(record.get("gst_status", "Unknown"))
    if gst_status.lower() not in ("active", "act"):
        return GSTNCheckResult(
            INACTIVE, **common,
            reason=f"GSTIN {gstin} belongs to '{legal}' but its registration status is '{gst_status}'.")

    match = check_name_gstin_match(consumer_name, legal, trade)
    common["name_match"] = match
    if match["level"] == "MISMATCH":
        return GSTNCheckResult(
            NAME_MISMATCH, **common,
            reason=f"Booking name '{consumer_name}' matches neither the legal name '{legal}' nor the trade "
                   f"name '{trade or 'N/A'}' of GSTIN {gstin}.")
    if match["level"] != "MATCH":
        return GSTNCheckResult(
            NAME_UNCERTAIN, **common,
            reason=f"Booking name '{consumer_name}' only partly matches '{legal}'"
                   f"{f' / {trade}' if trade else ''} (similarity {match['score']:.2f}).")

    note = ""
    if common["jurisdiction_mismatch"]:
        note = (f" JURISDICTION FLAG: registered in {get_state_from_gstin(gstin)}, "
                f"but delivery pincode {pincode} is in another state.")
    return GSTNCheckResult(
        VERIFIED, **common,
        reason=f"GSTIN {gstin} is active and belongs to '{legal}' (matched on {match['matched_on']}).{note}")


# ─── OFFLINE DEMO ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    restaurant = DEMO_GSTINS["Annapurna Family Restaurant"]
    cases = [
        ("Household, no GSTIN",                   "Meena Patil", "Flat 3, Sector 5, Vashi", "400703", "domestic", None),
        ("Restaurant books a domestic cylinder",  "Annapurna Restaurant", "Shop 4, Sector 17 Market, Vashi", "400703", "domestic", restaurant),
        ("Same restaurant, commercial cylinder",  "Annapurna Restaurant", "Shop 4, Sector 17 Market, Vashi", "400703", "commercial", restaurant),
        ("Someone else's GSTIN",                  "Rahul Verma", "Shop 9, Sector 2, Airoli", "400708", "commercial", restaurant),
        ("Wrong check digit",                     "Annapurna Restaurant", "Shop 4, Vashi", "400703", "commercial", restaurant[:14] + "0"),
        ("Suspended registration",                "Konkan Sweets", "Shop 11, Station Road, Nerul", "400706", "commercial", DEMO_GSTINS["Konkan Sweets and Farsan"]),
    ]
    print("\nGasSetu registry check — offline demo with fictional records\n" + "=" * 62)
    for label, name, address, pin, kind, gstin in cases:
        r = validate_booking_address(name, address, pin, "27", kind, gstin)
        print(f"\n{label}\n  outcome : {r.outcome}  (premises: {r.premises}, LPG-heavy: {r.lpg_heavy_business})\n  reason  : {r.reason}")
    print()
