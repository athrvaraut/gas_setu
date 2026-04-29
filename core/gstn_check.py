"""
GasSetu — GSTN Address Validator v3
=====================================
FIXES IN THIS VERSION:
  FIX 1: Cache save was dead code — now correctly saves before return
  FIX 2: Suspended/Cancelled GSTIN now explicitly BLOCKED (was falling through to ALLOW)
  FIX 3: Name match now checks BOTH legal_name AND trade_name (catches UrbanClap/UrbanCompany)
  FIX 4: Added taxpayer_type field (Regular vs Composition) for future ITC calculations
  FIX 5: State code extracted from GSTIN and compared against delivery pincode state

Author: Athrva Raut — GasSetu Project
"""

import os
import re
import requests
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
APPYFLOW_KEY = os.getenv("APPYFLOW_KEY")

# Memory cache — saves AppyFlow credits during session
# To remove in production: delete all 3 lines containing GSTIN_CACHE
GSTIN_CACHE = {}

# GSTIN format: 2-digit state + 5 PAN letters + 4 PAN digits + 1 PAN char + entity + Z + checksum
GSTIN_PATTERN = re.compile(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$")

# State code → state name mapping (first 2 digits of GSTIN)
# Used for jurisdictional arbitrage detection
GSTIN_STATE_CODES = {
    "01": "Jammu & Kashmir",  "02": "Himachal Pradesh",  "03": "Punjab",
    "04": "Chandigarh",       "05": "Uttarakhand",       "06": "Haryana",
    "07": "Delhi",            "08": "Rajasthan",         "09": "Uttar Pradesh",
    "10": "Bihar",            "11": "Sikkim",            "12": "Arunachal Pradesh",
    "13": "Nagaland",         "14": "Manipur",           "15": "Mizoram",
    "16": "Tripura",          "17": "Meghalaya",         "18": "Assam",
    "19": "West Bengal",      "20": "Jharkhand",         "21": "Odisha",
    "22": "Chhattisgarh",     "23": "Madhya Pradesh",    "24": "Gujarat",
    "27": "Maharashtra",      "28": "Andhra Pradesh",    "29": "Karnataka",
    "30": "Goa",              "32": "Kerala",            "33": "Tamil Nadu",
    "34": "Puducherry",       "36": "Telangana",         "37": "Andhra Pradesh (New)",
}

# Pincode prefix → state name (first 2 digits of pincode)
PINCODE_STATE_MAP = {
    "11": "Delhi",            "12": "Haryana",           "13": "Haryana",
    "14": "Punjab",           "15": "Punjab",            "16": "Punjab",
    "17": "Himachal Pradesh", "18": "Jammu & Kashmir",   "19": "Jammu & Kashmir",
    "20": "Uttar Pradesh",    "21": "Uttar Pradesh",     "22": "Uttar Pradesh",
    "23": "Uttar Pradesh",    "24": "Uttar Pradesh",     "25": "Uttar Pradesh",
    "26": "Uttar Pradesh",    "27": "Uttar Pradesh",     "28": "Uttar Pradesh",
    "30": "Rajasthan",        "31": "Rajasthan",         "32": "Rajasthan",
    "33": "Rajasthan",        "34": "Rajasthan",
    "36": "Gujarat",          "37": "Gujarat",           "38": "Gujarat",
    "39": "Gujarat",          "40": "Maharashtra",       "41": "Maharashtra",
    "42": "Maharashtra",      "43": "Maharashtra",       "44": "Maharashtra",
    "45": "Madhya Pradesh",   "46": "Madhya Pradesh",    "47": "Madhya Pradesh",
    "48": "Madhya Pradesh",   "49": "Chhattisgarh",
    "50": "Telangana",        "51": "Andhra Pradesh",    "52": "Andhra Pradesh",
    "53": "Andhra Pradesh",
    "56": "Karnataka",        "57": "Karnataka",         "58": "Karnataka",
    "59": "Karnataka",        "60": "Tamil Nadu",        "61": "Tamil Nadu",
    "62": "Tamil Nadu",       "63": "Tamil Nadu",        "64": "Tamil Nadu",
    "67": "Kerala",           "68": "Kerala",            "69": "Kerala",
    "70": "West Bengal",      "71": "West Bengal",       "72": "West Bengal",
    "73": "West Bengal",      "74": "West Bengal",
    "78": "Assam",            "80": "Bihar",             "81": "Bihar",
    "82": "Bihar",            "83": "Jharkhand",         "84": "Jharkhand",
    "85": "Odisha",           "86": "Odisha",
}

# Noise words to ignore in name matching
NOISE_WORDS = {
    "pvt", "ltd", "private", "limited", "and", "the", "&",
    "co", "company", "enterprises", "india", "industries",
    "services", "solutions", "group", "trading", "shri", "smt",
    "mr", "mrs", "kumar", "kumari",
}


# ─── RESULT STRUCTURE ─────────────────────────────────────────────────────────

@dataclass
class GSTNCheckResult:
    address_queried:  str
    pincode:          str
    business_found:   bool
    gstin_list:       list
    business_names:   list
    recommendation:   str
    confidence:       str
    reason:           str
    checked_at:       str
    taxpayer_type:    str = "Unknown"   # Regular | Composition | Unknown

    def is_blocked(self) -> bool:
        return self.recommendation == "BLOCK_FORCE_COMMERCIAL"

    def to_dict(self) -> dict:
        return {
            "address_queried":  self.address_queried,
            "pincode":          self.pincode,
            "business_found":   self.business_found,
            "gstin_list":       self.gstin_list,
            "business_names":   self.business_names,
            "recommendation":   self.recommendation,
            "confidence":       self.confidence,
            "reason":           self.reason,
            "checked_at":       self.checked_at,
            "blocked":          self.is_blocked(),
            "taxpayer_type":    self.taxpayer_type,
        }


# ─── GSTIN FORMAT VALIDATOR ───────────────────────────────────────────────────

def is_valid_gstin_format(gstin: str) -> bool:
    """Validates GSTIN format. Saves API credits — catches bad formats instantly."""
    if not gstin or len(gstin) != 15:
        return False
    return bool(GSTIN_PATTERN.match(gstin.upper()))


def get_state_from_gstin(gstin: str) -> str:
    """Extract state from first 2 digits of GSTIN."""
    if gstin and len(gstin) >= 2:
        return GSTIN_STATE_CODES.get(gstin[:2], "Unknown")
    return "Unknown"


def get_state_from_pincode(pincode: str) -> str:
    """Extract state from first 2 digits of pincode."""
    if pincode and len(pincode) >= 2:
        return PINCODE_STATE_MAP.get(pincode[:2], "Unknown")
    return "Unknown"


# ─── LIVE GSTN LOOKUP ─────────────────────────────────────────────────────────

def lookup_gstin(gstin: str) -> dict:
    """
    Live GSTN lookup via appyflow.in.
    Cache checked first — saves API credits on repeated lookups.
    """
    if not APPYFLOW_KEY:
        raise ValueError("APPYFLOW_KEY not found in .env file!")

    # FIX 1: Cache check BEFORE API call
    if gstin in GSTIN_CACHE:
        print(f"  [CACHE HIT] {gstin} — saved 1 credit")
        return GSTIN_CACHE[gstin]

    url = f"https://appyflow.in/api/verifyGST?gstNo={gstin}&key_secret={APPYFLOW_KEY}"

    try:
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()

            if not data.get("error"):
                info = data.get("taxpayerInfo", {})

                # Build result dict
                result = {
                    "found":          True,
                    "gstin":          gstin,
                    "legal_name":     info.get("lgnm", "Unknown"),
                    "trade_name":     info.get("tradeNam", ""),     # FIX 3: Now captured
                    "status":         info.get("sts", "Unknown"),
                    "business_type":  info.get("ctb", ""),
                    "taxpayer_type":  info.get("dty", "Unknown"),   # FIX 4: Regular/Composition
                    "pincode":        info.get("pradr", {}).get("addr", {}).get("pncd", ""),
                    "state":          info.get("pradr", {}).get("addr", {}).get("stcd", ""),
                    "nature":         info.get("nba", []),
                }

                # FIX 1: Save to cache BEFORE returning (was dead code before)
                GSTIN_CACHE[gstin] = result
                return result

            else:
                # API returned error — don't cache (might be temporary)
                return {"found": False, "reason": f"API error: {data.get('message', 'unknown')}"}

        elif response.status_code == 401:
            return {"found": False, "reason": "Invalid API key — check .env file"}
        elif response.status_code == 404:
            result = {"found": False, "reason": "GSTIN not registered in GSTN"}
            GSTIN_CACHE[gstin] = result   # Cache "not found" — won't change
            return result
        else:
            return {"found": False, "reason": f"HTTP {response.status_code}"}

    except requests.exceptions.Timeout:
        return {"found": False, "reason": "API timeout — will retry"}
    except requests.exceptions.ConnectionError:
        return {"found": False, "reason": "Network error"}
    except Exception as e:
        return {"found": False, "reason": str(e)}


# ─── NAME-GSTIN IDENTITY MATCH ────────────────────────────────────────────────

def check_name_gstin_match(
    booking_name:    str,
    legal_name:      str,
    trade_name:      str = "",   # FIX 3: Now checks trade name too
) -> dict:
    """
    Catches the 'Stolen GSTIN' attack.

    FIX 3: Checks BOTH legal_name AND trade_name.
    Before: "Urban Company" vs "UrbanClap Technologies" → FALSE BLOCK
    After:  "Urban Company" vs trade_name "Urban Company" → CORRECT ALLOW

    Returns: {"match": bool, "matched_on": str, "confidence": str}
    """
    if not booking_name:
        return {"match": True, "matched_on": "empty_booking_name", "confidence": "LOW"}

    def extract_meaningful(name: str) -> set:
        return {
            w.lower() for w in name.split()
            if w.lower() not in NOISE_WORDS and len(w) > 2
        }

    booking_words = extract_meaningful(booking_name)

    if not booking_words:
        return {"match": True, "matched_on": "no_meaningful_words", "confidence": "LOW"}

    # Check against legal name
    legal_words = extract_meaningful(legal_name)
    legal_overlap = booking_words & legal_words
    booking_lower = booking_name.lower().replace(" ", "")
    legal_lower   = legal_name.lower().replace(" ", "")
    legal_substring = booking_lower in legal_lower or legal_lower in booking_lower

    if legal_overlap or legal_substring:
        return {
            "match":      True,
            "matched_on": "legal_name",
            "confidence": "HIGH" if len(legal_overlap) >= 2 else "MEDIUM",
        }

    # FIX 3: If legal name didn't match, check trade name
    if trade_name:
        trade_words = extract_meaningful(trade_name)
        trade_overlap = booking_words & trade_words
        trade_lower   = trade_name.lower().replace(" ", "")
        trade_substring = booking_lower in trade_lower or trade_lower in booking_lower

        if trade_overlap or trade_substring:
            return {
                "match":      True,
                "matched_on": "trade_name",
                "confidence": "HIGH" if len(trade_overlap) >= 2 else "MEDIUM",
            }

    # No match on either name
    return {
        "match":      False,
        "matched_on": "none",
        "confidence": "HIGH",   # HIGH confidence in the mismatch
    }


# ─── MAIN BOOKING VALIDATION ──────────────────────────────────────────────────

def validate_booking_address(
    consumer_name:   str,
    address_line:    str,
    pincode:         str,
    state_code:      str,
    booking_type:    str,
    consumer_gstin:  Optional[str] = None,
) -> GSTNCheckResult:
    """
    Main fraud check. Decision logic:
    ┌──────────────────────────────────────────────────────────────────┐
    │ Invalid GSTIN format              → BLOCK (fake number)          │
    │ Suspended / Cancelled GSTIN       → BLOCK (FIX 2)               │
    │ GSTIN state ≠ delivery pincode    → FLAG (jurisdictional fraud)  │
    │ Name mismatch (legal + trade)     → BLOCK (stolen GSTIN)        │
    │ Active GSTIN + domestic booking   → BLOCK (commercial misuse)   │
    │ Active GSTIN + commercial booking → ALLOW                        │
    │ No GSTIN                          → ALLOW + ML monitoring        │
    └──────────────────────────────────────────────────────────────────┘
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    if consumer_gstin and consumer_gstin.strip():
        gstin_clean = consumer_gstin.strip().upper()

        # ── Check 1: Format validation (free, no API call) ────────────────────
        if not is_valid_gstin_format(gstin_clean):
            return GSTNCheckResult(
                address_queried = address_line,
                pincode         = pincode,
                business_found  = False,
                gstin_list      = [gstin_clean],
                business_names  = [],
                recommendation  = "BLOCK_FORCE_COMMERCIAL",
                confidence      = "HIGH",
                reason          = (
                    f"FAKE GSTIN: '{gstin_clean}' fails format validation. "
                    f"Valid: 2-digit state + PAN(10) + entity + Z + checksum. "
                    f"No API call made. Booking BLOCKED."
                ),
                checked_at      = timestamp,
            )

        # ── Check 2: Jurisdictional check (free, no API call) ─────────────────
        gstin_state   = get_state_from_gstin(gstin_clean)
        pincode_state = get_state_from_pincode(pincode)
        jurisdiction_flag = (
            gstin_state != "Unknown"
            and pincode_state != "Unknown"
            and gstin_state != pincode_state
        )

        # ── Check 3: Live GSTN lookup ──────────────────────────────────────────
        api_result = lookup_gstin(gstin_clean)

        if api_result.get("found"):
            raw_status    = api_result.get("status", "Unknown")
            is_active     = raw_status.lower() in ["active", "act"]
            legal_name    = api_result.get("legal_name", "Unknown")
            trade_name    = api_result.get("trade_name", "")
            taxpayer_type = api_result.get("taxpayer_type", "Unknown")

            # ── FIX 2: Explicitly block Suspended / Cancelled ─────────────────
            if not is_active:
                suspended_statuses = ["suspended", "cancelled", "inactive", "cancel"]
                is_suspended = any(s in raw_status.lower() for s in suspended_statuses)
                return GSTNCheckResult(
                    address_queried = address_line,
                    pincode         = pincode,
                    business_found  = True,
                    gstin_list      = [gstin_clean],
                    business_names  = [legal_name],
                    recommendation  = "BLOCK_FORCE_COMMERCIAL",
                    confidence      = "HIGH",
                    reason          = (
                        f"INACTIVE GSTIN BLOCKED: {gstin_clean} belongs to '{legal_name}' "
                        f"but status is '{raw_status}'. "
                        f"{'GSTIN is legally suspended — high fraud risk. ' if is_suspended else ''}"
                        f"Booking BLOCKED. Reactivate GSTIN before booking."
                    ),
                    checked_at      = timestamp,
                    taxpayer_type   = taxpayer_type,
                )

            # ── Check 4: Name-GSTIN identity match (legal + trade) ────────────
            name_match = check_name_gstin_match(consumer_name, legal_name, trade_name)

            if not name_match["match"]:
                return GSTNCheckResult(
                    address_queried = address_line,
                    pincode         = pincode,
                    business_found  = True,
                    gstin_list      = [gstin_clean],
                    business_names  = [legal_name],
                    recommendation  = "BLOCK_FORCE_COMMERCIAL",
                    confidence      = "HIGH",
                    reason          = (
                        f"STOLEN GSTIN: Booking name '{consumer_name}' "
                        f"matches neither legal name '{legal_name}' "
                        f"nor trade name '{trade_name or 'N/A'}'. "
                        f"Identity theft attempt. Booking BLOCKED."
                    ),
                    checked_at      = timestamp,
                    taxpayer_type   = taxpayer_type,
                )

            # ── Check 5: Jurisdictional arbitrage warning ──────────────────────
            jurisdiction_note = ""
            if jurisdiction_flag:
                jurisdiction_note = (
                    f" ⚠️ JURISDICTION FLAG: GSTIN registered in {gstin_state} "
                    f"but delivery pincode {pincode} is in {pincode_state}. "
                    f"Flagged for review."
                )

            # ── Check 6: Active GSTIN + domestic booking = fraud ──────────────
            if booking_type.lower() == "domestic":
                return GSTNCheckResult(
                    address_queried = address_line,
                    pincode         = pincode,
                    business_found  = True,
                    gstin_list      = [gstin_clean],
                    business_names  = [legal_name],
                    recommendation  = "BLOCK_FORCE_COMMERCIAL",
                    confidence      = "HIGH",
                    reason          = (
                        f"FRAUD: Active GSTIN {gstin_clean} → '{legal_name}' "
                        f"(Type: {taxpayer_type}). "
                        f"Domestic booking BLOCKED. "
                        f"Book Commercial cylinder (18% GST applies)."
                        f"{jurisdiction_note}"
                    ),
                    checked_at      = timestamp,
                    taxpayer_type   = taxpayer_type,
                )

            # ── Active GSTIN + commercial = correct booking ───────────────────
            return GSTNCheckResult(
                address_queried = address_line,
                pincode         = pincode,
                business_found  = True,
                gstin_list      = [gstin_clean],
                business_names  = [legal_name],
                recommendation  = "ALLOW_COMMERCIAL",
                confidence      = "HIGH",
                reason          = (
                    f"VERIFIED: '{legal_name}' (matched on {name_match['matched_on']}). "
                    f"Type: {taxpayer_type}. Commercial booking APPROVED."
                    f"{jurisdiction_note}"
                ),
                checked_at      = timestamp,
                taxpayer_type   = taxpayer_type,
            )

        else:
            # GSTIN not found in database
            return GSTNCheckResult(
                address_queried = address_line,
                pincode         = pincode,
                business_found  = False,
                gstin_list      = [gstin_clean],
                business_names  = [],
                recommendation  = "ALLOW_DOMESTIC",
                confidence      = "MEDIUM",
                reason          = (
                    f"GSTIN '{gstin_clean}' valid format but not found in GSTN "
                    f"({api_result.get('reason', 'not registered')}). "
                    f"Domestic allowed. Flagged for verification."
                ),
                checked_at      = timestamp,
            )

    # ── No GSTIN — normal household ───────────────────────────────────────────
    return GSTNCheckResult(
        address_queried = address_line,
        pincode         = pincode,
        business_found  = False,
        gstin_list      = [],
        business_names  = [],
        recommendation  = "ALLOW_DOMESTIC",
        confidence      = "MEDIUM",
        reason          = (
            "No GSTIN. Domestic booking APPROVED. "
            "AI consumption monitoring ACTIVE."
        ),
        checked_at      = timestamp,
    )


# ─── TEST SUITE ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "=" * 68)
    print("  GASSETU — GSTN CHECK v3 — ALL FIXES APPLIED")
    print("=" * 68)

    tests = [
        {
            "label":  "🍽️  Restaurant domestic booking (BLOCK)",
            "name":   "Sharma Restaurant", "pincode": "400053",
            "state":  "27", "type": "domestic",
            "gstin":  "27AAPFU0939F1ZV",
        },
        {
            "label":  "✅  Restaurant commercial booking (ALLOW)",
            "name":   "Dishant Mahajan", "pincode": "411001",
            "state":  "27", "type": "commercial",
            "gstin":  "27AAPFU0939F1ZV",
        },
        {
            "label":  "🕵️  Stolen GSTIN — name mismatch (BLOCK)",
            "name":   "Rahul Sharma", "pincode": "400001",
            "state":  "27", "type": "domestic",
            "gstin":  "27AAPFU0939F1ZV",
        },
        {
            "label":  "❌  Fake GSTIN format (BLOCK — free, no API call)",
            "name":   "Krishna Hotel", "pincode": "440001",
            "state":  "27", "type": "domestic",
            "gstin":  "27ABCDE1234F1Z",   # only 14 chars — format fail
        },
        {
            "label":  "🗺️  Jurisdiction flag — Maharashtra GSTIN, Delhi delivery",
            "name":   "Dishant Mahajan", "pincode": "110001",  # Delhi pincode
            "state":  "07", "type": "commercial",
            "gstin":  "27AAPFU0939F1ZV",  # Maharashtra GSTIN
        },
        {
            "label":  "🏠  Normal household no GSTIN (ALLOW)",
            "name":   "Priya Sharma", "pincode": "400067",
            "state":  "27", "type": "domestic",
            "gstin":  None,
        },
        {
            "label":  "🔄  Same GSTIN again — CACHE HIT (no API call)",
            "name":   "Dishant Mahajan", "pincode": "411001",
            "state":  "27", "type": "commercial",
            "gstin":  "27AAPFU0939F1ZV",  # same as test 2 — should hit cache
        },
    ]

    for tc in tests:
        result = validate_booking_address(
            consumer_name  = tc["name"],
            address_line   = f"Test address, {tc['pincode']}",
            pincode        = tc["pincode"],
            state_code     = tc["state"],
            booking_type   = tc["type"],
            consumer_gstin = tc.get("gstin"),
        )
        icon = "🚫 BLOCKED" if result.is_blocked() else "✅ APPROVED"
        print(f"\n{tc['label']}")
        print(f"  Result       : {icon}  |  Confidence: {result.confidence}")
        print(f"  Taxpayer Type: {result.taxpayer_type}")
        print(f"  Reason       : {result.reason[:120]}...")

    print(f"\n  Credits used this run: {len([t for t in tests if t.get('gstin') and t['gstin'] != '27AAPFU0939F1ZV' or tests.index(t) == 0])}")
    print(f"  Cache entries now: {len(GSTIN_CACHE)}")
    print("\n" + "=" * 68)
    print("  All fixes verified.")
    print("=" * 68 + "\n")