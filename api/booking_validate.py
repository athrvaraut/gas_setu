"""
GasSetu — /booking/validate  (the combined API)
================================================
One endpoint. Two evidence layers. One outcome that never refuses a household's refill.

POST /booking/validate
  → Registry check      (core/gstn_check.py)        GSTIN valid? active? whose? registered where?
  → Consumption risk    (core/anomaly_detection.py) booking history compared with similar households
  → Decision policy     (core/decision_policy.py)   CLEAR | MONITOR | REVIEW | HOLD_FOR_REVIEW | CONVERT_TO_COMMERCIAL

The booking history must come from the system of record (the oil company's booking database):
send the refill dates and the server derives the model inputs itself. The delivery state machine
and the distributor reconciliation are separate modules (see ROADMAP.md).

Run:  uvicorn api.booking_validate:app --reload
Test: http://localhost:8000/docs  (Swagger UI)

Author: Athrva — GasSetu Project
"""

import os
import sys
import time
import uuid
from datetime import date, datetime, timezone
from typing import Literal, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from api.demo_profiles import demo_profiles
from core import gstn_check
from core.anomaly_detection import ConsumptionRiskModel
from core.decision_policy import decide
from core.features import features_from_refill_dates
from core.gstn_check import validate_booking_address

VERSION = "2.0.0"


# ─── APP SETUP ────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "GasSetu Risk-Scoring API",
    description = "Scores LPG bookings for diversion risk and routes suspicious cases to human review. "
                  "It never refuses a household's refill.",
    version     = VERSION,
    docs_url    = "/docs",
)

STATIC_DIR = os.path.join(ROOT, "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def serve_dashboard():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# The dashboard is served from this same origin, so no cross-origin access is needed by default.
# To let another site call the API, list its origin(s) in ALLOWED_ORIGINS (comma-separated).
_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if _origins:
    app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_methods=["GET", "POST"], allow_headers=["*"])

# Weights, peer statistics and percentile tables are read from core/consumption_model.json
model = ConsumptionRiskModel.load()


# ─── REQUEST / RESPONSE MODELS ────────────────────────────────────────────────

class BookingRequest(BaseModel):
    # Who and where
    consumer_id:    str = Field(..., examples=["CON-MUM-004521"])
    consumer_name:  str = Field(..., examples=["Meena Devi"])
    address:        str = Field(..., examples=["Flat 2B, Nehru Nagar, Kurla East, Mumbai"])
    pincode:        str = Field(..., pattern=r"^[1-9][0-9]{5}$", examples=["400024"])
    state_code:     str = Field("", examples=["27"])
    cylinder_type:  Literal["domestic", "commercial"] = "domestic"
    company:        str = Field("", examples=["indane"])
    consumer_gstin: Optional[str] = Field(None, description="Only if the customer declared one")

    # Connection record (from KYC)
    household_size:     int  = Field(4, ge=1, le=30)
    is_pmuy:            bool = False
    is_rural:           bool = False
    mobile_changes_12m: int  = Field(0, ge=0, description="Registered-mobile changes in the last 12 months")

    # Booking history, option A: the refill dates of the last 24 months (preferred; the server derives the rest)
    refill_dates: Optional[list[date]] = None
    # Booking history, option B: the same numbers already derived by the system of record
    refills_12m:          Optional[int] = Field(None, ge=0)
    refills_6m:           Optional[int] = Field(None, ge=0)
    refills_prev_18m:     Optional[int] = Field(None, ge=0)
    earliest_day_refills: Optional[int] = Field(None, ge=0)
    dormant_days:         Optional[float] = Field(None, ge=0)


class BookingResponse(BaseModel):
    booking_id:          str
    consumer_id:         str
    decision:            str            # CLEAR | MONITOR | REVIEW | HOLD_FOR_REVIEW | CONVERT_TO_COMMERCIAL
    proceed:             bool           # does the booking go ahead now, as booked?
    risk_level:          str            # consumption tier: NORMAL | LOW | MEDIUM | HIGH | NOT_SCORED
    primary_reason:      str
    recommended_action:  str
    review_within_hours: Optional[int]
    reasons:             list
    checks_run:          list
    model_version:       str
    processing_ms:       float
    timestamp:           str


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def history_inputs(req: BookingRequest) -> Optional[dict]:
    """The model's history numbers: derived from the refill dates if given, else taken as supplied."""
    if req.refill_dates is not None:
        return features_from_refill_dates(req.refill_dates)
    if req.refills_12m is None:
        return None
    # Missing pieces default to a steady pattern, which adds no suspicion of its own
    r12 = req.refills_12m
    return {
        "refills_12m":          r12,
        "refills_6m":           req.refills_6m if req.refills_6m is not None else round(r12 / 2),
        "refills_prev_18m":     req.refills_prev_18m if req.refills_prev_18m is not None else round(r12 * 1.5),
        "earliest_day_refills": req.earliest_day_refills or 0,
        "dormant_days":         req.dormant_days if req.dormant_days is not None else round(365 / max(r12, 1)),
    }


# ─── MAIN ENDPOINT ────────────────────────────────────────────────────────────

@app.post("/booking/validate", response_model=BookingResponse)
def validate_booking(req: BookingRequest):
    """
    Score one LPG booking.

    1. Registry check: GSTIN format and check digit, live status, name, registered address.
    2. Consumption risk: the connection's history compared with similar households
       (domestic bookings with a history only).
    3. Decision policy: the strongest outcome any layer asks for. A domestic refill is never
       refused or held by an automatic check.
    """
    started = time.perf_counter()

    # ── CHECK 1: registry ─────────────────────────────────────────────────────
    registry_result = validate_booking_address(
        consumer_name  = req.consumer_name,
        address_line   = req.address,
        pincode        = req.pincode,
        state_code     = req.state_code,
        booking_type   = req.cylinder_type,
        consumer_gstin = req.consumer_gstin,
    )
    checks_run = [{
        "check_name": "REGISTRY",
        "outcome":    registry_result.outcome,
        "detail":     registry_result.reason + (" (fictional demo record)" if registry_result.is_demo_record else ""),
    }]

    # ── CHECK 2: consumption risk ─────────────────────────────────────────────
    consumption, history = None, history_inputs(req)
    if req.cylinder_type == "commercial":
        note = "Not scored: the model describes households, not businesses."
    elif history is None:
        note = "Not scored: no booking history was supplied."
    else:
        consumption = model.check(
            consumer_id        = req.consumer_id,
            household_size     = req.household_size,
            is_pmuy            = int(req.is_pmuy),
            is_rural           = int(req.is_rural),
            mobile_changes_12m = req.mobile_changes_12m,
            **history,
        )
        note = (f"Risk score above {consumption.percentile:g}% of connections. {consumption.refills_12m} refills in "
                f"12 months; similar households ({consumption.peer_group}) take about {consumption.peer_median:.0f}.")
    checks_run.append({
        "check_name": "CONSUMPTION",
        "outcome":    consumption.risk_level if consumption else "NOT_SCORED",
        "detail":     note,
    })

    # ── DECISION ──────────────────────────────────────────────────────────────
    outcome = decide(req.cylinder_type, req.consumer_name, registry_result, consumption)

    return BookingResponse(
        booking_id          = f"BK-{datetime.now(timezone.utc):%Y%m%d%H%M%S}-{uuid.uuid4().hex[:8]}",
        consumer_id         = req.consumer_id,
        decision            = outcome.decision,
        proceed             = outcome.proceed,
        risk_level          = consumption.risk_level if consumption else "NOT_SCORED",
        primary_reason      = outcome.primary_reason,
        recommended_action  = outcome.recommended_action,
        review_within_hours = outcome.review_within_hours,
        reasons             = [r.to_dict() for r in outcome.reasons],
        checks_run          = checks_run,
        model_version       = model.version,
        processing_ms       = round((time.perf_counter() - started) * 1000, 2),
        timestamp           = datetime.now(timezone.utc).isoformat(),
    )


# ─── DEMO DATA, HEALTH, INDEX ─────────────────────────────────────────────────

@app.get("/demo/profiles")
def get_demo_profiles():
    """Fictional consumers for the dashboard's quick scenarios."""
    return demo_profiles()


@app.get("/health")
def health():
    return {
        "status":            "running",
        "version":           VERSION,
        "model_trained":     model.trained,
        "model_version":     model.version,
        "model_data":        model.spec.get("notes", {}).get("data", ""),
        "registry_provider": "Appyflow (live GSTN lookup)" if gstn_check.APPYFLOW_KEY else "fictional demo records only",
        "checks_active":     ["REGISTRY", "CONSUMPTION", "DECISION_POLICY"],
        "timestamp":         datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api")
def api_index():
    return {
        "name":        "GasSetu Risk-Scoring API",
        "description": "Scores LPG bookings for diversion risk; suspicious cases go to human review",
        "docs":        "/docs",
        "health":      "/health",
        "validate":    "POST /booking/validate",
        "demo":        "/demo/profiles",
    }
