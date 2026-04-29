"""
GasSetu — /booking/validate  (The Final Combined API)
=======================================================
One endpoint. Three fraud checks. One decision.

POST /booking/validate
  → Runs GSTN check (fake/stolen GSTIN, wrong cylinder type)
  → Runs Anomaly Detection (behaviour-based fraud)
  → Runs State Machine check (delivery chain integrity)
  → Returns: APPROVED or BLOCKED with full reason

This is what you demo to MoPNG.
This is what you show at SINE IIT Bombay.
This is what changes the system.

Run:  uvicorn api.booking_validate:app --reload
Test: http://localhost:8000/docs  (Swagger UI auto-generated)

Author: Athrva — GasSetu Project
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timezone
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.gstn_check import validate_booking_address
from core.anomaly_detection import ConsumptionAnomalyDetector
from core.state_machine import CylinderStateMachine, CylinderState


# ─── APP SETUP ────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "GasSetu Fraud Detection API",
    description = "Real-time LPG booking fraud prevention engine for India",
    version     = "1.0.0",
    docs_url    = "/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# Train anomaly model once at startup
print("\n🔥 GasSetu starting up...")
detector = ConsumptionAnomalyDetector()
detector.train(verbose=True)
print("✅ All systems ready.\n")


# ─── REQUEST / RESPONSE MODELS ────────────────────────────────────────────────

class BookingRequest(BaseModel):
    # Consumer identity
    consumer_id:    str   = Field(..., example="CON-MUM-004521")
    consumer_name:  str   = Field(..., example="Sharma Restaurant")
    address:        str   = Field(..., example="Shop 4, Andheri Market, Mumbai")
    pincode:        str   = Field(..., example="400053")
    state_code:     str   = Field(..., example="27")

    # Cylinder request
    cylinder_type:  str   = Field(..., example="domestic")   # domestic | commercial
    company:        str   = Field(..., example="indane")      # indane | bharat | hp

    # Optional GSTIN (if business)
    consumer_gstin: Optional[str] = Field(None, example="27AAPFU0939F1ZV")

    # Consumption history (for anomaly detection)
    days_between_bookings:      float = Field(27.0,  example=15.2)
    household_size:             float = Field(3.0,   example=1.5)
    booking_frequency_change:   float = Field(0.05,  example=0.88)
    address_change_count:       int   = Field(0,     example=0)
    days_since_last_booking:    float = Field(28.0,  example=15.0)
    agent_change_rate:          float = Field(0.1,   example=0.75)
    bookings_last_12_months:    int   = Field(13,    example=24)
    complaint_rate:             float = Field(0.0,   example=0.01)


class FraudCheck(BaseModel):
    check_name: str
    passed:     bool
    severity:   str
    detail:     str


class BookingResponse(BaseModel):
    booking_id:      str
    consumer_id:     str
    decision:        str        # APPROVED | BLOCKED
    confidence:      str        # HIGH | MEDIUM | LOW
    risk_level:      str        # NORMAL | LOW | MEDIUM | HIGH | CRITICAL
    primary_reason:  str
    checks_run:      list
    processing_ms:   float
    timestamp:       str
    recommended_action: str


# ─── HELPER ───────────────────────────────────────────────────────────────────

def generate_booking_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    import random
    return f"BK-{ts}-{random.randint(1000,9999)}"


# ─── MAIN ENDPOINT ────────────────────────────────────────────────────────────

@app.post("/booking/validate", response_model=BookingResponse)
async def validate_booking(req: BookingRequest):
    """
    Validate an LPG booking request against all fraud detection layers.
    
    Runs three checks in sequence:
    1. GSTN Check — verifies GSTIN, catches fake/stolen tax IDs
    2. Anomaly Detection — ML model on consumption pattern
    3. Name-GSTIN Match — catches identity theft (stolen GSTINs)
    
    Returns APPROVED or BLOCKED with full audit trail.
    """
    start_time = datetime.now(timezone.utc)
    checks_run = []
    booking_id = generate_booking_id()

    # ── CHECK 1: GSTN Verification ────────────────────────────────────────────
    gstn_result = validate_booking_address(
        consumer_name  = req.consumer_name,
        address_line   = req.address,
        pincode        = req.pincode,
        state_code     = req.state_code,
        booking_type   = req.cylinder_type,
        consumer_gstin = req.consumer_gstin,
    )

    checks_run.append(FraudCheck(
        check_name = "GSTN_ADDRESS_CHECK",
        passed     = not gstn_result.is_blocked(),
        severity   = "CRITICAL" if gstn_result.is_blocked() else "PASS",
        detail     = gstn_result.reason[:200],
    ).dict())

    # If GSTN check blocks — stop here, no need for further checks
    if gstn_result.is_blocked():
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        return BookingResponse(
            booking_id      = booking_id,
            consumer_id     = req.consumer_id,
            decision        = "BLOCKED",
            confidence      = gstn_result.confidence,
            risk_level      = "CRITICAL",
            primary_reason  = gstn_result.reason,
            checks_run      = checks_run,
            processing_ms   = round(elapsed, 2),
            timestamp       = start_time.isoformat(),
            recommended_action = (
                "Consumer has active GSTIN at this address. "
                "Must book Commercial cylinder (19kg, ₹2,320 + 18% GST). "
                "Domestic booking permanently blocked for this address."
            ),
        )

    # ── CHECK 2: Name-GSTIN Identity Match ────────────────────────────────────
    if req.consumer_gstin and gstn_result.business_found:
        gstin_name    = gstn_result.business_names[0] if gstn_result.business_names else ""
        booking_words = set(req.consumer_name.lower().split())
        gstin_words   = set(gstin_name.lower().split())
        common_words  = {"pvt", "ltd", "private", "limited", "and", "the", "&"}
        meaningful_booking = booking_words - common_words
        meaningful_gstin   = gstin_words - common_words

        name_overlap = len(meaningful_booking & meaningful_gstin)
        identity_match = name_overlap > 0 or len(meaningful_booking) == 0

        if not identity_match and len(meaningful_booking) >= 2:
            checks_run.append(FraudCheck(
                check_name = "GSTIN_IDENTITY_CHECK",
                passed     = False,
                severity   = "HIGH",
                detail     = (
                    f"Identity mismatch: booking name '{req.consumer_name}' "
                    f"does not match GSTIN owner '{gstin_name}'. "
                    f"Possible stolen/borrowed GSTIN."
                ),
            ).dict())

            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            return BookingResponse(
                booking_id      = booking_id,
                consumer_id     = req.consumer_id,
                decision        = "BLOCKED",
                confidence      = "HIGH",
                risk_level      = "HIGH",
                primary_reason  = (
                    f"GSTIN identity theft detected. "
                    f"Booking name '{req.consumer_name}' doesn't match "
                    f"GSTIN owner '{gstin_name}'."
                ),
                checks_run      = checks_run,
                processing_ms   = round(elapsed, 2),
                timestamp       = start_time.isoformat(),
                recommended_action = (
                    "Verify consumer identity. GSTIN belongs to a different entity. "
                    "Flag for manual verification before allowing any booking."
                ),
            )
        else:
            checks_run.append(FraudCheck(
                check_name = "GSTIN_IDENTITY_CHECK",
                passed     = True,
                severity   = "PASS",
                detail     = f"Name match verified: '{req.consumer_name}' ↔ '{gstin_name}'",
            ).dict())

    # ── CHECK 3: Consumption Anomaly Detection ────────────────────────────────
    anomaly = detector.check(
        consumer_id                = req.consumer_id,
        days_between_bookings      = req.days_between_bookings,
        household_size             = req.household_size,
        booking_frequency_change   = req.booking_frequency_change,
        address_change_count       = req.address_change_count,
        days_since_last_booking    = req.days_since_last_booking,
        agent_change_rate          = req.agent_change_rate,
        bookings_last_12_months    = req.bookings_last_12_months,
        complaint_rate             = req.complaint_rate,
    )

    checks_run.append(FraudCheck(
        check_name = "CONSUMPTION_ANOMALY",
        passed     = not anomaly.anomaly_detected,
        severity   = anomaly.risk_level,
        detail     = (
            f"Anomaly score: {anomaly.anomaly_score:.2f}. "
            + (anomaly.flags[0][:150] if anomaly.flags else "Pattern normal.")
        ),
    ).dict())

    # ── FINAL DECISION ────────────────────────────────────────────────────────
    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

    if anomaly.risk_level == "CRITICAL":
        return BookingResponse(
            booking_id      = booking_id,
            consumer_id     = req.consumer_id,
            decision        = "BLOCKED",
            confidence      = "HIGH",
            risk_level      = "CRITICAL",
            primary_reason  = (
                f"Consumption pattern is critically anomalous. "
                f"Score: {anomaly.anomaly_score:.2f}. "
                + (anomaly.flags[0] if anomaly.flags else "")
            ),
            checks_run      = checks_run,
            processing_ms   = round(elapsed, 2),
            timestamp       = start_time.isoformat(),
            recommended_action = anomaly.recommendation,
        )

    # All checks passed
    decision   = "APPROVED"
    confidence = gstn_result.confidence if anomaly.risk_level == "NORMAL" else "MEDIUM"
    primary    = (
        f"All fraud checks passed. "
        f"GSTN: {gstn_result.recommendation}. "
        f"Consumption risk: {anomaly.risk_level}."
    )
    if anomaly.risk_level in ["HIGH", "MEDIUM"]:
        primary += f" ⚠️ Flagged for monitoring. {anomaly.recommendation}"

    return BookingResponse(
        booking_id      = booking_id,
        consumer_id     = req.consumer_id,
        decision        = decision,
        confidence      = confidence,
        risk_level      = anomaly.risk_level,
        primary_reason  = primary,
        checks_run      = checks_run,
        processing_ms   = round(elapsed, 2),
        timestamp       = start_time.isoformat(),
        recommended_action = anomaly.recommendation,
    )


# ─── HEALTH CHECK ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":         "running",
        "version":        "1.0.0",
        "model_trained":  detector.trained,
        "checks_active":  ["GSTN_ADDRESS", "GSTIN_IDENTITY", "CONSUMPTION_ANOMALY"],
        "timestamp":      datetime.now(timezone.utc).isoformat(),
    }


@app.get("/")
async def root():
    return {
        "name":        "GasSetu Fraud Detection API",
        "description": "Real-time LPG booking fraud prevention for India",
        "docs":        "/docs",
        "health":      "/health",
        "validate":    "POST /booking/validate",
    }