"""
GasSetu — Consumption Anomaly Detection
=========================================
ML model that catches fraud the GSTN check cannot:
  - Households selling their subsidised cylinder to restaurants
  - Distributors creating artificial shortages
  - Agents doing impossible delivery volumes

Uses Isolation Forest — unsupervised anomaly detection.
Learns what NORMAL looks like, flags everything that isn't.

No real data needed — trains on synthetic Indian LPG patterns.

Author: Athrva — GasSetu Project
"""

import numpy as np
import json
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ─── RESULT STRUCTURE ─────────────────────────────────────────────────────────

@dataclass
class AnomalyResult:
    consumer_id:      str
    anomaly_detected: bool
    anomaly_score:    float        # 0.0 to 1.0 — higher = more suspicious
    risk_level:       str          # NORMAL | LOW | MEDIUM | HIGH | CRITICAL
    flags:            list
    recommendation:   str
    checked_at:       str

    def to_dict(self) -> dict:
        return {
            "consumer_id":      self.consumer_id,
            "anomaly_detected": self.anomaly_detected,
            "anomaly_score":    round(self.anomaly_score, 3),
            "risk_level":       self.risk_level,
            "flags":            self.flags,
            "recommendation":   self.recommendation,
            "checked_at":       self.checked_at,
        }


# ─── SYNTHETIC DATA GENERATOR ─────────────────────────────────────────────────

def generate_synthetic_data(n_normal=8000, n_fraud=2000, seed=42):
    """
    Generate realistic Indian LPG booking patterns.

    Normal household features:
    - Books every 22-35 days
    - 1-2 people per cylinder per month
    - Consistent across seasons (slight winter increase)
    - Same address, same agent usually

    Fraud patterns we inject:
    - Maximum booking frequency (every 15 days, the legal minimum)
    - Single person household with family-level consumption
    - Sudden change in booking frequency after months of normal
    - Perfectly consistent booking (bot-like, no human variation)
    """
    np.random.seed(seed)

    # ── NORMAL HOUSEHOLDS ─────────────────────────────────────────────────────
    normal = np.column_stack([
        np.random.normal(27, 4, n_normal),          # days_between_bookings (mean 27)
        np.random.normal(2.3, 0.8, n_normal),       # household_size
        np.random.normal(0.15, 0.1, n_normal),      # booking_frequency_change (low)
        np.random.normal(0, 1, n_normal),            # address_change_count (rarely moves)
        np.random.normal(28, 6, n_normal),           # days_since_last_booking
        np.random.normal(0.1, 0.05, n_normal),      # agent_change_rate (same agent usually)
        np.random.normal(12, 3, n_normal),           # bookings_last_12_months (normal ~12)
        np.random.normal(0.05, 0.03, n_normal),     # complaint_rate (low)
    ])

    # ── FRAUD PATTERNS ────────────────────────────────────────────────────────

    # Pattern 1: Cylinder sellers (book max frequency, sell to restaurants)
    n_sellers = n_fraud // 4
    sellers = np.column_stack([
        np.random.normal(15.5, 0.8, n_sellers),    # books exactly at minimum (15 days)
        np.random.normal(1.1, 0.2, n_sellers),     # single person but high consumption
        np.random.normal(0.9, 0.1, n_sellers),     # high frequency change (suddenly books faster)
        np.random.normal(0, 0.5, n_sellers),       # stays at same address
        np.random.normal(15, 1, n_sellers),        # always 15 days between bookings
        np.random.normal(0.8, 0.1, n_sellers),     # high agent change (different agents)
        np.random.normal(24, 1, n_sellers),        # 24 bookings/year (max possible)
        np.random.normal(0.02, 0.01, n_sellers),   # low complaints (transaction works)
    ])

    # Pattern 2: Ghost connections (distributor's fake customers)
    n_ghost = n_fraud // 4
    ghosts = np.column_stack([
        np.random.normal(15.2, 0.3, n_ghost),     # perfectly regular (bot-like)
        np.random.normal(0.0, 0.1, n_ghost),      # zero household size (no one lives there)
        np.random.normal(0.95, 0.05, n_ghost),    # extremely consistent booking change
        np.random.normal(3, 1, n_ghost),          # multiple address changes
        np.random.normal(15.1, 0.2, n_ghost),     # near-perfect 15-day intervals
        np.random.normal(0.95, 0.05, n_ghost),    # always different agent
        np.random.normal(24, 0.5, n_ghost),       # maximum possible bookings
        np.random.normal(0.001, 0.001, n_ghost),  # zero complaints (not real people)
    ])

    # Pattern 3: Dormant account hijack
    n_hijack = n_fraud // 4
    hijacks = np.column_stack([
        np.random.normal(16, 1, n_hijack),         # suddenly books fast
        np.random.normal(2.5, 0.5, n_hijack),     # appears normal size
        np.random.normal(0.85, 0.1, n_hijack),    # major frequency change (key signal)
        np.random.normal(1, 0.5, n_hijack),       # one address change (phone number swap)
        np.random.normal(16, 1.5, n_hijack),      # new pattern emerging
        np.random.normal(0.7, 0.2, n_hijack),     # agent changes (new distributor)
        np.random.normal(22, 2, n_hijack),        # high bookings
        np.random.normal(0.01, 0.01, n_hijack),  # low complaints
    ])

    # Pattern 4: Distributor black market (large-scale diversion)
    n_dist = n_fraud // 4
    distributors = np.column_stack([
        np.random.normal(15.0, 0.1, n_dist),      # perfectly minimum interval
        np.random.normal(4.5, 0.5, n_dist),       # inflated household size
        np.random.normal(0.98, 0.02, n_dist),     # perfectly consistent
        np.random.normal(0, 0.1, n_dist),         # never moves
        np.random.normal(15.0, 0.1, n_dist),      # robotic precision
        np.random.normal(0.99, 0.01, n_dist),     # always different agents
        np.random.normal(24, 0.2, n_dist),        # maximum bookings
        np.random.normal(0.0, 0.001, n_dist),     # zero complaints
    ])

    X = np.vstack([normal, sellers, ghosts, hijacks, distributors])
    y = np.array([0] * n_normal + [1] * n_fraud)  # 0=normal, 1=fraud

    return X, y


# ─── ANOMALY DETECTOR ─────────────────────────────────────────────────────────

class ConsumptionAnomalyDetector:
    """
    Isolation Forest trained on synthetic Indian LPG consumption patterns.
    
    Usage:
        detector = ConsumptionAnomalyDetector()
        detector.train()
        result = detector.check(consumer_id="CON-001", days_between=15, ...)
    """

    FEATURE_NAMES = [
        "days_between_bookings",
        "household_size",
        "booking_frequency_change",
        "address_change_count",
        "days_since_last_booking",
        "agent_change_rate",
        "bookings_last_12_months",
        "complaint_rate",
    ]

    def __init__(self):
        self.model    = None
        self.scaler   = StandardScaler()
        self.trained  = False

    def train(self, verbose=True):
        """Train the model on synthetic data."""
        if verbose:
            print("  Training anomaly detection model on synthetic LPG data...")

        X, y = generate_synthetic_data()
        X_scaled = self.scaler.fit_transform(X)

        # Isolation Forest: contamination = expected fraud rate (~20%)
        self.model = IsolationForest(
            n_estimators  = 200,
            contamination = 0.2,
            random_state  = 42,
            n_jobs        = -1,
        )
        self.model.fit(X_scaled)
        self.trained = True

        if verbose:
            # Evaluate on training data
            preds = self.model.predict(X_scaled)
            n_flagged = sum(1 for p in preds if p == -1)
            print(f"  Model trained. Flagging {n_flagged}/{len(X)} samples as anomalies.")
            print(f"  Expected fraud rate: 20% ({len(X) * 0.2:.0f} samples)")

    def check(
        self,
        consumer_id:                str,
        days_between_bookings:      float,
        household_size:             float,
        booking_frequency_change:   float,   # 0.0-1.0: how much frequency changed recently
        address_change_count:       int,
        days_since_last_booking:    float,
        agent_change_rate:          float,   # 0.0-1.0: how often agent changes
        bookings_last_12_months:    int,
        complaint_rate:             float,   # 0.0-1.0: complaints per booking
    ) -> AnomalyResult:
        """
        Check one consumer's booking pattern for anomalies.
        Returns AnomalyResult with risk level and specific flags.
        """
        if not self.trained:
            raise RuntimeError("Model not trained. Call detector.train() first.")

        features = np.array([[
            days_between_bookings,
            household_size,
            booking_frequency_change,
            address_change_count,
            days_since_last_booking,
            agent_change_rate,
            bookings_last_12_months,
            complaint_rate,
        ]])

        features_scaled = self.scaler.transform(features)
        prediction      = self.model.predict(features_scaled)[0]   # 1=normal, -1=anomaly
        raw_score       = self.model.score_samples(features_scaled)[0]

        # Convert score to 0-1 range (higher = more anomalous)
        anomaly_score = max(0.0, min(1.0, (-raw_score - 0.1) / 0.6))
        anomaly_detected = (prediction == -1)

        # ── Determine risk level ──────────────────────────────────────────────
        if anomaly_score >= 0.85:   risk = "CRITICAL"
        elif anomaly_score >= 0.65: risk = "HIGH"
        elif anomaly_score >= 0.45: risk = "MEDIUM"
        elif anomaly_score >= 0.25: risk = "LOW"
        else:                        risk = "NORMAL"

        # ── Generate specific human-readable flags ────────────────────────────
        flags = []

        if days_between_bookings <= 16:
            flags.append(
                f"BOOKING_RATE_MAX: Booking every {days_between_bookings:.0f} days "
                f"(minimum allowed: 15). Pattern matches cylinder resellers."
            )

        if bookings_last_12_months >= 22:
            flags.append(
                f"ANNUAL_VOLUME_HIGH: {bookings_last_12_months} bookings in 12 months. "
                f"Normal household: 12-15. Max possible: 24."
            )

        if booking_frequency_change >= 0.7:
            flags.append(
                f"SUDDEN_FREQUENCY_CHANGE: Booking pattern changed by "
                f"{booking_frequency_change*100:.0f}%. "
                f"Matches dormant account hijack or new resale activity."
            )

        if household_size <= 0.5 and bookings_last_12_months >= 18:
            flags.append(
                "SINGLE_PERSON_HIGH_VOLUME: Single/no occupant booking "
                "at commercial-level frequency. Ghost connection likely."
            )

        if agent_change_rate >= 0.8:
            flags.append(
                f"HIGH_AGENT_CHURN: Delivery agent changes {agent_change_rate*100:.0f}% "
                f"of the time. Normal households have consistent agents. "
                f"Matches distributor rotation to avoid detection."
            )

        if days_between_bookings <= 15.5 and abs(days_between_bookings - 15.0) < 0.8:
            flags.append(
                "ROBOTIC_PRECISION: Booking intervals are unnaturally precise. "
                "Real households vary by ±3-5 days. Bot/automated booking suspected."
            )

        # ── Recommendation ────────────────────────────────────────────────────
        if risk == "CRITICAL":
            recommendation = "BLOCK_AND_AUDIT: Immediately suspend booking. Escalate for physical inspection."
        elif risk == "HIGH":
            recommendation = "FLAG_FOR_REVIEW: Allow booking but assign to human review queue within 24h."
        elif risk == "MEDIUM":
            recommendation = "MONITOR: Allow booking. Increase monitoring frequency. Flag if pattern continues."
        elif risk == "LOW":
            recommendation = "LOG: Minor anomaly noted. No action required. Pattern logged for trend analysis."
        else:
            recommendation = "CLEAR: Normal consumption pattern. No action required."

        return AnomalyResult(
            consumer_id      = consumer_id,
            anomaly_detected = anomaly_detected,
            anomaly_score    = anomaly_score,
            risk_level       = risk,
            flags            = flags,
            recommendation   = recommendation,
            checked_at       = datetime.now(timezone.utc).isoformat(),
        )


# ─── DEMO ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "=" * 65)
    print("  GASSETU — CONSUMPTION ANOMALY DETECTION")
    print("=" * 65)

    # Train the model
    detector = ConsumptionAnomalyDetector()
    detector.train(verbose=True)
    print()

    # Test cases — real-world Indian LPG fraud scenarios
    test_cases = [
        {
            "label": "👨‍👩‍👧‍👦 Normal Mumbai family of 4",
            "id":    "CON-NORMAL-001",
            "data":  dict(
                days_between_bookings    = 26,
                household_size           = 4,
                booking_frequency_change = 0.05,
                address_change_count     = 0,
                days_since_last_booking  = 28,
                agent_change_rate        = 0.1,
                bookings_last_12_months  = 13,
                complaint_rate           = 0.0,
            ),
        },
        {
            "label": "🍽️  Restaurant owner booking on home name (FRAUD)",
            "id":    "CON-FRAUD-001",
            "data":  dict(
                days_between_bookings    = 15.2,
                household_size           = 1.5,
                booking_frequency_change = 0.88,
                address_change_count     = 0,
                days_since_last_booking  = 15,
                agent_change_rate        = 0.75,
                bookings_last_12_months  = 24,
                complaint_rate           = 0.01,
            ),
        },
        {
            "label": "👻 Ghost connection (Distributor SIM farm)",
            "id":    "CON-GHOST-001",
            "data":  dict(
                days_between_bookings    = 15.0,
                household_size           = 0.0,
                booking_frequency_change = 0.97,
                address_change_count     = 2,
                days_since_last_booking  = 15.0,
                agent_change_rate        = 0.98,
                bookings_last_12_months  = 24,
                complaint_rate           = 0.0,
            ),
        },
        {
            "label": "🔄 Dormant account hijacked by distributor",
            "id":    "CON-HIJACK-001",
            "data":  dict(
                days_between_bookings    = 16,
                household_size           = 2.5,
                booking_frequency_change = 0.82,
                address_change_count     = 1,
                days_since_last_booking  = 16,
                agent_change_rate        = 0.65,
                bookings_last_12_months  = 22,
                complaint_rate           = 0.01,
            ),
        },
        {
            "label": "🏠 Rural single person, slow bookings (genuine)",
            "id":    "CON-RURAL-001",
            "data":  dict(
                days_between_bookings    = 38,
                household_size           = 1.0,
                booking_frequency_change = 0.05,
                address_change_count     = 0,
                days_since_last_booking  = 40,
                agent_change_rate        = 0.05,
                bookings_last_12_months  = 9,
                complaint_rate           = 0.0,
            ),
        },
    ]

    for tc in test_cases:
        result = detector.check(consumer_id=tc["id"], **tc["data"])
        icon   = "🚨" if result.risk_level in ["CRITICAL","HIGH"] else "✅"
        print(f"{icon}  {tc['label']}")
        print(f"   Risk Level : {result.risk_level}  (Score: {result.anomaly_score:.2f})")
        if result.flags:
            for flag in result.flags[:2]:  # show first 2 flags
                print(f"   ⚠️  {flag[:90]}...")
        print(f"   Action     : {result.recommendation[:80]}...")
        print()

    print("=" * 65)
    print("  Anomaly Detection ready.")
    print("  Next: Connect to /booking/validate combined API endpoint")
    print("=" * 65 + "\n")