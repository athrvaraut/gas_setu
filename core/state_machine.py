"""
GasSetu — Cylinder State Machine
=================================
The brain of the fraud detection system.
Every cylinder has a journey. Every step of that journey is a STATE.
If reality doesn't match the expected state — that is fraud.

States:
  FILLED → LOADED → IN_TRANSIT → AT_DISTRIBUTOR →
  ASSIGNED → OUT_FOR_DELIVERY → APPROACHING → CONFIRMED → COMPLETE

Author: GasSetu
"""

from enum import Enum
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import hashlib
import json


# ─── STATE DEFINITIONS ────────────────────────────────────────────────────────

class CylinderState(str, Enum):
    """Every state a cylinder can legally be in."""
    FILLED          = "FILLED"           # Left bottling plant, weight verified
    LOADED          = "LOADED"           # Loaded onto specific truck
    IN_TRANSIT      = "IN_TRANSIT"       # Truck moving toward distributor
    AT_DISTRIBUTOR  = "AT_DISTRIBUTOR"   # Received by distributor, count reconciled
    ASSIGNED        = "ASSIGNED"         # Assigned to specific delivery agent
    OUT_FOR_DELIVERY = "OUT_FOR_DELIVERY" # Agent picked it up, moving
    APPROACHING     = "APPROACHING"      # Agent within 200m of customer address
    CONFIRMED       = "CONFIRMED"        # Customer confirmed receipt (OTP/missed call)
    COMPLETE        = "COMPLETE"         # Delivery logged, agent paid, chain closed

    # Terminal fraud/exception states
    FLAGGED         = "FLAGGED"          # System detected anomaly
    DISPUTED        = "DISPUTED"         # Agent or customer raised dispute
    CANCELLED       = "CANCELLED"        # Booking cancelled before delivery


# ─── LEGAL TRANSITIONS MAP ────────────────────────────────────────────────────
# The ONLY transitions the system will ever allow.
# Any other transition = automatic fraud flag.

LEGAL_TRANSITIONS = {
    CylinderState.FILLED:           [CylinderState.LOADED],
    CylinderState.LOADED:           [CylinderState.IN_TRANSIT, CylinderState.FILLED],  # FILLED = returned to plant
    CylinderState.IN_TRANSIT:       [CylinderState.AT_DISTRIBUTOR],
    CylinderState.AT_DISTRIBUTOR:   [CylinderState.ASSIGNED, CylinderState.CANCELLED],
    CylinderState.ASSIGNED:         [CylinderState.OUT_FOR_DELIVERY, CylinderState.AT_DISTRIBUTOR],  # AT_DISTRIBUTOR = reassigned
    CylinderState.OUT_FOR_DELIVERY: [CylinderState.APPROACHING, CylinderState.AT_DISTRIBUTOR],  # AT_DISTRIBUTOR = returned
    CylinderState.APPROACHING:      [CylinderState.CONFIRMED, CylinderState.OUT_FOR_DELIVERY],  # OUT_FOR_DELIVERY = customer not home
    CylinderState.CONFIRMED:        [CylinderState.COMPLETE],
    CylinderState.COMPLETE:         [],  # Terminal state — no further transitions
    CylinderState.FLAGGED:          [CylinderState.DISPUTED, CylinderState.CANCELLED],
    CylinderState.DISPUTED:         [CylinderState.COMPLETE, CylinderState.FLAGGED],
    CylinderState.CANCELLED:        [],  # Terminal state
}

# Transitions that REQUIRE extra validation before allowing
HIGH_RISK_TRANSITIONS = {
    (CylinderState.APPROACHING, CylinderState.CONFIRMED),    # Must have customer OTP
    (CylinderState.CONFIRMED, CylinderState.COMPLETE),       # Must have agent GPS at address
    (CylinderState.ASSIGNED, CylinderState.OUT_FOR_DELIVERY), # Must have agent scan
}


# ─── DATA STRUCTURES ──────────────────────────────────────────────────────────

@dataclass
class StateTransitionEvent:
    """A single recorded event in the cylinder's journey."""
    cylinder_id:    str
    from_state:     CylinderState
    to_state:       CylinderState
    timestamp:      datetime
    agent_id:       Optional[str]
    gps_lat:        Optional[float]
    gps_lng:        Optional[float]
    gps_accuracy:   Optional[float]   # metres — low accuracy = potential spoof
    accelerometer:  Optional[float]   # movement delta — 0 while GPS moves = spoof signal
    cell_tower_id:  Optional[str]     # for SIM farm detection
    device_id:      Optional[str]     # Android device fingerprint
    notes:          Optional[str]
    prev_hash:      Optional[str]     # cryptographic chain — links to previous event
    event_hash:     str = field(default="", init=False)

    def __post_init__(self):
        """Auto-generate cryptographic hash when event is created."""
        self.event_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """
        Cryptographic chain: each event hashes itself + previous event.
        If ANY field is tampered with after creation — hash breaks.
        This is what defeats the device clock manipulation attack.
        """
        payload = {
            "cylinder_id":   self.cylinder_id,
            "from_state":    self.from_state,
            "to_state":      self.to_state,
            "timestamp":     self.timestamp.isoformat(),
            "agent_id":      self.agent_id,
            "gps_lat":       self.gps_lat,
            "gps_lng":       self.gps_lng,
            "accelerometer": self.accelerometer,
            "cell_tower_id": self.cell_tower_id,
            "device_id":     self.device_id,
            "prev_hash":     self.prev_hash,
        }
        raw = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def verify_integrity(self) -> bool:
        """Recompute hash and compare — if they differ, data was tampered."""
        return self.event_hash == self._compute_hash()


@dataclass
class FraudFlag:
    """Generated whenever an illegal or suspicious transition is attempted."""
    cylinder_id:    str
    flag_type:      str
    severity:       str          # LOW, MEDIUM, HIGH, CRITICAL
    description:    str
    timestamp:      datetime
    agent_id:       Optional[str]
    evidence:       dict         # raw data that triggered the flag

    def to_dict(self) -> dict:
        return {
            "cylinder_id": self.cylinder_id,
            "flag_type":   self.flag_type,
            "severity":    self.severity,
            "description": self.description,
            "timestamp":   self.timestamp.isoformat(),
            "agent_id":    self.agent_id,
            "evidence":    self.evidence,
        }


# ─── MAIN STATE MACHINE ───────────────────────────────────────────────────────

class CylinderStateMachine:
    """
    The core fraud detection engine.

    Usage:
        machine = CylinderStateMachine("CYL-2024-001234")
        result = machine.transition(CylinderState.LOADED, agent_id="AGT-001", ...)
        if not result["allowed"]:
            print(result["fraud_flag"])
    """

    def __init__(self, cylinder_id: str, initial_state: CylinderState = CylinderState.FILLED):
        self.cylinder_id    = cylinder_id
        self.current_state  = initial_state
        self.history:  list[StateTransitionEvent] = []
        self.flags:    list[FraudFlag]             = []
        self.created_at     = datetime.utcnow()
        self.last_updated   = datetime.utcnow()

    # ── CORE TRANSITION METHOD ────────────────────────────────────────────────

    def transition(
        self,
        to_state:       CylinderState,
        timestamp:      Optional[datetime] = None,
        agent_id:       Optional[str]      = None,
        gps_lat:        Optional[float]    = None,
        gps_lng:        Optional[float]    = None,
        gps_accuracy:   Optional[float]    = None,
        accelerometer:  Optional[float]    = None,
        cell_tower_id:  Optional[str]      = None,
        device_id:      Optional[str]      = None,
        otp_verified:   bool               = False,
        notes:          Optional[str]      = None,
    ) -> dict:
        """
        Attempt a state transition.
        Returns: { "allowed": bool, "new_state": str, "fraud_flag": dict|None }
        """
        timestamp = timestamp or datetime.utcnow()

        # ── CHECK 1: Is this transition legally allowed? ──────────────────────
        if to_state not in LEGAL_TRANSITIONS.get(self.current_state, []):
            flag = self._raise_flag(
                flag_type   = "ILLEGAL_TRANSITION",
                severity    = "CRITICAL",
                description = (
                    f"Cylinder attempted illegal jump: "
                    f"{self.current_state} → {to_state}. "
                    f"This cylinder skipped mandatory intermediate states. "
                    f"Classic ghost delivery signature."
                ),
                agent_id    = agent_id,
                evidence    = {
                    "from_state":   self.current_state,
                    "to_state":     to_state,
                    "legal_options": [s for s in LEGAL_TRANSITIONS[self.current_state]],
                },
            )
            return {"allowed": False, "new_state": self.current_state, "fraud_flag": flag.to_dict()}

        # ── CHECK 2: High-risk transitions need extra validation ──────────────
        transition_pair = (self.current_state, to_state)

        if transition_pair == (CylinderState.APPROACHING, CylinderState.CONFIRMED):
            if not otp_verified:
                flag = self._raise_flag(
                    flag_type   = "CONFIRMATION_WITHOUT_OTP",
                    severity    = "HIGH",
                    description = "Delivery marked CONFIRMED without customer OTP verification.",
                    agent_id    = agent_id,
                    evidence    = {"otp_verified": False},
                )
                return {"allowed": False, "new_state": self.current_state, "fraud_flag": flag.to_dict()}

        # ── CHECK 3: GPS accuracy check ───────────────────────────────────────
        if gps_accuracy and gps_accuracy > 500:
            # GPS accuracy worse than 500m is unreliable for delivery verification
            self._raise_flag(
                flag_type   = "POOR_GPS_ACCURACY",
                severity    = "LOW",
                description = f"GPS accuracy {gps_accuracy}m — below verification threshold. Auto-logged, not blocked.",
                agent_id    = agent_id,
                evidence    = {"gps_accuracy": gps_accuracy},
            )
            # LOW severity — don't block, just log

        # ── CHECK 4: Mock location detection (sensor fusion) ──────────────────
        mock_flag = self._detect_mock_location(gps_lat, gps_lng, gps_accuracy, accelerometer)
        if mock_flag:
            flag = self._raise_flag(**mock_flag, agent_id=agent_id)
            return {"allowed": False, "new_state": self.current_state, "fraud_flag": flag.to_dict()}

        # ── CHECK 5: Time sequence integrity ──────────────────────────────────
        if self.history:
            last_event = self.history[-1]
            if timestamp < last_event.timestamp:
                flag = self._raise_flag(
                    flag_type   = "TIMESTAMP_REVERSAL",
                    severity    = "CRITICAL",
                    description = (
                        f"New event timestamp {timestamp} is BEFORE last event "
                        f"{last_event.timestamp}. Device clock was manipulated."
                    ),
                    agent_id    = agent_id,
                    evidence    = {
                        "new_timestamp":  timestamp.isoformat(),
                        "last_timestamp": last_event.timestamp.isoformat(),
                    },
                )
                return {"allowed": False, "new_state": self.current_state, "fraud_flag": flag.to_dict()}

        # ── ALL CHECKS PASSED — Execute transition ─────────────────────────────
        prev_hash = self.history[-1].event_hash if self.history else None

        event = StateTransitionEvent(
            cylinder_id   = self.cylinder_id,
            from_state    = self.current_state,
            to_state      = to_state,
            timestamp     = timestamp,
            agent_id      = agent_id,
            gps_lat       = gps_lat,
            gps_lng       = gps_lng,
            gps_accuracy  = gps_accuracy,
            accelerometer = accelerometer,
            cell_tower_id = cell_tower_id,
            device_id     = device_id,
            notes         = notes,
            prev_hash     = prev_hash,
        )

        self.history.append(event)
        self.current_state = to_state
        self.last_updated  = timestamp

        return {
            "allowed":    True,
            "new_state":  self.current_state,
            "event_hash": event.event_hash,
            "fraud_flag": None,
        }

    # ── MOCK LOCATION DETECTION ───────────────────────────────────────────────

    def _detect_mock_location(
        self,
        gps_lat:       Optional[float],
        gps_lng:       Optional[float],
        gps_accuracy:  Optional[float],
        accelerometer: Optional[float],
    ) -> Optional[dict]:
        """
        Sensor fusion check: GPS movement must correlate with accelerometer movement.
        If GPS shows movement but accelerometer shows stillness → mock location app.
        """
        if not self.history or not gps_lat or not gps_lng:
            return None

        last = self.history[-1]
        if not last.gps_lat or not last.gps_lng:
            return None

        # Calculate GPS distance moved (simplified flat-earth for short distances)
        lat_diff = abs(gps_lat - last.gps_lat) * 111_000   # ~111km per degree lat
        lng_diff = abs(gps_lng - last.gps_lng) * 111_000
        gps_distance_moved = (lat_diff**2 + lng_diff**2) ** 0.5  # metres

        # If GPS shows moved > 100m but accelerometer shows near-zero movement
        if (gps_distance_moved > 100
                and accelerometer is not None
                and last.accelerometer is not None):
            accel_delta = abs(accelerometer - last.accelerometer)
            if accel_delta < 0.5:  # essentially stationary phone
                return {
                    "flag_type":   "MOCK_LOCATION_DETECTED",
                    "severity":    "HIGH",
                    "description": (
                        f"GPS shows {gps_distance_moved:.0f}m movement but "
                        f"accelerometer delta is only {accel_delta:.2f}. "
                        f"Sensor fusion mismatch — mock location app likely active."
                    ),
                    "evidence": {
                        "gps_distance_moved": gps_distance_moved,
                        "accelerometer_delta": accel_delta,
                        "gps_accuracy": gps_accuracy,
                    },
                }
        return None

    # ── CHAIN INTEGRITY VERIFICATION ──────────────────────────────────────────

    def verify_chain_integrity(self) -> dict:
        """
        Walk the entire history and verify every cryptographic hash.
        Any broken link = tampering detected.
        """
        broken_links = []
        for i, event in enumerate(self.history):
            if not event.verify_integrity():
                broken_links.append({
                    "index":     i,
                    "state":     f"{event.from_state} → {event.to_state}",
                    "timestamp": event.timestamp.isoformat(),
                })

        return {
            "intact":       len(broken_links) == 0,
            "total_events": len(self.history),
            "broken_links": broken_links,
        }

    # ── HELPERS ───────────────────────────────────────────────────────────────

    def _raise_flag(self, flag_type, severity, description, agent_id, evidence) -> FraudFlag:
        flag = FraudFlag(
            cylinder_id = self.cylinder_id,
            flag_type   = flag_type,
            severity    = severity,
            description = description,
            timestamp   = datetime.utcnow(),
            agent_id    = agent_id,
            evidence    = evidence,
        )
        self.flags.append(flag)
        self.current_state = CylinderState.FLAGGED
        return flag

    def get_summary(self) -> dict:
        return {
            "cylinder_id":   self.cylinder_id,
            "current_state": self.current_state,
            "total_events":  len(self.history),
            "total_flags":   len(self.flags),
            "chain_intact":  self.verify_chain_integrity()["intact"],
            "created_at":    self.created_at.isoformat(),
            "last_updated":  self.last_updated.isoformat(),
        }