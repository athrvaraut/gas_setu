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
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Optional
import hashlib
import json
import math


def _utc(ts: Optional[datetime] = None) -> datetime:
    """Normalise to an aware UTC datetime. Naive inputs are treated as UTC."""
    if ts is None:
        return datetime.now(timezone.utc)
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance in metres."""
    r = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


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
    CylinderState.FLAGGED:          [CylinderState.DISPUTED, CylinderState.CANCELLED],  # or resolve_flag() by a reviewer
    CylinderState.DISPUTED:         [CylinderState.COMPLETE, CylinderState.FLAGGED],
    CylinderState.CANCELLED:        [],  # Terminal state
}

# Transitions that need extra evidence before they are allowed (both are enforced in transition())
HIGH_RISK_TRANSITIONS = {
    (CylinderState.APPROACHING, CylinderState.CONFIRMED),     # needs the customer's OTP (DAC)
    (CylinderState.ASSIGNED, CylinderState.OUT_FOR_DELIVERY), # must be picked up by the agent it was assigned to
}

# An event may be dated this far ahead of the server clock (device clocks drift); anything later is rejected
MAX_CLOCK_SKEW_SECONDS = 600


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
    otp_verified:   bool = False      # the customer's OTP (DAC) was checked for this event
    event_hash:     str = field(default="", init=False)

    def __post_init__(self):
        """Auto-generate cryptographic hash when event is created."""
        self.event_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """
        Cryptographic chain: each event hashes itself + previous event.
        EVERY stored field is in the hash, so editing any of them afterwards breaks it.
        """
        payload = {
            "cylinder_id":   self.cylinder_id,
            "from_state":    self.from_state,
            "to_state":      self.to_state,
            "timestamp":     self.timestamp.isoformat(),
            "agent_id":      self.agent_id,
            "gps_lat":       self.gps_lat,
            "gps_lng":       self.gps_lng,
            "gps_accuracy":  self.gps_accuracy,
            "accelerometer": self.accelerometer,
            "cell_tower_id": self.cell_tower_id,
            "device_id":     self.device_id,
            "notes":         self.notes,
            "otp_verified":  self.otp_verified,
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

    A rejected attempt moves the cylinder to FLAGGED and is itself written into the hash chain,
    so a flag cannot be quietly removed. A reviewer puts the cylinder back on its journey with
    resolve_flag(); that action is chained too.
    """

    def __init__(self, cylinder_id: str, initial_state: CylinderState = CylinderState.FILLED):
        self.cylinder_id    = cylinder_id
        self.current_state  = initial_state
        self.history:  list[StateTransitionEvent] = []
        self.flags:    list[FraudFlag]             = []
        self.created_at     = _utc()
        self.last_updated   = self.created_at
        self._state_before_flag: Optional[CylinderState] = None

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
        timestamp = _utc(timestamp)
        blocked = lambda flag: {"allowed": False, "new_state": self.current_state, "fraud_flag": flag.to_dict()}

        # ── CHECK 1: Is this transition legally allowed? ──────────────────────
        if to_state not in LEGAL_TRANSITIONS.get(self.current_state, []):
            return blocked(self._raise_flag(
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
                attempted_at = timestamp,
            ))

        # ── CHECK 2: High-risk transitions need extra evidence ────────────────
        transition_pair = (self.current_state, to_state)

        if transition_pair == (CylinderState.APPROACHING, CylinderState.CONFIRMED) and not otp_verified:
            return blocked(self._raise_flag(
                flag_type   = "CONFIRMATION_WITHOUT_OTP",
                severity    = "HIGH",
                description = "Delivery marked CONFIRMED without customer OTP verification.",
                agent_id    = agent_id,
                evidence    = {"otp_verified": False},
                attempted_at = timestamp,
            ))

        if transition_pair == (CylinderState.ASSIGNED, CylinderState.OUT_FOR_DELIVERY):
            assigned_to = self.history[-1].agent_id if self.history else None
            if assigned_to is not None and agent_id != assigned_to:
                return blocked(self._raise_flag(
                    flag_type   = "AGENT_MISMATCH",
                    severity    = "HIGH",
                    description = (
                        f"Cylinder was assigned to {assigned_to} but picked up by "
                        f"{agent_id or 'an unidentified agent'}."
                    ),
                    agent_id    = agent_id,
                    evidence    = {"assigned_to": assigned_to, "picked_up_by": agent_id},
                    attempted_at = timestamp,
                ))

        # ── CHECK 3: GPS accuracy check ───────────────────────────────────────
        if gps_accuracy and gps_accuracy > 500:
            # GPS accuracy worse than 500m is unreliable for delivery verification
            self._raise_flag(
                flag_type   = "POOR_GPS_ACCURACY",
                severity    = "LOW",
                description = f"GPS accuracy {gps_accuracy}m — below verification threshold. Auto-logged, not blocked.",
                agent_id    = agent_id,
                evidence    = {"gps_accuracy": gps_accuracy},
                block       = False,   # LOW severity — log only, the cylinder keeps its real state
            )

        # ── CHECK 4: Mock location detection (sensor fusion) ──────────────────
        mock_flag = self._detect_mock_location(gps_lat, gps_lng, gps_accuracy, accelerometer)
        if mock_flag:
            return blocked(self._raise_flag(**mock_flag, agent_id=agent_id, attempted_at=timestamp))

        # ── CHECK 5: Time sequence integrity ──────────────────────────────────
        if self.history:
            last_event = self.history[-1]
            if timestamp < last_event.timestamp:
                return blocked(self._raise_flag(
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
                    attempted_at = timestamp,
                ))

        ahead = (timestamp - _utc()).total_seconds()
        if ahead > MAX_CLOCK_SKEW_SECONDS:
            return blocked(self._raise_flag(
                flag_type   = "TIMESTAMP_IN_FUTURE",
                severity    = "HIGH",
                description = (
                    f"Event is dated {ahead / 60:.0f} minutes ahead of the server clock. "
                    f"Device clock was set forward."
                ),
                agent_id    = agent_id,
                evidence    = {"new_timestamp": timestamp.isoformat(), "seconds_ahead": round(ahead)},
                attempted_at = timestamp,
            ))

        # ── ALL CHECKS PASSED — Execute transition ─────────────────────────────
        event = self._append_event(
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
            otp_verified  = otp_verified,
        )

        return {
            "allowed":    True,
            "new_state":  self.current_state,
            "event_hash": event.event_hash,
            "fraud_flag": None,
        }

    def resolve_flag(self, reviewer_id: str, note: str = "") -> dict:
        """
        A reviewer clears a flag (a mistaken scan, a retried request) and the cylinder returns to
        the state it had before. Recorded in the chain with the reviewer's id.
        """
        if self.current_state != CylinderState.FLAGGED or self._state_before_flag is None:
            return {"resolved": False, "new_state": self.current_state, "error": "Cylinder is not flagged."}
        restored = self._state_before_flag
        self._state_before_flag = None
        event = self._append_event(to_state=restored, timestamp=self._not_before_last(_utc()),
                                   agent_id=reviewer_id, notes=f"FLAG_RESOLVED: {note}".strip())
        return {"resolved": True, "new_state": self.current_state, "event_hash": event.event_hash}

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
        A weak signal that has not been evaluated yet (ROADMAP step 29).
        """
        if not self.history or gps_lat is None or gps_lng is None:
            return None

        last = self.history[-1]
        if last.gps_lat is None or last.gps_lng is None:
            return None

        gps_distance_moved = _haversine_m(last.gps_lat, last.gps_lng, gps_lat, gps_lng)  # metres

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
        Walk the history and check two things for every event:
          1. HASH_MISMATCH: its stored hash no longer matches its fields (an edited record).
          2. BROKEN_LINK: its prev_hash is not the hash of the event before it
             (an event was deleted, inserted or reordered).

        Limitation: dropping the most recent event(s) leaves a shorter chain that is still
        valid. Only a chain head published outside this server can catch that (planned:
        device signatures + external anchoring, ROADMAP step 28).
        """
        broken_links = []
        for i, event in enumerate(self.history):
            expected_prev = self.history[i - 1].event_hash if i else None
            if not event.verify_integrity():
                reason = "HASH_MISMATCH"
            elif event.prev_hash != expected_prev:
                reason = "BROKEN_LINK"
            else:
                continue
            broken_links.append({
                "index":     i,
                "state":     f"{event.from_state} → {event.to_state}",
                "timestamp": event.timestamp.isoformat(),
                "reason":    reason,
            })

        return {
            "intact":       len(broken_links) == 0,
            "total_events": len(self.history),
            "broken_links": broken_links,
        }

    # ── HELPERS ───────────────────────────────────────────────────────────────

    def _not_before_last(self, timestamp: datetime) -> datetime:
        """Timestamp for an event the server writes itself: not ahead of the server clock, not before the last event."""
        timestamp = min(timestamp, _utc() + timedelta(seconds=MAX_CLOCK_SKEW_SECONDS))
        if self.history:
            timestamp = max(timestamp, self.history[-1].timestamp)
        return timestamp

    def _append_event(self, to_state: CylinderState, timestamp: datetime, agent_id: Optional[str] = None,
                      gps_lat=None, gps_lng=None, gps_accuracy=None, accelerometer=None, cell_tower_id=None,
                      device_id=None, notes=None, otp_verified: bool = False) -> StateTransitionEvent:
        """Write one event at the end of the chain and move the cylinder to `to_state`."""
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
            prev_hash     = self.history[-1].event_hash if self.history else None,
            otp_verified  = otp_verified,
        )
        self.history.append(event)
        self.current_state = to_state
        self.last_updated  = timestamp
        return event

    def _raise_flag(self, flag_type, severity, description, agent_id, evidence, block: bool = True,
                    attempted_at: Optional[datetime] = None) -> FraudFlag:
        """
        Record a flag. A blocking flag moves the cylinder to FLAGGED and writes that into the hash
        chain; a log-only flag (block=False) leaves the cylinder's state alone.
        """
        flag = FraudFlag(
            cylinder_id = self.cylinder_id,
            flag_type   = flag_type,
            severity    = severity,
            description = description,
            timestamp   = _utc(),
            agent_id    = agent_id,
            evidence    = evidence,
        )
        self.flags.append(flag)
        if block:
            if self.current_state != CylinderState.FLAGGED:
                self._state_before_flag = self.current_state
            self._append_event(to_state=CylinderState.FLAGGED,
                               timestamp=self._not_before_last(attempted_at or _utc()),
                               agent_id=agent_id, notes=f"{flag_type}: {description}")
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
