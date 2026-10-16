"""
GasSetu — State Machine Tests
==============================
Every fraud scenario the delivery state machine was designed against.
(The original 26 checks, now as pytest assertions.)

Run from the repo root:  python -m pytest tests/test_state_machine.py
"""

from datetime import datetime, timedelta

from core.state_machine import CylinderStateMachine, CylinderState


def test_normal_legal_delivery_journey():
    m = CylinderStateMachine("CYL-TEST-001")
    t = datetime(2024, 1, 15, 8, 0, 0)
    results = [
        m.transition(CylinderState.LOADED,           timestamp=t,                       agent_id="AGT-01", gps_lat=19.05, gps_lng=72.88, accelerometer=2.1),
        m.transition(CylinderState.IN_TRANSIT,       timestamp=t+timedelta(minutes=10), agent_id="AGT-01", gps_lat=19.06, gps_lng=72.89, accelerometer=3.4),
        m.transition(CylinderState.AT_DISTRIBUTOR,   timestamp=t+timedelta(minutes=40), agent_id="AGT-01", gps_lat=19.07, gps_lng=72.90, accelerometer=1.1),
        m.transition(CylinderState.ASSIGNED,         timestamp=t+timedelta(minutes=50), agent_id="AGT-02"),
        m.transition(CylinderState.OUT_FOR_DELIVERY, timestamp=t+timedelta(hours=2),    agent_id="AGT-02", gps_lat=19.08, gps_lng=72.91, accelerometer=2.8),
        m.transition(CylinderState.APPROACHING,      timestamp=t+timedelta(hours=2, minutes=20), agent_id="AGT-02", gps_lat=19.09, gps_lng=72.92, accelerometer=1.5),
        m.transition(CylinderState.CONFIRMED,        timestamp=t+timedelta(hours=2, minutes=25), agent_id="AGT-02", otp_verified=True),
        m.transition(CylinderState.COMPLETE,         timestamp=t+timedelta(hours=2, minutes=26), agent_id="AGT-02"),
    ]
    assert all(r["allowed"] for r in results), "All 8 transitions allowed"
    assert m.current_state == CylinderState.COMPLETE, "Final state is COMPLETE"
    assert len(m.history) == 8, "8 events in history"
    assert len(m.flags) == 0, "Zero fraud flags raised"
    assert m.verify_chain_integrity()["intact"], "Cryptographic chain intact"


def test_ghost_delivery_jump_to_complete_is_blocked():
    m = CylinderStateMachine("CYL-TEST-002")
    m.transition(CylinderState.LOADED,         timestamp=datetime(2024, 1, 15, 8, 0),  agent_id="AGT-BAD")
    m.transition(CylinderState.IN_TRANSIT,     timestamp=datetime(2024, 1, 15, 8, 10), agent_id="AGT-BAD")
    m.transition(CylinderState.AT_DISTRIBUTOR, timestamp=datetime(2024, 1, 15, 8, 40), agent_id="AGT-BAD")
    m.transition(CylinderState.ASSIGNED,       timestamp=datetime(2024, 1, 15, 8, 50), agent_id="AGT-BAD")
    # FRAUD: skip OUT_FOR_DELIVERY, APPROACHING, CONFIRMED
    result = m.transition(CylinderState.COMPLETE, timestamp=datetime(2024, 1, 15, 9, 0), agent_id="AGT-BAD")
    assert not result["allowed"], "Ghost delivery BLOCKED"
    assert result["fraud_flag"] is not None, "Fraud flag raised"
    assert result["fraud_flag"]["flag_type"] == "ILLEGAL_TRANSITION"
    assert result["fraud_flag"]["severity"] == "CRITICAL"
    assert m.current_state == CylinderState.FLAGGED, "State changed to FLAGGED"


def test_confirmation_without_otp_is_blocked():
    m = CylinderStateMachine("CYL-TEST-003")
    t = datetime(2024, 1, 15, 10, 0)
    m.transition(CylinderState.LOADED,           timestamp=t)
    m.transition(CylinderState.IN_TRANSIT,       timestamp=t+timedelta(minutes=10))
    m.transition(CylinderState.AT_DISTRIBUTOR,   timestamp=t+timedelta(minutes=40))
    m.transition(CylinderState.ASSIGNED,         timestamp=t+timedelta(minutes=50), agent_id="AGT-03")
    m.transition(CylinderState.OUT_FOR_DELIVERY, timestamp=t+timedelta(hours=1),    agent_id="AGT-03", gps_lat=19.05, gps_lng=72.88, accelerometer=2.0)
    m.transition(CylinderState.APPROACHING,      timestamp=t+timedelta(hours=1, minutes=20), agent_id="AGT-03", gps_lat=19.051, gps_lng=72.881, accelerometer=1.0)
    result = m.transition(CylinderState.CONFIRMED, timestamp=t+timedelta(hours=1, minutes=25), agent_id="AGT-03", otp_verified=False)
    assert not result["allowed"], "Confirmation without OTP BLOCKED"
    assert result["fraud_flag"]["flag_type"] == "CONFIRMATION_WITHOUT_OTP"
    assert result["fraud_flag"]["severity"] == "HIGH"


def test_mock_location_gps_moves_phone_stationary():
    m = CylinderStateMachine("CYL-TEST-004")
    t = datetime(2024, 1, 15, 11, 0)
    m.transition(CylinderState.LOADED, timestamp=t, agent_id="AGT-04", gps_lat=19.1136, gps_lng=72.8697, accelerometer=0.2)
    # GPS jumps ~10 km north while the accelerometer barely changes
    result = m.transition(CylinderState.IN_TRANSIT, timestamp=t+timedelta(minutes=2), agent_id="AGT-04",
                          gps_lat=19.2041, gps_lng=72.8511, accelerometer=0.18)
    assert not result["allowed"], "Mock location BLOCKED"
    assert result["fraud_flag"]["flag_type"] == "MOCK_LOCATION_DETECTED"
    assert result["fraud_flag"]["severity"] == "HIGH"
    assert "gps_distance_moved" in result["fraud_flag"]["evidence"]


def test_device_clock_manipulation_is_blocked():
    m = CylinderStateMachine("CYL-TEST-005")
    t = datetime(2024, 1, 15, 14, 0)
    m.transition(CylinderState.LOADED,     timestamp=t,                       agent_id="AGT-05")
    m.transition(CylinderState.IN_TRANSIT, timestamp=t+timedelta(minutes=10), agent_id="AGT-05")
    result = m.transition(CylinderState.AT_DISTRIBUTOR, timestamp=t - timedelta(hours=2), agent_id="AGT-05")
    assert not result["allowed"], "Time reversal BLOCKED"
    assert result["fraud_flag"]["flag_type"] == "TIMESTAMP_REVERSAL"
    assert result["fraud_flag"]["severity"] == "CRITICAL"


def test_chain_detects_edited_record():
    m = CylinderStateMachine("CYL-TEST-006")
    t = datetime(2024, 1, 15, 9, 0)
    m.transition(CylinderState.LOADED,     timestamp=t,                       agent_id="AGT-06")
    m.transition(CylinderState.IN_TRANSIT, timestamp=t+timedelta(minutes=10), agent_id="AGT-06")
    clean = m.verify_chain_integrity()
    assert clean["intact"], "Clean chain is intact"
    assert clean["total_events"] == 2, "Chain has 2 events"
    m.history[0].gps_lat = 99.9999   # silently alter a GPS coordinate
    tampered = m.verify_chain_integrity()
    assert not tampered["intact"], "Tampered chain is detected"
    assert len(tampered["broken_links"]) > 0, "Broken link identified"


def test_legal_return_to_distributor_is_allowed():
    m = CylinderStateMachine("CYL-TEST-007")
    t = datetime(2024, 1, 15, 16, 0)
    m.transition(CylinderState.LOADED,           timestamp=t, agent_id="AGT-07")
    m.transition(CylinderState.IN_TRANSIT,       timestamp=t+timedelta(minutes=10))
    m.transition(CylinderState.AT_DISTRIBUTOR,   timestamp=t+timedelta(minutes=40))
    m.transition(CylinderState.ASSIGNED,         timestamp=t+timedelta(minutes=50), agent_id="AGT-07")
    m.transition(CylinderState.OUT_FOR_DELIVERY, timestamp=t+timedelta(hours=1), agent_id="AGT-07", gps_lat=19.05, gps_lng=72.88, accelerometer=2.0)
    result = m.transition(CylinderState.AT_DISTRIBUTOR, timestamp=t+timedelta(hours=1, minutes=30), agent_id="AGT-07")
    assert result["allowed"], "Legal return to distributor ALLOWED"
    assert m.current_state == CylinderState.AT_DISTRIBUTOR
