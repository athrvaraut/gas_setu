"""
Regression tests for the defects found in the October 2026 review (build_plan.md §5).
Each test fails on the old code and passes on the fix.
"""

from datetime import datetime, timedelta, timezone

import pytest

from core.distributor_reconciliation import DistributorReconciliation
from core.state_machine import CylinderStateMachine, CylinderState, _haversine_m


def _journey(n_steps=4):
    m = CylinderStateMachine("CYL-REG")
    t = datetime(2026, 10, 1, 8, 0)
    states = [CylinderState.LOADED, CylinderState.IN_TRANSIT, CylinderState.AT_DISTRIBUTOR, CylinderState.ASSIGNED]
    for i, s in enumerate(states[:n_steps]):
        assert m.transition(s, timestamp=t + timedelta(minutes=10 * i))["allowed"]
    return m


# ─── D1: hash-chain links ─────────────────────────────────────────────────────

def test_d1_deleted_event_is_detected():
    m = _journey()
    del m.history[1]
    result = m.verify_chain_integrity()
    assert not result["intact"]
    assert result["broken_links"][0]["reason"] == "BROKEN_LINK"


def test_d1_reordered_events_are_detected():
    m = _journey()
    m.history[1], m.history[2] = m.history[2], m.history[1]
    assert not m.verify_chain_integrity()["intact"]


def test_d1_edited_event_reports_hash_mismatch():
    m = _journey()
    m.history[2].agent_id = "AGT-FORGED"
    links = m.verify_chain_integrity()["broken_links"]
    assert links and links[0]["reason"] == "HASH_MISMATCH"


def test_d1_known_limitation_dropping_the_last_event():
    """Truncating the tail leaves a valid shorter chain. Only external anchoring
    (ROADMAP step 28) can catch this. If this test starts failing, anchoring works: update it."""
    m = _journey()
    m.history.pop()
    assert m.verify_chain_integrity()["intact"]


# ─── D2: ghost timer beyond 24 hours ──────────────────────────────────────────

@pytest.mark.parametrize("hours, expected", [(1, False), (3, True), (25, True), (49, True)])
def test_d2_ghost_detection_by_age(hours, expected):
    r = DistributorReconciliation("D", "Test")
    r.allocate(1)
    r.assign_cylinder("C1", "Q1", "A1", "K1", "9")
    r.mark_delivered("C1")
    r.cylinders["C1"].delivered_at = datetime.now(timezone.utc) - timedelta(hours=hours)
    assert r.cylinders["C1"].is_ghost() is expected


# ─── D4: a LOW (log-only) flag must not corrupt the cylinder's state ─────────

def test_d4_poor_gps_accuracy_is_logged_without_changing_state():
    m = CylinderStateMachine("CYL-GPS")
    result = m.transition(CylinderState.LOADED, timestamp=datetime(2026, 10, 1, 8, 0), gps_accuracy=800)
    assert result["allowed"]
    assert m.current_state == CylinderState.LOADED
    assert [f.flag_type for f in m.flags] == ["POOR_GPS_ACCURACY"]
    assert m.history[-1].from_state == CylinderState.FILLED


# ─── D5: reconciliation is idempotent ─────────────────────────────────────────

def test_d5_reconciling_twice_gives_the_same_result():
    r = DistributorReconciliation("D", "Test")
    r.register_agent("A1", "Agent", "9")
    r.allocate(4)
    for i in range(3):
        cid = f"C{i}"
        r.assign_cylinder(cid, f"Q{i}", "A1", f"K{i}", "9")
        r.mark_delivered(cid)
        r.cylinders[cid].delivered_at = datetime.now(timezone.utc) - timedelta(hours=3)
    first = r.reconcile()
    agent_trust, ghosts = r.agents["A1"].trust_score, r.agents["A1"].total_ghost
    second = r.reconcile()
    assert r.agents["A1"].trust_score == agent_trust
    assert r.agents["A1"].total_ghost == ghosts == 3
    assert second.ghost_deliveries == first.ghost_deliveries == 3
    assert second.cylinders_delivered == first.cylinders_delivered
    assert second.trust_score == first.trust_score, "distributor trust adjusted once per day"


# ─── D22: every stored field is hashed; flags are chained; future dates; pick-up agent ───

def test_d22_editing_any_stored_field_breaks_the_chain():
    for field_name, forged in [("gps_accuracy", 9999), ("notes", "FORGED"), ("otp_verified", True), ("device_id", "X")]:
        m = CylinderStateMachine("CYL-H")
        m.transition(CylinderState.LOADED, timestamp=datetime(2026, 10, 1, 8, 0), gps_accuracy=20, notes="scan ok")
        setattr(m.history[0], field_name, forged)
        assert not m.verify_chain_integrity()["intact"], field_name


def test_d22_the_otp_result_is_stored_with_the_event():
    m = _journey()
    t = datetime(2026, 10, 1, 9, 0)
    m.transition(CylinderState.OUT_FOR_DELIVERY, timestamp=t)
    m.transition(CylinderState.APPROACHING, timestamp=t + timedelta(minutes=20))
    assert m.transition(CylinderState.CONFIRMED, timestamp=t + timedelta(minutes=25), otp_verified=True)["allowed"]
    assert m.history[-1].otp_verified is True


def test_d22_event_dated_in_the_future_is_rejected():
    m = CylinderStateMachine("CYL-F")
    result = m.transition(CylinderState.LOADED, timestamp=datetime.now(timezone.utc) + timedelta(days=365))
    assert not result["allowed"] and result["fraud_flag"]["flag_type"] == "TIMESTAMP_IN_FUTURE"
    ok = CylinderStateMachine("CYL-F2").transition(CylinderState.LOADED, timestamp=datetime.now(timezone.utc) + timedelta(minutes=2))
    assert ok["allowed"], "a few minutes of clock drift is tolerated"


def test_d22_a_rejected_attempt_is_written_into_the_chain_and_a_reviewer_can_clear_it():
    m = CylinderStateMachine("CYL-R")
    now = datetime.now(timezone.utc)
    m.transition(CylinderState.LOADED, timestamp=now - timedelta(minutes=30))
    m.transition(CylinderState.COMPLETE, timestamp=now - timedelta(minutes=20))        # a mistaken request
    assert m.current_state == CylinderState.FLAGGED
    assert m.history[-1].to_state == CylinderState.FLAGGED and m.history[-1].notes.startswith("ILLEGAL_TRANSITION")
    del m.history[-1]                                                                    # try to hide the flag...
    m.history.append(m.history[0])
    assert not m.verify_chain_integrity()["intact"]                                      # ...the chain shows it

    m = CylinderStateMachine("CYL-R2")
    m.transition(CylinderState.LOADED, timestamp=now - timedelta(minutes=30))
    m.transition(CylinderState.COMPLETE, timestamp=now - timedelta(minutes=20))
    resolved = m.resolve_flag("REVIEWER-7", "duplicate scan")
    assert resolved["resolved"] and m.current_state == CylinderState.LOADED
    assert m.history[-1].agent_id == "REVIEWER-7" and "FLAG_RESOLVED" in m.history[-1].notes
    assert m.transition(CylinderState.IN_TRANSIT, timestamp=datetime.now(timezone.utc))["allowed"], "the journey continues"
    assert m.verify_chain_integrity()["intact"]
    assert not m.resolve_flag("REVIEWER-7")["resolved"], "nothing left to resolve"


def test_d22_cylinder_must_be_picked_up_by_the_agent_it_was_assigned_to():
    m = _journey(3)
    t = datetime(2026, 10, 1, 9, 0)
    m.transition(CylinderState.ASSIGNED, timestamp=t, agent_id="AGT-01")
    result = m.transition(CylinderState.OUT_FOR_DELIVERY, timestamp=t + timedelta(minutes=5), agent_id="AGT-99")
    assert not result["allowed"] and result["fraud_flag"]["flag_type"] == "AGENT_MISMATCH"


# ─── D11: no hard-coded rupee value ───────────────────────────────────────────

def test_d11_rupee_value_appears_only_when_a_price_is_supplied():
    def gap_text(**kw):
        r = DistributorReconciliation("D", "Test", **kw)
        r.allocate(10)
        return next(f["description"] for f in r.reconcile().flags if f["type"] == "STOCK_MISMATCH")
    assert "₹" not in gap_text()
    assert "estimated value ₹9,000" in gap_text(cylinder_value_rs=900)


# ─── D8: haversine distance ───────────────────────────────────────────────────

def test_d8_haversine_known_distances():
    assert _haversine_m(0, 0, 1, 0) == pytest.approx(111_195, rel=1e-3)     # 1° of latitude
    assert _haversine_m(60, 0, 60, 1) == pytest.approx(55_597, rel=1e-3)    # 1° of longitude at 60° N


# ─── D9: naive and aware timestamps can be mixed ──────────────────────────────

def test_d9_naive_then_aware_timestamps():
    m = CylinderStateMachine("CYL-TZ")
    assert m.transition(CylinderState.LOADED, timestamp=datetime(2026, 10, 1, 8, 0))["allowed"]
    later = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)
    assert m.transition(CylinderState.IN_TRANSIT, timestamp=later)["allowed"]
    assert m.verify_chain_integrity()["intact"]
