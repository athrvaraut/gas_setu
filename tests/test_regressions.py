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
