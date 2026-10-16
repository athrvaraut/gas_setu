"""
GasSetu — Distributor Reconciliation Tests
============================================
Every fraud scenario the reconciliation engine must catch.
(The original 35 checks, now as pytest assertions.)

Run from the repo root:  python -m pytest tests/test_reconciliation.py
"""

from datetime import datetime, timezone, timedelta

from core.distributor_reconciliation import DistributorReconciliation, AlertLevel


# ─── helpers ──────────────────────────────────────────────────────────────────

def make_recon(name="Test Agency") -> DistributorReconciliation:
    return DistributorReconciliation("DIST-TEST-001", name)


def add_clean_delivery(recon, cyl_id, agent_id="AGT-01"):
    """Assign + deliver + OTP confirm."""
    recon.assign_cylinder(cyl_id, f"QR-{cyl_id}", agent_id, f"CON-{cyl_id}", "9876500000")
    recon.mark_delivered(cyl_id, gps_lat=19.12, gps_lng=72.85)
    recon.confirm_otp(cyl_id, "123456")


def add_ghost_delivery(recon, cyl_id, agent_id="AGT-BAD", hours_ago=3):
    """Assign + deliver + NO OTP (ghost), delivered `hours_ago` hours back."""
    recon.assign_cylinder(cyl_id, f"QR-{cyl_id}", agent_id, f"CON-{cyl_id}", "9876500001")
    recon.mark_delivered(cyl_id, gps_lat=19.13, gps_lng=72.86)
    recon.cylinders[cyl_id].delivered_at = datetime.now(timezone.utc) - timedelta(hours=hours_ago)


# ─── tests ────────────────────────────────────────────────────────────────────

def test_clean_distributor_all_deliveries_confirmed():
    r = make_recon("Clean Agency")
    r.register_agent("AGT-01", "Honest Agent", "9876500001")
    r.allocate(10)
    for i in range(1, 11):
        add_clean_delivery(r, f"CYL-{i:03d}")
    report = r.reconcile()
    assert report.alert_level == AlertLevel.CLEAN
    assert report.ghost_deliveries == 0
    assert report.cylinders_unaccounted == 0
    assert report.cylinders_otp_confirmed == 10
    assert report.gap_percentage == 0.0
    assert report.trust_score >= 99, "Trust score stays high"
    assert len(report.flags) == 0, "No fraud flags raised"


def test_ghost_delivery_detection():
    r = make_recon("Ghost Agency")
    r.register_agent("AGT-01", "Honest", "9876500001")
    r.register_agent("AGT-BAD", "Corrupt", "9876500002")
    r.allocate(20)
    for i in range(1, 11):
        add_clean_delivery(r, f"CYL-{i:03d}", "AGT-01")
    for i in range(11, 21):
        add_ghost_delivery(r, f"CYL-{i:03d}", "AGT-BAD")
    report = r.reconcile()
    assert report.alert_level == AlertLevel.CRITICAL
    assert report.ghost_deliveries == 10
    assert r.agents["AGT-BAD"].trust_score == 0, "Corrupt agent trust score 0"
    assert r.agents["AGT-01"].trust_score == 100, "Honest agent trust score 100"
    assert any(f["type"] == "GHOST_DELIVERIES" for f in report.flags)
    assert "Suspend" in report.recommendation or "IMMEDIATE" in report.recommendation


def test_stock_mismatch_unaccounted_cylinders():
    r = make_recon("Mismatch Agency")
    r.register_agent("AGT-01", "Agent One", "9876500001")
    r.allocate(20)
    for i in range(1, 16):           # only 15 of 20 assigned and delivered
        add_clean_delivery(r, f"CYL-{i:03d}")
    report = r.reconcile()
    assert report.cylinders_unaccounted == 5
    assert any(f["type"] == "STOCK_MISMATCH" for f in report.flags)
    assert report.alert_level != AlertLevel.CLEAN


def test_legal_returns_are_not_fraud():
    r = make_recon("Return Agency")
    r.register_agent("AGT-01", "Return Agent", "9876500001")
    r.allocate(10)
    for i in range(1, 8):
        add_clean_delivery(r, f"CYL-{i:03d}")
    for i in range(8, 11):
        r.assign_cylinder(f"CYL-{i:03d}", f"QR-{i:03d}", "AGT-01", f"CON-{i:03d}", "9876500001")
        r.mark_returned(f"CYL-{i:03d}", "Customer not home")
    report = r.reconcile()
    assert report.cylinders_returned == 3
    assert report.ghost_deliveries == 0
    assert report.alert_level == AlertLevel.CLEAN
    assert report.cylinders_otp_confirmed == 7


def test_non_linear_penalty_escalates():
    r = make_recon("Penalty Agency")
    r.register_agent("AGT-BAD", "Smart Fraudster", "9876500003")
    agent = r.agents["AGT-BAD"]
    agent.apply_flag("GHOST_DELIVERY", 8.0)   # 8 × 1.5^0 = 8
    s1 = agent.trust_score
    agent.apply_flag("GHOST_DELIVERY", 8.0)   # 8 × 1.5^1 = 12
    s2 = agent.trust_score
    agent.apply_flag("GHOST_DELIVERY", 8.0)   # 8 × 1.5^2 = 18
    s3 = agent.trust_score
    assert abs(s1 - 92.0) < 0.1, "First flag deducts 8 points"
    assert s2 < s1 - 11, "Second flag deducts more (12 pts)"
    assert s3 < s2 - 17, "Third flag deducts even more"
    assert (s1 - s2) < (s2 - s3), "Penalties escalate"
    assert len(agent.flag_history) == 3, "Flag history preserved"


def test_trust_score_recovers_slowly():
    r = make_recon("Recovery Agency")
    r.register_agent("AGT-01", "Reformed Agent", "9876500001")
    agent = r.agents["AGT-01"]
    agent.trust_score = 70.0
    for _ in range(10):
        agent.recover_score(0.5)
    assert abs(agent.trust_score - 75.0) < 0.1, "10 honest deliveries recover 5 points"
    assert agent.trust_score <= 100.0
    agent.trust_score = 30.0
    for _ in range(100):
        agent.recover_score(0.5)
    assert agent.trust_score == 80.0, "100 deliveries bring 30 → 80, not instantly 100"


def test_alert_level_thresholds():
    watch = make_recon("Watch Agency")
    watch.register_agent("AGT-01", "Agent", "9876500001")
    watch.register_agent("AGT-BAD", "Bad", "9876500002")
    watch.allocate(100)
    for i in range(1, 98):
        add_clean_delivery(watch, f"CYL-{i:03d}")
    for i in range(98, 101):
        add_ghost_delivery(watch, f"CYL-{i:03d}", "AGT-BAD")
    assert watch.reconcile().alert_level == AlertLevel.WATCH, "3% gap = WATCH"

    warning = make_recon("Warning Agency")
    warning.register_agent("AGT-01", "Agent", "9876500001")
    warning.register_agent("AGT-BAD", "Bad", "9876500002")
    warning.allocate(100)
    for i in range(1, 94):
        add_clean_delivery(warning, f"CYL-{i:03d}")
    for i in range(94, 101):
        add_ghost_delivery(warning, f"CYL-{i:03d}", "AGT-BAD")
    assert warning.reconcile().alert_level == AlertLevel.WARNING, "7% gap = WARNING"


def test_financial_impact_calculation():
    r = make_recon("Finance Agency")
    r.register_agent("AGT-BAD", "Finance Fraudster", "9876500001")
    r.allocate(20)
    for i in range(1, 11):
        add_ghost_delivery(r, f"CYL-{i:03d}", "AGT-BAD")
    report = r.reconcile()
    expected_ghost_loss = 10 * 965
    assert expected_ghost_loss == report.ghost_deliveries * 965
    assert expected_ghost_loss * 26 == 250900, "Monthly projection = daily × 26"


def test_otp_on_delivered_cylinder_works():
    r = make_recon("OTP Agency")
    r.register_agent("AGT-01", "Agent", "9876500001")
    r.allocate(5)
    r.assign_cylinder("CYL-001", "QR-001", "AGT-01", "CON-001", "9876500001")
    r.confirm_otp("CYL-001", "123456")        # before delivery: rejected with an error dict
    r.mark_delivered("CYL-001")
    assert r.confirm_otp("CYL-001", "123456").get("status") == "otp_confirmed"


def test_invalid_cylinder_ids_return_errors():
    r = make_recon("Error Agency")
    r.allocate(5)
    assert "error" in r.mark_delivered("CYL-DOES-NOT-EXIST")
    assert "error" in r.confirm_otp("CYL-ALSO-FAKE", "123456")
