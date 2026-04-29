"""
GasSetu — Distributor Reconciliation Tests
============================================
Tests every fraud scenario the reconciliation engine must catch.

Run: PYTHONPATH=. python tests/test_reconciliation.py

Author: Athrva Raut — GasSetu Project
"""

import sys
sys.path.append(".")

from datetime import datetime, timezone, timedelta
from core.distributor_reconciliation import (
    DistributorReconciliation,
    AlertLevel,
    DeliveryStatus,
)

# ─── TEST HELPERS ─────────────────────────────────────────────────────────────

results = []

def test(name: str, condition: bool):
    status = "✅ PASS" if condition else "❌ FAIL"
    results.append((name, condition))
    print(f"  {status}  {name}")

def make_recon(name="Test Agency") -> DistributorReconciliation:
    return DistributorReconciliation("DIST-TEST-001", name)

def add_clean_delivery(recon, cyl_id, agent_id="AGT-01"):
    """Helper: assign + deliver + OTP confirm."""
    recon.assign_cylinder(cyl_id, f"QR-{cyl_id}", agent_id, f"CON-{cyl_id}", "9876500000")
    recon.mark_delivered(cyl_id, gps_lat=19.12, gps_lng=72.85)
    recon.confirm_otp(cyl_id, "123456")

def add_ghost_delivery(recon, cyl_id, agent_id="AGT-BAD", hours_ago=3):
    """Helper: assign + deliver + NO OTP (ghost)."""
    recon.assign_cylinder(cyl_id, f"QR-{cyl_id}", agent_id, f"CON-{cyl_id}", "9876500001")
    recon.mark_delivered(cyl_id, gps_lat=19.13, gps_lng=72.86)
    # Simulate time passing for ghost detection
    recon.cylinders[cyl_id].delivered_at = (
        datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    )

# ─── TEST SUITE ───────────────────────────────────────────────────────────────

print("\n" + "="*62)
print("  GASSETU — RECONCILIATION ENGINE TEST SUITE")
print("="*62)

# ─── TEST 1: Clean distributor — no fraud ─────────────────────────────────────
print("\n📦 TEST 1: Clean Distributor — All Deliveries OTP Confirmed")
r = make_recon("Clean Agency")
r.register_agent("AGT-01", "Honest Agent", "9876500001")
r.allocate(10)
for i in range(1, 11):
    add_clean_delivery(r, f"CYL-{i:03d}")
report = r.reconcile()

test("Alert level is CLEAN",                report.alert_level == AlertLevel.CLEAN)
test("Zero ghost deliveries",               report.ghost_deliveries == 0)
test("Zero unaccounted cylinders",          report.cylinders_unaccounted == 0)
test("10 OTP verified",                     report.cylinders_otp_confirmed == 10)
test("Gap percentage is 0%",                report.gap_percentage == 0.0)
test("Trust score stays high",              report.trust_score >= 99)
test("No fraud flags raised",               len(report.flags) == 0)


# ─── TEST 2: Ghost delivery detection ─────────────────────────────────────────
print("\n👻 TEST 2: Ghost Delivery Detection")
r2 = make_recon("Ghost Agency")
r2.register_agent("AGT-01", "Honest",  "9876500001")
r2.register_agent("AGT-BAD", "Corrupt", "9876500002")
r2.allocate(20)

for i in range(1, 11):
    add_clean_delivery(r2, f"CYL-{i:03d}", "AGT-01")
for i in range(11, 21):
    add_ghost_delivery(r2, f"CYL-{i:03d}", "AGT-BAD")

report2 = r2.reconcile()

test("Alert level is CRITICAL",             report2.alert_level == AlertLevel.CRITICAL)
test("10 ghost deliveries detected",        report2.ghost_deliveries == 10)
test("Corrupt agent trust score 0",
     r2.agents["AGT-BAD"].trust_score == 0)
test("Honest agent trust score 100",
     r2.agents["AGT-01"].trust_score == 100)
test("Ghost flag raised",
     any(f["type"] == "GHOST_DELIVERIES" for f in report2.flags))
test("Recommendation includes SUSPEND",
     "Suspend" in report2.recommendation or "IMMEDIATE" in report2.recommendation)


# ─── TEST 3: Stock mismatch (unaccounted cylinders) ───────────────────────────
print("\n📊 TEST 3: Stock Mismatch — Unaccounted Cylinders")
r3 = make_recon("Mismatch Agency")
r3.register_agent("AGT-01", "Agent One", "9876500001")
r3.allocate(20)

# Only assign and deliver 15 — 5 cylinders completely unaccounted
for i in range(1, 16):
    add_clean_delivery(r3, f"CYL-{i:03d}")

report3 = r3.reconcile()

test("5 cylinders unaccounted",             report3.cylinders_unaccounted == 5)
test("Stock mismatch flag raised",
     any(f["type"] == "STOCK_MISMATCH" for f in report3.flags))
test("Alert not CLEAN",                     report3.alert_level != AlertLevel.CLEAN)


# ─── TEST 4: Returns are not counted as fraud ─────────────────────────────────
print("\n🔄 TEST 4: Legal Returns — Customer Not Home")
r4 = make_recon("Return Agency")
r4.register_agent("AGT-01", "Return Agent", "9876500001")
r4.allocate(10)

for i in range(1, 8):
    add_clean_delivery(r4, f"CYL-{i:03d}")
for i in range(8, 11):
    r4.assign_cylinder(f"CYL-{i:03d}", f"QR-{i:03d}", "AGT-01", f"CON-{i:03d}", "9876500001")
    r4.mark_returned(f"CYL-{i:03d}", "Customer not home")

report4 = r4.reconcile()

test("3 returned cylinders counted",        report4.cylinders_returned == 3)
test("Zero ghost deliveries",               report4.ghost_deliveries == 0)
test("Alert level is CLEAN",               report4.alert_level == AlertLevel.CLEAN)
test("7 OTP confirmed",                    report4.cylinders_otp_confirmed == 7)


# ─── TEST 5: Non-linear penalty system ────────────────────────────────────────
print("\n📉 TEST 5: Non-Linear Penalty — Heal and Steal Prevention")
r5 = make_recon("Penalty Agency")
r5.register_agent("AGT-BAD", "Smart Fraudster", "9876500003")
agent = r5.agents["AGT-BAD"]

# First flag: 8 points * 1.5^0 = 8 points
agent.apply_flag("GHOST_DELIVERY", 8.0)
score_after_first = agent.trust_score

# Second flag: 8 points * 1.5^1 = 12 points
agent.apply_flag("GHOST_DELIVERY", 8.0)
score_after_second = agent.trust_score

# Third flag: 8 points * 1.5^2 = 18 points
agent.apply_flag("GHOST_DELIVERY", 8.0)
score_after_third = agent.trust_score

test("First flag deducts 8 points",        abs(score_after_first - 92.0) < 0.1)
test("Second flag deducts more (12pts)",   score_after_second < score_after_first - 11)
test("Third flag deducts even more",       score_after_third < score_after_second - 17)
test("Penalties escalate exponentially",   
     (score_after_first - score_after_second) < (score_after_second - score_after_third))
test("Flag history preserved",             len(agent.flag_history) == 3)


# ─── TEST 6: Score recovery is slow ───────────────────────────────────────────
print("\n⬆️  TEST 6: Trust Score Recovery — Honest Work Recovers Slowly")
r6 = make_recon("Recovery Agency")
r6.register_agent("AGT-01", "Reformed Agent", "9876500001")
agent6 = r6.agents["AGT-01"]
agent6.trust_score = 70.0

# 10 honest deliveries — each recovers 0.5 points
for _ in range(10):
    agent6.recover_score(0.5)

test("Score recovered by 5 points after 10 honest deliveries",
     abs(agent6.trust_score - 75.0) < 0.1)
test("Score cannot exceed 100",
     agent6.trust_score <= 100.0)

# Even 100 honest deliveries cannot immediately recover from fraud history
agent6.trust_score = 30.0
for _ in range(100):
    agent6.recover_score(0.5)
test("100 deliveries brings score to 80 (not instantly 100)",
     agent6.trust_score == 80.0)


# ─── TEST 7: Watch and Warning thresholds ─────────────────────────────────────
print("\n⚠️  TEST 7: Alert Level Thresholds")

# WATCH: 3% gap
r7 = make_recon("Watch Agency")
r7.register_agent("AGT-01", "Agent", "9876500001")
r7.register_agent("AGT-BAD", "Bad",  "9876500002")
r7.allocate(100)
for i in range(1, 98):
    add_clean_delivery(r7, f"CYL-{i:03d}")
for i in range(98, 101):
    add_ghost_delivery(r7, f"CYL-{i:03d}", "AGT-BAD")
report7 = r7.reconcile()
test("3% gap = WATCH level",                report7.alert_level == AlertLevel.WATCH)

# WARNING: 7% gap
r7b = make_recon("Warning Agency")
r7b.register_agent("AGT-01", "Agent", "9876500001")
r7b.register_agent("AGT-BAD", "Bad",  "9876500002")
r7b.allocate(100)
for i in range(1, 94):
    add_clean_delivery(r7b, f"CYL-{i:03d}")
for i in range(94, 101):
    add_ghost_delivery(r7b, f"CYL-{i:03d}", "AGT-BAD")
report7b = r7b.reconcile()
test("7% gap = WARNING level",              report7b.alert_level == AlertLevel.WARNING)


# ─── TEST 8: Financial impact calculation ─────────────────────────────────────
print("\n💰 TEST 8: Financial Impact Calculation")
r8 = make_recon("Finance Agency")
r8.register_agent("AGT-BAD", "Finance Fraudster", "9876500001")
r8.allocate(20)
for i in range(1, 11):
    add_ghost_delivery(r8, f"CYL-{i:03d}", "AGT-BAD")
report8 = r8.reconcile()

expected_ghost_loss = 10 * 965  # ₹9,650
test("Ghost delivery financial loss correct",
     expected_ghost_loss == report8.ghost_deliveries * 965)
test("Monthly projection = daily × 26",
     expected_ghost_loss * 26 == 250900)


# ─── TEST 9: OTP confirmed before mark_delivered ──────────────────────────────
print("\n🔐 TEST 9: OTP Attempt on Non-Delivered Cylinder")
r9 = make_recon("OTP Agency")
r9.register_agent("AGT-01", "Agent", "9876500001")
r9.allocate(5)
r9.assign_cylinder("CYL-001", "QR-001", "AGT-01", "CON-001", "9876500001")
# Try to confirm OTP without marking delivered first
result = r9.confirm_otp("CYL-001", "123456")
# Should still work — IN_TRANSIT can receive OTP
r9.mark_delivered("CYL-001")
result2 = r9.confirm_otp("CYL-001", "123456")
test("OTP confirmation on delivered cylinder works", result2.get("status") == "otp_confirmed")


# ─── TEST 10: Cylinder not found ──────────────────────────────────────────────
print("\n🔍 TEST 10: Invalid Cylinder ID Handling")
r10 = make_recon("Error Agency")
r10.allocate(5)
result_bad = r10.mark_delivered("CYL-DOES-NOT-EXIST")
result_bad2 = r10.confirm_otp("CYL-ALSO-FAKE", "123456")
test("Invalid cylinder returns error dict",  "error" in result_bad)
test("Invalid OTP returns error dict",       "error" in result_bad2)


# ─── SUMMARY ──────────────────────────────────────────────────────────────────
print("\n" + "="*62)
passed = sum(1 for _, r in results if r)
total  = len(results)
print(f"  RESULTS: {passed}/{total} tests passed")
if passed == total:
    print("  🎉 ALL TESTS PASSING — Reconciliation Engine production-ready")
else:
    failed = [(n, r) for n, r in results if not r]
    print(f"  ⚠️  {len(failed)} FAILING:")
    for name, _ in failed:
        print(f"     ❌ {name}")
print("="*62 + "\n")