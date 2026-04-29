"""
GasSetu — State Machine Tests
==============================
Tests every fraud scenario we designed against.
Run with: python tests/test_state_machine.py
"""

import sys
sys.path.append("..")

from datetime import datetime, timedelta
from core.state_machine import CylinderStateMachine, CylinderState

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []

def test(name: str, condition: bool):
    status = PASS if condition else FAIL
    results.append((name, condition))
    print(f"  {status}  {name}")


print("\n" + "="*60)
print("  GASSETU — FRAUD DETECTION ENGINE TESTS")
print("="*60)

# ─── TEST 1: Normal Legal Journey ─────────────────────────────────────────────
print("\n📦 TEST 1: Normal Legal Delivery Journey")
m = CylinderStateMachine("CYL-TEST-001")
t = datetime(2024, 1, 15, 8, 0, 0)

r1 = m.transition(CylinderState.LOADED,            timestamp=t,                       agent_id="AGT-01", gps_lat=19.05, gps_lng=72.88, accelerometer=2.1)
r2 = m.transition(CylinderState.IN_TRANSIT,        timestamp=t+timedelta(minutes=10), agent_id="AGT-01", gps_lat=19.06, gps_lng=72.89, accelerometer=3.4)
r3 = m.transition(CylinderState.AT_DISTRIBUTOR,    timestamp=t+timedelta(minutes=40), agent_id="AGT-01", gps_lat=19.07, gps_lng=72.90, accelerometer=1.1)
r4 = m.transition(CylinderState.ASSIGNED,          timestamp=t+timedelta(minutes=50), agent_id="AGT-02")
r5 = m.transition(CylinderState.OUT_FOR_DELIVERY,  timestamp=t+timedelta(hours=2),    agent_id="AGT-02", gps_lat=19.08, gps_lng=72.91, accelerometer=2.8)
r6 = m.transition(CylinderState.APPROACHING,       timestamp=t+timedelta(hours=2,minutes=20), agent_id="AGT-02", gps_lat=19.09, gps_lng=72.92, accelerometer=1.5)
r7 = m.transition(CylinderState.CONFIRMED,         timestamp=t+timedelta(hours=2,minutes=25), agent_id="AGT-02", otp_verified=True)
r8 = m.transition(CylinderState.COMPLETE,          timestamp=t+timedelta(hours=2,minutes=26), agent_id="AGT-02")

test("All 8 transitions allowed",          all(r["allowed"] for r in [r1,r2,r3,r4,r5,r6,r7,r8]))
test("Final state is COMPLETE",            m.current_state == CylinderState.COMPLETE)
test("8 events in history",                len(m.history) == 8)
test("Zero fraud flags raised",            len(m.flags) == 0)
test("Cryptographic chain intact",         m.verify_chain_integrity()["intact"])


# ─── TEST 2: Ghost Delivery (Skip States) ─────────────────────────────────────
print("\n👻 TEST 2: Ghost Delivery — Agent Jumps Straight to COMPLETE")
m2 = CylinderStateMachine("CYL-TEST-002")
m2.transition(CylinderState.LOADED,           timestamp=datetime(2024,1,15,8,0),  agent_id="AGT-BAD")
m2.transition(CylinderState.IN_TRANSIT,       timestamp=datetime(2024,1,15,8,10), agent_id="AGT-BAD")
m2.transition(CylinderState.AT_DISTRIBUTOR,   timestamp=datetime(2024,1,15,8,40), agent_id="AGT-BAD")
m2.transition(CylinderState.ASSIGNED,         timestamp=datetime(2024,1,15,8,50), agent_id="AGT-BAD")

# FRAUD: Skip OUT_FOR_DELIVERY, APPROACHING, CONFIRMED — jump straight to COMPLETE
result = m2.transition(CylinderState.COMPLETE, timestamp=datetime(2024,1,15,9,0), agent_id="AGT-BAD")

test("Ghost delivery BLOCKED",             not result["allowed"])
test("Fraud flag raised",                  result["fraud_flag"] is not None)
test("Flag type is ILLEGAL_TRANSITION",    result["fraud_flag"]["flag_type"] == "ILLEGAL_TRANSITION")
test("Severity is CRITICAL",               result["fraud_flag"]["severity"] == "CRITICAL")
test("State changed to FLAGGED",           m2.current_state == CylinderState.FLAGGED)


# ─── TEST 3: Confirmation Without OTP ────────────────────────────────────────
print("\n🔐 TEST 3: Agent Marks CONFIRMED Without Customer OTP")
m3 = CylinderStateMachine("CYL-TEST-003")
t3 = datetime(2024,1,15,10,0)
m3.transition(CylinderState.LOADED,           timestamp=t3)
m3.transition(CylinderState.IN_TRANSIT,       timestamp=t3+timedelta(minutes=10))
m3.transition(CylinderState.AT_DISTRIBUTOR,   timestamp=t3+timedelta(minutes=40))
m3.transition(CylinderState.ASSIGNED,         timestamp=t3+timedelta(minutes=50), agent_id="AGT-03")
m3.transition(CylinderState.OUT_FOR_DELIVERY, timestamp=t3+timedelta(hours=1),    agent_id="AGT-03", gps_lat=19.05, gps_lng=72.88, accelerometer=2.0)
m3.transition(CylinderState.APPROACHING,      timestamp=t3+timedelta(hours=1,minutes=20), agent_id="AGT-03", gps_lat=19.051, gps_lng=72.881, accelerometer=1.0)

# FRAUD: Mark confirmed without otp_verified=True
result = m3.transition(CylinderState.CONFIRMED, timestamp=t3+timedelta(hours=1,minutes=25), agent_id="AGT-03", otp_verified=False)

test("Confirmation without OTP BLOCKED",   not result["allowed"])
test("Correct flag type",                  result["fraud_flag"]["flag_type"] == "CONFIRMATION_WITHOUT_OTP")
test("Severity HIGH",                      result["fraud_flag"]["severity"] == "HIGH")


# ─── TEST 4: Mock Location Detection ─────────────────────────────────────────
print("\n📍 TEST 4: Mock Location App — GPS Moves But Phone Is Stationary")
m4 = CylinderStateMachine("CYL-TEST-004")
t4 = datetime(2024,1,15,11,0)

# First transition — establishes baseline (phone in Andheri, barely moving)
m4.transition(CylinderState.LOADED, timestamp=t4, agent_id="AGT-04",
              gps_lat=19.1136, gps_lng=72.8697, accelerometer=0.2)

# Agent turns on mock location app — GPS jumps 3km to Kandivali but phone doesn't move
result = m4.transition(CylinderState.IN_TRANSIT, timestamp=t4+timedelta(minutes=2), agent_id="AGT-04",
                        gps_lat=19.2041,  # 10km north — Kandivali
                        gps_lng=72.8511,
                        accelerometer=0.18)  # Phone barely moved

test("Mock location BLOCKED",              not result["allowed"])
test("Correct flag type",                  result["fraud_flag"]["flag_type"] == "MOCK_LOCATION_DETECTED")
test("Severity HIGH",                      result["fraud_flag"]["severity"] == "HIGH")
test("Evidence contains GPS distance",     "gps_distance_moved" in result["fraud_flag"]["evidence"])


# ─── TEST 5: Device Clock Manipulation (Time Travel Attack) ──────────────────
print("\n⏰ TEST 5: Agent Manipulates Device Clock (Time Travel Attack)")
m5 = CylinderStateMachine("CYL-TEST-005")
t5 = datetime(2024,1,15,14,0)

m5.transition(CylinderState.LOADED,     timestamp=t5,                      agent_id="AGT-05")
m5.transition(CylinderState.IN_TRANSIT, timestamp=t5+timedelta(minutes=10), agent_id="AGT-05")

# Agent changes phone clock backwards, tries to insert a fake past event
fake_past_time = t5 - timedelta(hours=2)  # claims this happened 2 hours ago
result = m5.transition(CylinderState.AT_DISTRIBUTOR, timestamp=fake_past_time, agent_id="AGT-05")

test("Time reversal BLOCKED",              not result["allowed"])
test("Correct flag type",                  result["fraud_flag"]["flag_type"] == "TIMESTAMP_REVERSAL")
test("Severity CRITICAL",                  result["fraud_flag"]["severity"] == "CRITICAL")


# ─── TEST 6: Cryptographic Chain Integrity ────────────────────────────────────
print("\n🔗 TEST 6: Cryptographic Chain — Tamper Detection")
m6 = CylinderStateMachine("CYL-TEST-006")
t6 = datetime(2024,1,15,9,0)
m6.transition(CylinderState.LOADED,     timestamp=t6,                       agent_id="AGT-06")
m6.transition(CylinderState.IN_TRANSIT, timestamp=t6+timedelta(minutes=10), agent_id="AGT-06")

# Verify clean chain
clean_result = m6.verify_chain_integrity()
test("Clean chain is intact",              clean_result["intact"])
test("Chain has 2 events",                 clean_result["total_events"] == 2)

# Tamper with a record
m6.history[0].gps_lat = 99.9999  # silently alter GPS coordinate
tampered_result = m6.verify_chain_integrity()
test("Tampered chain is detected",         not tampered_result["intact"])
test("Broken link identified",             len(tampered_result["broken_links"]) > 0)


# ─── TEST 7: Distributor Returns to Legal State ───────────────────────────────
print("\n🔄 TEST 7: Legal Return Flow (Agent Can't Deliver, Returns Cylinder)")
m7 = CylinderStateMachine("CYL-TEST-007")
t7 = datetime(2024,1,15,16,0)
m7.transition(CylinderState.LOADED,           timestamp=t7,                      agent_id="AGT-07")
m7.transition(CylinderState.IN_TRANSIT,       timestamp=t7+timedelta(minutes=10))
m7.transition(CylinderState.AT_DISTRIBUTOR,   timestamp=t7+timedelta(minutes=40))
m7.transition(CylinderState.ASSIGNED,         timestamp=t7+timedelta(minutes=50), agent_id="AGT-07")
m7.transition(CylinderState.OUT_FOR_DELIVERY, timestamp=t7+timedelta(hours=1),    agent_id="AGT-07", gps_lat=19.05, gps_lng=72.88, accelerometer=2.0)

# Customer not home — agent returns to distributor (legal)
result = m7.transition(CylinderState.AT_DISTRIBUTOR, timestamp=t7+timedelta(hours=1,minutes=30), agent_id="AGT-07")
test("Legal return to distributor ALLOWED", result["allowed"])
test("State is AT_DISTRIBUTOR",             m7.current_state == CylinderState.AT_DISTRIBUTOR)


# ─── SUMMARY ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
passed = sum(1 for _, r in results if r)
total  = len(results)
print(f"  RESULTS: {passed}/{total} tests passed")
if passed == total:
    print("  🎉 ALL TESTS PASSING — State Machine is production-ready")
else:
    failed = [(n, r) for n, r in results if not r]
    print(f"  ⚠️  {len(failed)} FAILING:")
    for name, _ in failed:
        print(f"     ❌ {name}")
print("="*60 + "\n")