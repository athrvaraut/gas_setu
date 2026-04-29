"""
GasSetu — Distributor Reconciliation Engine
=============================================
Catches ghost deliveries — the physical fraud software alone cannot stop.

How ghost deliveries work:
  1. Distributor gets 500 cylinders allocated
  2. Delivery agent marks 480 as "delivered" in the system
  3. Actually only 440 were delivered — 40 went to black market
  4. System shows "delivered" — customer never got cylinder
  5. Distributor gets fresh stock allocation based on fake delivery count

This module:
  - Tracks every cylinder from allocation → assignment → OTP confirmation
  - Calculates the gap between "marked delivered" and "OTP confirmed"
  - Flags distributors where gap exceeds threshold
  - Generates audit reports for government/OMC review
  - Scores each distributor with a trust rating

Author: Athrva Raut — GasSetu Project
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional
from enum import Enum
import json


# ─── ENUMS ────────────────────────────────────────────────────────────────────

class AlertLevel(str, Enum):
    CLEAN    = "CLEAN"      # Gap < 2% — normal operational variance
    WATCH    = "WATCH"      # Gap 2-5% — monitor closely
    WARNING  = "WARNING"    # Gap 5-10% — flag for inspection
    CRITICAL = "CRITICAL"   # Gap > 10% — suspend allocation, audit now


class DeliveryStatus(str, Enum):
    ASSIGNED     = "ASSIGNED"       # Cylinder given to agent
    IN_TRANSIT   = "IN_TRANSIT"     # Agent picked up, moving
    ATTEMPTED    = "ATTEMPTED"      # Agent claims tried to deliver
    DELIVERED    = "DELIVERED"      # Agent marked delivered in system
    OTP_VERIFIED = "OTP_VERIFIED"   # Customer confirmed via OTP/missed call
    RETURNED     = "RETURNED"       # Cylinder returned to distributor
    GHOST        = "GHOST"          # Marked delivered but NO OTP — suspicious


# ─── DATA STRUCTURES ──────────────────────────────────────────────────────────

@dataclass
class CylinderRecord:
    """Single cylinder tracking record."""
    cylinder_id:    str
    qr_code:        str
    agent_id:       str
    consumer_id:    str
    consumer_phone: str
    assigned_at:    datetime
    status:         DeliveryStatus = DeliveryStatus.ASSIGNED
    delivered_at:   Optional[datetime] = None
    otp_verified_at: Optional[datetime] = None
    otp_code:       Optional[str] = None
    agent_gps_lat:  Optional[float] = None
    agent_gps_lng:  Optional[float] = None
    notes:          str = ""

    def is_ghost(self) -> bool:
        """
        Ghost delivery: system says DELIVERED but no OTP confirmation.
        Time window: if delivered >2 hours ago and still no OTP = ghost.
        """
        if self.status == DeliveryStatus.DELIVERED and not self.otp_verified_at:
            if self.delivered_at:
                hours_since = (datetime.now(timezone.utc) - self.delivered_at).seconds / 3600
                return hours_since > 2
        return self.status == DeliveryStatus.GHOST

    def time_to_delivery(self) -> Optional[float]:
        """Minutes from assignment to delivery."""
        if self.delivered_at and self.assigned_at:
            return (self.delivered_at - self.assigned_at).seconds / 60
        return None


@dataclass
class AgentRecord:
    """Delivery agent performance tracking."""
    agent_id:           str
    agent_name:         str
    phone:              str
    distributor_id:     str
    total_assigned:     int = 0
    total_delivered:    int = 0
    total_otp_verified: int = 0
    total_ghost:        int = 0
    total_returned:     int = 0
    trust_score:        float = 100.0   # Starts at 100, decreases on fraud flags
    flag_history:       list = field(default_factory=list)

    def otp_verification_rate(self) -> float:
        if self.total_delivered == 0:
            return 0.0
        return (self.total_otp_verified / self.total_delivered) * 100

    def ghost_rate(self) -> float:
        if self.total_delivered == 0:
            return 0.0
        return (self.total_ghost / self.total_delivered) * 100

    def apply_flag(self, reason: str, points_deducted: float):
        """
        Non-linear penalty system — repeated flags cost exponentially more.
        Heal and Steal prevention: score recovers slowly, history never clears.
        """
        # Exponential penalty: 1st flag = 5pts, 2nd = 12pts, 3rd = 25pts
        flag_count = len(self.flag_history)
        actual_deduction = points_deducted * (1.5 ** flag_count)
        actual_deduction = min(actual_deduction, 40)  # Cap at 40 per flag

        self.trust_score = max(0, self.trust_score - actual_deduction)
        self.flag_history.append({
            "reason":    reason,
            "deducted":  actual_deduction,
            "score_after": self.trust_score,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def recover_score(self, points: float = 1.0):
        """Score recovers slowly for honest deliveries. Max 100."""
        self.trust_score = min(100.0, self.trust_score + points)


@dataclass
class ReconciliationReport:
    """Daily reconciliation report for one distributor."""
    distributor_id:     str
    distributor_name:   str
    report_date:        str
    cylinders_allocated: int
    cylinders_assigned:  int
    cylinders_delivered: int    # Agent marked delivered
    cylinders_otp_confirmed: int  # Customer confirmed via OTP
    cylinders_returned:  int
    cylinders_unaccounted: int  # Allocated - (delivered + returned)
    ghost_deliveries:    int    # Delivered but no OTP
    alert_level:         AlertLevel
    gap_percentage:      float
    trust_score:         float
    flags:               list
    agent_breakdown:     list
    recommendation:      str

    def to_dict(self) -> dict:
        return {
            "distributor_id":        self.distributor_id,
            "distributor_name":      self.distributor_name,
            "report_date":           self.report_date,
            "cylinders_allocated":   self.cylinders_allocated,
            "cylinders_assigned":    self.cylinders_assigned,
            "cylinders_delivered":   self.cylinders_delivered,
            "cylinders_otp_confirmed": self.cylinders_otp_confirmed,
            "cylinders_returned":    self.cylinders_returned,
            "cylinders_unaccounted": self.cylinders_unaccounted,
            "ghost_deliveries":      self.ghost_deliveries,
            "alert_level":           self.alert_level,
            "gap_percentage":        round(self.gap_percentage, 2),
            "trust_score":           round(self.trust_score, 1),
            "flags":                 self.flags,
            "agent_breakdown":       self.agent_breakdown,
            "recommendation":        self.recommendation,
        }


# ─── MAIN RECONCILIATION ENGINE ───────────────────────────────────────────────

class DistributorReconciliation:
    """
    Tracks cylinder movements for a distributor.
    Runs nightly reconciliation to detect ghost deliveries.

    Usage:
        recon = DistributorReconciliation("DIST-MUM-4221", "Andheri Gas Agency")
        recon.allocate(500)
        recon.record_delivery("CYL-001", "AGT-01", "CON-001", "9876543210")
        recon.confirm_otp("CYL-001", "492817")
        report = recon.reconcile()
    """

    # Alert thresholds
    WATCH_THRESHOLD    = 2.0    # % gap
    WARNING_THRESHOLD  = 5.0
    CRITICAL_THRESHOLD = 10.0

    def __init__(self, distributor_id: str, distributor_name: str):
        self.distributor_id   = distributor_id
        self.distributor_name = distributor_name
        self.cylinders:  dict[str, CylinderRecord] = {}
        self.agents:     dict[str, AgentRecord]    = {}
        self.allocated:  int = 0
        self.trust_score: float = 100.0
        self.flag_history: list = []
        self.created_at  = datetime.now(timezone.utc)

    # ── ALLOCATION ────────────────────────────────────────────────────────────

    def allocate(self, count: int):
        """Record how many cylinders were allocated to this distributor today."""
        self.allocated = count

    # ── AGENT MANAGEMENT ──────────────────────────────────────────────────────

    def register_agent(self, agent_id: str, name: str, phone: str):
        self.agents[agent_id] = AgentRecord(
            agent_id       = agent_id,
            agent_name     = name,
            phone          = phone,
            distributor_id = self.distributor_id,
        )

    # ── CYLINDER TRACKING ─────────────────────────────────────────────────────

    def assign_cylinder(
        self,
        cylinder_id:    str,
        qr_code:        str,
        agent_id:       str,
        consumer_id:    str,
        consumer_phone: str,
    ) -> dict:
        """Record cylinder being assigned to delivery agent."""
        if agent_id not in self.agents:
            self.register_agent(agent_id, f"Agent-{agent_id}", "Unknown")

        record = CylinderRecord(
            cylinder_id    = cylinder_id,
            qr_code        = qr_code,
            agent_id       = agent_id,
            consumer_id    = consumer_id,
            consumer_phone = consumer_phone,
            assigned_at    = datetime.now(timezone.utc),
            status         = DeliveryStatus.ASSIGNED,
        )
        self.cylinders[cylinder_id] = record
        self.agents[agent_id].total_assigned += 1

        return {"status": "assigned", "cylinder_id": cylinder_id, "agent_id": agent_id}

    def mark_delivered(
        self,
        cylinder_id: str,
        gps_lat: Optional[float] = None,
        gps_lng: Optional[float] = None,
    ) -> dict:
        """
        Agent marks cylinder as delivered in their app.
        NOTE: This does NOT confirm actual delivery — that requires OTP.
        This is where ghost deliveries are created.
        """
        if cylinder_id not in self.cylinders:
            return {"error": f"Cylinder {cylinder_id} not found"}

        record = self.cylinders[cylinder_id]
        record.status        = DeliveryStatus.DELIVERED
        record.delivered_at  = datetime.now(timezone.utc)
        record.agent_gps_lat = gps_lat
        record.agent_gps_lng = gps_lng

        self.agents[record.agent_id].total_delivered += 1

        return {
            "status":      "marked_delivered",
            "cylinder_id": cylinder_id,
            "warning":     "Pending OTP confirmation from customer",
        }

    def confirm_otp(self, cylinder_id: str, otp_code: str) -> dict:
        """
        Customer confirms receipt via OTP/missed call.
        This is the ONLY valid proof of physical delivery.
        """
        if cylinder_id not in self.cylinders:
            return {"error": f"Cylinder {cylinder_id} not found"}

        record = self.cylinders[cylinder_id]

        if record.status not in [DeliveryStatus.DELIVERED, DeliveryStatus.IN_TRANSIT]:
            return {
                "error": f"Cannot confirm OTP — cylinder status is {record.status}",
                "cylinder_id": cylinder_id,
            }

        record.status          = DeliveryStatus.OTP_VERIFIED
        record.otp_verified_at = datetime.now(timezone.utc)
        record.otp_code        = otp_code

        agent = self.agents.get(record.agent_id)
        if agent:
            agent.total_otp_verified += 1
            agent.recover_score(0.5)   # Honest delivery = slow score recovery

        return {
            "status":      "otp_confirmed",
            "cylinder_id": cylinder_id,
            "verified_at": record.otp_verified_at.isoformat(),
        }

    def mark_returned(self, cylinder_id: str, reason: str = "") -> dict:
        """Customer not home — cylinder returned to distributor."""
        if cylinder_id not in self.cylinders:
            return {"error": f"Cylinder {cylinder_id} not found"}

        record         = self.cylinders[cylinder_id]
        record.status  = DeliveryStatus.RETURNED
        record.notes   = reason

        self.agents[record.agent_id].total_returned += 1

        return {"status": "returned", "cylinder_id": cylinder_id}

    # ── RECONCILIATION ────────────────────────────────────────────────────────

    def reconcile(self) -> ReconciliationReport:
        """
        Run nightly reconciliation.
        The core fraud detection for physical ghost deliveries.

        Math:
          Allocated = Delivered(OTP) + Returned + InTransit + Unaccounted
          Unaccounted > threshold → FRAUD FLAG
        """
        now = datetime.now(timezone.utc)

        # Count cylinder statuses
        assigned     = sum(1 for c in self.cylinders.values() if c.status == DeliveryStatus.ASSIGNED)
        in_transit   = sum(1 for c in self.cylinders.values() if c.status == DeliveryStatus.IN_TRANSIT)
        delivered    = sum(1 for c in self.cylinders.values() if c.status in [DeliveryStatus.DELIVERED, DeliveryStatus.OTP_VERIFIED])
        otp_verified = sum(1 for c in self.cylinders.values() if c.status == DeliveryStatus.OTP_VERIFIED)
        returned     = sum(1 for c in self.cylinders.values() if c.status == DeliveryStatus.RETURNED)

        # Ghost deliveries: marked delivered but no OTP after 2+ hours
        ghost_list = [c for c in self.cylinders.values() if c.is_ghost()]
        ghost_count = len(ghost_list)

        # Mark them explicitly
        for c in ghost_list:
            c.status = DeliveryStatus.GHOST
            if c.agent_id in self.agents:
                self.agents[c.agent_id].total_ghost += 1
                self.agents[c.agent_id].apply_flag(
                    reason="GHOST_DELIVERY_DETECTED",
                    points_deducted=8.0,
                )

        # Unaccounted = allocated minus everything we can account for
        accounted    = otp_verified + returned + assigned + in_transit
        unaccounted  = max(0, self.allocated - accounted)

        # Gap percentage (based on allocated)
        if self.allocated > 0:
            gap_pct = unaccounted / self.allocated * 100
        else:
            gap_pct = 0.0

        # Determine alert level
        if gap_pct >= self.CRITICAL_THRESHOLD:
            level = AlertLevel.CRITICAL
        elif gap_pct >= self.WARNING_THRESHOLD:
            level = AlertLevel.WARNING
        elif gap_pct >= self.WATCH_THRESHOLD:
            level = AlertLevel.WATCH
        else:
            level = AlertLevel.CLEAN

        # Generate flags
        flags = []

        if ghost_count > 0:
            ghost_agents = list({c.agent_id for c in ghost_list})
            flags.append({
                "type":        "GHOST_DELIVERIES",
                "severity":    "CRITICAL" if ghost_count > 10 else "HIGH",
                "count":       ghost_count,
                "agents":      ghost_agents,
                "description": (
                    f"{ghost_count} cylinders marked DELIVERED in system "
                    f"but NO customer OTP received after 2+ hours. "
                    f"Agents involved: {', '.join(ghost_agents)}. "
                    f"Physical inspection required."
                ),
            })

        if unaccounted > 0:
            flags.append({
                "type":        "STOCK_MISMATCH",
                "severity":    "HIGH" if unaccounted > 5 else "MEDIUM",
                "count":       unaccounted,
                "description": (
                    f"{unaccounted} cylinders unaccounted for. "
                    f"Allocated: {self.allocated}, "
                    f"Verified+Returned+InTransit: {accounted}. "
                    f"Gap: {unaccounted} cylinders (₹{unaccounted * 965:,} potential subsidy loss)."
                ),
            })

        # Agent-level breakdown
        agent_breakdown = []
        for agent in self.agents.values():
            agent_flags = []
            if agent.ghost_rate() > 5:
                agent_flags.append(f"Ghost rate {agent.ghost_rate():.1f}%")
            if agent.otp_verification_rate() < 80:
                agent_flags.append(f"OTP rate only {agent.otp_verification_rate():.1f}%")
            if agent.trust_score < 70:
                agent_flags.append(f"Trust score LOW: {agent.trust_score:.0f}/100")

            agent_breakdown.append({
                "agent_id":       agent.agent_id,
                "agent_name":     agent.agent_name,
                "assigned":       agent.total_assigned,
                "delivered":      agent.total_delivered,
                "otp_verified":   agent.total_otp_verified,
                "ghost":          agent.total_ghost,
                "otp_rate":       f"{agent.otp_verification_rate():.1f}%",
                "trust_score":    f"{agent.trust_score:.0f}/100",
                "flags":          agent_flags,
                "action_needed":  len(agent_flags) > 0,
            })

        # Distributor trust score update
        if level == AlertLevel.CRITICAL:
            self.trust_score = max(0, self.trust_score - 25)
        elif level == AlertLevel.WARNING:
            self.trust_score = max(0, self.trust_score - 10)
        elif level == AlertLevel.WATCH:
            self.trust_score = max(0, self.trust_score - 3)
        else:
            self.trust_score = min(100, self.trust_score + 1)

        # Recommendation
        if level == AlertLevel.CRITICAL:
            rec = (
                f"IMMEDIATE ACTION: Suspend fresh stock allocation for {self.distributor_name}. "
                f"Dispatch field inspector within 24 hours. "
                f"File incident report with petroleum dept. "
                f"Freeze agent payment for flagged agents."
            )
        elif level == AlertLevel.WARNING:
            rec = (
                f"FLAG FOR INSPECTION: Schedule surprise visit to {self.distributor_name} "
                f"within 48 hours. Review agent GPS logs. "
                f"Hold 20% of fresh allocation pending verification."
            )
        elif level == AlertLevel.WATCH:
            rec = (
                f"MONITOR: Increase OTP verification spot-checks for {self.distributor_name}. "
                f"Send automated warning to flagged agents. "
                f"Review again tomorrow."
            )
        else:
            rec = f"CLEAN: {self.distributor_name} operating within normal parameters. No action required."

        return ReconciliationReport(
            distributor_id        = self.distributor_id,
            distributor_name      = self.distributor_name,
            report_date           = now.strftime("%Y-%m-%d"),
            cylinders_allocated   = self.allocated,
            cylinders_assigned    = assigned + in_transit,
            cylinders_delivered   = delivered,
            cylinders_otp_confirmed = otp_verified,
            cylinders_returned    = returned,
            cylinders_unaccounted = unaccounted,
            ghost_deliveries      = ghost_count,
            alert_level           = level,
            gap_percentage        = gap_pct,
            trust_score           = self.trust_score,
            flags                 = flags,
            agent_breakdown       = agent_breakdown,
            recommendation        = rec,
        )

    def get_summary(self) -> dict:
        return {
            "distributor_id":   self.distributor_id,
            "distributor_name": self.distributor_name,
            "allocated":        self.allocated,
            "total_tracked":    len(self.cylinders),
            "total_agents":     len(self.agents),
            "trust_score":      self.trust_score,
        }


# ─── DEMO ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "=" * 65)
    print("  GASSETU — DISTRIBUTOR RECONCILIATION ENGINE")
    print("  Simulating: Andheri Gas Agency, Mumbai")
    print("=" * 65)

    recon = DistributorReconciliation("DIST-MUM-4221", "Andheri Gas Agency")

    # Register agents
    recon.register_agent("AGT-01", "Suresh Kumar",  "9876501001")
    recon.register_agent("AGT-02", "Ravi Sharma",   "9876501002")
    recon.register_agent("AGT-03", "Mohan Das",     "9876501003")  # corrupt agent

    # Today's allocation
    recon.allocate(50)
    print(f"\n📦 Allocated: 50 cylinders to Andheri Gas Agency")

    # ── Honest deliveries (AGT-01) ────────────────────────────────────────────
    print("\n👤 AGT-01 Suresh Kumar — honest agent")
    for i in range(1, 16):
        cyl_id = f"CYL-{i:03d}"
        recon.assign_cylinder(cyl_id, f"QR{i:04d}", "AGT-01", f"CON-{i:03d}", f"98765{i:05d}")
        recon.mark_delivered(cyl_id, gps_lat=19.12, gps_lng=72.85)
        recon.confirm_otp(cyl_id, f"OTP{i:04d}")
    print(f"  ✅ 15 deliveries — all OTP confirmed")

    # 2 returns
    for i in range(16, 18):
        cyl_id = f"CYL-{i:03d}"
        recon.assign_cylinder(cyl_id, f"QR{i:04d}", "AGT-01", f"CON-{i:03d}", f"98765{i:05d}")
        recon.mark_returned(cyl_id, "Customer not home")
    print(f"  🔄 2 returned — customer not home")

    # ── Honest deliveries (AGT-02) ────────────────────────────────────────────
    print("\n👤 AGT-02 Ravi Sharma — honest agent")
    for i in range(18, 33):
        cyl_id = f"CYL-{i:03d}"
        recon.assign_cylinder(cyl_id, f"QR{i:04d}", "AGT-02", f"CON-{i:03d}", f"98765{i:05d}")
        recon.mark_delivered(cyl_id, gps_lat=19.13, gps_lng=72.86)
        recon.confirm_otp(cyl_id, f"OTP{i:04d}")
    print(f"  ✅ 15 deliveries — all OTP confirmed")

    # ── CORRUPT AGENT (AGT-03) — Ghost deliveries ─────────────────────────────
    print("\n👤 AGT-03 Mohan Das — 🚨 CORRUPT AGENT")
    for i in range(33, 43):
        cyl_id = f"CYL-{i:03d}"
        recon.assign_cylinder(cyl_id, f"QR{i:04d}", "AGT-03", f"CON-{i:03d}", f"98765{i:05d}")
        recon.mark_delivered(cyl_id, gps_lat=19.15, gps_lng=72.88)
        # NO OTP CONFIRMATION — cylinders went to black market restaurant
    print(f"  ❌ 10 deliveries marked — ZERO OTP confirmations")
    print(f"  ⚠️  Cylinders likely diverted to black market")

    # Simulate time passing (2+ hours) for ghost detection
    for cyl_id, record in recon.cylinders.items():
        if record.agent_id == "AGT-03" and record.status == DeliveryStatus.DELIVERED:
            record.delivered_at = datetime.now(timezone.utc) - timedelta(hours=3)

    # 5 cylinders still with AGT-03 (not yet assigned)
    print(f"\n  📦 5 cylinders unaccounted (not assigned to any agent)")

    # ── RUN RECONCILIATION ────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  Running nightly reconciliation...")
    print("─" * 65)

    report = recon.reconcile()

    print(f"\n🏭 {report.distributor_name} — {report.report_date}")
    print(f"   Alert Level  : {report.alert_level.value}")
    print(f"   Trust Score  : {report.trust_score:.0f}/100")
    print(f"   Gap          : {report.gap_percentage:.1f}%")
    print(f"\n📊 Cylinder Counts:")
    print(f"   Allocated    : {report.cylinders_allocated}")
    print(f"   OTP Verified : {report.cylinders_otp_confirmed}  ✅ (genuine deliveries)")
    print(f"   Returned     : {report.cylinders_returned}   🔄 (customer not home)")
    print(f"   Ghost        : {report.ghost_deliveries}  🚨 (marked delivered, no OTP)")
    print(f"   Unaccounted  : {report.cylinders_unaccounted}   ❓ (missing from all counts)")

    if report.flags:
        print(f"\n🚩 Fraud Flags ({len(report.flags)}):")
        for flag in report.flags:
            print(f"   [{flag['severity']}] {flag['type']}: {flag['description'][:100]}...")

    print(f"\n👥 Agent Breakdown:")
    for agent in report.agent_breakdown:
        status = "🚨" if agent["action_needed"] else "✅"
        print(f"   {status} {agent['agent_name']}: "
              f"{agent['otp_verified']}/{agent['delivered']} OTP confirmed | "
              f"Trust: {agent['trust_score']} | "
              f"Ghost: {agent['ghost']}")
        if agent["flags"]:
            for f in agent["flags"]:
                print(f"      ⚠️  {f}")

    print(f"\n📋 Recommendation:")
    print(f"   {report.recommendation}")

    print(f"\n💰 Financial Impact:")
    ghost_loss = report.ghost_deliveries * 965
    unaccounted_loss = report.cylinders_unaccounted * 965
    print(f"   Ghost delivery subsidy loss : ₹{ghost_loss:,}")
    print(f"   Unaccounted cylinder loss   : ₹{unaccounted_loss:,}")
    print(f"   Total today                 : ₹{ghost_loss + unaccounted_loss:,}")
    print(f"   Projected monthly loss      : ₹{(ghost_loss + unaccounted_loss) * 26:,}")

    print("\n" + "=" * 65)
    print("  Distributor Reconciliation Engine ready.")
    print("  Next: Connect to /distributor/reconcile API endpoint")
    print("=" * 65 + "\n")