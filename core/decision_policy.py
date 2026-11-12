"""
GasSetu — Decision Policy
==========================
Turns the evidence from the layers into ONE outcome for a booking.

The rule everything else follows:

    A household's refill is never refused or held by an automatic check.

Cooking gas is essential, the layers work on incomplete data, and a wrong refusal hurts a family.
So for a domestic booking the strongest automatic outcome is REVIEW: the cylinder is delivered and
a person looks at the case. The single exception needs hard registry evidence: an ACTIVE GSTIN,
registered AT THE DELIVERY ADDRESS, for a food or hospitality business. Then the customer is
offered a commercial cylinder (CONVERT_TO_COMMERCIAL) and can appeal.

Outcomes, weakest to strongest:
    CLEAR                  nothing to note
    MONITOR                logged; nobody needs to act on this booking
    REVIEW                 the booking proceeds AND a person reviews the case
    HOLD_FOR_REVIEW        the booking waits for a person. Commercial bookings only.
    CONVERT_TO_COMMERCIAL  verified LPG-heavy business at the delivery address: offer a commercial cylinder

This is a fixed, readable rule table (v1). A fitted risk-fusion model is planned (ROADMAP step 30);
whatever replaces this table must keep the rule above, and tests/test_decision_policy.py checks it
for every combination of inputs.
"""

from dataclasses import dataclass, field
from typing import Optional

from core import gstn_check as registry
from core.gstn_check import GSTNCheckResult

CLEAR, MONITOR, REVIEW, HOLD_FOR_REVIEW, CONVERT_TO_COMMERCIAL = (
    "CLEAR", "MONITOR", "REVIEW", "HOLD_FOR_REVIEW", "CONVERT_TO_COMMERCIAL")
STRENGTH = {CLEAR: 0, MONITOR: 1, REVIEW: 2, HOLD_FOR_REVIEW: 3, CONVERT_TO_COMMERCIAL: 4}
PROCEEDS = {CLEAR: True, MONITOR: True, REVIEW: True, HOLD_FOR_REVIEW: False, CONVERT_TO_COMMERCIAL: False}
REVIEW_WITHIN_HOURS = {REVIEW: 72, HOLD_FOR_REVIEW: 24, CONVERT_TO_COMMERCIAL: 24}

# What the reviewer or the booking desk should do, by reason code
ACTIONS = {
    "INVALID_GSTIN":            "Ask the customer for the correct GSTIN and fix the record.",
    "GSTIN_NOT_REGISTERED":     "Ask for proof of GST registration before treating this as a business booking.",
    "GSTIN_INACTIVE":           "Check the business status with the customer; the registration is not active.",
    "GSTIN_NAME_MISMATCH":      "Verify who is booking: the GSTIN is registered to a different name.",
    "GSTIN_NAME_UNCERTAIN":     "Compare the booking name with the GST registration; they match only partly.",
    "VERIFICATION_UNAVAILABLE": "No action now. Verify the GSTIN again when the registry answers.",
    "COMMERCIAL_PREMISES":      "Offer a commercial (19 kg) cylinder for this address. The customer may appeal with "
                                "proof that the home and the business are separate.",
    "LPG_HEAVY_BUSINESS_NEARBY": "Check whether this connection serves the business registered in the same pincode.",
    "ESTABLISHMENT_NAME":       "Check whether this domestic connection is being used by an establishment.",
    "HOME_BASED_BUSINESS":      "No action. A business is registered at this address but its type does not suggest heavy LPG use.",
    "JURISDICTION_MISMATCH":    "No action. Note that the GSTIN's state differs from the delivery state.",
    "COMMERCIAL_WITHOUT_GSTIN": "No action. Other business proof (shop licence, FSSAI) should be on file.",
    "CONSUMPTION_HIGH":         "Review the booking history with the reasons listed. The refill proceeds.",
    "CONSUMPTION_MEDIUM":       "No action. The connection stays on the watch list.",
}


@dataclass
class Reason:
    code:    str        # stable identifier, see ACTIONS
    layer:   str        # REGISTRY | CONSUMPTION
    asks:    str        # the outcome this reason alone would lead to
    detail:  str        # one sentence for the reviewer

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class Decision:
    decision:            str
    proceed:             bool                 # does the booking go ahead now, as booked?
    primary_reason:      str
    recommended_action:  str
    reasons:             list = field(default_factory=list)
    review_within_hours: Optional[int] = None

    def to_dict(self) -> dict:
        return {**self.__dict__, "reasons": [r.to_dict() for r in self.reasons]}


def _registry_reasons(domestic: bool, consumer_name: str, r: GSTNCheckResult) -> list:
    """What the registry evidence asks for. Identity problems hold a commercial booking and never a household's."""
    identity = REVIEW if domestic else HOLD_FOR_REVIEW
    reasons = []
    add = lambda code, asks, detail=None: reasons.append(Reason(code, "REGISTRY", asks, detail or r.reason))

    if r.outcome == registry.NO_GSTIN:
        if not domestic:
            add("COMMERCIAL_WITHOUT_GSTIN", MONITOR, "Commercial booking with no GSTIN declared.")
        elif registry.looks_lpg_heavy(consumer_name):
            add("ESTABLISHMENT_NAME", REVIEW,
                f"The domestic connection is in the name '{consumer_name}', which reads like a food or hospitality business.")
    elif r.outcome == registry.INVALID_GSTIN:
        add("INVALID_GSTIN", identity)
    elif r.outcome == registry.NOT_REGISTERED:
        add("GSTIN_NOT_REGISTERED", identity)
    elif r.outcome == registry.INACTIVE:
        add("GSTIN_INACTIVE", identity)
    elif r.outcome == registry.NAME_MISMATCH:
        add("GSTIN_NAME_MISMATCH", identity)
    elif r.outcome == registry.NAME_UNCERTAIN:
        add("GSTIN_NAME_UNCERTAIN", REVIEW)
    elif r.outcome == registry.UNAVAILABLE:
        add("VERIFICATION_UNAVAILABLE", MONITOR)
    elif r.outcome == registry.VERIFIED:
        business = r.trade_name or r.legal_name
        if domestic and r.premises == registry.SAME_ADDRESS and r.lpg_heavy_business:
            add("COMMERCIAL_PREMISES", CONVERT_TO_COMMERCIAL,
                f"'{business}' (active GSTIN {r.gstin}) is registered at this delivery address and is a food or "
                f"hospitality business.")
        elif domestic and r.premises == registry.SAME_ADDRESS:
            add("HOME_BASED_BUSINESS", MONITOR,
                f"'{business}' (active GSTIN {r.gstin}) is registered at this delivery address.")
        elif domestic and r.premises in (registry.SAME_PINCODE, registry.UNKNOWN) and r.lpg_heavy_business:
            add("LPG_HEAVY_BUSINESS_NEARBY", REVIEW,
                f"The customer's GSTIN {r.gstin} belongs to '{business}', a food or hospitality business; its "
                f"registered address could not be matched to the delivery address.")
        if r.jurisdiction_mismatch:
            add("JURISDICTION_MISMATCH", MONITOR,
                f"GSTIN {r.gstin} is registered in {registry.get_state_from_gstin(r.gstin)}; the delivery pincode is in another state.")
    return reasons


def _consumption_reasons(consumption) -> list:
    """The consumption model can ask for REVIEW at most. It never holds or refuses anything."""
    if consumption is None or consumption.risk_level not in ("HIGH", "MEDIUM"):
        return []
    why = " ".join(x["detail"] for x in consumption.reasons) or "Unusual booking pattern for similar households."
    detail = f"Risk score above {consumption.percentile:g}% of connections (compared with: {consumption.peer_group}). {why}"
    if consumption.risk_level == "HIGH":
        return [Reason("CONSUMPTION_HIGH", "CONSUMPTION", REVIEW, detail)]
    return [Reason("CONSUMPTION_MEDIUM", "CONSUMPTION", MONITOR, detail)]


def decide(booking_type: str, consumer_name: str, registry_result: GSTNCheckResult, consumption=None) -> Decision:
    """One outcome for the booking: the strongest thing any layer asks for."""
    domestic = booking_type.strip().lower() != "commercial"
    reasons = _registry_reasons(domestic, consumer_name, registry_result) + _consumption_reasons(consumption)
    reasons.sort(key=lambda r: -STRENGTH[r.asks])

    if not reasons:
        return Decision(CLEAR, True, "No check raised a concern.", "No action.")

    decision = reasons[0].asks
    if domestic and decision == HOLD_FOR_REVIEW:       # cannot happen with the rules above; kept as a guard
        decision = REVIEW
    return Decision(
        decision            = decision,
        proceed             = PROCEEDS[decision],
        primary_reason      = reasons[0].detail,
        recommended_action  = ACTIONS[reasons[0].code],
        reasons             = reasons,
        review_within_hours = REVIEW_WITHIN_HOURS.get(decision),
    )
