# 🔥 GasSetu — LPG Subsidy-Diversion Risk Scoring

[![Live Demo](https://img.shields.io/badge/Live%20Demo-gas--setu.onrender.com-orange?style=flat-square&logo=render)](https://gas-setu.onrender.com)
[![API Docs](https://img.shields.io/badge/API%20Docs-Swagger%20UI-blue?style=flat-square&logo=fastapi)](https://gas-setu.onrender.com/docs)
[![tests](https://github.com/athrvaraut/gas_setu/actions/workflows/tests.yml/badge.svg)](https://github.com/athrvaraut/gas_setu/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python)](https://python.org)

> **GasSetu scores LPG bookings for diversion risk and tells inspectors where to look first.**
> It is a student research prototype (B.Tech AIML major project, 2026–27), not a government system.
> It never refuses a household its cooking gas: unusual cases go to a person, and the refill goes ahead.

---

## 🎯 The problem

Domestic LPG is cheaper per kilogram than commercial LPG and is taxed at 5% instead of 18%. That gap rewards
diversion: eateries running on household cylinders, households selling part of their quota, and delivery staff
booking cylinders on connections that barely use any.

- **32.94 crore** active domestic consumers, 10.33 crore of them under PMUY (Ujjwala) ([PIB, Apr 2025](https://static.pib.gov.in/WriteReadData/specificdocs/documents/2025/apr/doc2025430548201.pdf))
- **1.28 lakh raids, 59,000+ cylinders seized** between March and 14 April 2026 ([Business Standard](https://www.business-standard.com/india-news/over-128-000-raids-59-000-lpg-cylinders-seized-since-march-2026-126041400024_1.html))
- The CAG's 2019 audit of PMUY found 1.98 lakh consumers averaging more than 12 cylinders a year and 13.96 lakh
  taking 3 to 41 refills in one month ([CAG Report 14 of 2019, ch. 6](https://cag.gov.in/uploads/download_audit_report/2019/Chapter_6_Transition_of_BPL_households_to_LPG_of_Report_No_14_of_2019_Performance_Audit_of_Pradhan_Mantri_Ujjwala_Yojana_Ministry_of_Petroleum_and_Natural_Gas.pdf))

**What changed in 2026, and why it matters here.** The delivery OTP (DAC) is now mandatory and a 25-day gap between
bookings applies to everyone. So a diverted cylinder today usually *has* a valid booking and a valid OTP. Counting
confirmations no longer finds it. What is left is the **pattern**: a connection that used three cylinders a year
and now books every 25 days, or one delivery agent whose beat is full of such connections.

---

## 🏗️ What the system does

```
POST /booking/validate
        │
        ▼
┌─────────────────────────────────────────┐
│ 1. GSTIN format + check digit           │  no API call
├─────────────────────────────────────────┤
│ 2. Registry check (GSTN lookup)         │  active? whose name? registered at this
│                                         │  address? a food / hospitality business?
├─────────────────────────────────────────┤
│ 3. Consumption risk                     │  booking history compared with SIMILAR
│    (peer groups, readable score)        │  households (PMUY, rural/urban, size)
├─────────────────────────────────────────┤
│ 4. Decision policy                      │  strongest outcome any layer asks for
└─────────────────────────────────────────┘
        ▼
CLEAR · MONITOR · REVIEW · HOLD_FOR_REVIEW · CONVERT_TO_COMMERCIAL

Modules with tests, not yet behind an API endpoint:
  core/state_machine.py              delivery state machine + SHA-256 hash chain
  core/inspection_ranking.py         which delivery agents to inspect first
  core/distributor_reconciliation.py allocated vs confirmed stock
```

### The five outcomes

| Outcome | Meaning | Booking proceeds? |
|---|---|---|
| `CLEAR` | Nothing to note | Yes |
| `MONITOR` | Logged; nobody needs to act | Yes |
| `REVIEW` | A person looks at the case within 72 h | Yes |
| `HOLD_FOR_REVIEW` | Waits for a person, within 24 h. **Commercial bookings only**, when the business identity fails verification | No |
| `CONVERT_TO_COMMERCIAL` | An active GSTIN for a food business is registered **at the delivery address**: a commercial cylinder is offered, with appeal | No |

The rule behind the table: **a household's refill is never refused or held by an automatic check.** The consumption
model can ask for `REVIEW` at most. `tests/test_decision_policy.py` checks this for every combination of inputs.

---

## 🧠 Consumption risk model (v2)

The first version used an Isolation Forest on hand-made data. A review found it did the opposite of its job: it
rated a genuine Ujjwala home with 3–4 refills a year as CRITICAL and passed the fraud patterns it was built for.
(Isolation Forest flags what is *rare*, and the hand-made "normal family" booked 12 cylinders a year; the real
average is about 6.) Version 2 replaces it.

1. **Peer groups.** A home is compared only with similar homes: PMUY or not, rural or urban, household-size band.
2. **Five readable inputs**, combined with non-negative weights fitted by logistic regression: refills above
   peers, a jump against the connection's own past, waking up after a long silence, a mobile-number change, and
   a mobile-number change together with a jump.
3. **Percentile within peers.** The tiers are cut at the same percentile in every group, so no group (PMUY,
   rural, large families) has more than 0.5% of its homes sent for review.

The history must come from the booking system: send `refill_dates` and the server derives the inputs itself.

### Measured results

**All of this is on simulated data.** No labelled LPG fraud data is public. The simulator
(`core/synthetic_population.py`) is built from published averages (PMUY 4.71 refills a year, other homes 6–7,
household size 4.4, the 25-day rule); how diversion looks in the data is an assumption. Each of 10 seeds draws
two different worlds: methods that need labels are trained on one and tested on the other. Reviewers can look at
0.5% of connections. Values are mean (range) over seeds.

**Households: share of reviewed connections that are real diversion** (diversion is 1.6% of connections)

| Method | Needs labels? | Precision at 0.5% budget | Diversion found |
|---|---|---|---|
| Rule: count refills | no | 0.11 (0.00–0.53) | 3% |
| Isolation Forest, raw features (the v1 approach) | no | 0.21 (0.06–0.64) | 6% |
| Local Outlier Factor / One-Class SVM | no | 0.10 / 0.16 | 3% / 5% |
| Rule: refills above similar homes (z-score) | no | 0.62 (0.37–0.90) | 19% |
| **GasSetu v2 score** (five readable inputs) | weights only | **0.64 (0.44–0.91)** | 20% |
| Logistic regression, all features | yes | 0.62 (0.23–0.91) | 19% |
| Gradient boosting | yes | 0.74 (0.50–0.95) | 23% |
| Random forest | yes | 0.78 (0.52–0.97) | 24% |

What this says:

- **Comparing a home with similar homes is what matters.** It lifts precision from 0.11 to 0.62 by itself. The
  extra inputs of the v2 score add little at household level; a black-box forest adds about 13 points.
- **Anomaly detectors are the wrong tool here.** They flag what is rare, and rare is mostly genuine.
- **Most diversion is not found at household level** (about one case in five within the budget). Households
  selling their own quota are almost invisible (5% found); captured connections 26%; eateries on domestic
  connections 39%.
- **False alarms are not even across groups.** With the v2 score, 0.31% of genuine PMUY homes are reviewed
  against 0.13% of other homes, and 0.42% of one- and two-person homes. Counting refills without peer groups is
  far worse (1.5% of large families). Spending the budget group by group was tried: precision halves (0.32) and
  the gap between groups does not shrink, so it was not adopted.
- **Cheaters who hide.** If diverters spread their bookings out, precision falls from 0.86 (no hiding) to 0.53,
  0.32 and 0.19 as they hide more.

**Delivery agents: which 5% to inspect first** (300 agents, about 8% corrupt)

| Method | Precision | Corrupt agents found |
|---|---|---|
| Allocated minus confirmed (the old reconciliation gap) | 0.10 (0.00–0.27), i.e. chance | 7% |
| Mean refills per household on the beat | 0.49 (0.20–0.73) | 34% |
| Mean household risk score on the beat | 0.94 (0.80–1.00) | 68% |
| GasSetu ranking, bookings only | 0.87 (0.60–1.00) | 63% |
| GasSetu ranking, bookings + how and where the OTP was entered | 1.00 (1.00–1.00) | 72% |

- **The old gap formula is blind**, because every diverted cylinder in the simulation has an OTP, as in reality.
- **Adding weak household evidence up to the agent works**: the same score that finds one case in five among
  households finds most corrupt agents. A plain average of household scores does as well as the binomial ranking.
- The last row depends on assumed rates for "OTP entered far from the address" and "within minutes of dispatch".
  Real delivery logs are needed to know how strong those signals are.
- **How small an operation can hide?** With bookings only, when each corrupt agent captures 8 connections on a
  beat of 400, 41% of the agents on the inspection list are corrupt; with 16 connections 72%; with 32, 95%.

Reproduce with `python experiments/consumption_eval.py` and `python experiments/agent_ranking_eval.py`
(CSV files in `experiments/results/`).

---

## 🚀 Getting started

```bash
git clone https://github.com/athrvaraut/gas_setu.git
cd gas_setu
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements-dev.txt
python -m pytest                # GSTN lookups are mocked or fictional; no API key needed
uvicorn api.booking_validate:app --reload
# http://localhost:8000          dashboard
# http://localhost:8000/docs     Swagger UI
```

For live GSTN lookups, put `APPYFLOW_KEY=...` in `.env`. Without it, the fictional demo records still work and
any other GSTIN is reported as "could not be verified".

```bash
curl -X POST http://localhost:8000/booking/validate -H "Content-Type: application/json" -d '{
  "consumer_id": "CON-001", "consumer_name": "Kamalabai Jadhav",
  "address": "House 14, Gram Panchayat Road, Wada", "pincode": "421303",
  "cylinder_type": "domestic", "household_size": 2, "is_pmuy": true, "is_rural": true,
  "mobile_changes_12m": 1,
  "refills_12m": 9, "refills_6m": 7, "refills_prev_18m": 5, "earliest_day_refills": 7, "dormant_days": 144
}'
```

```json
{
  "decision": "REVIEW",
  "proceed": true,
  "risk_level": "HIGH",
  "review_within_hours": 72,
  "recommended_action": "Review the booking history with the reasons listed. The refill proceeds."
}
```

---

## 📁 Project structure

```
api/booking_validate.py              FastAPI app: POST /booking/validate, /health, /demo/profiles, dashboard at /
api/demo_profiles.py                 fictional consumers for the dashboard
core/gstn_check.py                   GSTIN check digit, registry lookup, name / address / jurisdiction evidence
core/features.py                     booking history → model inputs (same code for experiments and API)
core/anomaly_detection.py            consumption risk model v2 (reads core/consumption_model.json)
core/decision_policy.py              the rule table and the "never refuse a household" rule
core/synthetic_population.py         simulator: distributors, agents, households, refills, three kinds of diversion
core/inspection_ranking.py           ranks delivery agents for inspection
core/state_machine.py                delivery state machine + hash chain
core/distributor_reconciliation.py   allocated vs confirmed stock
experiments/                         evaluation scripts and results (CSV)
tests/                               pytest suite, run by GitHub Actions on every push
static/index.html                    dashboard
```

---

## ⚠️ Limits, stated plainly

1. **No real data.** Every model number above comes from a simulator. A pilot with an oil company or a
   distributor is needed before any claim about real fraud.
2. **The weights are fitted on simulated labels.** On real data the peer statistics must be recomputed and the
   weights re-fitted on inspection outcomes.
3. **Households that sell part of their own quota** look like any heavy user; the model finds few of them.
4. **Cheaters who spread their bookings out** are harder to find. `experiments/consumption_eval.py` measures it.
5. **A business that never declares a GSTIN** is found only if the connection is in an establishment's name.
   Address-based discovery (food-licence and map data) is planned.
6. **Name and address matching** use engineering thresholds that still have to be measured on labelled pairs.
7. **The delivery log lives on one server.** The hash chain shows edits, deletions and reordering, but not an
   insider who rewrites the whole chain or drops its newest events. Device signatures and external anchoring
   are planned.
8. **Physical decanting** happens outside any digital system.

---

## 🔬 Background

An earlier version of this work was written up as a preprint
([ResearchGate, DOI 10.13140/RG.2.2.12096.49924](https://doi.org/10.13140/RG.2.2.12096.49924)), which is not
peer reviewed. **Its evaluation does not hold**: the reported precision, recall and F1 cannot be reproduced from
this repository and contradict each other, and the model it describes was replaced (see above).
`experiments/baseline_check.py` reproduces what the old model actually scored.

---

## 🗺️ Roadmap

- [x] Registry checks, delivery state machine, reconciliation, deployment (Apr–May 2026)
- [x] Review fixes, pytest suite, CI (Oct 2026)
- [x] v2 (Nov 2026): decision policy that never refuses a household; GSTIN check digit; explicit
      "registry unavailable" outcome; consumption model v2 with baselines; agent inspection ranking;
      fictional demo data
- [ ] Find businesses that declare no GSTIN (food-licence and map data, address matching) and measure it on labelled addresses
- [ ] Name matching measured on labelled pairs
- [ ] Real evidence: household survey on OTP practice, RTI data, logs from a distributor
- [ ] Validation on the real-label SGCC electricity-theft dataset
- [ ] Reconciliation and ranking behind API endpoints, with persistence
- [ ] Device-signed, externally anchored delivery log
- [ ] DPDP Act 2023 compliance note

---

## 👤 Author

**Athrva Raut**, B.Tech (Artificial Intelligence & Machine Learning),
Bharati Vidyapeeth (Deemed to be University), Department of Engineering & Technology, Navi Mumbai.
Developed further in 2026–27 as a two-member major project.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Athrva%20Raut-0077B5?style=flat-square&logo=linkedin)](https://linkedin.com/in/athrvaraut)
[![GitHub](https://img.shields.io/badge/GitHub-athrvaraut-181717?style=flat-square&logo=github)](https://github.com/athrvaraut)
