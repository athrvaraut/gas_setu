# 🔥 GasSetu — Real-Time LPG Subsidy-Diversion Risk Scoring

[![Live Demo](https://img.shields.io/badge/Live%20Demo-gas--setu.onrender.com-orange?style=flat-square&logo=render)](https://gas-setu.onrender.com)
[![API Docs](https://img.shields.io/badge/API%20Docs-Swagger%20UI-blue?style=flat-square&logo=fastapi)](https://gas-setu.onrender.com/docs)
[![Preprint](https://img.shields.io/badge/Preprint-ResearchGate-00CCBB?style=flat-square&logo=researchgate)](https://doi.org/10.13140/RG.2.2.12096.49924)
[![tests](https://github.com/athrvaraut/gas_setu/actions/workflows/tests.yml/badge.svg)](https://github.com/athrvaraut/gas_setu/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square)](./LICENSE)

> **In March–April 2026, enforcement teams ran more than 1.28 lakh raids and seized over 59,000 diverted LPG cylinders**
> ([Business Standard, 14 Apr 2026](https://www.business-standard.com/india-news/over-128-000-raids-59-000-lpg-cylinders-seized-since-march-2026-126041400024_1.html)).
> GasSetu is a real-time **risk-scoring layer**. It checks each booking against tax-registry and consumption signals and sends
> suspicious cases to human review before a cylinder moves.

---

## 🎯 The Problem

Domestic LPG is cheaper per kilogram than commercial LPG and is taxed at a lower rate:

| Cylinder type | Weight | GST |
|---------------|--------|-----|
| 🏠 Domestic (household) | 14.2 kg | 5% |
| 🏪 Commercial (business) | 19 kg | 18% |

That gap rewards diversion. Restaurants and small businesses book domestic cylinders on household connections,
households resell cylinders, ghost or duplicate connections draw extra stock, and deliveries are marked complete
without reaching the consumer.

**Scale:**
- **32.94 crore** active domestic LPG consumers, incl. 10.33 crore PMUY ([PIB, Apr 2025](https://static.pib.gov.in/WriteReadData/specificdocs/documents/2025/apr/doc2025430548201.pdf))
- **1.28 lakh+ raids, 59,000+ cylinders seized, 1,000+ FIRs, 238 arrests** between March and 14 April 2026 ([Business Standard](https://www.business-standard.com/india-news/over-128-000-raids-59-000-lpg-cylinders-seized-since-march-2026-126041400024_1.html))
- **94.5%** of deliveries DAC (OTP) confirmed by late April 2026, and DAC mandatory from 1 May 2026 ([The Sentinel](https://www.sentinelassam.com/more-news/business/delivery-authentication-code-based-gas-deliveries-reach-over-945-to-prevent-diversion-centre))

**The gap GasSetu targets:** existing safeguards (DBTL/PAHAL, Aadhaar de-duplication, DAC) each protect one point in
the chain and mostly act after the fact. GasSetu correlates signals **at booking time** and produces an explainable
risk score. It is meant to rank cases for human review, not to deny cooking gas automatically.

---

## ⚡ Live Demo

**Try it:** [gas-setu.onrender.com](https://gas-setu.onrender.com). The free tier can take about 30 s to wake up.

What the dashboard presets actually return (checked 17 Oct 2026 against the current code, GSTN lookup mocked):

| Preset | Result | Why |
|--------|--------|-----|
| 🍽️ Restaurant fraud | 🚫 BLOCKED (stolen GSTIN) | The sample GSTIN belongs to a different taxpayer, so the name check fires first. A proper commercial-misuse demo needs fictional GSTINs (roadmap) |
| 🕵️ Stolen GSTIN | 🚫 BLOCKED | Booking name matches neither the GSTN legal nor trade name |
| 📦 Cylinder seller | ⚠️ APPROVED, risk HIGH | Only CRITICAL scores block; HIGH is flagged for review |
| 🏠 Normal household | ✅ APPROVED, risk MEDIUM | ML risk tiers are not calibrated yet (see Known Limitations) |
| ✅ Commercial, correct | ✅ APPROVED, risk HIGH | The ML model only knows household patterns |

---

## 🏗️ Architecture

`POST /booking/validate` runs a fail-fast cascade. A block stops the cascade and returns an audit trail.

```
POST /booking/validate
        │
        ▼
┌──────────────────────────────────────┐
│ 1. GSTIN format check (regex)        │── BLOCK  fake / malformed GSTIN
│    check digit: planned              │
└──────────────────────────────────────┘
        ▼
┌──────────────────────────────────────┐
│ 2. Live GSTN lookup (Appyflow)       │── BLOCK  suspended / cancelled GSTIN
│    + state vs pincode check          │── FLAG   jurisdiction mismatch
└──────────────────────────────────────┘
        ▼
┌──────────────────────────────────────┐
│ 3. Identity match                    │── BLOCK  stolen GSTIN
│    booking name vs legal/trade name  │
├──────────────────────────────────────┤
│    Active GSTIN + domestic booking   │── BLOCK  commercial misuse
└──────────────────────────────────────┘
        ▼
┌──────────────────────────────────────┐
│ 4. Consumption anomaly model         │── BLOCK  CRITICAL score
│    Isolation Forest, synthetic data  │── FLAG   HIGH / MEDIUM
└──────────────────────────────────────┘
        ▼
     APPROVED

Separate modules (tested, not yet exposed through the API):
  • core/state_machine.py              cylinder delivery state machine + SHA-256 hash chain
  • core/distributor_reconciliation.py allocation vs OTP-confirmed deliveries, agent trust scores
```

### Detection capabilities (honest status)

| Attack vector | Method | Status |
|---------------|--------|--------|
| Domestic booking with a declared GSTIN | Live GSTN lookup | ✅ Implemented, tested (mocked API) |
| Business that leaves the GSTIN blank | PAN→GSTIN / FSSAI lookup | 🔜 Planned, the biggest open gap |
| Stolen / borrowed GSTIN | Name vs legal + trade name | ⚠️ Implemented; beaten if the attacker also copies the owner's name |
| Fake GSTIN | Format regex | ⚠️ Format only; check digit planned |
| Ghost deliveries | State machine + reconciliation | ⚠️ Modules tested; not yet wired to the API |
| Device clock manipulation | Timestamp ordering + hash chain | ✅ Tested within the chain; insider rewrite needs external anchoring (planned) |
| Deleted / reordered delivery events | Hash-chain link check | ✅ Tested |
| Cylinder reselling | Rule flags + ML anomaly | ⚠️ Rule flags work; ML tiers uncalibrated |
| GPS spoofing | GPS vs motion-sensor consistency | ⚠️ Weak signal, not evaluated |
| SIM-farm ghost accounts | Device fingerprinting | 🔜 Not implemented |
| Gas decanting | — | ❌ Out of scope (physical crime) |

---

## 📁 Project Structure

```
gas_setu/
├── api/booking_validate.py           # FastAPI app: POST /booking/validate, GET /health, GET /api, dashboard at /
├── core/
│   ├── gstn_check.py                 # GSTIN format, live GSTN lookup, name match, jurisdiction check
│   ├── anomaly_detection.py          # Synthetic data generator + Isolation Forest
│   ├── state_machine.py              # Cylinder delivery state machine + hash chain
│   └── distributor_reconciliation.py # Ghost-delivery / stock-gap reconciliation
├── static/index.html                 # Dashboard (single page)
├── tests/                            # pytest suite (GSTN lookups mocked)
├── .github/workflows/tests.yml       # CI: runs pytest on every push
├── pytest.ini, requirements-dev.txt
├── render.yaml                       # Render.com deployment
└── requirements.txt
```

---

## 🚀 Getting Started

```bash
git clone https://github.com/athrvaraut/gas_setu.git
cd gas_setu
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements-dev.txt
```

Create `.env` in the project root for live GSTN lookups (not needed for the tests):

```env
APPYFLOW_KEY=your_appyflow_api_key_here
```

### Run the tests

```bash
python -m pytest -rx
# 46 passed, 6 xfailed
```

The 6 `xfail` tests are **known defects**, marked `strict`. Each states the defect and the planned fix. When a fix
lands, the test starts passing and pytest flags it so the marker gets removed:
- D6: a GSTN API timeout is treated as "not registered" (fails open)
- D7: the GSTIN check digit is not verified
- D12: the ML risk tiers are uncalibrated (4 tests)

### Run locally

```bash
uvicorn api.booking_validate:app --reload
# http://localhost:8000          dashboard
# http://localhost:8000/docs     Swagger UI
# http://localhost:8000/health   health check
# http://localhost:8000/api      JSON index
```

### Try a booking (cURL)

```bash
# Domestic booking with a GSTIN registered to someone else → BLOCKED (stolen GSTIN)
curl -X POST http://localhost:8000/booking/validate \
  -H "Content-Type: application/json" \
  -d '{
    "consumer_id": "TEST-001", "consumer_name": "Sharma Restaurant",
    "address": "Shop 4, Andheri Market, Mumbai", "pincode": "400053", "state_code": "27",
    "cylinder_type": "domestic", "company": "indane", "consumer_gstin": "27AAPFU0939F1ZV",
    "days_between_bookings": 15.2, "household_size": 1.5, "booking_frequency_change": 0.88,
    "address_change_count": 0, "days_since_last_booking": 15, "agent_change_rate": 0.75,
    "bookings_last_12_months": 24, "complaint_rate": 0.01
  }'
```

```json
{
  "decision": "BLOCKED",
  "risk_level": "HIGH",
  "primary_reason": "STOLEN GSTIN: Booking name 'Sharma Restaurant' matches neither legal name '…' nor trade name '…'. …",
  "recommended_action": "Verify consumer identity. GSTIN belongs to a different entity. …"
}
```

---

## 🧠 ML Model Details

**Algorithm:** Isolation Forest (scikit-learn), 200 trees, contamination 0.2 (equal to the synthetic fraud share)
**Training data:** 10,000 **synthetic** records (8,000 normal + 2,000 across 4 fraud patterns), generated by
`generate_synthetic_data()`. The app trains on all of them at startup.

| Feature | Description |
|---------|-------------|
| `days_between_bookings` | Average interval between bookings |
| `household_size` | Registered household members |
| `booking_frequency_change` | Change in booking frequency (recent vs history) |
| `address_change_count` | Address changes on record |
| `days_since_last_booking` | Time since the last booking |
| `agent_change_rate` | Share of deliveries by different agents |
| `bookings_last_12_months` | Bookings in the past year |
| `complaint_rate` | Complaints per booking |

### Measured performance and why it is not a claim of accuracy

70/30 stratified split of the same synthetic data (`experiments/baseline_check.py`, 17 Oct 2026):

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Isolation Forest, contamination 0.2 | 0.942 | 0.857 | 0.850 | **0.854** |
| Isolation Forest, contamination 0.02 / 0.1 / 0.3 | 0.817 / 0.890 / 0.912 | — | — | 0.159 / 0.628 / 0.819 |
| Logistic regression | 1.000 | 1.000 | 1.000 | 1.000 |
| Depth-1 decision stump (one threshold) | 1.000 | 1.000 | 1.000 | 1.000 |

- The earlier "94.2%" figure is **accuracy**, not F1.
- A single threshold separates the synthetic classes perfectly, so these numbers say nothing about real fraud.
- Risk tiers are uncalibrated. A typical family scores 0.59 (MEDIUM), a reseller pattern 0.64 (MEDIUM), and a genuine
  low-usage rural household 0.72 (HIGH).

**Next:** an independently written synthetic generator, supervised and unsupervised baselines with PR-AUC and
precision@k, calibration on genuine households (including rural ones), and validation on the real-label SGCC
electricity-theft dataset.

---

## 🏛️ Distributor Reconciliation

Module that finds ghost deliveries by comparing allocated cylinders with OTP-confirmed deliveries:

```python
gap% = (allocated - (OTP_confirmed + returned + in_transit)) / allocated × 100
```

| Alert level | Gap | Suggested action |
|-------------|-----|------------------|
| CLEAN | < 2% | None |
| WATCH | 2–5% | Monitor |
| WARNING | 5–10% | Schedule inspection |
| CRITICAL | > 10% | Audit; refer for inspection |

Agent penalties escalate (8 → 12 → 18 points) so alternating honest and fraudulent deliveries doesn't pay.
Re-running reconciliation on the same day doesn't double-penalise.

---

## 📡 API Reference

| Endpoint | Purpose |
|----------|---------|
| `POST /booking/validate` | Runs the GSTN checks and the consumption model; returns `APPROVED` / `BLOCKED`, risk level, reasons, recommended action |
| `GET /health` | Status, model state, active checks |
| `GET /api` | JSON index of endpoints |
| `GET /` | Dashboard |

Interactive docs: [gas-setu.onrender.com/docs](https://gas-setu.onrender.com/docs)

---

## 🔬 Research

> **GasSetu: A Multi-Layer Real-Time Fraud Detection Architecture for India's LPG Distribution System**
> Athrva Raut, Bharati Vidyapeeth (Deemed to be University), Navi Mumbai
> Preprint on ResearchGate (not peer reviewed) · DOI: [10.13140/RG.2.2.12096.49924](https://doi.org/10.13140/RG.2.2.12096.49924)

The work is being extended in 2026–27 as a B.Tech (AIML) major project, with a stricter evaluation than the preprint
(see the measured results above).

---

## 🗺️ Roadmap

- [x] GSTN integration, name matching, Isolation Forest, state machine, reconciliation, Render deployment
- [x] Review fixes (Oct 2026): hash-chain link check, ghost timer beyond 24 h, idempotent reconciliation, trade-name match, timezone handling
- [x] pytest suite + CI; dashboard shows sourced numbers only
- [ ] GSTIN check digit; explicit policy when the GSTN API is down; fictional-GSTIN demo mode
- [ ] Find businesses that leave the GSTIN blank (PAN-linked GSTIN, FSSAI licence, address matching)
- [ ] Better name matching (transliteration-aware) with a labelled evaluation set
- [ ] Re-evaluate the ML layer: independent synthetic data, baselines, calibration, SGCC real-data validation
- [ ] Reconciliation API + dashboard on DAC data
- [ ] Device-signed, externally anchored delivery log
- [ ] Risk-fusion engine with a human review queue and appeal path
- [ ] Field pilot with a distributor

---

## ⚠️ Known Limitations

1. **No real fraud labels.** All ML results come from synthetic data; see the measured table above.
2. **ML risk tiers are uncalibrated.** A genuine low-usage rural household currently scores higher than a reseller.
3. **Businesses that don't declare a GSTIN** (including those below the GST threshold) are not checked against registries yet.
4. **GSTIN check digit not verified**, and a GSTN API timeout currently lets the booking through as "not registered".
5. **Delivery logs live on one server.** The hash chain detects edits, deletions and reordering, but not an insider
   who rewrites the whole chain or drops its newest events. That needs device signatures and external anchoring.
6. **State machine and reconciliation are not yet exposed through the API.**
7. **Demo presets use a real taxpayer's sample GSTIN.** A fictional-GSTIN demo mode is planned.
8. **Data limits:** UPI transaction data is not accessible; OTP confirmation can be socially engineered; gas decanting
   happens outside the digital system.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI + Pydantic |
| ML | scikit-learn IsolationForest |
| GSTN | Appyflow API (GSTN wrapper) |
| Hash chain | SHA-256 (hashlib) |
| Tests / CI | pytest, GitHub Actions |
| Deployment | Render.com (free tier) |
| Language | Python 3.11+ |

---

## 👤 Author

**Athrva Raut**
B-Tech CSE-AIML, Bharati Vidyapeeth College of Engineering (Deemed University), Navi Mumbai

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Athrva%20Raut-0077B5?style=flat-square&logo=linkedin)](https://linkedin.com/in/athrvaraut)
[![GitHub](https://img.shields.io/badge/GitHub-athrvaraut-181717?style=flat-square&logo=github)](https://github.com/athrvaraut)
[![ResearchGate](https://img.shields.io/badge/ResearchGate-DOI%2010.13140-00CCBB?style=flat-square&logo=researchgate)](https://doi.org/10.13140/RG.2.2.12096.49924)

---

*Built with the goal of making India's LPG system transparent, fraud-resistant, and fair for 33 crore households.*
