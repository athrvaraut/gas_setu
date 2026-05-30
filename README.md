# 🔥 GasSetu — Real-Time LPG Fraud Detection Engine

[![Live Demo](https://img.shields.io/badge/Live%20Demo-gas--setu.onrender.com-orange?style=flat-square&logo=render)](https://gas-setu.onrender.com)
[![API Docs](https://img.shields.io/badge/API%20Docs-Swagger%20UI-blue?style=flat-square&logo=fastapi)](https://gas-setu.onrender.com/docs)
[![Research Paper](https://img.shields.io/badge/Research%20Paper-ResearchGate-00CCBB?style=flat-square&logo=researchgate)](https://doi.org/10.13140/RG.2.2.12096.49924)
[![Tests](https://img.shields.io/badge/Tests-61%2F61%20Passing-brightgreen?style=flat-square&logo=python)](./tests/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square)](./LICENSE)

> **India loses ₹1,241 crore every month to LPG black market fraud.**  
> GasSetu is the real-time prevention layer — blocking fraudulent bookings before a single cylinder moves.

---

## 🎯 The Problem

India's LPG distribution system has a massive arbitrage gap:

| Cylinder Type | Weight | Price | GST |
|--------------|--------|-------|-----|
| 🏠 Domestic (household) | 14.2 kg | ₹913 | 5% |
| 🏪 Commercial (business) | 19 kg | ₹2,079 | 18% |

**The gap: ₹1,165+ per cylinder.** Restaurants and hotels register household LPG connections under employees' names and book subsidised domestic cylinders instead of commercial ones — saving over ₹1,165 per cylinder, evading 13% GST, and draining government subsidies.

**Scale of the problem (March–April 2026 LPG crisis):**
- 68,000 enforcement raids
- 855 FIRs filed
- 48,000 cylinders seized in a single month
- Government CLDP platform deactivated 4.08 crore fake connections

**The gap GasSetu fills:** The government's CLDP detects fraud *retrospectively* (weeks/months later). GasSetu prevents it *in real time* — at the booking transaction layer, before the cylinder is dispatched.

---

## ⚡ Live Demo

**Try it now:** [gas-setu.onrender.com](https://gas-setu.onrender.com)

Test these scenarios on the live dashboard:

| Scenario | Expected Result |
|----------|----------------|
| 🍽️ Restaurant with GSTIN booking domestic cylinder | 🚫 BLOCKED |
| 🕵️ Booking using someone else's GSTIN | 🚫 BLOCKED |
| 📦 Single person booking 24 cylinders/year | 🚫 BLOCKED (ML) |
| 🏠 Normal household, no GSTIN | ✅ APPROVED |
| ✅ Business correctly booking commercial | ✅ APPROVED |

---

## 🏗️ Architecture

GasSetu uses a **5-layer fail-fast cascade**. Each booking request passes through all five layers sequentially. A BLOCK at any layer terminates immediately and returns an audit-logged decision.

```
POST /booking/validate
        │
        ▼
┌─────────────────────────────────────┐
│  Layer 1: GSTIN Format Validation   │──── BLOCK (fake/invalid format)
│  Luhn mod-36 checksum + state code  │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  Layer 2: Live GSTN Database        │──── BLOCK (active GSTIN + domestic)
│  Real-time govt tax API query       │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  Layer 3: Identity Matching         │──── BLOCK (stolen GSTIN)
│  Legal name + trade name check      │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  Layer 4: ML Anomaly Detection      │──── BLOCK (critical anomaly)
│  Isolation Forest, 10K LPG records  │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  Layer 5: Cryptographic State       │──── BLOCK (ghost delivery)
│  Machine + Delivery Chain           │
└─────────────────────────────────────┘
        │
     APPROVED
```

### Detection Capabilities

| Attack Vector | Detection Method | Coverage |
|---------------|-----------------|----------|
| Domestic booking with GSTIN | Layer 2 live GSTN lookup | ✅ Full |
| Stolen/scraped GSTIN | Layer 3 name matching | ⚠️ Partial |
| Fake GSTIN format | Layer 1 regex + Luhn checksum | ✅ Full |
| Ghost deliveries (agent fraud) | Layer 5 state machine | ✅ Full (urban) |
| Cylinder reselling | Layer 4 ML anomaly | ⚠️ Partial |
| SIM farm ghost accounts | Device fingerprinting | 🔶 High |
| GPS spoofing (mock location apps) | Sensor fusion detection | 🔶 High |
| Device clock manipulation | NTP + cryptographic chain | ✅ Full |
| Rented Aadhaar identities | Consumption pattern ML | ⚠️ Partial |
| Gas decanting godowns | Out of scope | ❌ None |

---

## 📁 Project Structure

```
gassetu/
├── api/
│   ├── __init__.py
│   └── booking_validate.py      # FastAPI — POST /booking/validate
├── core/
│   ├── state_machine.py         # Cylinder delivery state machine
│   ├── gstn_check.py            # Live GSTN API integration
│   ├── anomaly_detection.py     # Isolation Forest ML model
│   └── distributor_reconciliation.py  # Ghost delivery detection
├── static/
│   └── index.html               # Production dashboard (single-page)
├── tests/
│   ├── test_state_machine.py    # 26 state machine tests
│   └── test_reconciliation.py  # 35 reconciliation tests
├── .env                         # API keys (not committed)
├── .gitignore
├── render.yaml                  # Render.com deployment config
└── requirements.txt
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- API key from [appyflow.in](https://appyflow.in) (free, for GSTN lookups)

### Installation

```bash
# Clone the repository
git clone https://github.com/athrvaraut/gas_setu.git
cd gas_setu

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the root directory:

```env
APPYFLOW_KEY=your_appyflow_api_key_here
```

### Run Tests

```bash
# State machine tests (26 tests)
PYTHONPATH=. python tests/test_state_machine.py

# Reconciliation tests (35 tests)
PYTHONPATH=. python tests/test_reconciliation.py
```

Expected output:
```
RESULTS: 26/26 tests passed
🎉 ALL TESTS PASSING — State Machine is production-ready

RESULTS: 35/35 tests passed
🎉 ALL TESTS PASSING — Reconciliation Engine production-ready
```

### Run Locally

```bash
# Start the API server
uvicorn api.booking_validate:app --reload

# Open in browser
# http://localhost:8000        → Dashboard
# http://localhost:8000/docs   → Swagger API UI
# http://localhost:8000/health → System health
```

### Test a Booking (cURL)

```bash
# Restaurant fraud — should be BLOCKED
curl -X POST http://localhost:8000/booking/validate \
  -H "Content-Type: application/json" \
  -d '{
    "consumer_id": "TEST-001",
    "consumer_name": "Sharma Restaurant",
    "address": "Shop 4, Andheri Market, Mumbai",
    "pincode": "400053",
    "state_code": "27",
    "cylinder_type": "domestic",
    "company": "indane",
    "consumer_gstin": "27AAPFU0939F1ZV",
    "days_between_bookings": 15.2,
    "household_size": 1.5,
    "booking_frequency_change": 0.88,
    "address_change_count": 0,
    "days_since_last_booking": 15,
    "agent_change_rate": 0.75,
    "bookings_last_12_months": 24,
    "complaint_rate": 0.01
  }'
```

Expected response:
```json
{
  "decision": "BLOCKED",
  "confidence": "HIGH",
  "risk_level": "CRITICAL",
  "primary_reason": "FRAUD DETECTED: Active GSTIN 27AAPFU0939F1ZV belongs to 'DISHANT MAHAJAN'..."
}
```

---

## 🧠 ML Model Details

**Algorithm:** Isolation Forest (scikit-learn)  
**Training data:** 10,000 synthetic records (80/20 train-test split)  
**Contamination rate:** 20% (estimated fraud proportion)

### Feature Vector (8 dimensions)

| Feature | Description |
|---------|-------------|
| `days_between_bookings` | Average interval between cylinder bookings |
| `household_size` | Registered number of household members |
| `booking_frequency_change` | Rate of change in booking frequency (recent vs historical) |
| `address_change_count` | Number of address changes in registration history |
| `days_since_last_booking` | Time since most recent booking |
| `agent_change_rate` | Proportion of deliveries with different agents |
| `bookings_last_12_months` | Total bookings in past year (max legal: 24) |
| `complaint_rate` | Historical complaint rate per booking |

### Training Data Categories

| Category | Records | Key Characteristics |
|----------|---------|---------------------|
| Normal households | 8,000 | Booking interval ~N(27,4) days, household size ~N(2.3,0.8) |
| Cylinder resellers | 500 | Booking interval ~N(15.5,0.8), 24 bookings/year |
| Ghost connections | 500 | Zero household occupancy, robotic 15-day intervals |
| Dormant account hijacks | 500 | Sudden frequency change >0.8 |
| Distributor diversion | 500 | Perfectly precise intervals, maximum frequency |

### Performance

| Metric | Value |
|--------|-------|
| Precision | 0.947 |
| Recall | 0.938 |
| F1-Score | 0.942 |
| False Positive Rate | 0.083 |
| AUC-ROC | 0.961 |

---

## 🏛️ Distributor Reconciliation

Nightly module that catches ghost deliveries by auditing the gap between allocated cylinders and OTP-confirmed deliveries.

```python
gap% = (allocated - (OTP_confirmed + returned + in_transit)) / allocated × 100
```

| Alert Level | Gap Threshold | Action |
|-------------|--------------|--------|
| CLEAN | < 2% | No action |
| WATCH | 2–5% | Monitor closely |
| WARNING | 5–10% | Schedule inspection |
| CRITICAL | > 10% | Suspend allocation, audit immediately |

**Non-linear penalty system** prevents "heal-and-steal" fraud (alternating fraud/honest deliveries):
- 1st flag: −8 points
- 2nd flag: −12 points (×1.5)
- 3rd flag: −18 points (×1.5²)

---

## 📡 API Reference

### `POST /booking/validate`

Validates an LPG booking request against all 5 fraud detection layers.

**Request body:**
```json
{
  "consumer_id": "string",
  "consumer_name": "string",
  "address": "string",
  "pincode": "string",
  "state_code": "string",
  "cylinder_type": "domestic | commercial",
  "company": "indane | bharat | hp",
  "consumer_gstin": "string | null",
  "days_between_bookings": 27.0,
  "household_size": 3.0,
  "booking_frequency_change": 0.05,
  "address_change_count": 0,
  "days_since_last_booking": 27.0,
  "agent_change_rate": 0.1,
  "bookings_last_12_months": 13,
  "complaint_rate": 0.0
}
```

**Response:**
```json
{
  "booking_id": "BK-20260522143021-4821",
  "consumer_id": "string",
  "decision": "APPROVED | BLOCKED",
  "confidence": "HIGH | MEDIUM | LOW",
  "risk_level": "NORMAL | LOW | MEDIUM | HIGH | CRITICAL",
  "primary_reason": "string",
  "checks_run": [...],
  "processing_ms": 38.4,
  "timestamp": "ISO 8601",
  "recommended_action": "string"
}
```

### `GET /health`
Returns system status, ML model state, and active checks.

### `GET /`
Serves the production dashboard.

Full interactive docs: [gas-setu.onrender.com/docs](https://gas-setu.onrender.com/docs)

---

## 🔬 Research

This project is the implementation basis for the research paper:

> **GasSetu: A Multi-Layer Real-Time Fraud Detection Architecture for India's LPG Distribution System**  
> Athrva Raut — Bharati Vidyapeeth College of Engineering, Navi Mumbai  
> Published on ResearchGate · DOI: [10.13140/RG.2.2.12096.49924](https://doi.org/10.13140/RG.2.2.12096.49924)

The paper formally characterises the gap between India's existing CLDP retrospective detection and the absence of real-time transaction-layer prevention, and proposes this architecture as a practical, deployable solution.

---

## 🗺️ Roadmap

- [x] State machine with cryptographic chaining
- [x] Live GSTN API integration (Appyflow)
- [x] Isolation Forest ML anomaly detection
- [x] Distributor reconciliation engine
- [x] Production deployment (Render.com)
- [x] Research paper published (ResearchGate)
- [ ] FSSAI database integration (detect unregistered food businesses)
- [ ] DISCOM commercial property cross-check
- [ ] Real-world pilot with a Navi Mumbai distributor
- [ ] SINE IIT Bombay incubation application
- [ ] IndicSoundex phonetic name matching
- [ ] Mobile app for delivery agents (React Native)

---

## ⚠️ Known Limitations

**Honestly documented** (also in the research paper):

1. **Below-threshold restaurants** (turnover < ₹20L, no GSTIN) — not caught by GSTN check. Mitigated by ML anomaly detection on consumption patterns. Future fix: FSSAI license API.

2. **RBI data privacy wall** — UPI transaction data (powerful fraud signal) is inaccessible under RBI data governance rules.

3. **Rural illiterate users** — OTP-based delivery confirmation can be exploited via social engineering. Gram Panchayat secretary verification proposed as a partial fix.

4. **Gas decanting** — physical crime occurring outside the digital system. Detectable only via consumer complaint patterns + physical inspection.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| API Framework | FastAPI + Pydantic |
| ML Model | scikit-learn IsolationForest |
| GSTN Integration | Appyflow API (GSTN wrapper) |
| Cryptographic Chain | SHA-256 (Python hashlib) |
| Deployment | Render.com (free tier) |
| Language | Python 3.11 |

---

## 📄 License

Licensed under the **Apache License 2.0** — see [LICENSE](./LICENSE) for details.

You are free to use, modify, and distribute this code. Attribution required. Patent claims in this repository are not licensed under this license.

---

## 👤 Author

**Athrva Raut**  
B-Tech CSE-AIML, Bharati Vidyapeeth College of Engineering (Deemed University), Navi Mumbai  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Athrva%20Raut-0077B5?style=flat-square&logo=linkedin)](https://linkedin.com/in/athrvaraut)
[![GitHub](https://img.shields.io/badge/GitHub-athrvaraut-181717?style=flat-square&logo=github)](https://github.com/athrvaraut)
[![ResearchGate](https://img.shields.io/badge/ResearchGate-DOI%2010.13140-00CCBB?style=flat-square&logo=researchgate)](https://doi.org/10.13140/RG.2.2.12096.49924)

---

*Built with the goal of making India's LPG system transparent, fraud-resistant, and fair for 33 crore households.*
