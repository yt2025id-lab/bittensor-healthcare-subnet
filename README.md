# Decentralized AI Healthcare Subnet

**Subnet #2 — Bittensor Ideathon**

A privacy-preserving, decentralized AI healthcare subnet on Bittensor. Miners compete to build the most accurate medical AI diagnostic models. Validators verify predictions against clinical ground truth using evidence-based medicine standards. Rewards ($TAO) are distributed via Yuma Consensus.

## Quick Start (For Judges)

```bash
# 1. Clone & enter directory
git clone https://github.com/yt2025id-lab/bittensor-healthcare-subnet.git
cd bittensor-healthcare-subnet

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
uvicorn main:app --reload --port 8000

# 4. Open in browser
open http://localhost:8000
```

### What You'll See

- **Interactive Web UI** at `http://localhost:8000` — click any of the 3 demo scenarios
- **Swagger API Docs** at `http://localhost:8000/docs` — test all endpoints interactively
- **ReDoc** at `http://localhost:8000/redoc` — clean API reference

### Demo Scenarios

| # | Scenario | Task Type |
|---|----------|-----------|
| 1 | 58yo male, chest pain & dyspnea → NSTEMI diagnosis | Symptom Diagnosis |
| 2 | 45yo female, cough & fever → Pneumonia detection | Medical Imaging |
| 3 | 68yo male, CHF+CKD+diabetes → Readmission risk | Risk Scoring |

Each demo broadcasts a clinical challenge to 6 simulated miners, scores their predictions through 3-4 validators, and distributes TAO rewards via Yuma Consensus.

## Features

- 6 specialized medical AI miners (SymptomNet, ClinicalBERT, BioBERT, etc.)
- 3-4 validators with clinical verification pipelines
- ICD-10 diagnosis with differential diagnoses
- Real-time scoring: accuracy, differential quality, calibration, latency
- TAO reward distribution via Yuma Consensus
- Full miner/validator CRUD, leaderboard, and network status APIs

## Folder Structure

```
main.py                  # FastAPI entry point
healthcare/
  __init__.py
  ai.py                  # AI prediction engine (3 demo scenarios, 6 miners)
  db.py                  # In-memory DB (miners, validators, challenges)
  models.py              # Pydantic data models
  routes.py              # 20+ API endpoints
static/
  index.html             # Interactive demo UI
  app.js                 # Frontend logic
  style.css              # Dark theme styling
overview/overview.md     # Full technical documentation
pitchdeck/               # Pitch deck materials
SUBNET_PROPOSAL.md       # Detailed subnet design proposal
```

## Scoring Formula

```
final_score = (0.50 × accuracy + 0.15 × differential_quality
             + 0.15 × calibration + 0.10 × latency + 0.10 × consistency)
             × 1.5 if critical finding correct
```

## Subnet Parameters

- **Subnet ID:** 2 | **Tempo:** 360 blocks (~72 min) | **Max UIDs:** 256
- **Emission Split:** Owner 18% | Miners 41% | Validators+Stakers 41%

## Miner Tasks

| Task | Weight | Description |
|------|--------|-------------|
| Symptom Diagnosis | 50% | Differential diagnosis from patient symptoms & labs |
| Medical Imaging | 30% | Classification from X-ray, CT, MRI images |
| Risk Scoring | 20% | Readmission, mortality, and complication risk |

## License

MIT

## Documentation

- [`SUBNET_PROPOSAL.md`](SUBNET_PROPOSAL.md) — Full technical subnet design proposal
- [`overview/overview.md`](overview/overview.md) — Problem/solution, architecture, mechanism design
- [`pitchdeck/`](pitchdeck/) — Pitch deck and demo video script
