# Decentralized AI Healthcare

## Overview
A privacy-preserving, decentralized AI healthcare assistant using Bittensor subnets. Patients and providers interact with AI for diagnosis, triage, and health insights, with data privacy and auditability.

## Features
- Privacy-preserving AI diagnosis
- Patient-provider AI chat
- Medical data anonymization
- Bittensor subnet integration with $TAO rewards

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Run the app: `python main.py`
3. Submit diagnosis queries via `/diagnose` endpoint

## Folder Structure
- `main.py`: Entry point (FastAPI)
- `healthcare/`: Core logic
  - `ai.py`: AI diagnosis engine
  - `models.py`: Data models (PatientQuery, DiagnosisResponse)
  - `routes.py`: API routes
- `overview/`: Full project documentation
- `requirements.txt`: Dependencies

## Bittensor Subnet Design
- **Miner:** Runs medical AI diagnostic models, returns ICD-10 diagnoses with confidence scores
- **Validator:** Evaluates diagnosis accuracy against curated medical datasets with known ground truth
- **Incentive:** $TAO rewards based on diagnostic accuracy, confidence calibration, and consistency

## License
MIT

## Subnet Design Proposal
See [`SUBNET_PROPOSAL.md`](SUBNET_PROPOSAL.md) for the full technical subnet design proposal, including incentive mechanism, miner/validator design, business logic, and go-to-market strategy.

## Full Documentation
See `overview/overview.md` for detailed problem/solution, architecture, and mechanism design.
