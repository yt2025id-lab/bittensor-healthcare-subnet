from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from healthcare.routes import router as healthcare_router

app = FastAPI(
    title="Decentralized AI Healthcare Subnet",
    description="""
## Decentralized Healthcare Intelligence — Powered by Bittensor & Yuma Consensus

**Decentralized AI Healthcare** is a Bittensor subnet that creates a decentralized marketplace for clinical AI models.

### How It Works

- **Miners** compete to build AI models that accurately diagnose conditions, classify medical images, and score patient risk
- **Validators** verify predictions against clinical ground truth using evidence-based medicine standards
- **Rewards** ($TAO) are distributed based on diagnostic accuracy via Yuma Consensus

### Miner Tasks

| Task | Weight | Description |
|------|--------|-------------|
| Symptom Diagnosis | 50% | Differential diagnosis given patient presentation and labs |
| Medical Imaging | 30% | Classify pathology from chest X-rays, CT scans, etc. |
| Risk Scoring | 20% | Predict readmission risk, mortality, cardiovascular events |

### Scoring Formula

```
final_score = (0.50 x diagnostic_accuracy + 0.15 x differential_quality
             + 0.15 x calibration + 0.10 x latency + 0.10 x consistency)
             x 1.5 if critical finding correctly identified
```

### Subnet Parameters
- **Subnet ID:** 2 | **Tempo:** 360 blocks (~72 min) | **Max UIDs:** 256
- **Emission Split:** Owner 18% | Miners 41% | Validators+Stakers 41%

---
*Subnet #2 — Decentralized AI Healthcare*
    """,
    version="1.0.0-beta",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "Healthcare API",
            "description": "User-facing endpoints. Query the subnet for clinical AI predictions.",
        },
        {
            "name": "Miners",
            "description": "Miner management — register, list, and run predictions on individual healthcare AI miners.",
        },
        {
            "name": "Validators",
            "description": "Validator operations — generate clinical challenges, dispatch to miners, and score predictions.",
        },
        {
            "name": "Network",
            "description": "Subnet network status, leaderboard, emission distribution, and hyperparameters.",
        },
        {
            "name": "Demo Simulation",
            "description": "Full simulation endpoints — run complete tempo cycles and compare miners side-by-side.",
        },
    ],
)

# API routes
app.include_router(healthcare_router)

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
def root():
    return FileResponse("static/index.html")
