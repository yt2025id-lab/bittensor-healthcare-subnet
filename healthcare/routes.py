"""
API Routes for Decentralized AI Healthcare Subnet Demo.
Demonstrates full subnet functionality: Miners, Validators, Scoring, and Network.
"""

import random
import time
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from .models import (
    TaskType, MinerTier,
    PatientContext, HealthcareSynapse,
    MinerPrediction, ScoreBreakdown, MinerScoreResult,
    MinerRegister, MinerInfo,
    ValidatorRegister, ValidatorInfo,
    ChallengeResult, NetworkStatus, SubnetHyperparameters,
    LeaderboardEntry,
    PatientQuery, DiagnosisResponse,
)
from .ai import run_miner_prediction, score_prediction, get_diagnosis, run_demo_scenario, get_demo_scenarios_list
from . import db

router = APIRouter()


# ===============================================================
# 1. HEALTHCARE QUERY (User-facing API)
# ===============================================================

@router.post(
    "/diagnose",
    response_model=DiagnosisResponse,
    tags=["Healthcare API"],
    summary="Get AI Diagnosis",
    description=(
        "User-facing endpoint. Submit patient symptoms and history to receive AI-powered "
        "diagnostic predictions from the decentralized miner network. Returns primary diagnosis, "
        "confidence score, and clinical advice."
    ),
)
def diagnose(query: PatientQuery):
    result = get_diagnosis(query)
    return DiagnosisResponse(**result)


# ===============================================================
# 2. MINER ENDPOINTS
# ===============================================================

@router.get(
    "/miners",
    response_model=List[MinerInfo],
    tags=["Miners"],
    summary="List All Miners",
    description="Get list of all registered healthcare AI miners on the subnet with their stats and performance.",
)
def list_miners():
    miners = db.get_miners()
    return [MinerInfo(**m) for m in miners.values()]


@router.get(
    "/miners/{uid}",
    response_model=MinerInfo,
    tags=["Miners"],
    summary="Get Miner Details",
    description="Get detailed information about a specific miner by UID.",
)
def get_miner(uid: int):
    miner = db.get_miner(uid)
    if not miner:
        raise HTTPException(status_code=404, detail=f"Miner UID {uid} not found")
    return MinerInfo(**miner)


@router.post(
    "/miners/register",
    response_model=MinerInfo,
    tags=["Miners"],
    summary="Register New Miner",
    description=(
        "Register a new healthcare AI miner on the subnet. Requires hotkey, coldkey, and network info. "
        "New miners start with 0 stake and enter the immunity period (5000 blocks)."
    ),
)
def register_miner(miner: MinerRegister):
    for m in db.get_miners().values():
        if m["hotkey"] == miner.hotkey:
            raise HTTPException(status_code=400, detail="Hotkey already registered")

    result = db.add_miner(miner.dict())
    return MinerInfo(**result)


@router.post(
    "/miners/{uid}/predict",
    response_model=MinerPrediction,
    tags=["Miners"],
    summary="Run Miner Prediction",
    description=(
        "Simulate a miner processing a healthcare challenge. "
        "The miner runs its AI model and returns a diagnosis, ICD-10 code, "
        "differential diagnoses, risk score, and clinical reasoning. Response varies by miner tier."
    ),
)
def miner_predict(uid: int, synapse: HealthcareSynapse):
    miner = db.get_miner(uid)
    if not miner:
        raise HTTPException(status_code=404, detail=f"Miner UID {uid} not found")

    result = run_miner_prediction(synapse.dict(), miner["tier"])
    result["miner_uid"] = uid
    result["miner_hotkey"] = miner["hotkey"]

    return MinerPrediction(**result)


# ===============================================================
# 3. VALIDATOR ENDPOINTS
# ===============================================================

@router.get(
    "/validators",
    response_model=List[ValidatorInfo],
    tags=["Validators"],
    summary="List All Validators",
    description="Get list of all registered validators on the healthcare subnet.",
)
def list_validators():
    validators = db.get_validators()
    return [ValidatorInfo(**v) for v in validators.values()]


@router.get(
    "/validators/{uid}",
    response_model=ValidatorInfo,
    tags=["Validators"],
    summary="Get Validator Details",
    description="Get detailed information about a specific validator by UID.",
)
def get_validator(uid: int):
    validator = db.get_validator(uid)
    if not validator:
        raise HTTPException(status_code=404, detail=f"Validator UID {uid} not found")
    return ValidatorInfo(**validator)


@router.post(
    "/validators/register",
    response_model=ValidatorInfo,
    tags=["Validators"],
    summary="Register New Validator",
    description="Register a new validator on the healthcare subnet. Requires stake to participate.",
)
def register_validator(validator: ValidatorRegister):
    for v in db.get_validators().values():
        if v["hotkey"] == validator.hotkey:
            raise HTTPException(status_code=400, detail="Hotkey already registered")

    result = db.add_validator(validator.dict())
    return ValidatorInfo(**result)


@router.post(
    "/validators/{uid}/generate-challenge",
    response_model=HealthcareSynapse,
    tags=["Validators"],
    summary="Generate Healthcare Challenge",
    description=(
        "Validator generates a healthcare challenge (HealthcareSynapse) to dispatch to miners. "
        "70% are retrospective challenges (past cases with known outcomes), "
        "30% are prospective challenges (current cases for forward prediction). "
        "Each challenge includes patient context, clinical data, and a random seed."
    ),
)
def generate_challenge(
    uid: int,
    task_type: TaskType = Query(default=TaskType.symptom_diagnosis, description="Type of challenge"),
):
    validator = db.get_validator(uid)
    if not validator:
        raise HTTPException(status_code=404, detail=f"Validator UID {uid} not found")

    # Realistic challenge generation
    complaints = [
        "chest_pain_shortness_of_breath",
        "cough_fever_chest_pain",
        "headache_visual_changes",
        "abdominal_pain_nausea",
        "altered_mental_status",
        "syncope_dizziness",
    ]
    histories = [
        "hypertension, type_2_diabetes",
        "asthma, former smoker",
        "atrial fibrillation, CHF",
        "COPD, coronary artery disease",
        "CKD stage 3, obesity",
        "no significant past medical history",
    ]
    medications = [
        "metformin 1000mg BID, lisinopril 20mg daily",
        "albuterol PRN, fluticasone inhaler",
        "apixaban 5mg BID, metoprolol 50mg BID",
        "tiotropium inhaler, atorvastatin 40mg",
        "furosemide 40mg daily, carvedilol 12.5mg BID",
        "none",
    ]

    idx = random.randint(0, len(complaints) - 1)
    age = random.randint(25, 85)
    sex = random.choice(["male", "female"])

    synapse = HealthcareSynapse(
        task_type=task_type,
        patient_context=PatientContext(
            age=age,
            sex=sex,
            chief_complaint=random.choice(complaints),
            medical_history=random.choice(histories),
            current_medications=random.choice(medications),
        ),
        clinical_data=f"vitals: BP {random.randint(110,180)}/{random.randint(60,100)}, HR {random.randint(60,120)}, RR {random.randint(14,28)}, SpO2 {random.randint(88,100)}%, Temp {round(random.uniform(36.5, 39.5), 1)}C",
        lab_results=f"WBC: {round(random.uniform(4.0, 18.0), 1)}, Hgb: {round(random.uniform(8.0, 16.0), 1)}, Cr: {round(random.uniform(0.6, 3.0), 1)}, BNP: {random.randint(50, 1200)}",
        random_seed=random.randint(10000000, 99999999),
    )

    validator["challenges_sent"] += 1
    return synapse


@router.post(
    "/validators/{uid}/run-challenge",
    response_model=ChallengeResult,
    tags=["Validators"],
    summary="Run Full Challenge Cycle",
    description=(
        "Execute a complete healthcare challenge cycle:\n"
        "1. Validator generates a clinical challenge (HealthcareSynapse)\n"
        "2. Challenge is dispatched to ALL active miners\n"
        "3. Each miner runs its diagnostic/prediction model\n"
        "4. Validator scores each miner's prediction against clinical ground truth\n"
        "5. Miners are ranked and TAO rewards are distributed\n\n"
        "This simulates one full tempo cycle of the subnet."
    ),
)
def run_challenge(
    uid: int,
    task_type: TaskType = Query(default=TaskType.symptom_diagnosis),
    synapse: Optional[HealthcareSynapse] = None,
):
    validator = db.get_validator(uid)
    if not validator:
        raise HTTPException(status_code=404, detail=f"Validator UID {uid} not found")

    # Generate challenge if not provided
    if synapse is None:
        synapse = HealthcareSynapse(
            task_type=task_type,
            patient_context=PatientContext(
                age=random.randint(30, 80),
                sex=random.choice(["male", "female"]),
                chief_complaint=random.choice(["chest_pain", "dyspnea", "cough_fever", "syncope"]),
                medical_history=random.choice(["hypertension, diabetes", "COPD, CAD", "CHF, CKD"]),
                current_medications="as prescribed",
            ),
            clinical_data=f"vitals: BP {random.randint(110,170)}/{random.randint(60,95)}, HR {random.randint(65,115)}, SpO2 {random.randint(90,99)}%",
            random_seed=random.randint(10000000, 99999999),
        )

    # Determine challenge type (70% retrospective, 30% prospective)
    is_retrospective = random.random() < 0.7
    challenge_type = "retrospective" if is_retrospective else "prospective"

    rng = random.Random(synapse.random_seed if synapse.random_seed else int(time.time()))
    ground_truth = None
    if is_retrospective:
        diagnoses = [
            ("Acute Coronary Syndrome", "I21.4", "emergent"),
            ("Community-Acquired Pneumonia", "J18.1", "moderate"),
            ("Pulmonary Embolism", "I26.99", "emergent"),
            ("Heart Failure Exacerbation", "I50.9", "high_risk"),
            ("Acute Kidney Injury", "N17.9", "moderate"),
        ]
        dx = rng.choice(diagnoses)
        ground_truth = {
            "actual_diagnosis": dx[0],
            "actual_icd10": dx[1],
            "severity": dx[2],
            "outcome": "treated_and_discharged",
        }

    # Dispatch to all active miners and collect predictions
    miners = db.get_miners()
    predictions = []
    for miner_uid, miner in miners.items():
        if not miner["is_active"]:
            continue
        result = run_miner_prediction(synapse.dict(), miner["tier"])
        result["miner_uid"] = miner_uid
        result["miner_hotkey"] = miner["hotkey"]
        predictions.append(MinerPrediction(**result))

    # Score each prediction
    scores = []
    total_emission = db.get_state()["total_emission_per_tempo"] * 0.41
    for i, pred in enumerate(predictions):
        if ground_truth:
            score_data = score_prediction(pred.dict(), ground_truth)
        else:
            score_data = {
                "accuracy": round(rng.uniform(0.5, 0.95), 4),
                "differential_quality": round(rng.uniform(0.3, 0.9), 4),
                "calibration": round(rng.uniform(0.4, 0.85), 4),
                "latency_score": round(rng.uniform(0.7, 0.99), 4),
                "consistency": round(rng.uniform(0.6, 0.92), 4),
                "critical_finding_bonus": False,
                "final_score": 0,
            }
            score_data["final_score"] = round(
                0.50 * score_data["accuracy"]
                + 0.15 * score_data["differential_quality"]
                + 0.15 * score_data["calibration"]
                + 0.10 * score_data["latency_score"]
                + 0.10 * score_data["consistency"],
                4
            )

        scores.append({
            "miner_uid": pred.miner_uid,
            "miner_hotkey": pred.miner_hotkey,
            "score": ScoreBreakdown(**score_data),
            "rank": 0,
            "tau_earned": 0,
        })

    # Rank by final score
    scores.sort(key=lambda s: s["score"].final_score, reverse=True)
    total_scores = sum(s["score"].final_score for s in scores)
    for rank, s in enumerate(scores, 1):
        s["rank"] = rank
        if total_scores > 0:
            s["tau_earned"] = round(total_emission * (s["score"].final_score / total_scores), 6)
        else:
            s["tau_earned"] = 0

        db.update_miner_score(s["miner_uid"], s["score"].final_score, s["tau_earned"])

    score_results = [MinerScoreResult(**s) for s in scores]

    # Update validator
    validator["challenges_sent"] += 1
    validator["last_weight_block"] = db.get_state()["block_height"]

    # Advance blocks
    db.advance_block(random.randint(1, 5))

    # Save challenge
    challenge_id = str(uuid.uuid4())[:8]
    challenge_record = {
        "challenge_id": challenge_id,
        "synapse": synapse,
        "challenge_type": challenge_type,
        "ground_truth": ground_truth,
        "miner_predictions": predictions,
        "scores": score_results,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "tempo": db.get_state()["current_tempo"],
    }
    db.add_challenge(challenge_record)

    return ChallengeResult(**challenge_record)


@router.post(
    "/validators/{uid}/score-prediction",
    response_model=MinerScoreResult,
    tags=["Validators"],
    summary="Score a Single Miner Prediction",
    description=(
        "Validator scores a specific miner's prediction against clinical ground truth.\n\n"
        "Scoring dimensions:\n"
        "- **Diagnostic Accuracy (50%):** ICD-10 category match + clinical concordance\n"
        "- **Differential Quality (15%):** Completeness and relevance of differential diagnoses\n"
        "- **Calibration (15%):** `1 - |confidence - actual_accuracy|`\n"
        "- **Latency (10%):** `max(1 - elapsed/10s, 0)`\n"
        "- **Consistency (10%):** EMA over 100 rounds\n"
        "- **Critical Finding Bonus:** 1.5x for correctly identifying emergent/critical findings"
    ),
)
def score_single_prediction(
    uid: int,
    prediction: MinerPrediction,
    actual_diagnosis: str = Query(..., description="Actual clinical diagnosis"),
    actual_icd10: str = Query(..., description="Actual ICD-10 code"),
    severity: str = Query(default="moderate", description="Clinical severity: moderate, emergent, critical"),
):
    validator = db.get_validator(uid)
    if not validator:
        raise HTTPException(status_code=404, detail=f"Validator UID {uid} not found")

    ground_truth = {
        "actual_diagnosis": actual_diagnosis,
        "actual_icd10": actual_icd10,
        "severity": severity,
    }

    score_data = score_prediction(prediction.dict(), ground_truth)
    tau_earned = round(db.get_state()["total_emission_per_tempo"] * 0.41 * score_data["final_score"] / 8, 6)

    db.update_miner_score(prediction.miner_uid, score_data["final_score"], tau_earned)

    return MinerScoreResult(
        miner_uid=prediction.miner_uid,
        miner_hotkey=prediction.miner_hotkey,
        score=ScoreBreakdown(**score_data),
        rank=1,
        tau_earned=tau_earned,
    )


# ===============================================================
# 4. NETWORK & SUBNET ENDPOINTS
# ===============================================================

@router.get(
    "/network/status",
    response_model=NetworkStatus,
    tags=["Network"],
    summary="Subnet Network Status",
    description=(
        "Get the current status of the Decentralized AI Healthcare subnet including "
        "block height, tempo, miner/validator counts, stake, emissions, "
        "and top performing miners."
    ),
)
def network_status():
    state = db.get_state()
    miners = db.get_miners()
    validators = db.get_validators()

    active_miners = [m for m in miners.values() if m["is_active"]]
    active_validators = [v for v in validators.values() if v["is_active"]]
    total_stake = sum(m["stake"] for m in miners.values()) + sum(v["stake"] for v in validators.values())

    top = sorted(active_miners, key=lambda m: m["avg_score"], reverse=True)[:5]

    return NetworkStatus(
        block_height=state["block_height"],
        current_tempo=state["current_tempo"],
        total_miners=len(miners),
        active_miners=len(active_miners),
        total_validators=len(validators),
        active_validators=len(active_validators),
        total_stake=round(total_stake, 2),
        total_emission_per_tempo=state["total_emission_per_tempo"],
        hyperparameters=SubnetHyperparameters(),
        top_miners=[MinerInfo(**m) for m in top],
    )


@router.get(
    "/network/leaderboard",
    response_model=List[LeaderboardEntry],
    tags=["Network"],
    summary="Miner Leaderboard",
    description=(
        "Get the ranked leaderboard of all healthcare AI miners sorted by average score. "
        "Shows performance metrics, TAO earned, and tier classification."
    ),
)
def leaderboard():
    miners = db.get_leaderboard()
    rng = random.Random(42)
    entries = []
    for rank, m in enumerate(miners, 1):
        entries.append(LeaderboardEntry(
            rank=rank,
            miner_uid=m["uid"],
            miner_hotkey=m["hotkey"],
            tier=m["tier"],
            avg_score=m["avg_score"],
            total_challenges=m["total_challenges"],
            total_tau_earned=m["total_tau_earned"],
            accuracy_avg=round(m["avg_score"] * rng.uniform(0.9, 1.1), 3),
            differential_quality_avg=round(m["avg_score"] * rng.uniform(0.75, 1.0), 3),
            streak=max(0, int((m["avg_score"] - 0.5) * 20) + rng.randint(0, 5)),
        ))
    return entries


@router.get(
    "/network/challenges",
    response_model=List[ChallengeResult],
    tags=["Network"],
    summary="Recent Challenges",
    description="Get the most recent healthcare challenges and their results.",
)
def recent_challenges(limit: int = Query(default=10, ge=1, le=50)):
    challenges = db.get_challenges(limit)
    return [ChallengeResult(**c) for c in challenges]


@router.get(
    "/network/hyperparameters",
    response_model=SubnetHyperparameters,
    tags=["Network"],
    summary="Subnet Hyperparameters",
    description=(
        "Get the current subnet hyperparameters including max UIDs, "
        "immunity period, tempo length, and emission split."
    ),
)
def hyperparameters():
    return SubnetHyperparameters()


@router.get(
    "/network/emission-distribution",
    tags=["Network"],
    summary="Current Emission Distribution",
    description="Shows how TAO emissions are distributed among subnet participants this tempo.",
)
def emission_distribution():
    state = db.get_state()
    total = state["total_emission_per_tempo"]
    miners = db.get_miners()
    validators = db.get_validators()

    top_miners = sorted(miners.values(), key=lambda m: m["avg_score"], reverse=True)[:5]

    return {
        "tempo": state["current_tempo"],
        "total_emission_tao": total,
        "distribution": {
            "subnet_owner": {"share": "18%", "amount_tao": round(total * 0.18, 6)},
            "miners_total": {"share": "41%", "amount_tao": round(total * 0.41, 6)},
            "validators_stakers_total": {"share": "41%", "amount_tao": round(total * 0.41, 6)},
        },
        "top_miner_earnings": [
            {
                "uid": m["uid"],
                "hotkey": m["hotkey"][:16] + "...",
                "tier": m["tier"],
                "score": m["avg_score"],
                "estimated_tao_this_tempo": round(total * 0.41 * m["avg_score"] / max(1, sum(mm["avg_score"] for mm in miners.values())), 6),
            }
            for m in top_miners
        ],
    }


# ===============================================================
# 5. DEMO / SIMULATION ENDPOINTS
# ===============================================================

@router.post(
    "/demo/full-tempo-cycle",
    tags=["Demo Simulation"],
    summary="Run Full Tempo Cycle",
    description=(
        "Simulates a complete tempo cycle (~72 minutes compressed into one API call):\n\n"
        "1. Validator generates 3 clinical challenges (2 retrospective + 1 prospective)\n"
        "2. All miners receive and process each challenge\n"
        "3. Validator scores all predictions against clinical ground truth\n"
        "4. Weights are updated via Yuma Consensus\n"
        "5. TAO emissions are distributed\n"
        "6. Block height and tempo advance\n\n"
        "Returns complete results for all 3 challenges."
    ),
)
def full_tempo_cycle():
    state = db.get_state()
    validators = list(db.get_validators().values())
    if not validators:
        raise HTTPException(status_code=400, detail="No validators registered")

    active_validators = [v for v in validators if v["is_active"]]
    if not active_validators:
        raise HTTPException(status_code=400, detail="No active validators")

    lead_validator = max(active_validators, key=lambda v: v["stake"])

    results = []
    task_types = [TaskType.symptom_diagnosis, TaskType.medical_imaging, TaskType.risk_scoring]

    complaints = [
        "chest_pain_shortness_of_breath",
        "cough_fever_chest_pain",
        "heart_failure_exacerbation",
    ]
    histories = [
        "hypertension, type_2_diabetes, hyperlipidemia",
        "asthma, former smoker",
        "CHF, CKD stage 3a, diabetes, obesity",
    ]

    for i, task_type in enumerate(task_types):
        synapse = HealthcareSynapse(
            task_type=task_type,
            patient_context=PatientContext(
                age=random.randint(35, 75),
                sex=random.choice(["male", "female"]),
                chief_complaint=complaints[i],
                medical_history=histories[i],
                current_medications="as prescribed per chart",
            ),
            clinical_data=f"vitals: BP {random.randint(115,165)}/{random.randint(65,95)}, HR {random.randint(68,110)}, SpO2 {random.randint(90,98)}%",
            lab_results=f"WBC: {round(random.uniform(5.0, 16.0), 1)}, Cr: {round(random.uniform(0.7, 2.0), 1)}, BNP: {random.randint(80, 900)}",
            random_seed=random.randint(10000000, 99999999),
        )

        is_retrospective = i < 2
        challenge_type = "retrospective" if is_retrospective else "prospective"

        rng = random.Random(synapse.random_seed)
        ground_truth = None
        if is_retrospective:
            diagnoses = [
                ("Acute Coronary Syndrome", "I21.4", "emergent"),
                ("Community-Acquired Pneumonia", "J18.1", "moderate"),
            ]
            dx = diagnoses[i]
            ground_truth = {
                "actual_diagnosis": dx[0],
                "actual_icd10": dx[1],
                "severity": dx[2],
            }

        miners = db.get_miners()
        predictions = []
        for miner_uid, miner in miners.items():
            if not miner["is_active"]:
                continue
            result = run_miner_prediction(synapse.dict(), miner["tier"])
            result["miner_uid"] = miner_uid
            result["miner_hotkey"] = miner["hotkey"]
            predictions.append(MinerPrediction(**result))

        scores = []
        total_emission = state["total_emission_per_tempo"] * 0.41 / 3
        for pred in predictions:
            if ground_truth:
                score_data = score_prediction(pred.dict(), ground_truth)
            else:
                score_data = {
                    "accuracy": round(rng.uniform(0.5, 0.95), 4),
                    "differential_quality": round(rng.uniform(0.3, 0.9), 4),
                    "calibration": round(rng.uniform(0.4, 0.85), 4),
                    "latency_score": round(rng.uniform(0.7, 0.99), 4),
                    "consistency": round(rng.uniform(0.6, 0.92), 4),
                    "critical_finding_bonus": False,
                    "final_score": 0,
                }
                score_data["final_score"] = round(
                    0.50 * score_data["accuracy"]
                    + 0.15 * score_data["differential_quality"]
                    + 0.15 * score_data["calibration"]
                    + 0.10 * score_data["latency_score"]
                    + 0.10 * score_data["consistency"],
                    4
                )

            scores.append({
                "miner_uid": pred.miner_uid,
                "miner_hotkey": pred.miner_hotkey,
                "score": ScoreBreakdown(**score_data),
                "rank": 0,
                "tau_earned": 0,
            })

        scores.sort(key=lambda s: s["score"].final_score, reverse=True)
        total_scores = sum(s["score"].final_score for s in scores)
        for rank, s in enumerate(scores, 1):
            s["rank"] = rank
            if total_scores > 0:
                s["tau_earned"] = round(total_emission * (s["score"].final_score / total_scores), 6)
            db.update_miner_score(s["miner_uid"], s["score"].final_score, s["tau_earned"])

        challenge_id = str(uuid.uuid4())[:8]
        challenge_record = {
            "challenge_id": challenge_id,
            "synapse": synapse,
            "challenge_type": challenge_type,
            "ground_truth": ground_truth,
            "miner_predictions": predictions,
            "scores": [MinerScoreResult(**s) for s in scores],
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "tempo": state["current_tempo"],
        }
        db.add_challenge(challenge_record)
        results.append(ChallengeResult(**challenge_record))

    db.advance_tempo()

    lead_validator["challenges_sent"] += 3
    lead_validator["last_weight_block"] = state["block_height"]

    return {
        "tempo_completed": state["current_tempo"] - 1,
        "new_tempo": state["current_tempo"],
        "block_height": state["block_height"],
        "lead_validator_uid": lead_validator["uid"],
        "challenges_run": len(results),
        "challenge_types": ["retrospective", "retrospective", "prospective"],
        "task_types": [str(t.value) for t in task_types],
        "total_tao_distributed": round(state["total_emission_per_tempo"], 6),
        "challenges": results,
        "updated_leaderboard": [
            {
                "rank": rank,
                "uid": m["uid"],
                "hotkey": m["hotkey"][:16] + "...",
                "tier": m["tier"],
                "avg_score": m["avg_score"],
                "total_tau": m["total_tau_earned"],
            }
            for rank, m in enumerate(sorted(db.get_miners().values(), key=lambda x: x["avg_score"], reverse=True), 1)
        ],
    }


@router.post(
    "/demo/compare-miners",
    tags=["Demo Simulation"],
    summary="Compare Miners on Same Challenge",
    description=(
        "Sends the same clinical challenge to all miners and compares their predictions side-by-side. "
        "Shows how different miner tiers (entry/mid/high) produce different quality diagnostic predictions."
    ),
)
def compare_miners(synapse: HealthcareSynapse):
    miners = db.get_miners()
    comparisons = []

    for uid, miner in miners.items():
        if not miner["is_active"]:
            continue
        result = run_miner_prediction(synapse.dict(), miner["tier"])
        comparisons.append({
            "miner_uid": uid,
            "miner_hotkey": miner["hotkey"][:16] + "...",
            "tier": miner["tier"],
            "model": miner["model_name"] or "unknown",
            "primary_diagnosis": result["primary_diagnosis"],
            "icd10_code": result["icd10_code"],
            "risk_score": result["risk_score"],
            "confidence": result["confidence"],
            "response_time_ms": result["response_time_ms"],
            "differential_count": len(result.get("differential_diagnoses", [])),
        })

    comparisons.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "challenge": synapse.dict(),
        "total_miners_queried": len(comparisons),
        "comparisons": comparisons,
        "analysis": {
            "avg_confidence": round(sum(c["confidence"] for c in comparisons) / len(comparisons), 2) if comparisons else 0,
            "avg_risk_score": round(sum(c["risk_score"] for c in comparisons) / len(comparisons), 2) if comparisons else 0,
            "confidence_spread": round(max(c["confidence"] for c in comparisons) - min(c["confidence"] for c in comparisons), 2) if comparisons else 0,
            "highest_confidence_miner": comparisons[0]["miner_uid"] if comparisons else None,
            "fastest_miner": min(comparisons, key=lambda c: c["response_time_ms"])["miner_uid"] if comparisons else None,
        },
    }


# ===============================================================
# 6. LANDING PAGE DEMO ENDPOINTS
# ===============================================================

@router.get(
    "/api/demo-scenarios",
    tags=["Demo Simulation"],
    summary="List Available Demo Scenarios",
    description="Returns metadata for all 3 pre-built healthcare demo scenarios.",
)
def list_demo_scenarios():
    return get_demo_scenarios_list()


@router.get(
    "/api/demo/{scenario_key}",
    tags=["Demo Simulation"],
    summary="Run Demo Scenario",
    description=(
        "Run one of 3 pre-built demo scenarios showing full subnet operation:\n\n"
        "- **demo1:** Symptom-Based Diagnosis (Chest Pain, Emergency Triage)\n"
        "- **demo2:** Medical Image Analysis (Chest X-Ray, Pneumonia Detection)\n"
        "- **demo3:** Patient Risk Scoring (Cardiovascular, 30-Day Readmission)\n\n"
        "Returns full miner responses, validator verifications, consensus, and TAO rewards."
    ),
)
def run_demo(scenario_key: str):
    result = run_demo_scenario(scenario_key)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result
