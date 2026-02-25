from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


# -- Enums --

class TaskType(str, Enum):
    symptom_diagnosis = "symptom_diagnosis"
    medical_imaging = "medical_imaging"
    risk_scoring = "risk_scoring"


class MinerTier(str, Enum):
    entry = "entry"
    mid = "mid"
    high = "high"


# -- Healthcare Synapse (Challenge from Validator -> Miner) --

class PatientContext(BaseModel):
    age: int = Field(..., example=58)
    sex: str = Field(..., example="male")
    chief_complaint: str = Field(..., example="chest_pain_shortness_of_breath")
    medical_history: str = Field(..., example="hypertension, type_2_diabetes")
    current_medications: str = Field(..., example="metformin, lisinopril")


class HealthcareSynapse(BaseModel):
    """Challenge dispatched by Validator to Miners via Bittensor network."""
    task_type: TaskType = Field(..., description="Type of healthcare prediction task")
    patient_context: PatientContext = Field(..., description="Anonymized patient clinical context")
    clinical_data: Optional[str] = Field(None, example="vitals: BP 158/94, HR 102, SpO2 94%, Temp 37.1C")
    imaging_data: Optional[str] = Field(None, example="chest_xray_PA_lateral")
    lab_results: Optional[str] = Field(None, example="troponin_I: 0.08 ng/mL, BNP: 420 pg/mL, D-dimer: 1.2 ug/mL")
    random_seed: Optional[int] = Field(None, example=83920174)


# -- Miner Response --

class MinerPrediction(BaseModel):
    """Prediction returned by a Miner in response to a Validator challenge."""
    miner_uid: int = Field(..., description="Miner UID on the subnet")
    miner_hotkey: str = Field(..., description="Miner hotkey address")
    primary_diagnosis: Optional[str] = Field(None, example="Acute Coronary Syndrome (NSTEMI)")
    icd10_code: Optional[str] = Field(None, example="I21.4")
    differential_diagnoses: Optional[List[str]] = Field(None, example=["Unstable Angina (I20.0)", "Pulmonary Embolism (I26.99)"])
    risk_score: Optional[float] = Field(None, ge=0, le=1, example=0.78)
    confidence: Optional[float] = Field(None, ge=0, le=1, example=0.85)
    clinical_reasoning: Optional[str] = Field(None, description="Clinical reasoning chain")
    recommended_actions: Optional[List[str]] = Field(None, example=["Serial troponins q6h", "ECG monitoring", "Cardiology consult"])
    response_time_ms: Optional[float] = Field(None, description="Response latency in milliseconds")


# -- Validator Scoring --

class ScoreBreakdown(BaseModel):
    accuracy: float = Field(..., ge=0, le=1, description="Diagnostic accuracy score (weight: 50%)")
    differential_quality: float = Field(..., ge=0, le=1, description="Differential diagnosis quality (weight: 15%)")
    calibration: float = Field(..., ge=0, le=1, description="Confidence calibration score (weight: 15%)")
    latency_score: float = Field(..., ge=0, le=1, description="Response latency score (weight: 10%)")
    consistency: float = Field(..., ge=0, le=1, description="Consistency EMA over 100 rounds (weight: 10%)")
    critical_finding_bonus: bool = Field(False, description="1.5x bonus for correctly identifying critical/emergent findings")
    final_score: float = Field(..., ge=0, description="Weighted final score")


class MinerScoreResult(BaseModel):
    miner_uid: int
    miner_hotkey: str
    score: ScoreBreakdown
    rank: int
    tau_earned: float = Field(..., description="Estimated TAO earned this tempo")


# -- Miner Registration & Info --

class MinerRegister(BaseModel):
    hotkey: str = Field(..., example="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty")
    coldkey: str = Field(..., example="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY")
    tier: MinerTier = Field(default=MinerTier.entry)
    ip: str = Field(..., example="192.168.1.100")
    port: int = Field(default=8091, example=8091)
    model_name: Optional[str] = Field(None, example="clinical-diagnosis-transformer-v2")


class MinerInfo(BaseModel):
    uid: int
    hotkey: str
    coldkey: str
    tier: MinerTier
    ip: str
    port: int
    model_name: Optional[str]
    stake: float = Field(0.0, description="TAO staked")
    is_active: bool = True
    total_challenges: int = 0
    avg_score: float = 0.0
    total_tau_earned: float = 0.0
    last_active_block: Optional[int] = None


# -- Validator Registration & Info --

class ValidatorRegister(BaseModel):
    hotkey: str = Field(..., example="5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy")
    coldkey: str = Field(..., example="5HGjWAeFDfFCWPsjFQdVV2Msvz2XtMktvgocEZcCj68kUMaw")
    ip: str = Field(..., example="192.168.1.200")
    port: int = Field(default=8092, example=8092)
    stake: float = Field(default=1000.0, example=1000.0)


class ValidatorInfo(BaseModel):
    uid: int
    hotkey: str
    coldkey: str
    ip: str
    port: int
    stake: float
    is_active: bool = True
    challenges_sent: int = 0
    last_weight_block: Optional[int] = None
    bond_strength: float = 0.0


# -- Challenge Result --

class ChallengeResult(BaseModel):
    challenge_id: str
    synapse: HealthcareSynapse
    challenge_type: str = Field(..., description="retrospective (70%) or prospective (30%)")
    ground_truth: Optional[dict] = Field(None, description="Actual clinical outcome (for retrospective)")
    miner_predictions: List[MinerPrediction]
    scores: List[MinerScoreResult]
    timestamp: str
    tempo: int


# -- Network Status --

class SubnetHyperparameters(BaseModel):
    max_allowed_uids: int = 256
    max_allowed_validators: int = 64
    immunity_period: int = 5000
    weights_rate_limit: int = 100
    commit_reveal_period: int = 1
    tempo: int = 360
    subnet_owner_cut: float = 0.18
    miner_cut: float = 0.41
    validator_cut: float = 0.41


class NetworkStatus(BaseModel):
    subnet_name: str = "Decentralized AI Healthcare Subnet"
    subnet_id: int = 2
    block_height: int
    current_tempo: int
    total_miners: int
    active_miners: int
    total_validators: int
    active_validators: int
    total_stake: float
    total_emission_per_tempo: float
    hyperparameters: SubnetHyperparameters
    top_miners: List[MinerInfo]


# -- Leaderboard --

class LeaderboardEntry(BaseModel):
    rank: int
    miner_uid: int
    miner_hotkey: str
    tier: MinerTier
    avg_score: float
    total_challenges: int
    total_tau_earned: float
    accuracy_avg: float
    differential_quality_avg: float
    streak: int = Field(0, description="Consecutive tempos in top 10")


# -- Simple Patient Query (backward compatible) --

class PatientQuery(BaseModel):
    patient_id: str
    symptoms: str
    history: str


class DiagnosisResponse(BaseModel):
    diagnosis: str
    confidence: float
    advice: str
