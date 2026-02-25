"""
AI Healthcare Prediction Engine
Simulates realistic miner AI models and validator scoring for healthcare subnet.
Each demo scenario has specialized miners and validators with unique clinical analysis.
"""

import random
import hashlib
import time
from datetime import datetime


# ============================================================
# SPECIALIZED MINERS & VALIDATORS PER SCENARIO
# Each scenario has dedicated miners with unique names,
# specialties, and analysis patterns -- just like a real subnet.
# ============================================================

SPECIALISTS = {
    "symptom_diagnosis": {
        "miners": [
            {"name": "MedDiag-Transformer",    "hotkey": "5FMdT7kQ", "tier": "high", "specialty": "Transformer-based Differential Diagnosis Engine"},
            {"name": "SymptomNet-v3",           "hotkey": "5FSnT9xP", "tier": "high", "specialty": "Multi-symptom Pattern Recognition Network"},
            {"name": "ClinicalBERT-Dx",         "hotkey": "5FCBl3mK", "tier": "mid",  "specialty": "BERT-based Clinical NLP Diagnosis Model"},
            {"name": "DifferentialDx-LSTM",     "hotkey": "5FDdV2nR", "tier": "mid",  "specialty": "LSTM Sequential Symptom Analysis"},
            {"name": "CardioRisk-GBM",          "hotkey": "5FCrR4pT", "tier": "mid",  "specialty": "Gradient Boosted Cardiovascular Risk Model"},
            {"name": "BasicDx-v1",              "hotkey": "5FBd1qUm", "tier": "entry","specialty": "Rule-based Clinical Decision Support"},
        ],
        "validators": [
            {"name": "EHR-Oracle",              "hotkey": "5VeO1aXp", "specialty": "Electronic Health Record outcome cross-validation"},
            {"name": "ClinicalTrial-Verifier",  "hotkey": "5VcT2bYq", "specialty": "Clinical trial evidence-based verification"},
            {"name": "ICD10-Validator",         "hotkey": "5ViV3cZr", "specialty": "ICD-10-CM coding accuracy verification"},
            {"name": "Guideline-Checker",       "hotkey": "5VgC4dAs", "specialty": "AHA/ACC/ESC clinical guideline compliance check"},
        ],
        "check_labels": ["Diagnosis Matches ICD-10", "Clinical Reasoning Valid", "Guideline-Concordant"],
        "analyses": [
            "Transformer analysis: Processed 847 symptom-feature embeddings across 23,419 historical case presentations. Attention heads weighted chest pain character (8.2%), troponin trend (12.1%), and ECG ST-segment changes (15.3%) as top discriminative features. Bayesian posterior for ACS given presentation: P(ACS|symptoms,labs,ECG) = 0.84. Model cross-referenced against ACC/AHA 2023 NSTEMI guidelines -- presentation meets Class I recommendation for emergent cardiology evaluation. Differential ranked by posterior probability with PE excluded via D-dimer threshold.",
            "Multi-symptom pattern analysis: Identified 14 concordant symptom clusters from presentation. Primary cluster (chest pain + dyspnea + diaphoresis) maps to acute coronary syndrome with 91% sensitivity in training cohort (n=18,247). Secondary cluster (pleuritic component + tachycardia) raises PE probability to 0.18 via Wells score integration. Troponin I at 0.08 ng/mL exceeds 99th percentile URL (0.04 ng/mL), confirming myocardial injury. BNP elevation at 420 pg/mL suggests concurrent heart failure decompensation. HEART score calculation: History(2) + ECG(1) + Age(2) + Risk factors(2) + Troponin(2) = 9 (high risk).",
            "Clinical NLP model: Extracted 32 clinical entities from presentation using BioBERT embeddings. Entity linking to UMLS CUI codes identified C0027051 (Myocardial Infarction) with cosine similarity 0.89 and C0034065 (Pulmonary Embolism) at 0.62. Negation detection confirmed absence of hemoptysis and unilateral leg swelling, reducing PE probability. Temporal reasoning: symptom onset 4 hours ago with progressive worsening pattern consistent with evolving ACS. Medication reconciliation: metformin and lisinopril suggest pre-existing cardiovascular risk factors.",
            "LSTM sequential analysis: Fed time-ordered symptom sequence through 3-layer bidirectional LSTM (hidden dim=256). Sequence pattern [chest_pain -> dyspnea -> diaphoresis -> nausea] has 78% historical association with ACS in our training data (n=12,840 ED presentations). Hidden state analysis shows strongest activation for acute coronary pathway at timestep 3 (diaphoresis onset). Risk stratification via TIMI score integration: age>65(0) + 3+CAD risk factors(1) + known CAD(0) + ASA use(0) + severe angina(1) + ST deviation(1) + elevated biomarker(1) = 4 (intermediate-high risk, 12.1% 14-day event rate).",
            "Gradient boosted model: 412 features including vital signs, lab values, demographic risk factors, and medication history. Top feature importance: troponin_I (18.7%), chest_pain_character (14.2%), age_hypertension_interaction (11.8%), BNP_level (9.4%), SpO2 (7.1%). Ensemble of 800 trees with max depth 6, learning rate 0.05. 5-fold cross-validation AUC for MACE prediction: 0.87. Calibration plot shows slight overconfidence in 0.7-0.8 probability range. Shapley analysis: troponin contributes +0.23 to predicted risk, age/HTN interaction +0.15, SpO2 at 94% adds +0.08.",
            "Rule-based calculation: Applied modified Sgarbossa criteria for STEMI evaluation (negative). Calculated HEART score: 9/10 (high risk). Wells PE score: 3.0 (moderate probability -- consider D-dimer). CURB-65 for pneumonia: 1 (low risk). Based on hierarchical rule engine: troponin > 99th percentile + typical chest pain + risk factors -> primary consideration ACS. Basic recommendation: serial troponins, continuous monitoring, cardiology consultation per institutional protocol.",
        ],
    },
    "medical_imaging": {
        "miners": [
            {"name": "RadiologyVision-v2",    "hotkey": "5FRvT7kQ", "tier": "high", "specialty": "Vision Transformer for Medical Image Classification"},
            {"name": "ChestNet-DenseNet",      "hotkey": "5FCnP9xP", "tier": "high", "specialty": "DenseNet-121 Chest X-ray Pathology Detection"},
            {"name": "PneumoDetect-ResNet",    "hotkey": "5FPdL3mK", "tier": "mid",  "specialty": "ResNet-50 Pneumonia/Consolidation Classifier"},
            {"name": "LungSeg-UNet",           "hotkey": "5FLsV2nR", "tier": "mid",  "specialty": "U-Net Lung Segmentation & Anomaly Detection"},
            {"name": "ThoraxScreen-EfficientNet","hotkey": "5FTsR4pT", "tier": "mid",  "specialty": "EfficientNet-B4 Multi-label Thoracic Screening"},
            {"name": "BasicXray-v1",           "hotkey": "5FBx1qUm", "tier": "entry","specialty": "Simple CNN Chest X-ray Binary Classifier"},
        ],
        "validators": [
            {"name": "DICOM-Oracle",           "hotkey": "5VdO1aXp", "specialty": "DICOM metadata and image quality verification"},
            {"name": "RadiologistConsensus",    "hotkey": "5VrC2bYq", "specialty": "Board-certified radiologist ground truth cross-check"},
            {"name": "PathologyCorrelation",   "hotkey": "5VpC3cZr", "specialty": "Histopathology/clinical outcome correlation"},
        ],
        "check_labels": ["Image Classification Correct", "Localization Accurate", "Confidence Calibrated"],
        "analyses": [
            "Vision Transformer analysis: Processed 512x512 chest PA radiograph through ViT-Large/16 pretrained on 2.1M CXR images (MIMIC-CXR + CheXpert + PadChest). Self-attention maps show strong activation in right lower lobe (RLL) with Grad-CAM heatmap confirming focal opacity localization at coordinates (312, 387). Multi-label classification probabilities: Pneumonia 0.89, Consolidation 0.84, Pleural Effusion 0.31, Cardiomegaly 0.12. Air bronchogram sign detected with 0.78 probability suggesting bacterial etiology. Model confidence calibrated using temperature scaling (T=1.8) on held-out validation set (n=4,200).",
            "DenseNet-121 pathology detection: Applied CheXNet-derived model with 14-pathology multi-label head. Feature extraction from final dense block (1024-dim) shows high activation in RLL region. Top pathology predictions: Infiltrate (0.91), Consolidation (0.87), Pneumonia (0.85). Comparison with lateral view confirms posterior segment involvement consistent with community-acquired pneumonia (CAP). Silhouette sign analysis: right hemidiaphragm partially obscured suggesting RLL involvement. No mediastinal widening (aortic knob ratio within normal limits). Cardiothoracic ratio: 0.48 (normal <0.50).",
            "ResNet-50 pneumonia classifier: Binary classification (Pneumonia vs Normal) with softmax probability 0.87 for pneumonia. Intermediate feature maps at layer conv4_x show strongest response in right costophrenic angle region. Data augmentation during training included random rotation (+-15 degrees), horizontal flip, and intensity jittering. Model trained on NIH ChestX-ray14 dataset (112,120 images). Sensitivity: 92.3%, Specificity: 87.1%, F1: 0.89 on test set. Current image shows patchy opacity in RLL with air bronchograms -- pattern consistent with lobar pneumonia rather than bronchopneumonia.",
            "U-Net lung segmentation: Performed pixel-level segmentation of bilateral lung fields (Dice coefficient: 0.96 on validation). Anomaly detection via reconstruction error in autoencoder branch. Reconstruction MSE in RLL region: 0.142 (threshold: 0.08), indicating significant abnormality. Segmented opacity covers approximately 23% of right lung field. Opacity density analysis: mean HU equivalent estimated at -200 (consistent with consolidation vs -700 for normal aerated lung). No pneumothorax detected (lung-chest wall interface intact bilaterally).",
            "EfficientNet-B4 thoracic screening: Multi-label prediction across 18 thoracic conditions. Compound scaling (depth=1.8, width=1.4, resolution=380) optimized for CXR interpretation. Top predictions: Lung Opacity (0.88), Pneumonia (0.83), Support Devices (0.05). Grad-CAM++ localization highlights 4.2cm x 3.8cm region in RLL. Normal findings confirmed for: mediastinum, cardiac silhouette, osseous structures, and left lung field. Bilateral costophrenic angles: right partially blunted (small effusion possible), left clear.",
            "Simple CNN classification: 3-layer convolutional network with global average pooling. Binary output: Pneumonia probability 0.72, Normal probability 0.28. Limited spatial resolution in activation maps. Model trained on Kaggle Chest X-ray Pneumonia dataset (5,863 images). Note: model does not provide localization or multi-label capability. Basic quality check: image orientation correct, adequate inspiration (>6 posterior ribs visible), no rotation artifact detected.",
        ],
    },
    "risk_scoring": {
        "miners": [
            {"name": "CardioRisk-Transformer",  "hotkey": "5FCrT7kQ", "tier": "high", "specialty": "Transformer-based Cardiovascular Event Prediction"},
            {"name": "ReadmitPredict-XGBoost",   "hotkey": "5FRpP9xP", "tier": "high", "specialty": "XGBoost 30-day Hospital Readmission Predictor"},
            {"name": "MortalityNet-v3",          "hotkey": "5FMnL3mK", "tier": "mid",  "specialty": "Deep Neural Network ICU Mortality Prediction"},
            {"name": "ComorbidityMapper-RF",     "hotkey": "5FCmV2nR", "tier": "mid",  "specialty": "Random Forest Comorbidity Risk Stratification"},
            {"name": "LACE-Enhanced-v2",         "hotkey": "5FLeR4pT", "tier": "mid",  "specialty": "Enhanced LACE Index Readmission Model"},
            {"name": "BasicRisk-v1",             "hotkey": "5FBr1qUm", "tier": "entry","specialty": "Framingham Risk Score Calculator"},
        ],
        "validators": [
            {"name": "OutcomeTracker-Oracle",    "hotkey": "5VoT1aXp", "specialty": "30/60/90-day patient outcome tracking verification"},
            {"name": "Registry-Validator",       "hotkey": "5VrV2bYq", "specialty": "Disease registry outcome cross-validation"},
            {"name": "Actuarial-Checker",        "hotkey": "5VaC3cZr", "specialty": "Actuarial mortality table calibration check"},
        ],
        "check_labels": ["Risk Score Calibrated", "Outcome Prediction Accurate", "Comorbidity Weighting Valid"],
        "analyses": [
            "Transformer risk model: Processed longitudinal patient record through temporal attention mechanism spanning 5-year EHR history. Attention weights: recent HbA1c trend (14.2%), medication adherence pattern (11.8%), prior hospitalization frequency (10.3%), eGFR trajectory (9.7%). 10-year ASCVD risk recalculated with pooled cohort equations enhanced by 847 additional features. Predicted 30-day readmission probability: 0.34. Key risk amplifiers: uncontrolled diabetes (HbA1c 8.9%), stage 3a CKD (eGFR 52), dual antiplatelet non-adherence pattern detected in refill data. Model calibration: Hosmer-Lemeshow p=0.42 (well-calibrated).",
            "XGBoost readmission model: 1,247 features extracted from index admission. Top SHAP values: length_of_stay (+0.08), discharge_disposition (+0.06), Elixhauser_comorbidity_index (+0.05), prior_30day_ED_visits (+0.04), polypharmacy_count (+0.03). Model trained on 2.3M Medicare admissions (2019-2024). Predicted 30-day all-cause readmission: 0.31. Risk category: HIGH (>0.25 threshold). Protective factors identified: discharge to skilled nursing (-0.03), complete medication reconciliation (-0.02). AUC on validation set: 0.78 (C-statistic).",
            "Deep neural network ICU prediction: 4-layer fully connected network (512-256-128-64) with dropout 0.3 and batch normalization. Input: 186 features from first 24h ICU data (MIMIC-IV derived). Predicted in-hospital mortality: 0.18. APACHE IV comparison score: 22 (predicted mortality 0.15). Model captures non-linear interactions: the combination of age >65 + mechanical ventilation + vasopressor use + AKI stage 2+ yields mortality risk 2.3x higher than additive model. Sequential Organ Failure Assessment (SOFA) score: 8 (predicted ICU mortality ~33%).",
            "Random forest comorbidity analysis: Mapped 847 ICD-10 codes to 31 Elixhauser comorbidity categories. Patient has 8 active comorbidities: CHF, diabetes (complicated), hypertension, CKD, obesity, depression, peripheral vascular disease, and COPD. Comorbidity interaction analysis: diabetes + CKD interaction increases readmission risk by 1.4x multiplicatively. Random forest with 500 trees, weighted by comorbidity prevalence-adjusted odds ratios from meta-analysis. Cumulative comorbidity burden index: 12.4 (high risk, >90th percentile).",
            "Enhanced LACE Index: Length of stay (L=5, score=4) + Acuity of admission (A=emergent, score=3) + Comorbidities (C=Charlson 6, score=5) + ED visits in 6 months (E=2, score=3) = LACE total: 15 (high risk, >10). Enhanced with social determinants: lives alone (+1), limited health literacy (+1), no primary care follow-up scheduled (+2). Enhanced LACE: 19. Added ML layer: gradient boosted correction factor based on 45,000 local hospital admissions improves C-statistic from 0.68 to 0.74. Predicted 30-day readmission: 0.38.",
            "Framingham Risk Score: 10-year ASCVD risk calculated using traditional risk factors. Age: 58 (male) + Total cholesterol: 248 mg/dL + HDL: 38 mg/dL + Systolic BP: 158 mmHg (treated) + Diabetes: yes + Smoking: former. Framingham 10-year risk: 28.4% (high risk, >20%). Pooled Cohort Equations 10-year ASCVD risk: 24.1%. Note: traditional score does not account for CKD, obesity, or inflammatory markers which may underestimate true risk. Basic recommendation: high-intensity statin therapy per ACC/AHA guidelines.",
        ],
    },
}


# -- 3 PRE-BUILT DEMO SCENARIOS --

DEMO_SCENARIOS = {
    "demo1": {
        "title": "Symptom-Based Diagnosis -- Chest Pain & Shortness of Breath",
        "subtitle": "58yo male, hypertension + diabetes, presenting with acute chest pain and dyspnea. Emergency triage.",
        "task_type": "symptom_diagnosis",
        "synapse": {
            "task_type": "symptom_diagnosis",
            "patient_context": {
                "age": 58,
                "sex": "male",
                "chief_complaint": "chest_pain_shortness_of_breath",
                "medical_history": "hypertension, type_2_diabetes, hyperlipidemia",
                "current_medications": "metformin 1000mg BID, lisinopril 20mg daily, atorvastatin 40mg daily",
            },
            "clinical_data": "vitals: BP 158/94, HR 102, RR 22, SpO2 94%, Temp 37.1C. ECG: sinus tachycardia, non-specific ST-T changes in V3-V6",
            "lab_results": "troponin_I: 0.08 ng/mL (ref <0.04), BNP: 420 pg/mL (ref <100), D-dimer: 1.2 ug/mL (ref <0.5), WBC: 11.2, Cr: 1.4",
            "random_seed": 42001,
        },
        "ground_truth": {
            "actual_diagnosis": "NSTEMI (Non-ST-Elevation Myocardial Infarction)",
            "actual_icd10": "I21.4",
            "severity": "emergent",
            "outcome": "cardiac_catheterization_performed",
        },
    },
    "demo2": {
        "title": "Medical Image Analysis -- Chest X-Ray Classification",
        "subtitle": "45yo female with cough, fever, and pleuritic chest pain. Chest X-ray PA and lateral obtained.",
        "task_type": "medical_imaging",
        "synapse": {
            "task_type": "medical_imaging",
            "patient_context": {
                "age": 45,
                "sex": "female",
                "chief_complaint": "cough_fever_chest_pain",
                "medical_history": "asthma, former smoker (quit 5 years ago)",
                "current_medications": "albuterol PRN, fluticasone inhaler",
            },
            "clinical_data": "vitals: BP 128/78, HR 96, RR 24, SpO2 92%, Temp 38.9C. Auscultation: decreased breath sounds RLL with crackles",
            "imaging_data": "chest_xray_PA_lateral_2_views",
            "lab_results": "WBC: 14.8 (ref 4.5-11), CRP: 89 mg/L (ref <10), procalcitonin: 0.8 ng/mL (ref <0.1)",
            "random_seed": 42002,
        },
        "ground_truth": {
            "actual_diagnosis": "Right Lower Lobe Community-Acquired Pneumonia",
            "actual_icd10": "J18.1",
            "severity": "moderate",
            "outcome": "inpatient_antibiotics_recovery",
        },
    },
    "demo3": {
        "title": "Patient Risk Scoring -- Cardiovascular Disease Risk",
        "subtitle": "68yo male with CHF, diabetes, CKD. Evaluate 30-day readmission risk post-discharge.",
        "task_type": "risk_scoring",
        "synapse": {
            "task_type": "risk_scoring",
            "patient_context": {
                "age": 68,
                "sex": "male",
                "chief_complaint": "heart_failure_exacerbation",
                "medical_history": "CHF (EF 35%), type_2_diabetes, CKD stage 3a, hypertension, obesity (BMI 34), former smoker, depression",
                "current_medications": "carvedilol 25mg BID, sacubitril/valsartan 97/103mg BID, furosemide 40mg BID, empagliflozin 10mg, insulin glargine 28u, aspirin 81mg, sertraline 100mg",
            },
            "clinical_data": "vitals: BP 132/82, HR 78, RR 18, SpO2 96%, Weight: 102kg (+3kg from dry weight). BNP: 890 pg/mL, HbA1c: 8.9%, eGFR: 52, K: 4.8",
            "lab_results": "BNP: 890, HbA1c: 8.9%, eGFR: 52, Cr: 1.6, K: 4.8, Na: 136, albumin: 3.2, Hgb: 10.8",
            "random_seed": 42003,
        },
        "ground_truth": {
            "actual_diagnosis": "High risk for 30-day readmission",
            "actual_icd10": "I50.9",
            "severity": "high_risk",
            "outcome": "readmitted_day_22_volume_overload",
        },
    },
}


# ============================================================
# MAIN DEMO ENGINE
# ============================================================

def _generate_miner_responses(task_type, synapse, ground_truth, num_miners=6):
    """Generate specialized miner responses with unique clinical analysis per miner."""
    spec = SPECIALISTS.get(task_type, SPECIALISTS["symptom_diagnosis"])
    pool = spec["miners"]
    num = min(num_miners, len(pool))
    selected = pool[:num]
    analyses = spec["analyses"]

    # Extract patient context for realistic prediction generation
    patient = synapse.get("patient_context", {})
    age = patient.get("age", 50)

    # Determine appropriate diagnosis/prediction based on task type
    actual_diagnosis = ground_truth.get("actual_diagnosis", "Unknown")
    actual_icd10 = ground_truth.get("actual_icd10", "R69")
    severity = ground_truth.get("severity", "moderate")

    # ICD-10 codes pool per task type for differential diagnoses
    icd10_pools = {
        "symptom_diagnosis": {
            "primary": actual_icd10,
            "differentials": [
                ("Unstable Angina", "I20.0"),
                ("Pulmonary Embolism", "I26.99"),
                ("Acute Pericarditis", "I30.9"),
                ("Aortic Dissection", "I71.01"),
                ("GERD with chest pain", "K21.0"),
                ("Costochondritis", "M94.0"),
                ("Panic Disorder", "F41.0"),
            ],
        },
        "medical_imaging": {
            "primary": actual_icd10,
            "differentials": [
                ("Lung Abscess", "J85.2"),
                ("Pleural Effusion", "J90"),
                ("Lung Carcinoma", "C34.31"),
                ("Pulmonary Edema", "J81.0"),
                ("Tuberculosis", "A15.0"),
                ("Atelectasis", "J98.11"),
            ],
        },
        "risk_scoring": {
            "primary": actual_icd10,
            "differentials": [
                ("Acute Kidney Injury", "N17.9"),
                ("Hyperkalemia", "E87.5"),
                ("Type 2 Diabetes with complications", "E11.65"),
                ("Hypertensive Heart Disease", "I11.0"),
                ("Major Depressive Disorder", "F32.1"),
                ("Chronic Obstructive Pulmonary Disease", "J44.1"),
            ],
        },
    }

    icd10_data = icd10_pools.get(task_type, icd10_pools["symptom_diagnosis"])

    miners = []
    for i, miner in enumerate(selected):
        rng = random.Random(synapse.get("random_seed", 42) + i * 7)

        tier = miner["tier"]
        if tier == "high":
            score = round(rng.uniform(0.82, 0.97), 4)
            response_time = round(rng.uniform(0.3, 1.2), 2)
            confidence = round(rng.uniform(0.82, 0.96), 2)
            num_differentials = rng.randint(3, 5)
        elif tier == "mid":
            score = round(rng.uniform(0.62, 0.82), 4)
            response_time = round(rng.uniform(0.8, 2.2), 2)
            confidence = round(rng.uniform(0.65, 0.82), 2)
            num_differentials = rng.randint(2, 4)
        else:
            score = round(rng.uniform(0.40, 0.62), 4)
            response_time = round(rng.uniform(1.5, 3.5), 2)
            confidence = round(rng.uniform(0.40, 0.65), 2)
            num_differentials = rng.randint(1, 2)

        # Top miner gets best score
        if i == 0:
            score = round(rng.uniform(0.93, 0.99), 4)
            response_time = round(rng.uniform(0.2, 0.6), 2)
            confidence = round(rng.uniform(0.88, 0.96), 2)
            num_differentials = rng.randint(4, 5)

        # Generate risk score for risk_scoring task
        if task_type == "risk_scoring":
            if tier == "high" or i == 0:
                risk_score = round(rng.uniform(0.28, 0.42), 2)
            elif tier == "mid":
                risk_score = round(rng.uniform(0.20, 0.50), 2)
            else:
                risk_score = round(rng.uniform(0.15, 0.55), 2)
        else:
            risk_score = round(rng.uniform(0.3, 0.9), 2)

        # Select differentials
        available_diffs = icd10_data["differentials"][:]
        rng.shuffle(available_diffs)
        selected_diffs = available_diffs[:num_differentials]
        diff_strings = [f"{name} ({code})" for name, code in selected_diffs]

        hk = miner["hotkey"]
        miners.append({
            "uid": i + 1,
            "hotkey": f"{hk}...{hashlib.md5(hk.encode()).hexdigest()[:6]}",
            "name": miner["name"],
            "tier": tier,
            "specialty": miner["specialty"],
            "primary_diagnosis": actual_diagnosis if (tier == "high" or i == 0) else (actual_diagnosis if rng.random() < 0.6 else selected_diffs[0][0] if selected_diffs else actual_diagnosis),
            "icd10_code": icd10_data["primary"] if (tier == "high" or i == 0) else (icd10_data["primary"] if rng.random() < 0.6 else selected_diffs[0][1] if selected_diffs else icd10_data["primary"]),
            "differential_diagnoses": diff_strings,
            "risk_score": risk_score,
            "confidence": confidence,
            "score": score,
            "response_time_s": response_time,
            "analysis": analyses[i] if i < len(analyses) else analyses[-1],
            "rank": i + 1,
        })

    # Sort by score descending
    miners.sort(key=lambda m: m["score"], reverse=True)
    for i, m in enumerate(miners):
        m["rank"] = i + 1

    return miners


def _generate_validator_results(task_type, num_validators=3):
    """Generate specialized validator verification results."""
    spec = SPECIALISTS.get(task_type, SPECIALISTS["symptom_diagnosis"])
    pool = spec["validators"]
    num = min(num_validators, len(pool))
    selected = pool[:num]
    check_labels = spec["check_labels"]

    validators = []
    for j, val in enumerate(selected):
        rng = random.Random(42 + j * 13)
        hk = val["hotkey"]
        stake = round(rng.uniform(6000, 20000), 2)
        vtrust = round(rng.uniform(0.88, 0.99), 4)

        checks = {}
        checks_passed = 0
        for label in check_labels:
            passed = rng.random() < 0.85
            checks[label] = passed
            if passed:
                checks_passed += 1

        validators.append({
            "uid": j + 1,
            "hotkey": f"{hk}...{hashlib.md5(hk.encode()).hexdigest()[:6]}",
            "name": val["name"],
            "specialty": val["specialty"],
            "stake_tao": stake,
            "vtrust": vtrust,
            "checks_passed": checks_passed,
            "checks_total": len(check_labels),
            "check_details": checks,
            "consensus": "Approved" if checks_passed >= 2 else "Disputed",
        })

    return validators


def run_demo_scenario(scenario_key: str) -> dict:
    """Run one of the 3 pre-built demo scenarios with full miner/validator output."""
    scenario = DEMO_SCENARIOS.get(scenario_key)
    if not scenario:
        return {"error": f"Unknown scenario: {scenario_key}"}

    task_type = scenario["task_type"]
    synapse = scenario["synapse"]
    ground_truth = scenario["ground_truth"]

    miner_responses = _generate_miner_responses(task_type, synapse, ground_truth, num_miners=6)
    validator_results = _generate_validator_results(task_type, num_validators=3)

    total_tao = round(random.Random(42).uniform(0.08, 0.42), 4)

    # Assign TAO to miners based on score
    total_score = sum(m["score"] for m in miner_responses)
    for m in miner_responses:
        m["tao_earned"] = round(total_tao * 0.41 * (m["score"] / total_score), 6) if total_score > 0 else 0

    return {
        "scenario": scenario_key,
        "title": scenario["title"],
        "subtitle": scenario["subtitle"],
        "task_type": task_type,
        "synapse": synapse,
        "ground_truth": ground_truth,
        "miner_responses": miner_responses,
        "miner_nodes_consulted": len(miner_responses),
        "validator_results": validator_results,
        "validator_nodes_consulted": len(validator_results),
        "tao_reward_pool": total_tao,
        "consensus_reached": all(v["consensus"] == "Approved" for v in validator_results),
        "block_number": random.randint(3_100_000, 3_500_000),
        "tempo": random.randint(8600, 8800),
        "timestamp": datetime.utcnow().isoformat(),
        "subnet_version": "1.0.0-beta",
    }


def get_demo_scenarios_list():
    """Return metadata for all 3 demo scenarios."""
    return [
        {
            "key": key,
            "title": s["title"],
            "subtitle": s["subtitle"],
            "task_type": s["task_type"],
            "patient_age": s["synapse"]["patient_context"]["age"],
            "patient_sex": s["synapse"]["patient_context"]["sex"],
            "chief_complaint": s["synapse"]["patient_context"]["chief_complaint"],
        }
        for key, s in DEMO_SCENARIOS.items()
    ]


# ============================================================
# LEGACY FUNCTIONS (used by Swagger API endpoints)
# ============================================================

def run_miner_prediction(synapse_dict: dict, tier: str) -> dict:
    """Simulate a miner processing a healthcare challenge (for Swagger endpoints)."""
    rng = random.Random(synapse_dict.get("random_seed", int(time.time())))

    task_type = synapse_dict.get("task_type", "symptom_diagnosis")
    patient = synapse_dict.get("patient_context", {})
    age = patient.get("age", 50)

    # Tier-based prediction quality
    if tier == "high":
        confidence = round(rng.uniform(0.82, 0.96), 2)
        latency = round(rng.uniform(200, 800), 0)
        risk_score = round(rng.uniform(0.25, 0.45), 2)
    elif tier == "mid":
        confidence = round(rng.uniform(0.65, 0.82), 2)
        latency = round(rng.uniform(500, 2000), 0)
        risk_score = round(rng.uniform(0.20, 0.55), 2)
    else:
        confidence = round(rng.uniform(0.40, 0.65), 2)
        latency = round(rng.uniform(1500, 4000), 0)
        risk_score = round(rng.uniform(0.15, 0.60), 2)

    # Generate diagnosis based on task type
    diagnoses = {
        "symptom_diagnosis": [
            ("Acute Coronary Syndrome", "I21.4"),
            ("Unstable Angina", "I20.0"),
            ("Pulmonary Embolism", "I26.99"),
            ("Community-Acquired Pneumonia", "J18.1"),
            ("Acute Pericarditis", "I30.9"),
        ],
        "medical_imaging": [
            ("Pneumonia - RLL", "J18.1"),
            ("Pleural Effusion", "J90"),
            ("Lung Nodule", "R91.1"),
            ("Cardiomegaly", "I51.7"),
            ("Normal Study", "Z87.09"),
        ],
        "risk_scoring": [
            ("High Readmission Risk", "I50.9"),
            ("Moderate Readmission Risk", "I50.9"),
            ("Cardiovascular Event Risk - High", "I25.10"),
            ("Acute Kidney Injury Risk", "N17.9"),
            ("Decompensated Heart Failure", "I50.21"),
        ],
    }

    dx_pool = diagnoses.get(task_type, diagnoses["symptom_diagnosis"])
    selected_dx = rng.choice(dx_pool)

    differentials = [
        f"{dx[0]} ({dx[1]})" for dx in rng.sample(dx_pool, min(3, len(dx_pool)))
    ]

    return {
        "miner_uid": 0,
        "miner_hotkey": "",
        "primary_diagnosis": selected_dx[0],
        "icd10_code": selected_dx[1],
        "differential_diagnoses": differentials,
        "risk_score": risk_score,
        "confidence": confidence,
        "clinical_reasoning": f"Analysis based on {task_type.replace('_', ' ')} evaluation of patient presentation.",
        "recommended_actions": ["Further evaluation recommended", "Follow-up in 48 hours"],
        "response_time_ms": latency,
    }


def score_prediction(prediction: dict, ground_truth: dict) -> dict:
    """Score a miner prediction against ground truth (for Swagger endpoints)."""
    rng = random.Random(hash(str(prediction.get("miner_hotkey", ""))) % 2**31)

    # Accuracy: does the primary diagnosis match?
    actual_dx = ground_truth.get("actual_diagnosis", "")
    predicted_dx = prediction.get("primary_diagnosis", "")
    actual_icd10 = ground_truth.get("actual_icd10", "")
    predicted_icd10 = prediction.get("icd10_code", "")

    # ICD-10 category match (first 3 chars)
    if predicted_icd10 and actual_icd10 and predicted_icd10[:3] == actual_icd10[:3]:
        accuracy = round(rng.uniform(0.85, 1.0), 4)
    elif predicted_dx and actual_dx and (predicted_dx.lower() in actual_dx.lower() or actual_dx.lower() in predicted_dx.lower()):
        accuracy = round(rng.uniform(0.70, 0.90), 4)
    else:
        accuracy = round(rng.uniform(0.30, 0.65), 4)

    # Differential quality
    differentials = prediction.get("differential_diagnoses", [])
    diff_has_actual = any(actual_icd10[:3] in d for d in differentials) if actual_icd10 and differentials else False
    differential_quality = round(rng.uniform(0.70, 0.95), 4) if diff_has_actual or len(differentials) >= 3 else round(rng.uniform(0.30, 0.65), 4)

    # Calibration
    confidence = prediction.get("confidence", 0.5)
    calibration = round(1.0 - abs(confidence - accuracy), 4)
    calibration = max(0, calibration)

    # Latency
    latency_ms = prediction.get("response_time_ms", 1000)
    latency_score = round(max(0, 1.0 - latency_ms / 10000), 4)

    # Consistency
    consistency = round(rng.uniform(0.65, 0.95), 4)

    # Critical finding bonus
    severity = ground_truth.get("severity", "moderate")
    is_critical = severity in ("emergent", "critical", "high_risk")
    critical_correct = is_critical and accuracy > 0.75
    critical_bonus = critical_correct

    final = 0.50 * accuracy + 0.15 * differential_quality + 0.15 * calibration + 0.10 * latency_score + 0.10 * consistency
    if critical_bonus:
        final *= 1.5
    final = round(min(1.0, final), 4)

    return {
        "accuracy": accuracy,
        "differential_quality": differential_quality,
        "calibration": calibration,
        "latency_score": latency_score,
        "consistency": consistency,
        "critical_finding_bonus": critical_bonus,
        "final_score": final,
    }


def get_diagnosis(query) -> dict:
    """Process a user-facing diagnosis query (for backward compatible /diagnose endpoint)."""
    synapse_dict = {
        "task_type": "symptom_diagnosis",
        "patient_context": {
            "age": 50,
            "sex": "unknown",
            "chief_complaint": query.symptoms if hasattr(query, "symptoms") else str(query),
            "medical_history": query.history if hasattr(query, "history") else "",
            "current_medications": "",
        },
    }

    result = run_miner_prediction(synapse_dict, "high")

    return {
        "diagnosis": result["primary_diagnosis"],
        "confidence": result["confidence"],
        "advice": f"Primary assessment: {result['primary_diagnosis']} ({result['icd10_code']}). "
                  f"Differentials considered: {', '.join(result.get('differential_diagnoses', [])[:3])}. "
                  f"Recommended: {', '.join(result.get('recommended_actions', []))}.",
    }
