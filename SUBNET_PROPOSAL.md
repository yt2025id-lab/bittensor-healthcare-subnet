# Decentralized AI Healthcare — Subnet Design Proposal

> **Bittensor Subnet Ideathon 2026**
> Team: Decentralized AI Healthcare | Twitter: @Ozan_OnChain | Discord: ozan_onchain

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Incentive & Mechanism Design](#2-incentive--mechanism-design)
3. [Miner Design](#3-miner-design)
4. [Validator Design](#4-validator-design)
5. [Business Logic & Market Rationale](#5-business-logic--market-rationale)
6. [Go-To-Market Strategy](#6-go-to-market-strategy)

---

## 1. Executive Summary

**Decentralized AI Healthcare** is a Bittensor subnet that creates a competitive marketplace for medical AI diagnostic models. Miners submit and continuously improve clinical prediction models (e.g., disease classification from symptoms, medical image analysis). Validators evaluate these models against curated medical datasets with known ground truth. The best-performing models earn $TAO emissions, creating a permissionless, privacy-preserving ecosystem that produces state-of-the-art healthcare AI — accessible to anyone.

**Digital Commodity Produced:** High-accuracy medical diagnostic AI models.

**Proof of Intelligence:** Every miner must demonstrate genuine medical AI capability by producing accurate diagnoses on randomized, validator-generated clinical challenges. There is no shortcut — the only way to earn rewards is to build better models.

---

## 2. Incentive & Mechanism Design

### 2.1 Emission and Reward Logic

The subnet uses Bittensor's native emission system with the following distribution per tempo (~360 blocks, ~72 minutes):

| Recipient | Share | Description |
|-----------|-------|-------------|
| Subnet Owner | 18% | Funds ongoing development, dataset curation, and security audits |
| Miners | 41% | Distributed proportionally to Yuma Consensus performance scores |
| Validators + Stakers | 41% | Proportional to stake and bond strength |

**Reward Flow:**

```
Block Emission ($TAO)
    └─> Subnet AMM (alpha token injection)
        └─> Tempo Distribution (every ~72 min)
            ├─> 18% → Subnet Owner
            ├─> 41% → Miners (via Yuma Consensus scores)
            └─> 41% → Validators & Stakers (via bond strength)
```

Miner emissions are determined by their **normalized weight** in the Yuma Consensus output. A miner scoring 0.95 accuracy on clinical tasks consistently will earn proportionally more than one scoring 0.70.

### 2.2 Incentive Alignment

**For Miners:**
- Higher diagnostic accuracy = higher weight from validators = more $TAO emissions.
- Multi-dimensional scoring (accuracy, speed, consistency) ensures miners must genuinely optimize, not just game one metric.
- Continuous model improvement is rewarded — there is no score cap.

**For Validators:**
- Validators earn emissions proportional to their **bond strength**, which grows when their weights align with Yuma Consensus.
- Validators who independently and honestly evaluate miners build stronger EMA bonds over time.
- The commit-reveal mechanism prevents weight copying — lazy validators who copy others get stale data and weaker bonds.

**For Stakers:**
- Staking $TAO into the subnet signals confidence in its value, directing more emissions to it (via Taoflow).
- Stakers earn a share of the 41% validator emissions proportional to their stake.

### 2.3 Mechanisms to Discourage Low-Quality or Adversarial Behavior

| Threat | Defense Mechanism |
|--------|-------------------|
| **Miners submitting random outputs** | Multi-dimensional scoring with ground-truth comparison; random outputs score near 0 |
| **Miners caching/memorizing answers** | Validators use randomized synthetic challenges with unique seeds per query; challenges are never repeated |
| **Miners proxying to external APIs** | Strict response time limits (e.g., 10s timeout); latency penalty in scoring |
| **Colluding validators inflating a miner** | Yuma Consensus clipping — outlier weights are clipped to stake-weighted median |
| **Weight-copying validators** | Commit-reveal mechanism (weights encrypted for 1+ tempos); Consensus-Based Weights penalize copiers with slower bond growth |
| **Model stagnation (same model forever)** | Anti-monopoly decay: after 30 consecutive tempos as top miner, reward share gradually decreases by 2% per tempo, forcing continuous improvement |
| **Sybil attacks (multiple miner identities)** | Registration burn cost + immunity period; each UID requires TAO burn |
| **Data poisoning** | Validators use curated, verified medical datasets as ground truth; miner outputs are compared against known-correct labels |

### 2.4 Proof of Intelligence

This subnet qualifies as a genuine **Proof of Intelligence** because:

1. **Non-trivial computation:** Training medical AI models requires genuine GPU-intensive work (fine-tuning transformers, CNNs on medical data).
2. **Verifiable output quality:** Validator-generated challenges have deterministic ground truth — there is no way to fake accuracy.
3. **Continuous improvement pressure:** The competitive landscape means miners must constantly retrain and improve models.
4. **Domain expertise required:** Medical AI requires understanding of clinical data, preprocessing, and model architecture choices.

The system produces a measurable, useful intelligence artifact: **accurate medical diagnostic models**.

### 2.5 High-Level Algorithm

```
EVERY TEMPO (~72 minutes):

  VALIDATOR LOOP:
    1. GENERATE synthetic clinical challenges:
       - Sample from curated medical datasets (symptoms → diagnosis)
       - Apply random perturbations (noise, missing fields, varied phrasing)
       - Record ground truth labels

    2. DISPATCH challenges to all registered miners:
       - Create HealthcareSynapse with patient symptoms, history, metadata
       - Send via dendrite to each miner's axon
       - Set timeout = 10 seconds

    3. COLLECT miner responses:
       - Each response contains: diagnosis, confidence, reasoning

    4. SCORE each miner response:
       - accuracy_score = compare(prediction, ground_truth)  [0.0 - 1.0]
       - latency_score = max(1 - elapsed/timeout, 0)         [0.0 - 1.0]
       - consistency_score = EMA of accuracy over last N rounds
       - confidence_calibration = 1 - |confidence - actual_accuracy|

       - final_score = 0.50 * accuracy_score
                     + 0.15 * latency_score
                     + 0.20 * consistency_score
                     + 0.15 * confidence_calibration

    5. UPDATE moving averages:
       - scores[uid] = 0.9 * scores[uid] + 0.1 * final_score

    6. SUBMIT weights to blockchain:
       - Normalize scores to weight vector
       - Commit encrypted weights (commit-reveal)
       - Reveal previous tempo's weights

  MINER LOOP:
    1. RECEIVE HealthcareSynapse from validator
    2. RUN input through local diagnostic model
    3. RETURN DiagnosisResponse with prediction, confidence, reasoning
    4. CONTINUOUSLY retrain model to improve accuracy

  YUMA CONSENSUS (on-chain):
    1. Collect all validator weight vectors
    2. Stake-weight and clip outliers
    3. Compute miner rankings → emission allocation
    4. Update validator bonds (EMA)
    5. Distribute $TAO emissions
```

---

## 3. Miner Design

### 3.1 Miner Tasks

Miners operate medical AI diagnostic models. Their primary task is to **receive clinical challenge synapses from validators and return accurate diagnoses**.

**Task Types (Multiple Incentive Mechanisms):**

| Mechanism | Weight | Description |
|-----------|--------|-------------|
| **Symptom-to-Diagnosis** | 60% | Given patient symptoms and history, predict the most likely diagnosis |
| **Medical Image Classification** | 30% | Given a medical image (X-ray, dermatology photo), classify the condition |
| **Risk Scoring** | 10% | Given patient data, predict risk scores for specific conditions |

### 3.2 Input → Output Format (Synapse Protocol)

```python
class HealthcareSynapse(bt.Synapse):
    """Data contract between validators and miners."""

    # ── Immutable Inputs (set by validator) ──
    task_type: str                        # "symptom_diagnosis" | "image_classification" | "risk_scoring"
    patient_symptoms: str                 # Free-text symptom description
    patient_history: str                  # Medical history summary
    patient_metadata: dict                # Age, sex, vitals (anonymized)
    image_data: Optional[str] = None      # Base64-encoded medical image (for image tasks)
    random_seed: int                      # Unique seed to prevent caching

    # ── Mutable Outputs (filled by miner) ──
    diagnosis: Optional[str] = None                   # Predicted diagnosis (ICD-10 code + name)
    confidence: Optional[float] = None                # Model confidence [0.0 - 1.0]
    differential_diagnoses: Optional[List[dict]] = None  # Top-5 alternatives with probabilities
    reasoning: Optional[str] = None                   # Brief clinical reasoning
    risk_scores: Optional[dict] = None                # For risk_scoring tasks
```

**Example Input (Validator → Miner):**
```json
{
  "task_type": "symptom_diagnosis",
  "patient_symptoms": "persistent cough for 3 weeks, fever 38.5°C, shortness of breath, fatigue",
  "patient_history": "45-year-old male, smoker (20 pack-years), no prior lung disease",
  "patient_metadata": {"age": 45, "sex": "M", "bmi": 26.1, "bp": "130/85"},
  "random_seed": 48291037
}
```

**Example Output (Miner → Validator):**
```json
{
  "diagnosis": "J18.9 - Community-acquired pneumonia",
  "confidence": 0.82,
  "differential_diagnoses": [
    {"diagnosis": "J18.9 - Pneumonia, unspecified", "probability": 0.82},
    {"diagnosis": "J44.1 - COPD with acute exacerbation", "probability": 0.10},
    {"diagnosis": "C34.9 - Lung cancer, unspecified", "probability": 0.05},
    {"diagnosis": "J06.9 - Upper respiratory infection", "probability": 0.02},
    {"diagnosis": "A15.0 - Pulmonary tuberculosis", "probability": 0.01}
  ],
  "reasoning": "Persistent productive cough with fever and dyspnea in a long-term smoker suggests pneumonia as primary diagnosis. Smoking history warrants monitoring for COPD exacerbation and lung malignancy."
}
```

### 3.3 Performance Dimensions

| Dimension | Weight | Metric | Description |
|-----------|--------|--------|-------------|
| **Diagnostic Accuracy** | 50% | Top-1 match with ground truth | Primary diagnosis matches the correct ICD-10 code |
| **Differential Quality** | 15% | Ground truth in top-5 predictions | Correct diagnosis appears in the differential list |
| **Confidence Calibration** | 15% | `1 - abs(confidence - actual_accuracy)` | Confidence score should reflect true accuracy |
| **Response Latency** | 10% | `max(1 - elapsed/timeout, 0)` | Faster responses score higher |
| **Consistency** | 10% | EMA of accuracy over last 100 rounds | Sustained performance over time |

### 3.4 Miner Hardware Requirements

| Tier | GPU | Expected Performance |
|------|-----|---------------------|
| Entry | RTX 3090 / A5000 (24GB) | Can run fine-tuned 7B parameter medical LLMs |
| Mid | A100 40GB | Can run 13B-30B medical models with image capabilities |
| High | A100 80GB / H100 | Can run 70B+ models, multi-modal (text + image) |

### 3.5 Recommended Miner Strategy

1. Fine-tune an open-source medical LLM (e.g., Med-PaLM, BioMistral, ClinicalBERT) on medical QA datasets.
2. Use ICD-10 structured output to standardize diagnosis predictions.
3. Train a separate image classification model (e.g., DenseNet, EfficientNet) for radiology/dermatology tasks.
4. Ensemble multiple models for higher accuracy.
5. Continuously retrain on new medical literature and datasets.

---

## 4. Validator Design

### 4.1 Scoring and Evaluation Methodology

Validators generate synthetic clinical challenges and score miner responses against known ground truth.

**Ground Truth Sources:**
- Curated medical QA datasets (MedQA, PubMedQA, MedMCQA)
- Synthetically generated patient cases using clinical ontologies (SNOMED CT, ICD-10)
- Medical image datasets with verified labels (CheXpert, ISIC, MIMIC-CXR)

**Scoring Algorithm:**

```python
def score_miner_response(synapse, response, ground_truth, elapsed_time):
    """Multi-dimensional scoring function."""

    # 1. Diagnostic Accuracy (50%)
    accuracy = 1.0 if response.diagnosis == ground_truth.diagnosis else 0.0

    # 2. Differential Quality (15%)
    differential = 0.0
    if response.differential_diagnoses:
        predicted_codes = [d["diagnosis"].split(" - ")[0] for d in response.differential_diagnoses]
        if ground_truth.diagnosis_code in predicted_codes:
            rank = predicted_codes.index(ground_truth.diagnosis_code)
            differential = 1.0 - (rank * 0.2)  # 1.0 for rank 0, 0.8 for rank 1, etc.

    # 3. Confidence Calibration (15%)
    # Over many rounds, confidence should approximate actual accuracy
    actual_accuracy = miner_accuracy_history[uid].mean()
    calibration = 1.0 - abs(response.confidence - actual_accuracy)

    # 4. Response Latency (10%)
    timeout = 10.0  # seconds
    latency = max(1.0 - elapsed_time / timeout, 0.0)

    # 5. Consistency (10%)
    consistency = miner_ema_scores[uid]  # EMA of past accuracy

    # Final weighted score
    score = (0.50 * accuracy +
             0.15 * differential +
             0.15 * calibration +
             0.10 * latency +
             0.10 * consistency)

    return score
```

**Synthetic Challenge Generation:**

```python
def generate_challenge():
    """Generate a synthetic clinical challenge with known ground truth."""

    # Sample from curated dataset
    case = random.choice(medical_dataset)

    # Apply perturbations to prevent memorization
    seed = random.randint(0, 2**32)
    symptoms = paraphrase(case["symptoms"], seed=seed)         # Rephrase symptoms
    history = add_noise(case["history"], seed=seed)             # Add irrelevant details
    metadata = perturb_vitals(case["metadata"], noise=0.05)    # Small numeric noise

    synapse = HealthcareSynapse(
        task_type=case["task_type"],
        patient_symptoms=symptoms,
        patient_history=history,
        patient_metadata=metadata,
        random_seed=seed
    )

    ground_truth = case["diagnosis"]  # Known correct answer

    return synapse, ground_truth
```

### 4.2 Evaluation Cadence

| Action | Frequency | Description |
|--------|-----------|-------------|
| Challenge dispatch | Every tempo (~72 min) | Send 1-3 synthetic challenges per miner per tempo |
| Score calculation | After each challenge | Immediate scoring upon response receipt |
| EMA update | After each challenge | `ema[uid] = 0.9 * ema[uid] + 0.1 * new_score` |
| Weight submission | Every `WeightsRateLimit` blocks (100) | Submit normalized weight vector to blockchain |
| Commit-reveal | 1 tempo delay | Weights encrypted for 1 tempo before reveal |
| Dataset rotation | Weekly | Rotate/expand ground truth datasets to prevent overfitting |

### 4.3 Validator Incentive Alignment

1. **Bond Growth:** Validators who independently evaluate miners and submit weights aligned with consensus build stronger EMA bonds → higher emissions.
2. **Commit-Reveal:** Weights are encrypted for 1+ tempos. Weight copiers can only access stale data, resulting in weaker bonds.
3. **Consensus-Based Weights:** Bond accrual rate is dynamically adjusted based on consensus alignment. Honest, independent validators compound their advantage.
4. **Minimum Stake:** Validators must hold sufficient stake to receive a validator permit (top 64 staked neurons).

**Validator Hardware Requirements:**
- CPU: 8+ cores
- RAM: 32GB+
- Storage: 500GB+ (for medical datasets)
- GPU: Optional (for running verification models)
- Network: Stable connection, low latency

---

## 5. Business Logic & Market Rationale

### 5.1 The Problem and Why It Matters

**Healthcare AI is broken by centralization:**

- **Data Silos:** Hospital systems lock patient data behind proprietary walls. An estimated 97% of hospital-generated data goes unused.
- **Privacy Barriers:** HIPAA, GDPR, and similar regulations make cross-institutional data sharing extremely difficult and expensive.
- **Monopolistic AI:** A handful of companies (Google Health, IBM Watson Health) control healthcare AI development, with high costs and limited access.
- **Lack of Trust:** Centralized AI models are black boxes with no audit trail for how diagnoses are generated.
- **No Incentives:** Institutions that contribute valuable training data receive nothing in return.

**Scale of the Problem:**
- Global healthcare AI market: $20.9B (2024) → projected $148.4B by 2029 (CAGR 45%).
- Medical errors cause ~250,000 deaths/year in the US alone.
- 50% of the world's population lacks access to essential health services.

### 5.2 Competing Solutions

**Within Bittensor:**

| Subnet | Focus | How We Differ |
|--------|-------|---------------|
| Safe Scan (SN76) | Skin cancer detection (single modality) | We cover multi-modal diagnosis: symptoms, images, risk scoring across all conditions |
| Healthi (SN34) | Clinical prediction tasks | We focus specifically on diagnostic accuracy with ICD-10 standardized output |
| bthealthcare | Medical image classification | We combine text-based diagnosis with imaging, not imaging alone |

**Outside Bittensor:**

| Solution | Limitation | Our Advantage |
|----------|-----------|---------------|
| Google Health AI | Centralized, proprietary, expensive | Decentralized, open, permissionless, $TAO-incentivized |
| IBM Watson Health | Shut down (2022); trust issues | Community-driven, transparent, continuously improving |
| Ada Health | Symptom checker only; no model marketplace | Full diagnostic pipeline + competitive model improvement |
| Hugging Face Medical Models | No quality verification or incentive layer | Bittensor's Yuma Consensus ensures only high-quality models earn rewards |

### 5.3 Why This Use Case Is Well-Suited to a Bittensor Subnet

1. **Clear digital commodity:** Medical diagnostic models are a well-defined, measurable output. Accuracy can be objectively scored.
2. **Competitive improvement:** Multiple miners competing to build the best diagnostic model drives continuous quality improvement — better than any single company could achieve alone.
3. **Privacy-preserving by design:** Miners train models locally; only model outputs (not patient data) traverse the network.
4. **Objective evaluation:** Medical diagnoses have ground truth (correct ICD-10 codes), enabling deterministic validator scoring.
5. **High-value application:** Healthcare AI commands premium pricing, supporting sustainable subnet economics.
6. **Natural network effects:** More miners = better model diversity; more validators = higher quality assurance; more users = more demand for emissions.

### 5.4 Path to Long-Term Adoption and Sustainable Business

**Phase 1 (Month 1-3): Foundation**
- Launch subnet on Bittensor testnet with symptom-to-diagnosis mechanism.
- Onboard initial miners with open-source medical models (BioMistral, ClinicalBERT).
- Curate initial ground-truth datasets (MedQA, PubMedQA).

**Phase 2 (Month 4-6): Expansion**
- Add medical image classification mechanism (radiology, dermatology).
- Launch consumer-facing API for third-party integrations.
- Partner with telemedicine platforms for real-world validation.

**Phase 3 (Month 7-12): Monetization**
- API marketplace: charge per-query fees for diagnostic predictions (paid in $TAO).
- Enterprise tier for hospitals and clinics.
- Revenue flows back to subnet (increasing emissions attractiveness).

**Phase 4 (Year 2+): Scale**
- Multi-language diagnostic support.
- Regulatory pathway for clinical decision support tool (FDA 510(k) for Class II).
- Integration with electronic health record (EHR) systems.
- Collaboration with other Bittensor subnets (Data Universe for medical data sourcing).

**Revenue Model:**
```
API Query Fees → $TAO → Subnet AMM Pool → Higher Alpha Price → More Emissions → More Miners → Better Models
```

This creates a **flywheel effect**: better models attract more users, more users generate more revenue, more revenue increases emissions, more emissions attract better miners.

---

## 6. Go-To-Market Strategy

### 6.1 Initial Target Users & Use Cases

**Primary (Early Adopters):**

| Segment | Use Case | Value Proposition |
|---------|----------|-------------------|
| **Telemedicine platforms** | AI-assisted triage and pre-diagnosis | Reduce doctor workload by 40%; faster patient routing |
| **Rural/underserved clinics** | Second-opinion diagnostic tool | Access to specialist-level AI without specialist costs |
| **Health tech startups** | Embed diagnostic API in their apps | Pay-per-query, no need to build AI in-house |

**Secondary:**

| Segment | Use Case | Value Proposition |
|---------|----------|-------------------|
| **Medical students** | Clinical reasoning training tool | Practice differential diagnosis with AI feedback |
| **Insurance companies** | Risk assessment and underwriting | AI-powered health risk scoring |
| **Pharmaceutical companies** | Patient stratification for clinical trials | Faster, more accurate patient selection |

### 6.2 Distribution & Growth Channels

**Developer Adoption:**
- Open-source SDK and API documentation on GitHub.
- Developer tutorials and integration guides.
- Listing on Bittensor subnet directories (SubnetAlpha, TaoStats).

**Community Building:**
- Active presence on Bittensor Discord and Twitter/X.
- Weekly development updates and subnet performance reports.
- Miner onboarding guides with recommended model architectures.

**Partnerships:**
- Telemedicine platforms (e.g., Teladoc, Doctor On Demand) for pilot integrations.
- Medical AI research groups for dataset curation and model validation.
- Healthcare accelerators and hackathons for visibility.

**Content Marketing:**
- Blog posts on decentralized healthcare AI.
- Case studies showing diagnostic accuracy improvements over time.
- Comparison benchmarks against centralized alternatives.

### 6.3 Incentives for Early Participation

**For Early Miners:**
- Lower competition in early tempos = higher per-miner emissions.
- Immunity period protection for new registrations.
- Detailed model training guides and recommended datasets provided by subnet owner.
- Community support channel for miner optimization.

**For Early Validators:**
- Early bond accumulation advantage (EMA bonds compound over time).
- Lower stake requirements when subnet is young.
- Direct communication channel with subnet development team.

**For Early Users/Stakers:**
- Alpha token price is lowest at subnet launch (early stakers benefit from price appreciation).
- Governance input on subnet development priorities.
- Early access to premium API features.

**Bootstrapping Strategy:**
1. **Week 1-2:** Subnet owner runs reference miner and validator to demonstrate the system works.
2. **Week 3-4:** Publish miner guide with pre-trained model weights; invite Bittensor community miners.
3. **Month 2:** Launch public API endpoint; invite health tech developers to test.
4. **Month 3:** First partnership pilot with a telemedicine platform.

---

## Appendix

### A. Subnet Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `MaxAllowedUids` | 256 | Sufficient competition without diluting rewards |
| `MaxAllowedValidators` | 64 | Standard Bittensor default |
| `ImmunityPeriod` | 5000 blocks | ~7 hours protection for new miners |
| `MinBurn` | 0.0005 TAO | Low barrier for miner registration |
| `WeightsRateLimit` | 100 blocks | ~20 min between weight updates |
| `CommitRevealPeriod` | 1 tempo | Encrypted for 1 tempo before reveal |
| `Tempo` | 360 blocks | ~72 min per evaluation cycle |

### B. Dataset Sources

| Dataset | Type | Size | Usage |
|---------|------|------|-------|
| MedQA (USMLE) | Symptom → Diagnosis QA | 12,723 questions | Primary ground truth for symptom diagnosis |
| PubMedQA | Biomedical QA | 1,000 expert-labeled | Supplementary clinical reasoning |
| MedMCQA | Medical MCQ | 194,000 questions | Training and evaluation |
| CheXpert | Chest X-ray | 224,316 images | Image classification ground truth |
| ISIC | Dermatology | 70,000+ images | Skin condition classification |
| MIMIC-CXR | Chest X-ray + Reports | 377,110 images | Multi-modal evaluation |

### C. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Low miner participation at launch | Low model diversity | Subnet owner runs reference miners; provide pre-trained model weights |
| Dataset bias | Inaccurate diagnoses for underrepresented populations | Actively curate diverse, balanced datasets; track demographic accuracy |
| Regulatory concerns | Cannot market as medical device without approval | Position as "clinical decision support" (not diagnostic); pursue FDA pathway in Phase 4 |
| Adversarial miners gaming the system | Reduced output quality | Multi-dimensional scoring, random challenges, anti-monopoly decay |
| Competing Bittensor subnets | Emission dilution | Differentiate through multi-modal approach and enterprise partnerships |

---

*This proposal was prepared for the Bittensor Subnet Ideathon 2026.*
*GitHub: https://github.com/yt2025id-lab/bittensor-healthcare-subnet*
