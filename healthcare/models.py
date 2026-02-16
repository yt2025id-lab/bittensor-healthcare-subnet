from pydantic import BaseModel

class PatientQuery(BaseModel):
    patient_id: str
    symptoms: str
    history: str

class DiagnosisResponse(BaseModel):
    diagnosis: str
    confidence: float
    advice: str
