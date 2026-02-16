from fastapi import APIRouter
from .models import PatientQuery, DiagnosisResponse
from .ai import get_diagnosis

router = APIRouter()

@router.post("/diagnose")
def diagnose(query: PatientQuery):
    result = get_diagnosis(query)
    return DiagnosisResponse(**result)
