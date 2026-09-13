import sys
sys.path.append("/workspace/models")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from analyze_drug_disease import analyze
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Pharmacovigilance Advisory API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    drug: str
    disease: str


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/analyze")
def analyze_drug_disease(request: AnalyzeRequest):
    if not request.drug.strip() or not request.disease.strip():
        raise HTTPException(status_code=400, detail="Both drug and disease fields are required.")

    try:
        result = analyze(request.drug, request.disease)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
