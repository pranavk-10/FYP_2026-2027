import os
import sys
import io
import json
from typing import Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure root workspace directory and multi_agent directory are in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MULTI_AGENT_DIR = os.path.join(BASE_DIR, "multi_agent")

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if MULTI_AGENT_DIR not in sys.path:
    sys.path.insert(0, MULTI_AGENT_DIR)


# Import ECG specialist inference pipeline
from cnn_pipeline.ecg.ecg_inference import predict_ecg

# Import compiled LangGraph MDT Debate graph
from multi_agent.graph import app as debate_app

# Initialize FastAPI Application
app = FastAPI(
    title="Multimodal Diagnostic Assistant API",
    description="FastAPI Backend combining Specialist CNN Vision Models with a RAG-Grounded LangGraph MDT Debate Engine.",
    version="1.0.0"
)

# Enable CORS for React/Vite Frontend Dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# SCHEMAS
# ==========================================
class DebateRequestPayload(BaseModel):
    input_data: Dict[str, Any]
    max_rounds: Optional[int] = 2

# ==========================================
# ENDPOINTS
# ==========================================

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Multimodal Diagnostic Assistant API",
        "endpoints": [
            "POST /api/predict/ecg (Upload ECG image -> Raw CNN JSON)",
            "POST /api/debate (Submit CNN JSON -> LangGraph MDT Debate)",
            "POST /api/analyze/ecg (Upload ECG image -> End-to-End Prediction + Debate)"
        ]
    }

@app.post("/api/predict/ecg")
async def predict_ecg_endpoint(file: UploadFile = File(...)):
    """
    Direct image upload endpoint for ECG printouts/scans.
    Runs ResNet18 model using trained weights (model_ecg_image_best.pth) in RAM.
    Returns standardized JSON evidence packet.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image (JPG, PNG, etc.).")

    try:
        contents = await file.read()
        prediction_json = predict_ecg(contents)
        return {
            "status": "success",
            "filename": file.filename,
            "data": prediction_json
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ECG Prediction Error: {str(e)}")

@app.post("/api/debate")
def run_debate_endpoint(payload: DebateRequestPayload):
    """
    Takes a standardized CNN JSON evidence packet, passes it into DiagnosticState,
    and runs the cyclic LangGraph MDT Debate engine.
    Returns final Moderator verdict and transcript.
    """
    try:
        initial_state = {
            "input_data": payload.input_data,
            "current_round": 1,
            "max_rounds": payload.max_rounds,
            "debate_history": [],
            "retrieved_context": None,
            "verdict": None
        }

        print(f"Executing LangGraph MDT Debate via API...")
        final_state = debate_app.invoke(initial_state)

        verdict_dict = None
        if final_state.get("verdict"):
            verdict_dict = final_state["verdict"].model_dump()

        return {
            "status": "success",
            "final_verdict": verdict_dict,
            "debate_transcript": final_state.get("debate_history", []),
            "retrieved_context": final_state.get("retrieved_context", "")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debate Engine Error: {str(e)}")

@app.post("/api/analyze/ecg")
async def analyze_ecg_end_to_end(file: UploadFile = File(...), max_rounds: int = Form(2)):
    """
    Combined End-to-End Endpoint:
    Upload ECG Image -> ResNet18 Inference -> LangGraph Debate -> Final Verdict.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        # Step 1: Run ECG CNN Model
        contents = await file.read()
        ecg_json = predict_ecg(contents)

        # Step 2: Seed LangGraph State
        initial_state = {
            "input_data": ecg_json,
            "current_round": 1,
            "max_rounds": max_rounds,
            "debate_history": [],
            "retrieved_context": None,
            "verdict": None
        }

        # Step 3: Run MDT Debate Loop
        final_state = debate_app.invoke(initial_state)

        verdict_dict = None
        if final_state.get("verdict"):
            verdict_dict = final_state["verdict"].model_dump()

        return {
            "status": "success",
            "filename": file.filename,
            "cnn_prediction": ecg_json,
            "final_verdict": verdict_dict,
            "debate_transcript": final_state.get("debate_history", []),
            "retrieved_context": final_state.get("retrieved_context", "")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"End-to-End ECG Pipeline Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
