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


# Import ECG specialist inference pipeline (PyTorch ResNet18)
from cnn_pipeline.ecg.ecg_inference import predict_ecg

# Import X-Ray specialist inference pipeline (TensorFlow DenseNet121 — Ojas's model)
from cnn_pipeline.xray.xray_inference import predict_xray

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
            "POST /api/predict/xray (Upload X-Ray image -> Raw CNN JSON)",
            "POST /api/debate (Submit CNN JSON -> LangGraph MDT Debate)",
            "POST /api/analyze/ecg (Upload ECG image -> End-to-End Prediction + Debate)",
            "POST /api/analyze/xray (Upload X-Ray image -> End-to-End Prediction + Debate)",
            "POST /api/analyze/multimodal (Upload ECG + X-Ray -> Combined Debate)"
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

@app.post("/api/predict/xray")
async def predict_xray_endpoint(file: UploadFile = File(...)):
    """
    Direct image upload endpoint for Chest X-Ray scans.
    Runs DenseNet121 model (Ojas's TensorFlow/Keras model) using .keras weights in RAM.
    Returns standardized JSON evidence packet with multi-label sigmoid probabilities.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image (JPG, PNG, etc.).")

    try:
        contents = await file.read()
        prediction_json = predict_xray(contents)
        return {
            "status": "success",
            "filename": file.filename,
            "data": prediction_json
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"X-Ray Prediction Error: {str(e)}")

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

@app.post("/api/analyze/xray")
async def analyze_xray_end_to_end(file: UploadFile = File(...), max_rounds: int = Form(2)):
    """
    Combined End-to-End Endpoint:
    Upload X-Ray Image -> DenseNet121 Inference -> LangGraph Debate -> Final Verdict.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        # Step 1: Run X-Ray CNN Model (TensorFlow DenseNet121)
        contents = await file.read()
        xray_json = predict_xray(contents)

        # Step 2: Seed LangGraph State
        initial_state = {
            "input_data": xray_json,
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
            "cnn_prediction": xray_json,
            "final_verdict": verdict_dict,
            "debate_transcript": final_state.get("debate_history", []),
            "retrieved_context": final_state.get("retrieved_context", "")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"End-to-End X-Ray Pipeline Error: {str(e)}")

@app.post("/api/analyze/multimodal")
async def analyze_multimodal(
    ecg_file: Optional[UploadFile] = File(None),
    xray_file: Optional[UploadFile] = File(None),
    max_rounds: int = Form(2)
):
    """
    Multimodal End-to-End Endpoint:
    Upload ECG + X-Ray images (both optional, at least one required).
    Runs whichever CNN models are needed, merges results,
    and feeds combined evidence into one LangGraph Debate session.
    """
    if ecg_file is None and xray_file is None:
        raise HTTPException(status_code=400, detail="At least one image must be uploaded (ecg_file or xray_file).")

    try:
        modality_results = {}

        # Run ECG CNN if ECG image was uploaded
        if ecg_file is not None:
            if not ecg_file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="ECG file must be an image.")
            ecg_contents = await ecg_file.read()
            ecg_json = predict_ecg(ecg_contents)
            modality_results["ecg"] = ecg_json

        # Run X-Ray CNN if X-Ray image was uploaded
        if xray_file is not None:
            if not xray_file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="X-Ray file must be an image.")
            xray_contents = await xray_file.read()
            xray_json = predict_xray(xray_contents)
            modality_results["xray"] = xray_json

        # Build combined input_data for the debate engine
        # If single modality, use that directly; if both, wrap them
        if len(modality_results) == 1:
            combined_input = list(modality_results.values())[0]
        else:
            # Multimodal: combine both CNN outputs into one merged evidence packet
            combined_input = {
                "modality": "multimodal",
                "input_type": "combined_images",
                "modalities": modality_results,
                "prediction": {
                    "primary_diagnosis": "See individual modality results",
                    "findings": []
                },
                "evidence": {
                    "positive_findings": [],
                    "negative_findings": []
                }
            }
            # Merge findings from all modalities
            for mod_key, mod_data in modality_results.items():
                # Add all findings to combined list
                if "prediction" in mod_data and "findings" in mod_data["prediction"]:
                    for finding in mod_data["prediction"]["findings"]:
                        finding_copy = dict(finding)
                        finding_copy["source_modality"] = mod_key
                        combined_input["prediction"]["findings"].append(finding_copy)
                # Merge evidence
                if "evidence" in mod_data:
                    for pf in mod_data["evidence"].get("positive_findings", []):
                        combined_input["evidence"]["positive_findings"].append(f"[{mod_key.upper()}] {pf}")
                    for nf in mod_data["evidence"].get("negative_findings", []):
                        combined_input["evidence"]["negative_findings"].append(f"[{mod_key.upper()}] {nf}")

        # Seed LangGraph State with combined input
        initial_state = {
            "input_data": combined_input,
            "current_round": 1,
            "max_rounds": max_rounds,
            "debate_history": [],
            "retrieved_context": None,
            "verdict": None
        }

        # Run MDT Debate Loop
        print(f"Executing Multimodal LangGraph MDT Debate (modalities: {list(modality_results.keys())})...")
        final_state = debate_app.invoke(initial_state)

        verdict_dict = None
        if final_state.get("verdict"):
            verdict_dict = final_state["verdict"].model_dump()

        return {
            "status": "success",
            "modalities_processed": list(modality_results.keys()),
            "individual_cnn_predictions": modality_results,
            "combined_input": combined_input,
            "final_verdict": verdict_dict,
            "debate_transcript": final_state.get("debate_history", []),
            "retrieved_context": final_state.get("retrieved_context", "")
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multimodal Pipeline Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
