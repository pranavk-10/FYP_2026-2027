import os
import sys
import json

# Ensure parent directory is in path so we can import cnn_pipeline
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from graph import app
from cnn_pipeline.ecg.ecg_inference import predict_ecg

# =====================================================================
# ORIGINAL DUMMY TEST INPUT (PRESERVED AS COMMENTED REFERENCE)
# =====================================================================
# sample_cnn_input = {
#     "patient_id": "P001",
#     "modality": "chest_xray",
#     "prediction": {
#         "findings": [
#             {"label": "Pneumonia", "calibrated_probability": 0.83},
#             {"label": "Cardiomegaly", "calibrated_probability": 0.29},
#             {"label": "Pleural Effusion", "calibrated_probability": 0.10}
#         ]
#     },
#     "evidence": {
#         "positive_findings": ["Right lower lung opacity"],
#         "negative_findings": ["No strong evidence of pleural effusion"]
#     },
#     "explainability": {
#         "method": "Grad-CAM",
#         "heatmap": "heatmaps/P001_xray.png",
#         "regions": [
#             {"location": "right_lower_lung", "importance": 0.82}
#         ]
#     }
# }

if __name__ == "__main__":
    # If an image path argument is passed (e.g. python main.py <path_to_image>), run predict_ecg first!
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        image_path = sys.argv[1]
        print(f"Running ResNet18 ECG Model on image: {image_path}")
        input_data = predict_ecg(image_path)
    else:
        # Default sample X-Ray dictionary fallback for quick testing
        input_data = {
            "patient_id": "P001",
            "modality": "xray",
            "prediction": {
                "findings": [
                    {"label": "Pneumonia", "calibrated_probability": 0.83},
                    {"label": "Cardiomegaly", "calibrated_probability": 0.29},
                    {"label": "Pleural Effusion", "calibrated_probability": 0.10}
                ]
            },
            "evidence": {
                "positive_findings": ["Right lower lung opacity"],
                "negative_findings": ["No strong evidence of pleural effusion"]
            },
            "explainability": {
                "method": "Grad-CAM",
                "regions": [{"location": "right_lower_lung", "importance": 0.82}]
            }
        }

    initial_state = {
        "input_data": input_data,
        "current_round": 1,
        "max_rounds": 2,
        "debate_history": [],
        "retrieved_context": None,
        "verdict": None
    }

    print("Starting Modular Multi-Agent MDT Debate...")
    
    # Invoke the compiled LangGraph application
    final_state = app.invoke(initial_state)
    
    print("\n==================================")
    print("FINAL MODERATOR VERDICT JSON:")
    if final_state.get("verdict"):
        print(final_state["verdict"].model_dump_json(indent=2))