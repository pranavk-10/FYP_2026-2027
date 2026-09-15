import json
from graph import app

if __name__ == "__main__":
    sample_cnn_input = {
        "patient_id": "P001",
        "modality": "chest_xray",
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
            "heatmap": "heatmaps/P001_xray.png",
            "regions": [
                {"location": "right_lower_lung", "importance": 0.82}
            ]
        }
    }

    initial_state = {
        "input_data": sample_cnn_input,
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
    print(final_state["verdict"].model_dump_json(indent=2))