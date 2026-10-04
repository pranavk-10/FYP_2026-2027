import json
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from state import DiagnosticState, clean_response, format_patient_summary
from dotenv import load_dotenv

load_dotenv()

def skeptic_node(state: DiagnosticState):
    print("\n--- Skeptic ---")
    llm = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0.3, max_tokens=600)
    
    debate_history = state.get("debate_history", [])
    advocate_argument = debate_history[-1] if debate_history else ""
    patient_summary = format_patient_summary(state.get("input_data", {}))
    
    system_prompt = (
        "You are the Skeptic / Challenger in a Multidisciplinary Medical Team. "
        "Your role is to rigorously challenge the Advocate's primary diagnosis and prevent diagnostic anchoring. "
        "1. Examine lower probabilities, watchlist conditions, and negative findings that could point to an alternative diagnosis. "
        "2. Formulate the TOP 2-3 most plausible alternative differential diagnoses in a concise Markdown table "
        "   (Condition | Plausibility / Mechanism | Key Clues from Data | Relative Risk vs Primary). "
        "3. Provide a brief 1-2 sentence concluding challenge to the Advocate."
    )
    human_prompt = f"Patient Clinical Summary:\n{patient_summary}\n\nAdvocate Argument:\n{advocate_argument}"
    
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
    clean_text = clean_response(response.content)
    
    print(clean_text)
    return {"debate_history": [f"Skeptic: {clean_text}"]}