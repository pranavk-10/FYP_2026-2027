import json
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from state import DiagnosticState, clean_response, format_patient_summary
from dotenv import load_dotenv

load_dotenv()

def advocate_node(state: DiagnosticState):
    current_round = state.get("current_round", 1)
    print(f"\n--- Round {current_round} : Advocate ---")
    
    llm = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0.2, max_tokens=600)
    
    debate_history = state.get("debate_history", [])
    patient_summary = format_patient_summary(state.get("input_data", {}))
    
    if current_round == 1 or not debate_history:
        # Initial Round: formulate primary diagnostic hypothesis
        system_prompt = (
            "You are the Lead Diagnostician (Advocate) in a Multidisciplinary Medical Team. "
            "Analyze the patient data and state the most likely primary diagnosis based on high-confidence findings. "
            "If an ECG shows high probability (>0.5), focus on the cardiac pathology. "
            "If an X-Ray shows 'No Finding' as highest but contains watchlist items (e.g., atelectasis, effusion), "
            "state that the study is predominantly normal but note relevant secondary signals for clinical correlation. "
            "Be clinically precise, concise, and justify your position with specific evidence from the data."
        )
        human_prompt = f"Patient Clinical Summary:\n{patient_summary}"
    else:
        # Multi-turn Round: directly counter the Skeptic and integrate RAG evidence
        prior_context = "\n".join(debate_history[-3:])
        system_prompt = (
            "You are the Lead Diagnostician (Advocate). "
            "Review the Skeptic's challenges and the Evidence-Checker's findings from the previous round. "
            "Directly address the Skeptic's counter-arguments: defend why your primary diagnosis remains most compelling, "
            "or explain why the Skeptic's alternatives are less likely based on the available data. "
            "Do not merely repeat your previous statement; advance the clinical argument."
        )
        human_prompt = (
            f"Patient Clinical Summary:\n{patient_summary}\n\n"
            f"Recent Debate History:\n{prior_context}"
        )
    
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
    clean_text = clean_response(response.content)
    
    print(clean_text)
    return {"debate_history": [f"Advocate: {clean_text}"]}