from langchain_core.messages import SystemMessage, HumanMessage
from state import ModeratorVerdict, format_patient_summary
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Enforce structured output matching the ModeratorVerdict Pydantic schema
llm = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0, max_tokens=1024, max_retries=3).with_structured_output(ModeratorVerdict)

def moderator_node(state):
    current_round = state["current_round"]
    max_rounds = state["max_rounds"]
    patient_summary = format_patient_summary(state.get("input_data", {}))
    
    system_prompt = f"""
    You are the Senior Clinical Moderator chairing a Multidisciplinary Medical Team.
    Evaluate the debate transcript and patient objective data across modalities.
    Current Round: {current_round}/{max_rounds}.
    
    CLINICAL DECISION RULES:
    1. Weigh objective findings by evidence strength:
       - High calibrated probability (>0.70) backed by characteristic electrophysiological, radiological, or dermatological signs is strong positive evidence.
       - A normal imaging study in one modality in the presence of acute pathology in another helps narrow the differential by ruling out gross anatomical mimics.
       - For dermatological presentations: high confidence benign findings support observation, whereas malignant or dysplastic risk markers mandate biopsy / specialist dermatology referral.
    2. If the Advocate's diagnosis is overwhelmingly supported by the data and the Skeptic's counter-arguments are theoretical without clinical corroboration, declare the validated primary diagnosis as final_verdict and set action to 'finalize'.
    3. If there is genuine ambiguity, conflicting high-probability findings across modalities, or no diagnostic consensus at round {max_rounds}, set final_verdict to 'Refer to Specialist / Inconclusive' and action to 'finalize'.
    4. If further deliberation is required and current_round < max_rounds, set action to 'continue_debate'.
    5. Provide an articulate audit_trail explaining the clinical rationale, which arguments prevailed, and recommended next steps.
    """
    
    human_prompt = f"""
    Patient Clinical Summary:
    {patient_summary}
    
    Debate History:
    {chr(10).join(state['debate_history'])}
    """
    
    verdict = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
    
    # Increment round counter and store the verdict
    return {
        "verdict": verdict,
        "current_round": current_round + 1
    }