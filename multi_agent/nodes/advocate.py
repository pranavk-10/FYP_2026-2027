import json
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from state import DiagnosticState, clean_response
from dotenv import load_dotenv

load_dotenv()

def advocate_node(state: DiagnosticState):
    print(f"\n--- Round {state['current_round']} : Advocate ---")
    
    llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0.2, max_tokens=500)
    
    system_prompt = "You are the Lead Diagnostician. State the most likely diagnosis based on the highest 'calibrated_probability' in the JSON, and justify it concisely. IMPORTANT: Output your response directly. DO NOT use <think> tags or output internal reasoning."
    human_prompt = f"Patient Data:\n{json.dumps(state['input_data'], indent=2)}"
    
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
    clean_text = clean_response(response.content)
    
    print(clean_text)
    return {"debate_history": [f"Advocate: {clean_text}"]}