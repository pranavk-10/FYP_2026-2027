import json
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from state import DiagnosticState, clean_response
from dotenv import load_dotenv

load_dotenv()

def skeptic_node(state: DiagnosticState):
    print("\n--- Skeptic ---")
    llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0.4, max_tokens=500)
    
    advocate_argument = state["debate_history"][-1] if state["debate_history"] else ""
    system_prompt = "You are the Skeptic. Review negative_findings and lower probabilities. Formulate an alternative differential diagnosis to challenge the Advocate. IMPORTANT: Output your response directly. DO NOT use <think> tags or output internal reasoning."
    human_prompt = f"Patient Data:\n{json.dumps(state['input_data'])}\n\nAdvocate: {advocate_argument}"
    
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
    clean_text = clean_response(response.content)
    
    print(clean_text)
    return {"debate_history": [f"Skeptic: {clean_text}"]}