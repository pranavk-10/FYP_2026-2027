import os
import time
import json
import operator
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv

import re

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.output_parsers import PydanticOutputParser

# Load API keys
load_dotenv()

def clean_response(text: str) -> str:
    """Removes <think>...</think> blocks from the model output."""
    cleaned = re.sub(r'<think>.*?(?:</think>|$)', '', text, flags=re.DOTALL)
    return cleaned.strip()

# ==========================================
# 1. STATE & SCHEMAS
# ==========================================
class ModeratorVerdict(BaseModel):
    final_verdict: str = Field(description="The validated primary diagnosis or 'Refer to Specialist / Inconclusive'")
    confidence_score: float = Field(description="Final calibrated confidence score between 0.0 and 1.0")
    audit_trail: str = Field(description="Explanation of why this diagnosis was chosen")
    action: str = Field(description="'finalize' if consensus/max rounds reached, else 'continue_debate'")

class DiagnosticState(TypedDict):
    input_data: Dict[str, Any]
    current_round: int
    max_rounds: int
    debate_history: Annotated[List[str], operator.add]
    retrieved_context: Optional[str]
    verdict: Optional[ModeratorVerdict]

# ==========================================
# 2. NODES
# ==========================================
def advocate_node(state: DiagnosticState):
    print(f"\n--- Round {state['current_round']} : Advocate ---")
    # Switched to Llama 3.1 8B to avoid mandatory <think> blocks and token limits
    llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0.2, max_tokens=500)
    
    system_prompt = "You are the Lead Diagnostician. State the most likely diagnosis based on the highest 'calibrated_probability' in the JSON, and justify it concisely."
    human_prompt = f"Patient Data:\n{json.dumps(state['input_data'], indent=2)}"
    
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
    clean_text = clean_response(response.content)
    print(clean_text)
    return {"debate_history": [f"Advocate: {clean_text}"]}

def skeptic_node(state: DiagnosticState):
    print("\n--- Skeptic ---")
    llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0.4, max_tokens=500)
    
    advocate_argument = state["debate_history"][-1] if state["debate_history"] else ""
    system_prompt = "You are the Skeptic. Review negative_findings and lower probabilities. Formulate an alternative differential diagnosis to challenge the Advocate."
    human_prompt = f"Patient Data:\n{json.dumps(state['input_data'])}\n\nAdvocate: {advocate_argument}"
    
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
    clean_text = clean_response(response.content)
    print(clean_text)
    return {"debate_history": [f"Skeptic: {clean_text}"]}

def evidence_checker_node(state: DiagnosticState):
    print("\n--- Evidence-Checker (RAG) ---")
    llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0, max_tokens=500)
    latest_arguments = "\n".join(state["debate_history"][-2:])
    
    try:
        embeddings = MistralAIEmbeddings(model="mistral-embed")
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        vector_store = PineconeVectorStore(index=pc.Index("medical-knowledge-base"), embedding=embeddings)
        docs = vector_store.as_retriever(search_kwargs={"k": 2}).invoke(latest_arguments)
        retrieved_context = "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        print(f"(Vector DB skipped/error: {e})")
        retrieved_context = "No retrieved evidence available for testing."

    system_prompt = f"Cross-reference claims against this context: <context> {retrieved_context} </context> State which claims are supported or lack evidence."
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=latest_arguments)])
    clean_text = clean_response(response.content)
    print(clean_text)
    return {"debate_history": [f"Evidence-Checker: {clean_text}"], "retrieved_context": retrieved_context}

def moderator_node(state: DiagnosticState):
    print("\n--- Moderator ---")
    llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0, max_tokens=1000)
    
    parser = PydanticOutputParser(pydantic_object=ModeratorVerdict)
    current_round, max_rounds = state["current_round"], state["max_rounds"]
    
    system_prompt = f"""
    You are the Clinical Moderator. Evaluate the transcript. Round {current_round}/{max_rounds}. 
    Set action to 'finalize' if consensus exists or max rounds reached, otherwise 'continue_debate'.
    
    {parser.get_format_instructions()}
    """
    
    human_prompt = f"Transcript: {' | '.join(state['debate_history'])}"
    
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
    clean_text = clean_response(response.content)
    
    try:
        # FIX: Ensure we parse the clean_text string, not the raw response object
        verdict = parser.invoke(clean_text)
    except Exception as e:
        print(f"Failed to parse JSON: {e}")
        # Log the raw text to see why it failed
        print(f"Raw model output was: {clean_text}") 
        verdict = ModeratorVerdict(
            final_verdict="Error parsing verdict",
            confidence_score=0.0,
            audit_trail="Parsing failed.",
            action="finalize"
        )
        
    print(f"Action: {verdict.action} | Verdict: {verdict.final_verdict}")
    return {"verdict": verdict, "current_round": current_round + 1}
# ==========================================
# 3. GRAPH ROUTING & COMPILATION
# ==========================================
def route_debate(state: DiagnosticState):
    if state.get("verdict").action == "finalize":
        return END
    return "advocate"

workflow = StateGraph(DiagnosticState)
workflow.add_node("advocate", advocate_node)
workflow.add_node("skeptic", skeptic_node)
workflow.add_node("evidence", evidence_checker_node)
workflow.add_node("moderator", moderator_node)

workflow.add_edge(START, "advocate")
workflow.add_edge("advocate", "skeptic")
workflow.add_edge("skeptic", "evidence")
workflow.add_edge("evidence", "moderator")
workflow.add_conditional_edges("moderator", route_debate, {END: END, "advocate": "advocate"})

app = workflow.compile()

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    sample_cnn_input = {
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
            "regions": [{"location": "right_lower_lung", "importance": 0.82}]
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

    print("Starting Multi-Agent MDT Debate...")
    # Invoke the graph
    final_state = app.invoke(initial_state)
    
    print("\n==================================")
    print("FINAL MODERATOR VERDICT JSON:")
    print(final_state["verdict"].model_dump_json(indent=2))