from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser
from state import DiagnosticState, ModeratorVerdict, clean_response

def moderator_node(state: DiagnosticState):
    print("\n--- Moderator ---")
    llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0, max_tokens=800)
    
    parser = PydanticOutputParser(pydantic_object=ModeratorVerdict)
    current_round, max_rounds = state["current_round"], state["max_rounds"]
    
    system_prompt = f"""
    You are the Clinical Moderator. Evaluate the transcript. Round {current_round}/{max_rounds}. 
    Set action to 'finalize' if consensus exists or max rounds reached, otherwise 'continue_debate'.
    
    IMPORTANT: Output ONLY valid JSON. DO NOT use <think> tags or output internal reasoning.
    {parser.get_format_instructions()}
    """
    
    human_prompt = f"Transcript: {' | '.join(state['debate_history'])}"
    
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
    clean_text = clean_response(response.content)
    
    try:
        verdict = parser.invoke(clean_text)
    except Exception as e:
        print(f"Failed to parse JSON: {e}")
        print(f"Raw model output was: {clean_text}") 
        verdict = ModeratorVerdict(
            final_verdict="Error parsing verdict",
            confidence_score=0.0,
            audit_trail="Parsing failed.",
            action="finalize"
        )
        
    print(f"Action: {verdict.action} | Verdict: {verdict.final_verdict}")
    return {"verdict": verdict, "current_round": current_round + 1}