from langchain_core.messages import SystemMessage, HumanMessage
from langchain_mistralai import ChatMistralAI
from state import ModeratorVerdict
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Enforce structured output matching the ModeratorVerdict Pydantic schema
llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0).with_structured_output(ModeratorVerdict)

def moderator_node(state):
    current_round = state["current_round"]
    max_rounds = state["max_rounds"]
    
    system_prompt = f"""
    You are the Clinical Moderator. Evaluate the debate transcript and the original CNN JSON.
    Current Round: {current_round}/{max_rounds}.
    
    INSTRUCTIONS:
    1. If consensus is reached, set action to 'finalize'.
    2. If {current_round} == {max_rounds} and no consensus exists, set final_verdict to 'Refer to specialist / Inconclusive' and action to 'finalize'.
    3. Otherwise, set action to 'continue_debate'.
    4. Provide an audit_trail detailing why arguments were accepted/rejected.
    """
    
    human_prompt = f"""
    Input Data: {state['input_data']}
    Debate History: {' | '.join(state['debate_history'])}
    """
    
    verdict = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
    
    # Increment round counter and store the verdict
    return {
        "verdict": verdict,
        "current_round": current_round + 1
    }