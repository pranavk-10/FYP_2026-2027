import operator
import re
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

def clean_response(text: str) -> str:
    """Removes <think>...</think> blocks from the model output."""
    cleaned = re.sub(r'<think>.*?(?:</think>|$)', '', text, flags=re.DOTALL)
    return cleaned.strip()

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