import operator
import re
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

def clean_response(text: str) -> str:
    """Removes <think>...</think> blocks from the model output and normalizes unicode characters."""
    cleaned = re.sub(r'<think>.*?(?:</think>|$)', '', text, flags=re.DOTALL)
    # Normalize common typographic unicode characters
    cleaned = (cleaned.replace('\u2011', '-')
                      .replace('\u2013', '-')
                      .replace('\u2014', '--')
                      .replace('\u2018', "'")
                      .replace('\u2019', "'")
                      .replace('\u201c', '"')
                      .replace('\u201d', '"'))
    # Remove any stray non-Latin CJK ideographs that cause Windows terminal crashes
    cleaned = re.sub(r'[\u4e00-\u9fff\u3000-\u303f]', '', cleaned)
    return cleaned.strip()

def format_patient_summary(input_data: Dict[str, Any]) -> str:
    """
    Condenses the raw CNN/OCR JSON into concise, high-signal medical bullet points.
    Prevents Groq ITPM (input tokens per minute) rate limit errors caused by dumping
    massive duplicated 20-class float arrays.
    """
    if not isinstance(input_data, dict):
        return str(input_data)
        
    lines = []
    
    # Check if multimodal combined
    modalities = input_data.get("modalities", {})
    if modalities:
        for mod_name, mod_info in modalities.items():
            if mod_name == "lab_values":
                continue
            pred = mod_info.get("prediction", {})
            diag = pred.get("primary_diagnosis", "Unknown")
            prob = pred.get("calibrated_probability", pred.get("probability", 0))
            norm = pred.get("normality_score")
            
            summary_line = f"• [{mod_name.upper()}] Primary Diagnosis: {diag} ({prob*100:.1f}%)"
            if norm is not None:
                summary_line += f" | Normality Score: {norm*100:.1f}%"
            lines.append(summary_line)
            
            watchlist = pred.get("watchlist_conditions", [])
            if watchlist:
                lines.append(f"   Watchlist Conditions: {', '.join(watchlist[:4])}")
                
            ev = mod_info.get("evidence", {})
            pos = ev.get("positive_findings", [])
            neg = ev.get("negative_findings", [])
            if pos:
                lines.append(f"   Positive Evidence: {'; '.join(pos[:2])}")
            if neg:
                lines.append(f"   Negative Evidence: {'; '.join(neg[:1])}")
                
        # Include lab values if present
        labs = modalities.get("lab_values", {})
        if labs and isinstance(labs, dict) and "error" not in labs:
            lab_items = []
            for k in ["hemoglobin", "wbc_count", "platelets"]:
                if k in labs and labs[k]:
                    lab_items.append(f"{k.capitalize()}: {labs[k].get('value')} {labs[k].get('unit')} ({labs[k].get('flag')})")
            if lab_items:
                lines.append(f"• [LABS] {', '.join(lab_items)}")
    else:
        # Single modality
        modality = input_data.get("modality", "Unknown")
        pred = input_data.get("prediction", {})
        diag = pred.get("primary_diagnosis", "Unknown")
        prob = pred.get("calibrated_probability", pred.get("probability", 0))
        norm = pred.get("normality_score")
        
        summary_line = f"• [{modality.upper()}] Primary Diagnosis: {diag} ({prob*100:.1f}%)"
        if norm is not None:
            summary_line += f" | Normality Score: {norm*100:.1f}%"
        lines.append(summary_line)
        
        watchlist = pred.get("watchlist_conditions", [])
        if watchlist:
            lines.append(f"   Watchlist Conditions: {', '.join(watchlist[:4])}")
            
        ev = input_data.get("evidence", {})
        pos = ev.get("positive_findings", [])
        neg = ev.get("negative_findings", [])
        if pos:
            lines.append(f"   Positive Evidence: {'; '.join(pos[:2])}")
        if neg:
            lines.append(f"   Negative Evidence: {'; '.join(neg[:1])}")
            
    return "\n".join(lines) if lines else str(input_data)[:400]

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