import operator
import re
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

def clean_response(text: str) -> str:
    """Removes <think>...</think> blocks from the model output and normalizes unicode characters."""
    cleaned = re.sub(r'<think>.*?(?:</think>|$)', '', text, flags=re.DOTALL)
    # Normalize common typographic unicode characters and special spaces
    cleaned = (cleaned.replace('\u2011', '-')
                      .replace('\u2013', '-')
                      .replace('\u2014', '--')
                      .replace('\u2018', "'")
                      .replace('\u2019', "'")
                      .replace('\u201c', '"')
                      .replace('\u201d', '"')
                      .replace('\u202f', ' ')
                      .replace('\u00a0', ' ')
                      .replace('\u2009', ' ')
                      .replace('\u200b', ''))
    # Remove any stray non-Latin CJK ideographs that cause Windows terminal crashes
    cleaned = re.sub(r'[\u4e00-\u9fff\u3000-\u303f]', '', cleaned)
    return cleaned.strip()

# Shared instruction for every agent: the CNNs output class probabilities only, so agents must
# not present image features (waves, leads, lesion borders, X-ray locations) as observed.
GROUNDING_RULE = (
    "GROUNDING RULE: Use only the diagnoses, probabilities and evidence listed in the Patient Clinical Summary "
    "and the debate. The image models output class probabilities only; they do not measure individual ECG waves "
    "or leads, lesion borders or pigment patterns, or X-ray locations or ratios. Evidence sentences from the models "
    "describe what the predicted class means, not measurements made on this image. Never state such features as "
    "observed; you may mention them only as things a clinician should check. Never make a diagnosis more specific "
    "than the model output (for example, do not call an MI 'acute' when the model says acute or old). "
    "Treat a 'Model referral flag' as a strong reason for caution."
)


def _top_differential(pred: Dict[str, Any], n: int = 3) -> List[str]:
    """Top-n diagnoses with calibrated probabilities from a CNN prediction's findings list."""
    findings = pred.get("findings") or []
    items = []
    for f in findings:
        if not isinstance(f, dict) or "label" not in f:
            continue
        p = f.get("calibrated_probability", f.get("probability"))
        if isinstance(p, (int, float)):
            items.append((f["label"], float(p)))
    items.sort(key=lambda x: x[1], reverse=True)
    return [f"{label} {p * 100:.1f}%" for label, p in items[:n]]


def _summarize_modality(name: str, data: Dict[str, Any], lines: List[str]) -> None:
    """Appends the summary lines for one CNN modality output."""
    pred = data.get("prediction", {}) or {}
    diag = pred.get("primary_diagnosis", "Unknown")
    prob = pred.get("calibrated_probability", pred.get("probability", 0)) or 0
    norm = pred.get("normality_score")
    if "primary_diagnosis" not in pred:
        # older/minimal inputs: fall back to the highest-probability finding
        top = sorted((f for f in pred.get("findings") or [] if isinstance(f, dict) and "label" in f),
                     key=lambda f: f.get("calibrated_probability", f.get("probability", 0)) or 0, reverse=True)
        if top:
            diag = top[0]["label"]
            prob = top[0].get("calibrated_probability", top[0].get("probability", 0)) or 0

    summary_line = f"• [{name.upper()}] Primary Diagnosis: {diag} ({prob*100:.1f}%)"
    if norm is not None:
        summary_line += f" | Normality Score: {norm*100:.1f}%"
    lines.append(summary_line)

    top = _top_differential(pred)
    if len(top) > 1:
        lines.append(f"   Top Differential: {', '.join(top)}")

    watchlist = pred.get("watchlist_conditions", [])
    if watchlist:
        lines.append(f"   Watchlist Conditions: {', '.join(watchlist[:4])}")

    if pred.get("abstain"):
        reason = pred.get("abstain_reason") or "low-confidence result"
        lines.append(f"   Model referral flag: specialist review advised ({reason})")

    ev = data.get("evidence", {}) or {}
    pos = ev.get("positive_findings", [])
    # the referral line is shown above, so skip it here and show the next negative finding
    neg = [n for n in ev.get("negative_findings", []) if not str(n).startswith("[REFER]")]
    if pos:
        lines.append(f"   Positive Evidence: {'; '.join(pos[:2])}")
    if neg:
        lines.append(f"   Negative Evidence: {'; '.join(neg[:1])}")


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
            if mod_name == "lab_values" or not isinstance(mod_info, dict):
                continue
            _summarize_modality(mod_name, mod_info, lines)

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
        _summarize_modality(input_data.get("modality", "Unknown"), input_data, lines)

    return "\n".join(lines) if lines else str(input_data)[:400]

class ModeratorVerdict(BaseModel):
    final_verdict: str = Field(description="The validated primary diagnosis or 'Refer to Specialist / Inconclusive'")
    confidence_score: float = Field(description="Final calibrated confidence score between 0.0 and 1.0")
    audit_trail: str = Field(description="Explanation of why this diagnosis was chosen")
    action: str = Field(description="'finalize' if consensus/max rounds reached, else 'continue_debate'")
    secondary_findings: List[str] = Field(
        default_factory=list,
        description="Every other clinically relevant finding, one per item, as 'MODALITY: finding (probability) - "
                    "recommended follow-up'. Include findings from modalities not used for the final_verdict.")

class DiagnosticState(TypedDict):
    input_data: Dict[str, Any]
    current_round: int
    max_rounds: int
    debate_history: Annotated[List[str], operator.add]
    retrieved_context: Optional[str]
    verdict: Optional[ModeratorVerdict]