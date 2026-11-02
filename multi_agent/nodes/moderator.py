from langchain_core.messages import SystemMessage, HumanMessage
from state import ModeratorVerdict, format_patient_summary, GROUNDING_RULE
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Enforce structured output matching the ModeratorVerdict Pydantic schema.
# gpt-oss also spends hidden reasoning tokens from max_tokens; 1024 was too small for multimodal
# verdicts and cut the JSON off (Groq 400 'tool_use_failed').
llm = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0, max_tokens=2048, max_retries=3).with_structured_output(ModeratorVerdict)

SHORTER = ("Your previous answer was cut off or was not valid JSON. Answer again with a shorter verdict: "
           "audit_trail at most 80 words and at most 3 secondary_findings.")


def _invoke_verdict(messages, input_data, current_round):
    """Call the moderator LLM; on an invalid or truncated answer retry once with a request to be
    shorter, then fall back to a referral instead of failing the whole request."""
    try:
        return llm.invoke(messages)
    except Exception as first_error:
        print(f"[Moderator] invalid structured output, retrying shorter: {str(first_error)[:200]}")
    try:
        return llm.invoke(messages + [HumanMessage(content=SHORTER)])
    except Exception as second_error:
        print(f"[Moderator] retry failed, falling back to referral: {str(second_error)[:200]}")
    top = sorted(_model_findings(input_data), key=lambda f: -f[1])[:4]
    return ModeratorVerdict(
        final_verdict="Refer to Specialist / Inconclusive",
        confidence_score=0.0,
        audit_trail=(f"[System: the moderator could not produce a valid verdict in round {current_round}; "
                     "the case is referred to a specialist. The model findings below are shown unreviewed.]"),
        action="finalize",
        secondary_findings=[f"{mod.upper()}: {label} ({p:.2f}) - review by a specialist" for label, p, mod in top],
    )


def _model_findings(input_data):
    """(label, calibrated probability, modality) for every CNN finding in single or multimodal input."""
    parts = input_data.get("modalities") if isinstance(input_data.get("modalities"), dict) else {
        input_data.get("modality", ""): input_data}
    out = []
    for modality, data in parts.items():
        if modality == "lab_values" or not isinstance(data, dict):
            continue
        for f in (data.get("prediction", {}) or {}).get("findings") or []:
            if isinstance(f, dict) and f.get("label"):
                p = f.get("calibrated_probability", f.get("probability"))
                if isinstance(p, (int, float)):
                    out.append((f["label"], float(p), modality))
    return out


def _enforce_model_limits(verdict, input_data):
    """Rule 6 and the grounding rule in code: confidence never exceeds the model's calibrated
    probability for the verdict, and an ECG MI is never upgraded to 'acute'."""
    text = (verdict.final_verdict or "").lower()
    findings = _model_findings(input_data)
    # longest matching model label (e.g. 'cardiomegaly' from 'Cardiomegaly (Enlarged Heart)')
    matches = [(lab.split("(")[0].strip().lower(), p, mod, lab) for lab, p, mod in findings]
    matches = [m for m in matches if m[0] and m[0] in text]
    if matches:
        key, p, mod, label = max(matches, key=lambda m: len(m[0]))
        if verdict.confidence_score > p:
            verdict.audit_trail += (f" [System: confidence capped at the {mod.upper()} model's calibrated "
                                    f"probability for {label} ({p:.2f}).]")
            verdict.confidence_score = round(p, 4)
        if mod == "ecg" and key == "myocardial infarction" and "acute" in text:
            verdict.final_verdict = "Myocardial Infarction (acute or old)"
            verdict.audit_trail += " [System: the ECG model cannot distinguish acute from old MI; label adjusted.]"
    return verdict


def moderator_node(state):
    current_round = state["current_round"]
    max_rounds = state["max_rounds"]
    patient_summary = format_patient_summary(state.get("input_data", {}))
    
    system_prompt = f"""
    You are the Senior Clinical Moderator chairing a Multidisciplinary Medical Team.
    Evaluate the debate transcript and patient objective data across modalities.
    Current Round: {current_round}/{max_rounds}.
    
    CLINICAL DECISION RULES:
    1. Evidence strength comes from each model's calibrated probability. The models do not report waves, borders
       or radiographic signs, so do not require such signs and do not assume they were seen.
       - >= 0.70: strong evidence.  0.40-0.70: moderate.  < 0.40: weak.
       - A 'Model referral flag' on a finding means that finding is low-confidence.
    2. Assess each modality independently, then integrate:
       - final_verdict is the most clinically urgent finding with strong evidence, using the model's own label
         (never more specific than the model, e.g. not 'acute' when the model says acute or old).
       - Put every other clinically relevant finding from every modality (including benign or lower-urgency ones,
         such as a benign skin lesion or an X-ray finding) in secondary_findings with its probability and follow-up.
       - Findings in unrelated organ systems (e.g. a skin lesion vs a cardiac finding) must not raise or lower
         confidence in each other. Related findings (e.g. cardiac ECG and cardiac X-ray signs) may support each other.
    3. This system receives images only; missing symptoms, biomarkers or history is expected. Do not downgrade a
       strong finding to Inconclusive for that reason alone; instead list the confirmatory tests (e.g. troponin,
       serial ECG, echocardiography, biopsy) in the audit_trail as next steps.
    4. Set final_verdict to 'Refer to Specialist / Inconclusive' only when: the lead finding is below 0.70 or has a
       Model referral flag, two strong findings give conflicting answers about the same organ system, or there is
       no consensus at round {max_rounds}. Still fill secondary_findings in that case.
    5. If the Advocate's diagnosis meets rule 2 and the Skeptic's counter-arguments are theoretical, set action to
       'finalize'. If further deliberation is required and current_round < max_rounds, set action to 'continue_debate'
       and state in the audit_trail the specific open question the next round must resolve.
    6. confidence_score must not exceed the model's calibrated probability for the final_verdict diagnosis.
    7. Give no weight to arguments that rely on image features not listed in the patient summary.
       Provide an articulate audit_trail: clinical rationale, which arguments prevailed, and recommended next steps.

    {GROUNDING_RULE}
    """
    
    human_prompt = f"""
    Patient Clinical Summary:
    {patient_summary}
    
    Debate History:
    {chr(10).join(state['debate_history'])}
    """
    
    verdict = _invoke_verdict([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)],
                              state.get("input_data", {}), current_round)

    # Enforce rule 3 in code: at the last round the debate must end. If the LLM still asks to
    # continue, there is no consensus, so the case is referred instead of looping.
    if current_round >= max_rounds and verdict.action != "finalize":
        verdict.action = "finalize"
        verdict.final_verdict = "Refer to Specialist / Inconclusive"
        verdict.audit_trail = (f"{verdict.audit_trail} [System: maximum of {max_rounds} rounds reached "
                               "without consensus; case referred to a specialist.]")

    _enforce_model_limits(verdict, state.get("input_data", {}))

    # Increment round counter and store the verdict
    update = {
        "verdict": verdict,
        "current_round": current_round + 1
    }
    # When the debate continues, pass the moderator's reasoning to the next round
    if verdict.action != "finalize":
        update["debate_history"] = [f"Moderator (end of round {current_round}): {verdict.audit_trail}"]
    return update