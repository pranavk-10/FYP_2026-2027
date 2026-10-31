import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from state import DiagnosticState, clean_response, GROUNDING_RULE
from dotenv import load_dotenv

load_dotenv()

# CNN modality names -> Pinecone "modality" metadata values
MODALITY_ALIASES = {"ecg": "ecg", "xray": "xray", "chest_xray": "xray",
                    "dermoscopy": "skin", "skin_lesion": "skin", "skin": "skin"}
CONTEXT_CHAR_LIMIT = 1200          # single-modality context budget
MULTIMODAL_CHAR_PER_MODALITY = 700  # per-modality budget when several modalities are present


def target_modalities(input_data: dict) -> list:
    """Guideline modalities to search: one per imaging modality in the case (multimodal aware)."""
    modalities = input_data.get("modalities")
    if isinstance(modalities, dict) and modalities:
        raw = [name for name in modalities if name != "lab_values"]
    else:
        raw = [input_data.get("modality", "xray")]
    targets = []
    for name in raw:
        mapped = MODALITY_ALIASES.get(str(name).lower())
        if mapped and mapped not in targets:
            targets.append(mapped)
    return targets


def _format_docs(docs, char_budget: int, label: str = None) -> list:
    """Formats retrieved chunks with source citations, capped at char_budget characters."""
    blocks, total = [], 0
    for d in docs:
        content = d.page_content.strip()
        if total + len(content) > char_budget:
            content = content[:max(0, char_budget - total)] + "..."
        doc_title = d.metadata.get("document_title", "WHO Clinical Guidelines")
        page_num = d.metadata.get("page_number", "N/A")
        prefix = f"{label.upper()} | " if label else ""
        blocks.append(f"[{prefix}{doc_title}, Page {page_num}]:\n{content}")
        total += len(content)
        if total >= char_budget:
            break
    return blocks


def build_retrieved_context(vector_store, search_query: str, modalities: list) -> str:
    """Retrieves guideline passages separately for each modality so multimodal cases get each one's guidelines."""
    if not modalities:
        docs = vector_store.as_retriever(search_kwargs={"k": 3}).invoke(search_query)
        return "\n\n".join(_format_docs(docs, CONTEXT_CHAR_LIMIT))
    single = len(modalities) == 1
    k = 3 if single else 2
    budget = CONTEXT_CHAR_LIMIT if single else MULTIMODAL_CHAR_PER_MODALITY
    blocks = []
    for modality in modalities:
        docs = vector_store.as_retriever(search_kwargs={"k": k, "filter": {"modality": modality}}).invoke(search_query)
        blocks.extend(_format_docs(docs, budget, label=None if single else modality))
    return "\n\n".join(blocks)

def extract_clean_search_query(latest_arguments: str) -> str:
    """
    Extracts concise, clean medical search terms from debate history.
    Uses llama-3.1-8b-instant (low token overhead, fast) with regex fallback
    to prevent exhausting 120B rate limits.
    """
    if not latest_arguments:
        return "clinical evaluation guidelines"

    try:
        fast_llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1, max_tokens=60, max_retries=2)
        prompt = (
            "Extract 3 to 6 key medical search terms (diagnoses, symptoms, ECG findings, skin lesion patterns, or radiological signs) "
            "from this debate text. Output ONLY space-separated keywords without punctuation or preamble:\n\n"
            f"{latest_arguments[:400]}"
        )
        response = fast_llm.invoke([HumanMessage(content=prompt)])
        clean_terms = clean_response(response.content).strip()
        clean_terms = clean_terms.replace("*", "").replace("-", "").replace("`", "").replace('"', '').strip()
        words = clean_terms.split()
        if words:
            return " ".join(words[:6])
    except Exception:
        pass

    # Pure Python fallback: extract key diagnostic tokens
    fallback_words = [w for w in latest_arguments.split() if len(w) > 4 and w.isalpha()]
    return " ".join(fallback_words[:5]) if fallback_words else "clinical diagnostic criteria"


def evidence_checker_node(state: DiagnosticState):
    print("\n--- Evidence-Checker (RAG) ---")
    llm = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0.1, max_tokens=1500, max_retries=3)
    
    debate_history = state.get("debate_history", [])
    latest_arguments = "\n".join(debate_history[-2:]) if debate_history else ""
    
    search_query = extract_clean_search_query(latest_arguments)
    print(f"Clean Extracted RAG Query: '{search_query}'")

    input_data = state.get("input_data", {})
    modalities = target_modalities(input_data)
    print(f"Guideline modalities searched: {modalities or 'all'}")

    try:
        embeddings = MistralAIEmbeddings(model="mistral-embed")
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        vector_store = PineconeVectorStore(index=pc.Index("medical-knowledge-base"), embedding=embeddings)
        # Context is capped per modality to prevent token exhaustion
        retrieved_context = build_retrieved_context(vector_store, search_query, modalities)
    except Exception as e:
        print(f"(Vector DB skipped/error: {e})")
        retrieved_context = "No retrieved evidence available for testing."

    system_prompt = f"""
    You are the Evidence-Checker in a Multidisciplinary Medical Team.
    Evaluate the clinical claims made by the Advocate and Skeptic against the retrieved medical context below.

    <context>
    {retrieved_context}
    </context>

    Instructions:
    1. Select the TOP 3 most decisive clinical claims from the debate to verify.
    2. Cross-reference each claim against the context and established clinical standards (Supported, Refuted, or Inconclusive / Not in Guidelines).
    3. Cite specific guideline source if present (e.g., [WHO HEARTS CVD Management Package, Page 113]).
    4. Keep evaluations concise and complete so the table closes properly without truncation.
    5. If a claim describes an image feature (ECG wave or lead, lesion border or pigment pattern, X-ray location)
       that the CNN models did not report, mark it "Unsupported by model output".

    {GROUNDING_RULE}

    Present your verification in a clean Markdown table (Claim | Origin | Evidence Assessment | Guideline Source | Key Takeaway).
    """
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=latest_arguments)])
    clean_text = clean_response(response.content)
    
    print(clean_text)
    return {"debate_history": [f"Evidence-Checker: {clean_text}"], "retrieved_context": retrieved_context}