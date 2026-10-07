import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from state import DiagnosticState, clean_response
from dotenv import load_dotenv

load_dotenv()

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
    raw_modality = input_data.get("modality", "xray").lower()
    
    # Normalize skin/dermoscopy to match Pinecone metadata
    if raw_modality in ["dermoscopy", "skin_lesion", "skin"]:
        active_modality = "skin"
    else:
        active_modality = raw_modality

    filter_dict = {"modality": active_modality} if active_modality in ["ecg", "xray", "skin"] else None

    try:
        embeddings = MistralAIEmbeddings(model="mistral-embed")
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        vector_store = PineconeVectorStore(index=pc.Index("medical-knowledge-base"), embedding=embeddings)
        
        search_kwargs = {"k": 3}
        if filter_dict:
            search_kwargs["filter"] = filter_dict

        docs = vector_store.as_retriever(search_kwargs=search_kwargs).invoke(search_query)

        # Smart context windowing: cap context at 1200 characters to prevent token exhaustion
        context_blocks = []
        total_chars = 0
        for d in docs:
            content = d.page_content.strip()
            if total_chars + len(content) > 1200:
                content = content[:max(0, 1200 - total_chars)] + "..."
            doc_title = d.metadata.get("document_title", "WHO Clinical Guidelines")
            page_num = d.metadata.get("page_number", "N/A")
            context_blocks.append(f"[{doc_title}, Page {page_num}]:\n{content}")
            total_chars += len(content)
            if total_chars >= 1200:
                break
            
        retrieved_context = "\n\n".join(context_blocks)
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

    Present your verification in a clean Markdown table (Claim | Origin | Evidence Assessment | Guideline Source | Key Takeaway).
    """
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=latest_arguments)])
    clean_text = clean_response(response.content)
    
    print(clean_text)
    return {"debate_history": [f"Evidence-Checker: {clean_text}"], "retrieved_context": retrieved_context}