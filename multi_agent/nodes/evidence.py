import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from state import DiagnosticState, clean_response
from dotenv import load_dotenv

load_dotenv()

def extract_clean_search_query(llm, latest_arguments: str) -> str:
    """
    Extracts concise, clean medical search terms from debate history
    to maximize vector similarity matching in Pinecone.
    """
    prompt = (
        "Extract 3 to 6 key medical search terms (diagnoses, symptoms, ECG findings, or radiological signs) "
        "from the following text. Omit all conversational words, punctuation, numbers, and agent names.\n\n"
        f"Text:\n{latest_arguments[:600]}\n\n"
        "Output ONLY the space-separated medical keywords (e.g., 'myocardial infarction ST elevation pericarditis atelectasis'):"
    )
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        clean_terms = clean_response(response.content).strip()
        # Take the first line and remove markdown/bullets
        clean_terms = clean_terms.replace("*", "").replace("-", "").replace("`", "").strip()
        lines = [line.strip() for line in clean_terms.split("\n") if line.strip()]
        first_line = lines[0] if lines else clean_terms
        words = first_line.split()
        if len(words) > 10:
            first_line = " ".join(words[:8])
        return first_line if len(first_line) > 3 else "myocardial infarction atelectasis"
    except Exception:
        return "myocardial infarction atelectasis"


def evidence_checker_node(state: DiagnosticState):
    print("\n--- Evidence-Checker (RAG) ---")
    llm = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0.1, max_tokens=600)
    
    debate_history = state.get("debate_history", [])
    latest_arguments = "\n".join(debate_history[-2:]) if debate_history else ""
    
    search_query = extract_clean_search_query(llm, latest_arguments)
    print(f"Clean Extracted RAG Query: '{search_query}'")

    input_data = state.get("input_data", {})
    active_modality = input_data.get("modality", "xray").lower()
    filter_dict = {"modality": active_modality} if active_modality in ["ecg", "xray", "skin"] else None

    try:
        embeddings = MistralAIEmbeddings(model="mistral-embed")
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        vector_store = PineconeVectorStore(index=pc.Index("medical-knowledge-base"), embedding=embeddings)
        
        search_kwargs = {"k": 4}
        if filter_dict:
            search_kwargs["filter"] = filter_dict

        docs = vector_store.as_retriever(search_kwargs=search_kwargs).invoke(search_query)

        context_blocks = []
        for d in docs:
            doc_title = d.metadata.get("document_title", "WHO Clinical Guidelines")
            page_num = d.metadata.get("page_number", "N/A")
            context_blocks.append(f"[{doc_title}, Page {page_num}]:\n{d.page_content}")
            
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

    For each clinical claim:
    1. Cross-reference against the context and established clinical standards (Supported, Refuted, or Inconclusive / Not in Guidelines).
    2. Cite the specific guideline source if present (e.g., [WHO HEARTS CVD Management Package, Page 113]).
    3. Note any red flags, diagnostic pitfalls, or mandatory confirmatory tests mentioned in the literature.

    Present your verification in a clear Markdown table.
    """
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=latest_arguments)])
    clean_text = clean_response(response.content)
    
    print(clean_text)
    return {"debate_history": [f"Evidence-Checker: {clean_text}"], "retrieved_context": retrieved_context}