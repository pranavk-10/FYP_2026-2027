import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from state import DiagnosticState, clean_response
from dotenv import load_dotenv

load_dotenv()

def extract_clean_search_query(llm, latest_arguments):
    """
    Priority 2: Uses Groq to extract concise, clean medical search terms 
    from the conversational debate history to maximize vector similarity matching.
    """
    prompt = f"""
    Extract the core medical diagnoses, symptoms, anatomical locations, and imaging signs 
    from the following clinical debate text into a single, clean search query string. 
    Omit conversational filler, agent names, and probability numbers.
    
    Text: {latest_arguments}
    
    Output ONLY the concise space-separated medical search terms.
    """
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        clean_terms = clean_response(response.content)
        return clean_terms if len(clean_terms) > 3 else latest_arguments
    except Exception:
        return latest_arguments

def evidence_checker_node(state: DiagnosticState):
    print("\n--- Evidence-Checker (RAG) ---")
    llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0, max_tokens=600)
    latest_arguments = "\n".join(state["debate_history"][-2:])
    
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
    Cross-reference the claims made by the Advocate and Skeptic against the retrieved WHO Clinical Guidelines context below.

    <context>
    {retrieved_context}
    </context>

    For each claim made in the debate:
    1. State whether it is Supported, Refuted, or Unverified by the provided WHO context.
    2. If Supported, include the exact document citation from the context (e.g., [WHO HEARTS CVD Management Package, Page 17]).

    Output your response in a Markdown table.
    IMPORTANT: Output your response directly. DO NOT use <think> tags or output internal reasoning.
    """
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=latest_arguments)])
    clean_text = clean_response(response.content)
    
    print(clean_text)
    return {"debate_history": [f"Evidence-Checker: {clean_text}"], "retrieved_context": retrieved_context}