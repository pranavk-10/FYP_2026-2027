import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from state import DiagnosticState, clean_response
from dotenv import load_dotenv

load_dotenv()

def evidence_checker_node(state: DiagnosticState):
    print("\n--- Evidence-Checker (RAG) ---")
    llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0, max_tokens=500)
    latest_arguments = "\n".join(state["debate_history"][-2:])
    
    try:
        embeddings = MistralAIEmbeddings(model="mistral-embed")
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        vector_store = PineconeVectorStore(index=pc.Index("medical-knowledge-base"), embedding=embeddings)
        docs = vector_store.as_retriever(search_kwargs={"k": 2}).invoke(latest_arguments)
        retrieved_context = "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        print(f"(Vector DB skipped/error: {e})")
        retrieved_context = "No retrieved evidence available for testing."

    system_prompt = f"Cross-reference claims against this context: <context> {retrieved_context} </context> State which claims are supported or lack evidence. IMPORTANT: Output your response directly. DO NOT use <think> tags or output internal reasoning."
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=latest_arguments)])
    clean_text = clean_response(response.content)
    
    print(clean_text)
    return {"debate_history": [f"Evidence-Checker: {clean_text}"], "retrieved_context": retrieved_context}