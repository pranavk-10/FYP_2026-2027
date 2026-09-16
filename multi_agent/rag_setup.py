## isko setup karna baki hai

# import os
# import time
# from pinecone import Pinecone, ServerlessSpec
# from langchain_pinecone import PineconeVectorStore
# from langchain_mistralai import MistralAIEmbeddings
# from langchain_core.documents import Document
# from dotenv import load_dotenv

# load_dotenv()

# pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
# index_name = "medical-knowledge-base"

# if index_name not in pc.list_indexes().names():
#     print(f"Creating Pinecone index: {index_name}...")
#     pc.create_index(
#         name=index_name,
#         dimension=1024,
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1")
#     )
#     while not pc.describe_index(index_name).status['ready']:
#         time.sleep(1)

# index = pc.Index(index_name)
# embeddings = MistralAIEmbeddings(model="mistral-embed")

# sample_docs = [
#     Document(
#         page_content="Clinical Guidelines for Pneumonia: Radiographic evaluation typically reveals lobar consolidation or patchy opacities. A right lower lobe opacity is highly indicative of pneumonia when accompanied by clinical symptoms.",
#         metadata={"source": "WHO_Guidelines_Pneumonia"}
#     ),
#     Document(
#         page_content="Cardiomegaly Imaging: An enlarged cardiac silhouette on a chest X-ray suggests cardiomegaly. While it can be associated with pulmonary edema, isolated cardiomegaly without pleural effusion or vascular congestion is less indicative of acute heart failure.",
#         metadata={"source": "Radiology_Reference"}
#     )
# ]

# print("Uploading documents to Pinecone...")
# vector_store = PineconeVectorStore(index=index, embedding=embeddings)
# vector_store.add_documents(sample_docs)
# print("Ingestion complete. The Evidence-Checker can now retrieve facts.")

# =====================================================================
# PRODUCTION WHO GUIDELINES PDF INGESTION PIPELINE (PRIORITIES 1 & 3)
# =====================================================================

import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

def get_pdf_path():
    """Returns absolute path to the WHO guidelines PDF file."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_path = os.path.join(base_dir, "data", "RAG_WHO.pdf")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"WHO Guidelines PDF not found at: {pdf_path}")
    return pdf_path




def determine_modality_and_doc(page_num_1indexed):
    """
    Maps 1-indexed page numbers from the merged WHO PDF to target modality and document.
    - Pages 1-44: Chest Radiography in TB Detection (X-Ray)
    - Pages 45-124: HEARTS CVD Risk Management (ECG)
    - Pages 125-181: Recognizing Skin NTDs (Skin)
    - Pages 182-215: Childhood Pneumonia Guidelines (X-Ray)
    - Pages 216-283: WHO Consolidated TB Screening Guidelines (X-Ray)
    """
    p = page_num_1indexed
    if 1 <= p <= 44:
        return "xray", "WHO Chest Radiography in TB Detection"
    elif 45 <= p <= 124:
        return "ecg", "WHO HEARTS CVD Management Package"
    elif 125 <= p <= 181:
        return "skin", "WHO Recognizing Skin NTDs Guide"
    elif 182 <= p <= 215:
        return "xray", "WHO Childhood Pneumonia Classification"
    else:
        return "xray", "WHO Consolidated TB Screening Guidelines"

def setup_rag_pipeline(pdf_path=None, index_name="medical-knowledge-base", batch_size=50):
    """
    Loads WHO guidelines PDF, assigns page-level modality metadata (Priority 1),
    chunks with optimal size 500 (Priority 3), embeds with Mistral Embeddings,
    and ingests into Pinecone index.
    """
    pinecone_key = os.environ.get("PINECONE_API_KEY")
    mistral_key = os.environ.get("MISTRAL_API_KEY")

    if not pinecone_key:
        raise ValueError("PINECONE_API_KEY is missing from environment / .env file.")
    if not mistral_key:
        raise ValueError("MISTRAL_API_KEY is missing from environment / .env file.")

    if pdf_path is None:
        pdf_path = get_pdf_path()

    print(f"=== Starting Production WHO Medical Knowledge Base RAG Setup ===")
    print(f"Loading PDF from: {pdf_path}")

    # 1. Load PDF pages with page-accurate modality metadata (Priority 1)
    loader = PyPDFLoader(pdf_path)
    raw_docs = []
    print("Parsing PDF pages and attaching modality metadata...")
    for idx, doc in enumerate(loader.lazy_load()):
        page_num = idx + 1
        modality, doc_title = determine_modality_and_doc(page_num)
        doc.metadata["page_number"] = page_num
        doc.metadata["modality"] = modality
        doc.metadata["document_title"] = doc_title
        raw_docs.append(doc)
        if page_num % 25 == 0:
            print(f"Parsed {page_num} pages...")

    print(f"Total: Loaded {len(raw_docs)} pages with modality metadata.")

    # 2. Chunk text into optimal 500-character passages (Priority 3)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(raw_docs)
    print(f"Split PDF into {len(chunks)} semantic chunks (chunk_size=500, overlap=100).")

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    # 3. Connect to Pinecone and verify index
    pc = Pinecone(api_key=pinecone_key)
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if index_name not in existing_indexes:
        print(f"Creating Pinecone Index: '{index_name}' (dimension 1024, metric='cosine')...")
        pc.create_index(
            name=index_name,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        while not pc.describe_index(index_name).status['ready']:
            print("Waiting for Pinecone index to become ready...")
            time.sleep(2)
        print("Pinecone index created and ready!")
    else:
        print(f"Pinecone Index '{index_name}' already exists. Clearing old vectors for clean ingestion...")
        index = pc.Index(index_name)
        try:
            index.delete(delete_all=True)
            print("Cleared older vectors successfully.")
        except Exception as e:
            print(f"(Note: Index reset warning: {e})")

    index = pc.Index(index_name)
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)


    # 4. Ingest chunks in batches to prevent payload overflow
    total_chunks = len(chunks)
    print(f"Ingesting {total_chunks} chunks into Pinecone in batches of {batch_size}...")

    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        print(f"Uploading batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size} (Chunks {i+1} to {min(i+batch_size, total_chunks)})...")
        vector_store.add_documents(batch)
        time.sleep(0.5)

    print("\n✅ RAG Ingestion Complete!")
    print(f"Pinecone index '{index_name}' now holds {total_chunks} modality-tagged WHO guideline passages.")

if __name__ == "__main__":
    setup_rag_pipeline()