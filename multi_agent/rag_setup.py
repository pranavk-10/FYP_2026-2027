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