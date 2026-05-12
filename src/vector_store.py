import time
import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.ingestion import process_documents
from google.api_core import exceptions

import os
import time
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.ingestion import process_documents

def get_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    persist_directory = "db/sap_knowledge_base"
    
    # Load existing or create new
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    # If DB is empty, start ingestion
    if len(vector_store.get()['ids']) < 10:
        chunks = process_documents()
        batch_size = 50 
        print(f"Uploading {len(chunks)} chunks in batches...")
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            success = False
            while not success:
                try:
                    vector_store.add_documents(batch)
                    print(f"✅ Batch {i//batch_size + 1} uploaded.")
                    success = True
                except Exception as e:
                    if "429" in str(e):
                        print("⏳ Quota reached. Sleeping 60s...")
                        time.sleep(60)
                    else: raise e
    return vector_store

def get_retriever():
    return get_vector_store().as_retriever()
 