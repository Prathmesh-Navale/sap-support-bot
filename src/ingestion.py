import os
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_documents(data_path="data/"):
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    print(f"Checking for files in: {os.path.abspath(data_path)}")
    all_docs = []
    
    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)
        
        if filename.endswith('.csv') or filename.endswith('.xlsx'):
            print(f"Processing dataset: {filename}")
            try:
                df = pd.read_csv(file_path) if filename.endswith('.csv') else pd.read_excel(file_path)
                for index, row in df.iterrows():
                    # Merges columns into a readable format for the AI
                    content = " ".join([f"{col}: {val}" for col, val in row.items() if pd.notnull(val)])
                    doc = Document(page_content=content, metadata={"source": filename, "row": index})
                    all_docs.append(doc)
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    if not all_docs:
        return []

    print(f"Loaded {len(all_docs)} rows. Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_documents(all_docs)