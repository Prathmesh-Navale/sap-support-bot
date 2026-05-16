import os
import time
import logging

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.ingestion import process_documents

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Vector DB folder
PERSIST_DIRECTORY = "db/sap_knowledge_base"

os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

# HuggingFace Embedding Model
 

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def get_vector_store():

    vector_store = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )

    try:
        existing_ids = vector_store.get().get("ids", [])
    except Exception:
        existing_ids = []

    if len(existing_ids) == 0:

        logger.info("Vector DB empty. Starting ingestion...")

        chunks = process_documents()

        if not chunks:
            logger.warning("No documents found.")
            return vector_store

        batch_size = 50

        for i in range(0, len(chunks), batch_size):

            batch = chunks[i:i + batch_size]

            try:
                vector_store.add_documents(batch)

                logger.info(
                    f"Uploaded batch {i // batch_size + 1}"
                )

            except Exception as e:
                logger.error(str(e))
                raise e

    else:
        logger.info("Existing vector DB loaded.")

    return vector_store


def get_retriever():

    vector_store = get_vector_store()

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )

    return retriever