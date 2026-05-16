import os
import logging
import pandas as pd

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_documents(data_path="data"):

    os.makedirs(data_path, exist_ok=True)

    logger.info(f"Checking files inside: {os.path.abspath(data_path)}")

    all_docs = []

    supported_extensions = (".csv", ".xlsx")

    for filename in os.listdir(data_path):

        if not filename.endswith(supported_extensions):
            continue

        file_path = os.path.join(data_path, filename)

        logger.info(f"Processing file: {filename}")

        try:

            # Read file
            if filename.endswith(".csv"):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)

            # Replace NaN
            df = df.fillna("")

            logger.info(f"Rows found: {len(df)}")

            # Convert rows to documents
            for index, row in df.iterrows():

                parts = []

            for column, value in row.items():
            
                value = str(value).strip()

                if value == "":
                    continue
                
                if len(value) < 3:
                    continue
                
                if value.isdigit():
                    continue
                
                column = str(column).lower()

                # Ignore noisy columns
                if column in [
                    "ticket_id",
                    "id",
                    "timestamp",
                    "created_at"
                ]:
                    continue
                
                parts.append(f"{column}: {value}")

            if not parts:
                continue
            
            content = ". ".join(parts)

            doc = Document(
                page_content=content,
                metadata={
                    "source": filename,
                    "row_number": index
                }
            )

            all_docs.append(doc)

        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")

    if not all_docs:
        logger.warning("No valid documents found.")
        return []

    logger.info(f"Loaded {len(all_docs)} cleaned documents.")

    # Better chunking for SAP datasets
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80
    )

    chunks = text_splitter.split_documents(all_docs)

    logger.info(f"Created {len(chunks)} chunks.")

    return chunks