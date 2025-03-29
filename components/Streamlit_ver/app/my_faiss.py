"""
This script creates a FAISS index for a Retrieval-Augmented Generation (RAG) model.

The FAISS index is utilized to retrieve the most similar documents for a given query.
Documents are extracted from Excel files located in a specified file path.
"""

import json
import os
import pandas as pd
from typing import List, Dict
from logger import logger

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.schema import Document

from config import settings


def createDoc(file_path: str) -> (List[Document], Dict):
    """Loads and processes text data from Excel files, returning documents and metadata."""
    all_documents = []
    school_dict = {}

    if not os.path.exists(file_path):
        logger.error(f"File path does not exist: {file_path}")
        return all_documents, school_dict

    def process_row(row):
        tags = str(row.get("Tags", "")).split(",")
        school, department = tags[:2] if len(tags) > 1 else (tags[0], "")

        if school not in school_dict:
            school_dict[school] = [department] if department else []
        elif department and department not in school_dict[school]:
            school_dict[school].append(department)

        question_text = (
            str(row.get("Question", "")).replace("\xa0", "").replace("\n", "")
        )
        answer_text = str(row.get("Answer", "")).replace("\xa0", "").replace("\n", "")

        doc_json = {
            "Q": f"{school}, {department} {question_text}",
            "A": answer_text,
        }

        doc_content = json.dumps(doc_json, ensure_ascii=False)

        return Document(
            page_content=doc_content,
            metadata={
                "school": school,
                "department": department,
                "link": row.get("Link", ""),
            },
        )

    def process_files():
        for file in os.listdir(file_path):
            if file.endswith((".xlsx", ".xls")):
                try:
                    df = pd.read_excel(os.path.join(file_path, file))
                except Exception as e:
                    logger.error(f"Error reading {file}: {e}")
                    continue

                for _, row in df.iterrows():
                    all_documents.append(process_row(row))

    try:
        process_files()
    except Exception as e:
        logger.error(f"Error processing files in {file_path}: {e}")

    return all_documents, school_dict


def build_faiss_index(
    documents: List[Document], embedding_model, prefix="uniApply_rag"
):
    """Builds or loads a FAISS index for the given documents using the specified embedding model."""
    index_save_path = f"./index_data/{prefix}"

    def load_or_create_index():
        if os.path.exists(index_save_path):
            logger.info(f"Loading FAISS index from: {index_save_path}")
            return FAISS.load_local(
                index_save_path, embedding_model, allow_dangerous_deserialization=True
            )
        else:
            logger.info("Building new FAISS index...")
            vectorstore = FAISS.from_documents(documents, embedding_model)
            logger.info(f"Saving FAISS index to: {index_save_path}")
            vectorstore.save_local(index_save_path)
            return vectorstore

    try:
        return load_or_create_index()
    except Exception as e:
        logger.error(f"Error building/loading FAISS index: {e}")
        raise


def build_retriever(file_path: str, embedding_model, prefix="uniApply_rag"):
    try:
        current_documents, school_dict = createDoc(file_path)
        logger.info(f"file_path: {file_path}, documents: {len(current_documents)}")
        return (
            build_faiss_index(current_documents, embedding_model, prefix),
            school_dict,
        )
    except Exception as e:
        logger.error(f"Error building retriever: {e}")
        raise


# Set Embedding Model
embedding_model_config = settings.RAG_INDEX_HF_EMBEDDING_MODEL_CONFIG
hf_embedding_model = HuggingFaceBgeEmbeddings(**embedding_model_config)
file_path = settings.RAG_FILES_FILEPATH
try:
    retriever, school_dict = build_retriever(
        file_path, hf_embedding_model, prefix=settings.RAG_INDEX_PREFIX
    )
except Exception as e:
    logger.error(f"Failed to build retriever: {e}")
