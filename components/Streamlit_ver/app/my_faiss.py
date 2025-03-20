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
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from config import settings


def createDoc(file_path: str) -> (List[Document], Dict):
    """Loads and processes text data from Excel files, returning documents and metadata."""

    all_documents = []
    school_dict = {}

    for file in os.listdir(file_path):
        if file.endswith((".xlsx", ".xls")):
            df = pd.read_excel(os.path.join(file_path, file))

            for _, row in df.iterrows():
                tags = str(row.get("Tags", "")).split(",")
                school, department = tags[:2] if len(tags) > 1 else (tags[0], "")

                if school not in school_dict:
                    school_dict[school] = [department] if department else []
                elif department and department not in school_dict[school]:
                    school_dict[school].append(department)

                question_text = (
                    str(row.get("Question", "")).replace("\xa0", "").replace("\n", "")
                )
                answer_text = (
                    str(row.get("Answer", "")).replace("\xa0", "").replace("\n", "")
                )

                doc_json = {
                    "Q": f"{school}, {department} {question_text}",
                    "A": answer_text,
                }

                doc_content = json.dumps(doc_json, ensure_ascii=False)

                all_documents.append(
                    Document(
                        page_content=doc_content,
                        metadata={"source": tags, "link": row.get("Link", "")},
                    )
                )

    return all_documents, school_dict


def build_faiss_index(
    documents: List[Document], embedding_model, prefix="uniApply_rag"
):
    index_save_path = f"./index_data/{prefix}"
    if os.path.exists(index_save_path):
        logger.info(f"Loading FAISS index: {index_save_path}")
        vectorstore = FAISS.load_local(
            index_save_path, embedding_model, allow_dangerous_deserialization=True
        )
        logger.info(f"Loaded FAISS index.")
    else:
        logger.info(f"Building FAISS index...")
        vectorstore = FAISS.from_documents(documents, embedding_model)
        logger.info(f"Saving FAISS index...")
        vectorstore.save_local(index_save_path)
        logger.info(f"Saved FAISS index.")
    return vectorstore


def build_retriever(file_path: str, embedding_model, prefix="uniApply_rag"):
    current_documents, school_dict = createDoc(file_path)
    logger.info(f"file_path: {file_path}, documents: {len(current_documents)}")
    return build_faiss_index(current_documents, embedding_model, prefix), school_dict


# Set Embedding Model
embedding_model_config = settings.RAG_INDEX_HF_EMBEDDING_MODEL_CONFIG
hf_embedding_model = HuggingFaceBgeEmbeddings(**embedding_model_config)
file_path = settings.RAG_FILES_FILEPATH
retriever, school_dict = build_retriever(
    file_path, hf_embedding_model, prefix=settings.RAG_INDEX_PREFIX
)
