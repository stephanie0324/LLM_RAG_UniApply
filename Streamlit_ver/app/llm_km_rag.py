from fastapi import FastAPI, Form, HTTPException, Request
import httpx
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from langserve.pydantic_v1 import BaseModel, Field
from typing import List, Tuple, TypedDict, Dict

import json
import os
import requests

from operator import itemgetter

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain.schema import format_document

from config import settings

import logging
import langchain


logging.basicConfig(
    level=logging.INFO,  # 設定日誌等級為 INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 設定日誌格式
    handlers=[
        logging.StreamHandler()  # 將日誌輸出到控制台
    ]
)

logger = logging.getLogger(__name__)

langchain.debug = settings.DEBUG

model_config = settings.MODEL_CONFIG.root

first_model_key = list(model_config.keys())[0]
llm_model = model_config[first_model_key].as_instance()

# 指定給 LLM 看時的文件組合格式
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n"
):
    doc_strings = [format_document(Document(page_content=doc["doc_name"], metadata={'source': '101_FAQ'}), document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def retrieve(question, top_n):
    url =  "http://140.96.111.113:31850/v2/search/text/"
    kg_data = {
          "keyword": "{}".format(question),
          "negKeyword": "",
          "knn": True,
          "k_boost": 1000,
          "k_similarity": 0.7,
          "nokeyword": False,
          "rerank": True,
          "sort": "relevance",
          "order": "desc",
          "showScore": True,
          "start": "0",
          "rows": top_n,
          "user": "testuser",
          "fuzzyMode": "0",
          "showQuery": False,
          "showHighlight": False,
          "filter": [
            {
              "field": "DATA_TYPE",
              "logic": "OR",
              "values": [
                "4"
              ]
            }]
        }
    kg_data = json.dumps(kg_data)
    payload = {
        'rerank_model_suffix': '1',
        'paras': kg_data
    }
    headers = {}
 
    response = requests.request("POST", url, headers=headers, data=payload)
    
    return  response.json()['records']

def get_language_name(language): 
    language_name_dict = {
        "zh-tw": "Traditional Chinese",
        "ja-jp": "Japanese",
        "en-us": "English",
        "ko-kr": "korean"
    }
    return  language_name_dict.get(language)
    


template = """Please answer the user's question based on the related questions and answers provided below. 
Please do not fabricate facts; answer only based on the provided information.
Reply in "{language_name}".

# Related Questions and Answers
{context}

# former interaction
User question: {question} Reply in "{language_name}".
ANSWER:"""

rag_prompt = HumanMessagePromptTemplate.from_template(template)

messages = [
    SystemMessage(content="You are a helpful AI assistant."),
    rag_prompt
]

full_rag_prompts = ChatPromptTemplate.from_messages(messages)

transform_inputs = {
    "question": lambda x: x["query"][-1],
    "history_question": lambda x: x["query"][:-1],
    "language": itemgetter("language"),
    "language_name": lambda x: get_language_name(x["language"])
}

retrieved_documents = {
    "context": lambda x: _combine_documents((retrieve(x["question"], settings.RETRIEVER_RETURN_TOP_N))),
    "question": itemgetter("question"),
    "language": itemgetter("language"),
    "language_name": itemgetter("language_name")
}

predict = {
    "response":  full_rag_prompts | llm_model | StrOutputParser(),
    "language": itemgetter("language"),
    "language_name": itemgetter("language_name")
}

rag_chain = (
    RunnablePassthrough() | transform_inputs
    | retrieved_documents
    | predict
)


class Input(BaseModel):
    """Chat with the bot."""
    query: List[str]
    language: str = "zh-tw"

class Output(TypedDict):
    status: bool= True
    msg: str = "成功"
    response: str

chain = rag_chain.with_types(input_type=Input)

app = FastAPI(
    title="RAG API",
    version="0.1",
    description="僅提供 RAG 的 input",
)

add_routes(app, chain, path="/qa", enable_feedback_endpoint=False)


# Redirect route from /v1/qa to /va/rag/invoke
@app.post("/v1/qa")
async def qa_handler(query: str = Form(...), language: str = Form(...)):
    try:
        # Parse the JSON string from 'query' form data into a Python list
        query_list = json.loads(query)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format in query parameter")

    
    url = 'http://127.0.0.1:8000/qa/invoke'  # Local redirection URL
    headers = {'Content-Type': 'application/json'}
    payload = {
        "input": {
            "query": query_list,
            "language": language
        }
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload, timeout=300)
        if response.status_code == 200:
            data = response.json()
            # Reformat the output as required
            return {
                "status": True,
                "msg": "成功",
                "intent": "",
                "response": data["output"]['response']
            }
        else:
            # Handle errors by reformatting the error message
            return {
                "status": False,
                "msg": response.text,  # or a more specific error message
                "response": None
            }



if settings.BACKEND_CORS_ORIGINS or settings.BACKEND_CORS_ORIGIN_REGEX:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin)
                       for origin in settings.BACKEND_CORS_ORIGINS],
        allow_origin_regex=settings.BACKEND_CORS_ORIGIN_REGEX,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

