from fastapi import FastAPI, Form, HTTPException, Request
import httpx
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from langserve.pydantic_v1 import BaseModel, Field
from typing import List, Tuple, TypedDict, Dict

import json
import os
import re

from operator import itemgetter

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain.schema.retriever import BaseRetriever
from langchain.schema import Document
from langchain_core.prompts import PromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain.schema import format_document
from langchain.chat_models import ChatOpenAI

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

def load_json_train_data(file_path: str) -> List[Document]:
    """
    載入訓練資料，轉換成 langchain 的 Ducument 格式
    """
    
    faq_json_array = []
    with open(file_path, 'r', encoding='utf-8') as file:
        faq_json_array = json.load(file)
    documents = []
    intent_index_map = {}
    index_intent_map = {}
    idx = 0

    for record in faq_json_array:
        if record['category'] not in index_intent_map.values():
            index_intent_map[idx] = record['category']
            intent_index_map[ record['category']] = idx
            idx+=1
        page_content = json.dumps(record, ensure_ascii=False)
        current_document = Document(
            page_content=page_content,
            metadata={
                    'source': 'FAQ',
                    'intent': record['category'],
                    'intent_index': intent_index_map[record['category']],
                    'answer': record['answer']
                }
        )
        documents.append(current_document)
    return index_intent_map, documents

def build_faiss_index(index_intent_map: Dict, documents: List[Document], language, embedding_model,prefix="101_ic_demo_expand"):
    """
    建立 FAISS 索引，並輸出保存
    """
    exist_status = False
    index_save_path = f"./index_data/{prefix}_{language}"
    index_intent_map_file_path = f"./index_data/{prefix}_{language}_intent.json"
    try:
        # 如果索引檔案已經存在就進行載入
        if os.path.exists(index_save_path):
            logger.info(f"嘗試載入 FAISS 索引: {index_save_path}")
            vectorstore = FAISS.load_local(index_save_path, embedding_model, allow_dangerous_deserialization=True)
            logger.info(f"載入 FAISS 索引完成.")
            
            # 如果該索引相關的 id to intent 名稱的表已經存在就載入
            if os.path.exists(index_intent_map_file_path):
                logger.info("載入 intent 的 index map: {index_intent_map_file_path}")
                with open(index_intent_map_file_path, 'r', encoding='utf-8') as json_file:
                    intent_index_map = json.load(json_file)
                logger.info(f"載入 intent 的 index map 完成.")
            exist_status = True
    except Exception as e:
        logger.warn("載入失敗：", e)
        logger.warn(f"FAISS Index: {index_save_path} not found.")

    # 索引檔案不存(載入失敗也會視為不存在)在就進行建立
    if not exist_status:
        logger.info(f"建立 FAISS 索引...")
        vectorstore = FAISS.from_documents(documents, embedding_model)
        logger.info(f"保存 FAISS 索引...")
        vectorstore.save_local(index_save_path)
        logger.info(f"保存 FAISS 索引完成.")
        logger.info(f"保存 intent 的 index map...")
        
        with open(index_intent_map_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(index_intent_map, json_file, ensure_ascii=False, indent=4)
        logger.info(f"保存 intent 的 index map 完成.")
    return index_intent_map, vectorstore
        

def build_retrievers(retriever_config: Dict, embedding_model, prefix="101_ic_demo_expand"):
    """
    包在索引處理的最外層，各語系逐一建立 FAISS 索引
    """
    
    retriever_map_dict = {}
    index_intent_map_dict = {}
    for language, file_path in retriever_config.items():
        index_intent_map, current_documents = load_json_train_data(file_path)
        logger.info(f"language: {language}, file_path: {file_path}, documents: {len(current_documents)}")
        current_index_intent_map, current_vectorstore = build_faiss_index(index_intent_map, current_documents, language,embedding_model, prefix)
        retriever_map_dict[language] = current_vectorstore
        index_intent_map_dict[language] = current_index_intent_map
    return index_intent_map_dict, retriever_map_dict
        

def load_language_intent_response_mapping(file_path):
    """
    讀取意圖與回覆 Mapping，雅筠提供的 python 檔案 (內有 dict 物件)，為快速同步檔案版本，因此採用直接讀取 python 檔案 dict 方式
    """
    
    logger.info(f"load_language_intent_response_mapping file_paht: {file_path}")
    result_map_dict = {}
    with open(file_path, 'r') as file:
        
        # 讀取文件中的所有内容到一個字符串
        file_content = file.read()
    
        # 使用 exec 而不是 eval，因為我們執行的是代碼賦值，而非僅評估表達式
        exec(file_content, globals())
        result_map_dict['zh-tw'] = dict
        result_map_dict['en-us'] = dict_us
        result_map_dict['ko-kr'] = dict_kr
        result_map_dict['ja-jp'] = dict_jp
    return result_map_dict

embedding_model_config = settings.RAG_INDEX_HF_EMBEDDING_MODEL_CONFIG
hf_embedding_model = HuggingFaceBgeEmbeddings(**embedding_model_config)
retriever_config = settings.RAG_FILES
index_intent_maps, retrievers = build_retrievers(retriever_config, hf_embedding_model, settings.RETRIEVER_INDEX_NAME_PREFIX)

answer_file_path = settings.ANSWER_FILE_PATH

language_intent_response_mapping = load_language_intent_response_mapping(answer_file_path)


first_model_key = list(model_config.keys())[0]
llm_model = model_config[first_model_key].as_instance()
openai_model = ChatOpenAI(model_name=os.getenv("OPENAI_API_MODEL"), openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{{\"id\": \"{intent_index}\",\"example_question\": \"{page_content}\"}}")
def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    # remove the duplicated intents
    new_doc_strings = []
    ids = []
    for doc_string in doc_strings:
        match = re.search(r'"id":\s*"(\d+)"', doc_string)
        if match:
            id_number = match.group(1)
        if id_number not in ids:
            new_doc_strings.append(doc_string)
            ids.append(id_number)
    
    return document_separator.join(new_doc_strings)


def _format_history_question(history_question_list):
    format_history_question_list = [ f"Human: {current}" for current in history_question_list]
    return "\n".join(format_history_question_list)

def _intent_maping_to_answer(language, intent_class):
    print(f"language: {language}, intent_class: {intent_class}")
    return language_intent_response_mapping.get(language).get(intent_class)
    
def _transfer_intent_id_to_name(language, response):
    responses = response.split(',')
    pattern = r'[^0-9]*?(\d+)'
    match = re.search(pattern, responses[0])
    if match:
        return index_intent_maps[language].get(int(match.group(1)), "none")
    else:
        return "none"

def _get_response(language,response):
    responses = response.split(',')

    if ',' in response and 'None' not in response and 'none' not in response:
        return ','.join(responses[1:])
    else:
        return language_intent_response_mapping.get(language).get(_transfer_intent_id_to_name(language,response))[0]

def choose_retriever(language, top_n):
    return retrievers.get(language).as_retriever(search_kwargs={"k": top_n})

def retrieve(language, question, top_n):
    return choose_retriever(language, top_n).invoke(question)

def get_language_name(language):
    language_name_dict = {
        "zh-tw": "Traditional Chinese",
        "ja-jp": "Japanese",
        "en-us": "English",
        "ko-kr": "Korean"
    }
    return language_name_dict.get(language)

taiwan_index = [ i for i in index_intent_maps['zh-tw'] if index_intent_maps['zh-tw'][i] == '台灣中國'][0]
danger_tokens = ["自殺", "炸彈", "手槍", "殺人", "暴力", "恐怖", "毒品", "走私", "洗錢", "販賣人口", "暴亂", "暴動", "戰爭", "恐怖主義", "核武器", "襲擊", "綁架", "騷擾", "槍擊", "暗殺", "縱火", "殺人未遂", "詐騙", "叛國", "叛亂", "酗酒", "強姦", "性侵", "輪姦", "販毒"]


template = """你現在是101智能客服。這是一個基於使用者歷史詢問紀錄和當前問題的分類與回答任務。你的任務是判斷使用者的意圖最符合的類別選項，參考該類別選項的answer生成答案。
如果有遇到任何敏感問題或字眼如 :"""+ str(danger_tokens) + """，直接回覆:"0,我們會記錄您的表達，轉知相關單位處理"。
任何有關政治問題、兩岸問題不要回答，直接回傳""" + str(taiwan_index)+""" ",這是一個複雜而敏感的問題。"。 
直接參考所選類別選項的'answer'並生成口語化且多元的回覆，請勿回傳問句，一定要是能回答問題的回覆。
不能直接引用context的資訊，要做適當的改寫回覆問句。
直接回答使用者的問題，一開始就提到重點。
Output format should be a string containing index_id and answer separated by a comma, "index_id,answer". 
Do not return any other format, do not add any punctuations.
# 規則
1. 仔細閱讀並分析提供的選項。每一個選項包括類別id和範例問句（example_question），範例問句由問句與回答組成。
2. 檢視歷史詢問紀錄(history_question)和當前問題，提取核心詞彙和概念。
3. 根據你對範例問句的理解，選擇一個最符合當前問題的範例問句與選項，考慮其問題相似度與回覆的完整度。
4. 當「類別選項」中沒有適合的問題或答案選項，則直接回傳 "None"。

#歷史詢問記錄
{history_question}

# 類別選項
{context}

# 回答格式
回傳intent_index與生成的答案，前後使用逗號分開。
答案請直接參考給予的的context中的'answer'，不用考慮'question'並產生多樣化的回覆，智能一點。
回覆有多樣性，並且口語話一點的句子。
試著讓語氣溫柔一點。

# 當前互動
當前問題: {question} 
請用{language_name}回答，若為"zh-tw"一定要用繁體中文，習用台灣人習慣的措辭回覆。
有關政治問題、兩岸問題不要回答。
直接參考所選類別選項的'answer'並生成口語化且多元的回覆，請勿回傳問句，一定要是能回答問題的回覆。
The output must include both intent_index and generated answer, and do not contain unusual punctuations.
ANSWER:
"""


rag_prompt = HumanMessagePromptTemplate.from_template(template)
messages = [SystemMessage(content="你現在是101智能客服"), rag_prompt]
prompt = ChatPromptTemplate.from_messages(messages)


transform_inputs = {
    "question": lambda x: x["query"][-1],
    "history_question": lambda x: x["query"][:-1],
    "language": itemgetter("language"),
    "language_name": lambda x: get_language_name(x["language"])
}

retrieved_documents = {
    "context": lambda x: _combine_documents(retrieve(x["language"], x["question"], settings.RETRIEVER_RETURN_TOP_N)),
    "question": itemgetter("question"),
    "history_question": lambda x: _format_history_question(x["history_question"]),
    "language": itemgetter("language"),
    "language_name": itemgetter("language_name")
}

local_predict = {
    "response":  prompt | llm_model | StrOutputParser(),
    "language": itemgetter("language"),
    "language_name": itemgetter("language_name")
}

openai_predict = {
    "response":  prompt | openai_model | StrOutputParser(),
    "language": itemgetter("language"),
    "language_name": itemgetter("language_name")
}

transfer_result = {
    "intent" : lambda x: _transfer_intent_id_to_name(x["language"], x["response"]),
    "response": lambda x: _get_response(x["language"], x["response"]),
    "language": itemgetter("language")
}

output = {
    "intent": itemgetter("intent") ,
    "response": lambda x: x["response"],
}

local_rag_chain = (
    RunnablePassthrough() | transform_inputs
    | retrieved_documents
    | local_predict
    | transfer_result
    | output
)

openai_rag_chain = (
    RunnablePassthrough() | transform_inputs
    | retrieved_documents
    | openai_predict
    | transfer_result
    | output
)

class Input(BaseModel):
    """Chat with the bot."""
    query: List[str]
    language: str = "zh-tw"

class Output(TypedDict):
    status: bool= True
    msg: str = "成功"
    response: str

local_chain = local_rag_chain.with_types(input_type=Input)
openai_chain = openai_rag_chain.with_types(input_type=Input)

app = FastAPI(
    title="RAG API",
    version="0.1",
    description="僅提供 RAG 的 input",
)

add_routes(app, local_chain, path="/local_qa", enable_feedback_endpoint=False)
add_routes(app, openai_chain, path="/openai_qa", enable_feedback_endpoint=False)


# Redirect route from /v1/qa to /va/rag/invoke
@app.post("/v1/local_qa")
async def qa_handler(query: str = Form(...), language: str = Form(...)):
    try:
        # Parse the JSON string from 'query' form data into a Python list
        query_list = json.loads(query)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format in query parameter")

    
    url = 'http://127.0.0.1:8000/local_qa/invoke'  # Local redirection URL
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
                "intent": data["output"]['intent'],
                "response": data["output"]['response']
            }
        else:
            # Handle errors by reformatting the error message
            return {
                "status": False,
                "msg": response.text,  # or a more specific error message
                "response": None
            }
# Redirect route from /v1/qa to /va/rag/invoke
@app.post("/v1/openai_qa")
async def qa_handler(query: str = Form(...), language: str = Form(...)):
    try:
        # Parse the JSON string from 'query' form data into a Python list
        query_list = json.loads(query)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format in query parameter")

    
    url = 'http://127.0.0.1:8000/openai_qa/invoke'  # Local redirection URL
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
                "intent": data["output"]['intent'],
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

