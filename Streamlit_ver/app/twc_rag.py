import json
import os
import re
from operator import itemgetter
from typing import List, Dict

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_core.prompts import PromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain.schema import format_document

from config import settings
import logging
import langchain

os.environ['CURL_CA_BUNDLE'] = ''

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)
langchain.debug = settings.DEBUG

model_config = settings.MODEL_CONFIG.root

# FAISS
def load_json_train_data(file_path: str) -> List[Document]:
    with open(file_path, 'r', encoding='utf-8') as file:
        faq_json_array = json.load(file)
    documents = []
    for record in faq_json_array:
        page_content = json.dumps(record, ensure_ascii=False)
        current_document = Document(
            page_content=page_content,
            metadata={'source': 'FAQ'}
        )
        documents.append(current_document)
    return documents

def build_faiss_index(language, documents: List[Document], embedding_model, prefix="twc_rag_demo_expand_2"):
    index_save_path = f"./index_data/{prefix}_{language}"
    if os.path.exists(index_save_path):
        logger.info(f"Loading FAISS index: {index_save_path}")
        vectorstore = FAISS.load_local(index_save_path, embedding_model, allow_dangerous_deserialization=True)
        logger.info(f"Loaded FAISS index.")
    else:
        logger.info(f"Building FAISS index...")
        vectorstore = FAISS.from_documents(documents, embedding_model)
        logger.info(f"Saving FAISS index...")
        vectorstore.save_local(index_save_path)
        logger.info(f"Saved FAISS index.")
    return vectorstore

def build_retrievers(retriever_config: Dict, embedding_model, prefix="twc_rag_demo_expand_2"):
    retriever_map_dict = {}
    for language, file_path in retriever_config.items():
        current_documents = load_json_train_data(file_path)
        logger.info(f"language: {language}, file_path: {file_path}, documents: {len(current_documents)}")
        current_vectorstore = build_faiss_index(language, current_documents, embedding_model, prefix)
        retriever_map_dict[language] = current_vectorstore
    return retriever_map_dict

# Set Embedding Model
embedding_model_config = settings.RAG_INDEX_HF_EMBEDDING_MODEL_CONFIG
hf_embedding_model = HuggingFaceBgeEmbeddings(**embedding_model_config)
retriever_config = settings.RAG_FILES
retrievers = build_retrievers(retriever_config, hf_embedding_model, prefix=settings.RAG_INDEX_PREFIX)

# Set the Prompt
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
def _combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n", is_local = True):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    # 地端模型要限制prompt token
    if is_local:
        return document_separator.join(doc_strings)[:int(os.getenv("RAG_REDOUCE_BELOW_LIMIT_TOKEN"))]
    else:
        return document_separator.join(doc_strings)


def _format_history_question(history_question_list):
    format_history_question_list = [ f"Human: {current}" for current in history_question_list]
    return "\n".join(format_history_question_list)

def choose_retriever(top_n):
    return retrievers.get("twc").as_retriever(search_kwargs={"k": top_n})

def retrieve(question, top_n):
    return choose_retriever(top_n).invoke(question)

def get_chain(llm_model, is_local = True):


    template = """
    你現在是台水智能客服。 你要針對問題回答與分類他的意圖與主題類別。
    請勿回傳列表，閱讀列表後將內容整理回覆問題。
    用繁體中文，習用台灣人習慣的措辭回覆。
    回傳格式如下，{{"id": 最符合的問答組 index_id,"response": generated ansers}} 不要有額外字詞，只能是這個樣式。
    在context中找尋一組最符合當前問題的資訊並回傳他的id，若皆不符合則回傳None.
    針對當前問題生成回覆，參考context中所有的'answer'並生成口語化且多元的回覆，請勿重複提問內容，一定要是能回答問題的回覆盡，正面回覆問題並補充多一點知識，不要回傳整個列表。

    #歷史詢問記錄
    {history_question}

    # 類別選項
    {context}

    # 回答格式
    回傳格式如下，{{"id": 最符合的問答組 index_id,"response": generated ansers}}  不要有額外字詞，只能是這個樣式。


    # 當前互動
    當前問題: {question} 
    請用繁體中文，習用台灣人習慣的措辭回覆。
    直接參考'answer'並生成口語化且多元的回覆，請勿回傳問句，一定要是能回答問題的回覆。
    回覆不應該包含列表，或是過長的資訊，在產生的時候要注意長度與閱讀容易度。
    回傳格式如下，{{"id": 最符合的問答組 index_id,"response": generated ansers}} 不要有額外字詞，只能是這個樣式。
    ANSWER:
    """

    category_chain = (
        ChatPromptTemplate.from_template(
            """你要針對問題區分類別，只需要回傳類別，不需要回傳原因。
            主題類別如下["政治","客訴","機敏","其他"]
            兩岸議題或是詢問政治立場才可以歸納為政治。
            主題類別可以有多個，回傳一個list包含所選類別，若皆不符合回傳[]。.
            請勿回傳其他字詞，只能從上述選項中選擇。
            # 當前問題
            {question}
            # 回傳格式 
            例如 {{['其他']}}
            """
        )
        | llm_model | StrOutputParser()
    )


    def route(input):
        if "政治" in input["category"]:
            rag_prompt = HumanMessagePromptTemplate.from_template("""你只要回傳: {{"id": "政治","response": "這是一個敏感且複雜的問題。"}}""")
            messages = [SystemMessage(content="你現在是台水智能客服"), rag_prompt]
            prompt = ChatPromptTemplate.from_messages(messages)
            return prompt | llm_model |StrOutputParser()
        elif "機敏" in input["category"]:
            rag_prompt = HumanMessagePromptTemplate.from_template("""你只要回傳: {{"id": "機敏","response": "很抱歉，我將紀錄以上的對話，並轉相關單位做後續處理。"}}""")
            messages = [SystemMessage(content="你現在是台水智能客服"), rag_prompt]
            prompt = ChatPromptTemplate.from_messages(messages)
            return prompt | llm_model |StrOutputParser()
        elif "客訴" in input["category"]:
            rag_prompt = HumanMessagePromptTemplate.from_template("""你只要回傳: {{"id": "客訴","response": "請問您方便提供具體情況嗎（發生的詳細地址、時間、現像等）？ 我們給您記錄，方便我們盡快查詢處理，感謝您的配合！"}}""")
            messages = [SystemMessage(content="你現在是台水智能客服"), rag_prompt]
            prompt = ChatPromptTemplate.from_messages(messages)
            return  prompt | llm_model |StrOutputParser()
        else:
            rag_prompt = HumanMessagePromptTemplate.from_template(template)
            messages = [SystemMessage(content="你現在是台水智能客服"), rag_prompt]
            prompt = ChatPromptTemplate.from_messages(messages)
            return prompt | llm_model | StrOutputParser() 
    
    transform_inputs = {
        "question": lambda x: x["question"],
        "history_question": lambda x: x["chat_history"],
    }

    retrieved_documents = {
        "context": lambda x: _combine_documents(retrieve(x["question"], settings.RETRIEVER_RETURN_TOP_N), is_local=is_local),
        "question": itemgetter("question"),
        "history_question": lambda x: _format_history_question(x["history_question"]),
    }

    full_chain = (
        RunnablePassthrough() | transform_inputs 
        | retrieved_documents
        |{
            "category": category_chain,
            "question": itemgetter("question"),
            "history_question": itemgetter("history_question"),
            "context": itemgetter("context")
        }
        | {
            "response":RunnableLambda(route),
            "context": itemgetter("context")
        }
    )

    return full_chain