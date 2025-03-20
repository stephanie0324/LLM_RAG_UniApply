import os
import json
import tempfile
import streamlit as st
from code_editor import code_editor
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, ChatMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain.memory import ConversationBufferWindowMemory

from operator import itemgetter

from chat_prompt import chat_prompt

from base_rag_chain import get_rag_chain
from conversation_rag_chain import get_conversation_rag_chain
from chat_chain import get_chat_chain
from self_rag_graph import get_self_rag_graph
from kgi_rag import get_rag_chain as get_kgi_rag
from kgi_rag_v1c import get_rag_chain as get_kgi_rag_v1c

from retriever.itri_retriever import ITRIRetriever

from config import settings
from itri_api import ITRI_API
from prompts import prompts

import langchain
langchain.debug = settings.DEBUG


model_config = settings.MODEL_CONFIG.root
# retrieval_config = settings.RETRIEVER_CONFIG.root
retrieval_doc_visiable = settings.RETRIEVAL_DOC_VISIABLE
rag_redouce_below_limit_token = settings.RAG_REDOUCE_BELOW_LIMIT_TOKEN

default_chat_mode = settings.DEFAULT_CHAT_MODE
default_chat_history_windows = settings.DEFAULT_CHAT_HISTORY_WINDOWS
# chat_history_windows = settings.CHAT_HISTORY_WINDOWS


def get_messages(msgs, max_pairs):
    """
    處理傳入給 chain 的 chat_hostory
    """
    
    if max_pairs == 0:
        return []
    
    # 确定 msgs 的长度
    n = len(msgs)

    # 计算开始索引，确保我们至少有 required_min_length 个元素，否则从 1 开始跳过第一个 ai
    start_index = 1

    # 如果列表中有足够的元素来形成至少一对 (human, ai)
    if n > 1:
        # 取相应的元素进行处理
        last_elements = msgs[start_index:]

        # 使用 zip 生成 (human, ai) 元组，确保只取最后 max_pairs 对
        # 这里修改了 zip 的顺序，将 ai, human 改为 human, ai
        result = [(human.content, ai.content) for human, ai in zip(last_elements[::2], last_elements[1::2])]
        result = result[-max_pairs:]
    else:
        # 如果没有足够的元素进行配对，不执行任何操作
        result = []

    # print(f"msgs: {msgs}")
    # print(f"result: {result}")
    # 显示结果或进行其他操作
    return result

def get_retrieval_config(api: ITRI_API, retrieval_top_n=1):
    """
    取得當前所有檢索來源 (知識庫)
    """
    data_type_list = api.get_data_type_list()
    retrieval_config = {}
    for data_type in data_type_list:
        data_type_name = data_type["data_type_name"]
        data_type_id = data_type["data_type_id"]
        retrieval_config[data_type_name] = ITRIRetriever(
            url=f"{settings.ITRI_KM_CONFIG['admin_api_base_url']}/v1/knowledge/search",
            data_type_ids=[data_type_id],
            threshold=0.80,
            limit=retrieval_top_n,
            auth=True,
            api=api
        )     
    return retrieval_config


def get_chains(chat_mode, llm_model, retriever, prompts, rag_document_tokens_limit=rag_redouce_below_limit_token):
    """
    基於選取的 chat_mode 來讀取最新 prompts 並產生放入到chat_mode對應的 chain
    """
    def _build_message_prompt(system_message, user_prompt):
        messages = []
        if system_message:
            messages.append(SystemMessage(content=system_message))
        
        messages.append(HumanMessagePromptTemplate.from_template(user_prompt))

        full_prompts = ChatPromptTemplate.from_messages(messages)
        
        return full_prompts
    
    if chat_mode == "Chat":
        chat_chain = get_chat_chain(llm_model, chat_prompt)
        return chat_chain
    elif chat_mode == "RAG":
        rag_prompt = _build_message_prompt("", prompts['RAG']['generate'])
        return get_rag_chain(llm_model, rag_prompt, retriever)
    elif chat_mode == "RAG_Conversation":
        return get_conversation_rag_chain(llm_model, retriever)
    elif chat_mode == "Self-RAG":
        return get_self_rag_graph(llm_model, retriever)
    elif chat_mode =="KGI POC":
        kgi_prompt = _build_message_prompt(prompts['KGI POC']['system'], prompts['KGI POC']['generate'])
        return get_kgi_rag(llm_model, retriever, kgi_prompt, max_input_tokens_limit=rag_document_tokens_limit)
    elif chat_mode =="KGI POC v1c":
        kgi_prompt = _build_message_prompt(prompts['KGI POC v1c']['system'], prompts['KGI POC v1c']['generate'])
        return get_kgi_rag_v1c(llm_model, retriever, kgi_prompt)
    elif chat_mode =="Prompt Test":
        prompte_test_prompt = ChatPromptTemplate.from_template(prompts['Prompt Test']['generate'])
        return get_chat_chain(llm_model, prompte_test_prompt)
    elif chat_mode =="Perplexity Test":
        prompte_test_prompt = _build_message_prompt(prompts['Perplexity Test']['system'], prompts['Perplexity Test']['generate'])
        return get_rag_chain(llm_model, prompte_test_prompt,retriever)
    else:
        st.info("不支援的模式")
        st.stop()
    



def chat_web(api: ITRI_API):
    title = f"""工研院資通所➰場域諮詢展示"""
    st.title(title)
    description = f""""""
    st.caption(description, unsafe_allow_html=True)
    
    """
    以下是處理當前 session 使用的 prompts
    """
    if "current_prompts" in st.session_state :
        current_prompts = st.session_state["current_prompts"]
    else:
        current_prompts = prompts
    
    current_prompts = prompts
    with st.expander("Prompt 修改"):
        st.caption("<ul><li>於修改 prompt 區塊修改，並點擊 save 後。</li><li>可於目前套用的 prompt區塊查看當前生效的 prompt</li><li>頁面重新整理或切換至知識庫都需要重新設定</li></ul>", unsafe_allow_html=True)
        
        st.write("原始 prompt:")
        st.json(prompts, expanded=False)
        st.write("修改 promt (請點擊旁邊的 save 圖示來保存):")
        response_dict = code_editor(
            json.dumps(
                current_prompts, ensure_ascii=False, indent=4
            ), lang="json", height=[10, 20], buttons=[
                {
                    "name": "Save",
                    "feather": "Save",
                    "hasText": True,
                    "commands": [
                    "save-state",
                    [
                        "response",
                        "saved"
                    ]
                    ],
                    "response": "saved",
                    "style": {
                    "bottom": "calc(50% - 4.25rem)",
                    "right": "0.4rem"
                    }
                }
        ])
        if response_dict["text"]:
            current_prompts = json.loads(response_dict["text"])
            st.session_state["current_prompts"] = current_prompts
        st.write("目前套用的 prompt:")
        st.json(current_prompts, expanded=False)
    """
    以上是處理當前 session 使用的 prompts 設定
    """

    retrieval_config = get_retrieval_config(api)

    with st.sidebar:
        def on_chat_mode_change():
            msgs.clear()
            chat_mode = st.session_state.chat_mode
            text = f"已切換到 {chat_mode} 模式"
            st.toast(text)
        
        chat_modes = [
            "RAG",
            "RAG_Conversation",
            "KGI POC",
            "KGI POC v1c",
            "Self-RAG",
            "Chat",
            "Prompt Test",
            "Perplexity Test",
        ]
        
        chat_mode = st.selectbox("請選擇對話模式",
            chat_modes,
            index=default_chat_mode,
            on_change=on_chat_mode_change,
            key="chat_mode",
        )
        
        def on_retrieval_change():
            msgs.clear()
            retrieval = st.session_state.retrieval_source
            text = f"已切換到 {retrieval} 知識庫"
            st.toast(text)
        
        retrieval_sources = retrieval_config.keys()
        
        retrieval_source = st.selectbox("請選擇 RAG 的知識庫：",
            retrieval_sources,
            index=0,
            on_change=on_retrieval_change,
            key="retrieval_source",
            disabled=(chat_mode not in ["RAG", "RAG_Conversation", "KGI POC", "Self-RAG", "Perplexity Test"])
        )
        
        chat_history_windows = st.number_input(
            "歷史對話輪數：", 
            0, 
            4, 
            default_chat_history_windows,
            disabled=(chat_mode in ["RAG"]))
        
        retrieval_top_n = st.number_input(
            "檢索回傳文件數", 
            1, 
            4, 
            4,
            disabled=(chat_mode not in ["RAG", "RAG_Conversation", "KGI POC", "Self-RAG", "Perplexity Test"])
        )
        
        rag_document_tokens_limit = st.slider("RAG 文件的 Tokens 上限", 2048, 8192, rag_redouce_below_limit_token, 128)
        
        def on_model_change():
            msgs.clear()
            model = st.session_state.select_llm_model
            text = f"已切換到 {model} 模型"
            st.toast(text)
        
        select_llm_models = model_config.keys()
        
        select_llm_model = st.selectbox("請選擇模型來源：",
            select_llm_models,
            index=0,
            on_change=on_model_change,
            key="select_llm_model",
        )
        
        temperature = st.slider("Temperature：", 0.0, 0.7, 0.0, 0.05)
        

    llm_model = model_config.get(select_llm_model).asInstance()
    llm_model.temperature = temperature
    retriever = retrieval_config.get(retrieval_source)
    retriever.limit = retrieval_top_n
    

    # Setup memory for contextual conversation
    msgs = StreamlitChatMessageHistory()
    # memory = ConversationBufferMemory(return_messages=True, output_key="answer", input_key="question", chat_memory=msgs)

    if len(msgs.messages) == 0 or st.sidebar.button("清除歷史訊息"):
        msgs.clear()
        # api_usage_messages.clear()
        msgs.add_ai_message("有什麼可以幫您?")

    avatars = {"human": "user", "ai": "🤖"}
    for msg in msgs.messages:
        st.chat_message(avatars[msg.type]).write(msg.content)

    if user_query := st.chat_input(placeholder="輸入您的問題..."):
        st.chat_message("user").write(user_query) 
        

        with st.chat_message("🤖"):        
            full_response = ""
            docs = []
            if retrieval_doc_visiable:
                retrieval_handler = st.container()
            response_handler = st.empty()
            chain = get_chains(chat_mode, llm_model, retriever, current_prompts, rag_document_tokens_limit=rag_document_tokens_limit)
            
            if chat_mode in ["Self-RAG"]:
                response = chain.invoke(
                    {
                        "question": user_query,
                        "chat_history": get_messages(msgs.messages, chat_history_windows)
                    }
                )
                full_response = response["generation"]
            else:

                if chat_mode in ["RAG", "RAG_Conversation", "KGI POC", "Perplexity Test"] and retrieval_doc_visiable:
                    retrieval_handler_status = retrieval_handler.status("**資料檢索**")
                
                
                # print(msgs.messages)
                for response in chain.stream(
                    {
                        "question": user_query,
                        "chat_history": get_messages(msgs.messages, chat_history_windows)
                    }
                ):
                    # print(f"---------------------------------------------{response}")
                    if "answer" in response:
                        full_response += response["answer"]
                        response_handler.markdown(full_response + "▌")
                    elif "docs" in response and chat_mode in ["RAG", "RAG_Conversation", "KGI POC", "Perplexity Test"] and retrieval_doc_visiable:
                        docs = response["docs"]
                        retrieval_handler_status.write(f"**Question:** {user_query}")
                        retrieval_handler_status.update(label=f"**資料檢索:** {user_query}")
                        
                        for idx, doc in enumerate(docs):
                            source = os.path.basename(doc.metadata["source"])
                            retrieval_handler_status.write(f"**命中文件片段 {idx} 於 {source}**")
                            retrieval_handler_status.markdown(doc.page_content)
                        retrieval_handler_status.update(state="complete")

            response_handler.markdown(full_response)

        msgs.add_user_message(user_query)
        msgs.add_ai_message(full_response)
    

