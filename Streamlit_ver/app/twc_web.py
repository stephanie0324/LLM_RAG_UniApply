import os
import tempfile
import json
import pandas as pd
import re
import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.output_parsers import StrOutputParser

from config import settings
from twc_rag import get_chain

os.environ['CURL_CA_BUNDLE'] = ''

retriever_config = settings.RAG_FILES
model_config = settings.MODEL_CONFIG.root
retrieval_doc_visiable = settings.RETRIEVAL_DOC_VISIABLE
is_local = True
for language, file_path in retriever_config.items():
    with open(file_path, 'r', encoding='utf-8') as file:
        faq_json_array = json.load(file)

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

# 定義頁面基本資訊
# 瀏覽器上的 page title 和 圖示
st.set_page_config(page_title="台水智慧客服Demo", page_icon="favicon.ico")
# 當前頁面的 title
st.title("台水智慧客服Demo")

# sidebar 相關物件
with st.sidebar:  
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
        key="select_llm_model",   # 將修改結果保存於 session_state
    )

llm_model_switch = { key: value.asInstance() for key, value in model_config.items() }
llm_model = llm_model_switch.get(select_llm_model)

if llm_model in ["gpt-4o","gpt-3.5-turbo-0125"]:
    is_local = False
chain = get_chain(llm_model, is_local=is_local)

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
  
if len(msgs.messages) == 0 or st.sidebar.button("清除歷史訊息"):
    msgs.clear()
    msgs.add_ai_message("有什麼可以幫您?")

avatars = {"human": "user", "ai": "🤖"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

# st.chat_input 是 chatbot 對話物件
if user_query := st.chat_input(placeholder="輸入您的問題..."):
    st.chat_message("user").write(user_query)

    with st.chat_message("🤖"):        
        full_response = ""
        new_response = ""
        docs = []
        if retrieval_doc_visiable:
            retrieval_handler = st.container()
        response_handler = st.empty()
        # print(msgs.messages)

        no_reply = False
        for response in chain.stream(
            {
                "question": user_query,
                "chat_history": get_messages(msgs.messages, 0)
            }
        ):
            print(f"---------------------------------------------{response}")

            if "response" in response:
                new_response+= response['response']
            elif "context" in response and retrieval_doc_visiable:
                try:
                    docs = response["context"]
                    json_objects = docs.strip().split('\n')
                    dict_list = [json.loads(obj) for obj in json_objects]
                    retrieval_handler_status = retrieval_handler.status("**資料檢索**")
                    retrieval_handler_status.write(f"**Question:** {user_query}")
                    retrieval_handler_status.update(label=f"**資料檢索:** {user_query}")
                    
                    for items in dict_list:
                        retrieval_handler_status.markdown(items["question"])
                        retrieval_handler_status.markdown(items["answer"])
                    retrieval_handler_status.update(state="complete")
                except:
                    pass
        
        try:
            new_response = new_response.strip('{}')
            new_response = new_response.split(',',1)
            id , response = new_response[0], new_response[1]
            id_value = id.split(':',1)[1]
            response_value = response.split(':',1)[1]
            full_response += response_value+"\n"
            full_response += "\n"+id_value if id_value else None
        except:
            full_response = '不好意思，請您在輸入一次問題'
        
        response_handler.markdown(full_response)

    msgs.add_user_message(user_query)
    msgs.add_ai_message(full_response)