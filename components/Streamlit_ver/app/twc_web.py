import json
import streamlit as st

import langchain
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

from twc_rag import get_chain
from config import settings
from my_faiss import school_dict

model_config = settings.MODEL_CONFIG.root
langchain.debug = settings.DEBUG


def get_messages(msgs, max_pairs):
    """
    處理傳入給 chain 的 chat_hostory
    """

    if max_pairs == 0:
        return []

    n = len(msgs)

    start_index = 1

    # there must be a least one pair of ai and usr conversation
    if n > 1:
        last_elements = msgs[start_index:]

        # use zip to keep iut as a (user, ai) pair
        # change the order to (ai, user)
        result = [
            (human.content, ai.content)
            for human, ai in zip(last_elements[::2], last_elements[1::2])
        ]
        result = result[-max_pairs:]
    else:
        result = []
    return result


# Set the page title
st.set_page_config(page_title="Uni Apply (.◜◡◝)", page_icon="favicon.ico")
st.title("Uni Apply (.◜◡◝)")

# sidebar

with st.sidebar:
    # Generic function to update selection state
    def update_session_state(key: str, label: str):
        value = st.session_state.get(key, "")
        st.toast(f"Changed to {value}")

    # LLM model selection
    select_llm_model = st.selectbox(
        "Model Selection:",
        model_config.keys(),
        index=0,
        key="select_llm_model",
        on_change=lambda: update_session_state("select_llm_model", "Model"),
    )

    # School selection
    select_school = st.selectbox(
        "School Selection:",
        school_dict.keys(),
        index=0,
        key="select_school",
        on_change=lambda: update_session_state("select_school", "School"),
    )

    # Retrieve available departments based on selected school
    available_departments = school_dict.get(st.session_state.select_school, [])

    # Department selection
    select_department = st.selectbox(
        "Department Selection:",
        available_departments,
        index=0 if available_departments else None,
        key="select_department",
        on_change=lambda: update_session_state("select_department", "Department"),
    )

llm_model = model_config[select_llm_model].as_instance()
chain = get_chain(llm_model, is_local=False)

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()

if len(msgs.messages) == 0 or st.sidebar.button("Clear History Chat"):
    msgs.clear()
    msgs.add_ai_message("How may I assist you?")

avatars = {"human": "user", "ai": "🤖"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

# st.chat_input 是 chatbot 對話物件
if user_query := st.chat_input(placeholder="Enter your question..."):
    st.chat_message("user").write(user_query)

    with st.chat_message("🤖"):
        full_response = ""
        new_response = ""
        docs = []
        if settings.RETRIEVAL_DOC_VISIABLE:
            retrieval_handler = st.container()
        response_handler = st.empty()

        no_reply = False
        for response in chain.stream(
            {"question": user_query, "chat_history": get_messages(msgs.messages, 0)}
        ):

            if "response" in response:
                new_response += response["response"]
            elif "context" in response and settings.RETRIEVAL_DOC_VISIABLE:
                try:
                    docs = response["context"]
                    json_objects = docs.strip().split("\n")
                    dict_list = [json.loads(obj) for obj in json_objects]
                    retrieval_handler_status = retrieval_handler.status(
                        "**Retrieved Docs**"
                    )
                    retrieval_handler_status.write(f"**Question:** {user_query}")
                    retrieval_handler_status.update(
                        label=f"**Retrieved Docs:** {user_query}"
                    )

                    # print(f"dict_list : {dict_list}")

                    for idx in range(len(dict_list)):
                        # Formatting the question and answer with Markdown
                        retrieval_handler_status.markdown(f"Doc No. {idx}")
                        retrieval_handler_status.markdown(
                            f"**Q:** {dict_list[idx]['Q']}"
                        )
                        retrieval_handler_status.markdown(
                            f"**A:** {dict_list[idx]['A']}"
                        )
                        retrieval_handler_status.markdown("---")
                    retrieval_handler_status.update(state="complete")
                except:
                    pass

        try:
            new_response = new_response.strip("{}")
            new_response = new_response.split(",", 1)
            id, response = new_response[0], new_response[1]
            id_value = id.split(":", 1)[1]
            response_value = response.split(":", 1)[1]
            full_response += response_value + "\n"
            full_response += "\n" + id_value if id_value else None
        except:
            full_response = "Sorry, could you please input the question again? 😊🙏"

        response_handler.markdown(full_response)

    msgs.add_user_message(user_query)
    msgs.add_ai_message(full_response)
