import json

import streamlit as st
import langchain
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

from rag import get_chain
from config import settings
from my_faiss import school_dict

# Setup configurations
model_config = settings.MODEL_CONFIG.root
langchain.debug = settings.DEBUG


# Function to process chat history for the LLM
def get_messages(msgs, max_pairs):
    """
    Process and return the most recent chat pairs.
    """
    if max_pairs == 0:
        return []

    n = len(msgs)
    if n > 1:
        # Start from index 1 to ensure the first message pair is processed
        start_index = 1
        last_elements = msgs[start_index:]

        # Create a pair of (user, AI) messages
        result = [
            (human.content, ai.content)
            for human, ai in zip(last_elements[::2], last_elements[1::2])
        ]
        result = result[-max_pairs:]  # Limit to the max number of pairs
    else:
        result = []

    return result


# Set the page title and config
st.set_page_config(page_title="Uni Apply (.◜◡◝)", page_icon="./img/UniApplyLogo.png")
st.title("Uni Apply (.◜◡◝)")

# Sidebar Configuration
with st.sidebar:

    def update_session_state(key: str, label: str):
        """Function to update session state with user selection."""
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

    # Language selection
    language_options = ["Traditional Chinese", "English", "Japanese"]
    select_language = st.selectbox(
        "Language Selection:",
        language_options,
        index=0,  # Default to Traditional Chinese
        key="select_language",
        on_change=lambda: update_session_state("select_language", "Language"),
    )

# Initialize memory for contextual conversation
msgs = StreamlitChatMessageHistory()

# Clear chat history if button pressed or if no messages
if len(msgs.messages) == 0 or st.sidebar.button("Clear History Chat"):
    msgs.clear()
    msgs.add_ai_message("How may I assist you?")

# Display conversation history
avatars = {"human": "user", "ai": "🤖"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

# Handle user query input
if user_query := st.chat_input(placeholder="Enter your question..."):
    st.chat_message("user").write(user_query)

    with st.chat_message("🤖"):
        full_response = ""
        new_response = ""
        docs = []

        # Display retrieved docs if enabled
        if settings.RETRIEVAL_DOC_VISIABLE:
            retrieval_handler = st.container()

        response_handler = st.empty()
        llm_model = model_config[select_llm_model].as_instance()
        chain = get_chain(llm_model, select_school, select_department)

        try:
            for response in chain.stream(
                {
                    "question": user_query,
                    "chat_history": get_messages(msgs.messages, 0),
                    "language": select_language,
                }
            ):

                if "response" in response:
                    new_response += response["response"]
                elif "context" in response and settings.RETRIEVAL_DOC_VISIABLE:
                    try:

                        retrieval_handler_status = retrieval_handler.status(
                            "**Retrieved Docs**"
                        )
                        retrieval_handler_status.update(
                            label=f"**Retrieved Docs:** {user_query}"
                        )

                        # Display retrieved documents
                        for idx, doc in enumerate(response["context"]):
                            content = json.loads(doc.page_content)
                            retrieval_handler_status.markdown(f"Doc No. {idx}")
                            retrieval_handler_status.markdown(f"**Q:** {content['Q']}")
                            retrieval_handler_status.markdown(f"**A:** {content['A']}")
                            retrieval_handler_status.markdown("---")

                        retrieval_handler_status.update(state="complete")
                    except Exception as e:
                        st.error(f"Error processing context: {e}")
                        pass

            # Format response
            new_response = new_response.strip("{}").split(",", 1)
            id, response = new_response[0], new_response[1]
            id_value = id.split(":", 1)[1]

            full_response += "Related Doc No. : " + id_value + "\n"

            response_value = response.split(":", 1)[1]
            split_response = response_value.replace('"', "").split("<br>")
            for r in split_response:
                full_response += "\n" + r + "\n"

        except Exception as e:
            full_response = (
                f"Sorry, could you please input the question again? 😊🙏 Error: {e}"
            )

        response_handler.markdown(full_response)

    # Save the conversation in memory
    msgs.add_user_message(user_query)
    msgs.add_ai_message(full_response)
