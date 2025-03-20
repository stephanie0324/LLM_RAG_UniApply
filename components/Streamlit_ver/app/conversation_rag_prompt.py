from langchain.prompts.prompt import PromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import format_document

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language. 

Chat History: 
{chat_history}

Follow Up Input: {question}
Standalone question:
"""
CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_template(_template)

# template = """Use the following pieces of context to answer the question at the end.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.
# 請確保你的回答完全基於提供的context。
# 如果context沒有提供足夠的信息來回答問題，請直接回答"很抱歉，我目前查不到相關知識"。
# 在生成答案前，仔細思考一下你的答案。

# Context:
# {context}

# QUESTION:{question}
# ANSWER:
# """

template = """#zh-tw 請使用以下context區塊提供的資訊來回答結尾的問題。如果你不知道答案，請直接表示不知道，不要嘗試捏造答案。
確保你的回答完全基於提供的context區塊。在回答前，請仔細思考你的答案。

Context:
{context}

QUESTION:{question}
ANSWER:
"""

rag_human_message_template = HumanMessagePromptTemplate.from_template(template)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        rag_human_message_template
    ]
)