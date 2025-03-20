from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from typing import List, Tuple

def _format_chat_history(chat_history: List[Tuple]) -> str:
    """Format chat history into a string."""
    buffer = ""
    for dialogue_turn in chat_history:
        human = "Human: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    return buffer

def get_chat_chain(llm_model, chat_prompt):
    chat_input = {
        "chat_history": lambda x: _format_chat_history(x["chat_history"]),
        "question":  itemgetter("question")
    }
    chat_answer = {
        "question":  itemgetter("question"),
        "answer": chat_prompt | llm_model | StrOutputParser()
    }
    chat_chain = RunnablePassthrough() | chat_input | chat_answer
    
    return chat_chain