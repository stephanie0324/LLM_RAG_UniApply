from operator import itemgetter


from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage


from config import settings
from my_faiss import retriever
from prompt import RAG_GEN_PROMPT


def _format_history_question(history_question_list):
    format_history_question_list = [
        f"Human: {current}" for current in history_question_list
    ]
    return "\n".join(format_history_question_list)


def choose_retriever(top_n, school, department):
    metadata_filter = {"school": school, "department": department}
    return retriever.as_retriever(search_kwargs={"k": top_n, "filter": metadata_filter})


def retrieve(question, top_n, school, department):
    return choose_retriever(top_n, school, department).invoke(question)


def get_chain(llm_model, school, department):

    rag_prompt = HumanMessagePromptTemplate.from_template(RAG_GEN_PROMPT)
    messages = [
        SystemMessage(
            content="You are an FAQ assistant specializing in university applications."
        ),
        rag_prompt,
    ]
    prompt = ChatPromptTemplate.from_messages(messages)

    full_chain = (
        RunnablePassthrough()
        | {
            "context": lambda x: retrieve(
                x["question"], settings.RETRIEVER_RETURN_TOP_N, school, department
            ),
            "question": itemgetter("question"),
            "chat_history": lambda x: _format_history_question(x["chat_history"]),
            "language": lambda x: x["language"],
        }
        | {
            "response": prompt | llm_model | StrOutputParser(),
            "context": itemgetter("context"),
        }
    )

    return full_chain
