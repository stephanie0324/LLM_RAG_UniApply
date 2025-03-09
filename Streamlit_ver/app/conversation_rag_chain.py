from operator import itemgetter
from typing import List, Tuple
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.prompts import format_document
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from conversation_rag_prompt import CONDENSE_QUESTION_PROMPT, ANSWER_PROMPT
from utils import _reduce_below_limit_token
from langchain.load import dumps, loads

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _format_chat_history(chat_history: List[Tuple]) -> str:
    """Format chat history into a string."""
    buffer = ""
    for dialogue_turn in chat_history:
        human = "Human: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    return buffer

def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    
    result_string = "NO_RETRIEVAL_RESULT"
    if len(doc_strings) > 0:
        result_string = document_separator.join(doc_strings)
    
    return result_string

def reciprocal_rank_fusion(results: List[list], k=60):
    fused_scores = {}
    for docs in results:
        # Assumes the docs are returned in sorted order of relevance
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            previous_score = fused_scores[doc_str]
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        loads(doc)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results

def get_conversation_rag_chain(llm_model, retriever, max_input_tokens_limit=7000):
    # standalone_question = {
    #     "standalone_question": {
    #         "question": lambda x: x["question"],
    #         "chat_history": lambda x: get_buffer_string(x["chat_history"]),
    #     }
    #     | CONDENSE_QUESTION_PROMPT
    #     | llm_model
    #     | StrOutputParser(),
    # }
    
    _inputs = RunnableParallel(
        standalone_question=RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | llm_model
        | StrOutputParser(),
        orignal_question=itemgetter("question")
    )

    retrieved_documents = {
        "new_question_docs": itemgetter("standalone_question") | retriever,
        "original_question_docs": itemgetter("orignal_question") | retriever,
        "question": lambda x: x["standalone_question"],
        "orignal_question": lambda x: x["orignal_question"],
    }
    
    combine_documents = {
        "docs": lambda x: _reduce_below_limit_token(docs=reciprocal_rank_fusion([x["new_question_docs"],x["original_question_docs"]],5), llm=llm_model, max_tokens_limit=max_input_tokens_limit),
        "question": lambda x: x["question"],
        "orignal_question": lambda x: x["orignal_question"],
    }
    
    # Now we construct the inputs for the final prompt
    final_inputs = {
        "context": lambda x: _combine_documents(x["docs"]),
        "question": itemgetter("question"),
    }
    # And finally, we do the part that returns the answers
    answer = {
        "question":  itemgetter("question"),
        "orignal_question": itemgetter("orignal_question"),
        "answer": final_inputs | ANSWER_PROMPT | llm_model| StrOutputParser(),
        "docs": itemgetter("docs"),
    }
    # And now we put it all together!
    conversation_rag_chain = _inputs | retrieved_documents | combine_documents | answer
    
    return conversation_rag_chain