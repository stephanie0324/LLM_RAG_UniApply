from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from utils import _reduce_below_limit_token
from utils import _combine_documents

def get_rag_chain(llm_model, rag_prompt, retriever, max_input_tokens_limit=7000):

    # Now we retrieve the documents
    retrieved_documents = {
        "docs": itemgetter("question") | retriever,
        "question": itemgetter("question")
    }

    # check & reduce ducuments
    check_documents = {
        "docs": lambda x: _reduce_below_limit_token(docs= x["docs"], llm=llm_model, max_tokens_limit=max_input_tokens_limit),
        "question": itemgetter("question"),
    }

    final_inputs = {
        "context": lambda x: _combine_documents(x["docs"]),
        "question": itemgetter("question"),
    }
    # And finally, we do the part that returns the answers
    answer = {
        "question":  itemgetter("question"),
        "answer": final_inputs | rag_prompt | llm_model | StrOutputParser(),
        "docs": itemgetter("docs"),
    }
    # And now we put it all together!
    rag_chain  = RunnablePassthrough() | retrieved_documents | check_documents | answer
    
    
    return rag_chain