from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from typing import List, Tuple
from langchain.prompts.chat import (
    ChatPromptTemplate
)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_self_rag_chains(llm):
    
    template = """#zh-tw Output in Traditional Chinese language.
    Please answer the question below based on the content of the provided document. Follow these steps to ensure a thorough and accurate response:
    1. Read the question carefully and identify the specific plan name or term mentioned.
    2. Search the document for the specific plan name or term to verify its presence.
    3. If the plan name or term exists, reference relevant text from the document, quoting directly and comprehensively.
    4. If the plan name or term does not exist, state clearly that it is not found.
    5. Provide a detailed explanation of why the snippet supports your answer.
    
    Your response should include parts:
    * Answer: Provide a complete answer quoting the document directly and confirming the specific plan name or term. If the plan name or term does not exist, state that it is not found and provide an explanation along with the referenced sections.
    * File Name: The file name(s) where the snippet is found. List all relevant file names.
    * Paragraph Title: The paragraph title(s) where the snippet is located. List all relevant paragraph titles.
    
    Document Content:
    {context}
    
    Question: {question}
    RESPONSE: """

    # Prompt
    prompt = ChatPromptTemplate.from_template(template)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()
    
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n
        If the document contains keywords related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' in english  to indicate whether the document is relevant to the question.. Direct to output result, and no premable or explaination.""",
        input_variables=["question", "document"],
    )

    retrieval_grader = prompt | llm | StrOutputParser()

    prompt = PromptTemplate(
        template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n 
        Here are the facts:
        \n ------- \n
        {documents} 
        \n ------- \n
        Here is the answer: {generation}
         Give a binary score 'yes' or 'no' in english to indicate whether the answer is grounded in / supported by a set of facts.. Direct to output result, no preamble or explanation.""",
        input_variables=["generation", "documents"],
    )

    hallucination_grader = prompt | llm | StrOutputParser()

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing whether an answer is useful to resolve a question. \n 
        Here is the answer:
        \n ------- \n
        {generation} 
        \n ------- \n
        Here is the question: {question}
        Give a binary score 'yes' or 'no' in english  to indicate whether the answer is useful to resolve a question. . Direct to output result, and no preamble or explanation.""",
        input_variables=["generation", "question"],
    )

    answer_grader = prompt | llm | StrOutputParser()

    # Prompt 
    re_write_prompt = PromptTemplate(
        template="""You a question re-writer that converts an input question to a better version that is optimized \n 
        for vectorstore retrieval. Look at the initial and formulate an improved question. \n
        Here is the initial question: \n\n {question}. Improved question in Traditional Chinese language. Direct to output result, and no preamble or explanation. \n """,
        input_variables=["generation", "question"],
    )

    question_rewriter = re_write_prompt | llm | StrOutputParser()


    standalone_question_prompt = ChatPromptTemplate.from_template("""#zh-tw Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. Output in Traditional Chinese language. Direct to output result. No other text or format is acceptable.

Chat History: 
{chat_history}

Follow Up Input: {question}
Standalone question:""")

    def _format_chat_history(chat_history: List[Tuple]) -> str:
        """Format chat history into a string."""
        buffer = ""
        for dialogue_turn in chat_history:
            human = "Human: " + dialogue_turn[0]
            ai = "Assistant: " + dialogue_turn[1]
            buffer += "\n" + "\n".join([human, ai])
        return buffer

    _input = {
        "question": lambda x: x["question"],
        "chat_history": lambda x: _format_chat_history(x["chat_history"])
    }

    standalone_question_chain = _input | standalone_question_prompt | llm | StrOutputParser()
    
    return standalone_question_chain, rag_chain, retrieval_grader, hallucination_grader, answer_grader, question_rewriter