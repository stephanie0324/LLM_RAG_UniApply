from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import format_document
from langchain.load import dumps, loads

from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from utils import _reduce_below_limit_token
from typing import List, Tuple
import re

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    # doc_strings = [format_document(doc, document_prompt) for doc in docs]
    doc_strings = [ f"文章片段:{doc_id}" + format_document(doc, document_prompt) for doc_id, doc in enumerate(docs)]
    result_string = "未檢索到相關資料"
    if len(doc_strings) > 0:
        result_string = document_separator.join(doc_strings)
    
    return result_string


rag_human_template = """#zh-tw 你的答案必須基於以下提供的相關文章資訊來回答使用者問題，並且找出答案來源的段落標題。
如果你不知道答案，請直接表示不知道，不要嘗試捏造答案。
請確認你的回答準確且詳細，請詳細比較你的答案和使用者問題的不同意義和應用，不淂省略特殊情況資訊。

相關文章資訊:
{context}

RESPONSE FORMAT:
```
回答內容: <your_answer>\n
文件名稱: <file_name>\n
文件段落：<答案的段落標題>\n
```

段落標題分析:
1. Identify the answer snippet in the document.
2. Analyze the content surrounding the snippet to understand its context and main topics.
3. Determine the type of information the snippet relates to, such as policy details, operational procedures, terms and conditions, etc.
4. Using this understanding, infer the most likely paragraph title that the snippet would fall under in the document structure.

使用者問題:{question}
RESPONSE:
"""

rag_human_message_template = HumanMessagePromptTemplate.from_template(rag_human_template)


rag_prompt = ChatPromptTemplate.from_messages(
    [
        rag_human_message_template
    ]
)


def _format_chat_history(chat_history: List[Tuple]) -> str:
    """Format chat history into a string."""
    buffer = ""
    for dialogue_turn in chat_history:
        human = "Human: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    return buffer



def get_rag_chain(llm_model, retriever, prompt=rag_prompt, max_input_tokens_limit=8000):
#     _template = """#zh-tw 請根據下方對話紀錄和後續問題重新改寫成可以獨立搜尋的問題，需要保持和後續問題同樣語言。直接輸出結果。不用描述說明或其他任何格式。

# 對話紀錄:
# {chat_history}

# 後續問題：{question}
# 獨立問題：<只輸出一個最佳獨立問題>
# """
    _template = """#zh-tw Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. Output in Traditional Chinese language. Direct to output result. No other text or format is acceptable.

Chat History: 
{chat_history}

Follow Up Input: {question}
Standalone question:
"""

    CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_template(_template)    
    
    _split_question_templte="""請仔細評估使用者問題，並將問題分解成多個簡單且易於查詢的3個不同版本的繁體中文問題，並只用一個換行符分隔，請不要輸出其他描述、資訊或補充文字。
    Direct to output result, and no preamble or explanation.

使用者問題：{question}"""

    QUESTION_SPLIT_PROMPT = ChatPromptTemplate.from_template(_split_question_templte)    
    
    def condense_question_route(input):
        if len(input["chat_history"]) > 0:
            print(input)
            
            return RunnablePassthrough.assign(
                chat_history=lambda x: _format_chat_history(input["chat_history"])
            ) | CONDENSE_QUESTION_PROMPT | llm_model | StrOutputParser()
        else: 
            print(input["question"] )
            return input["question"] 
        
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

    condense_question = {
        "question": RunnableLambda(condense_question_route)
    }

    #  | QUESTION_SPLIT_PROMPT | llm_model | retriever.map() | reciprocal_rank_fusion

    generate_queries = ( QUESTION_SPLIT_PROMPT | llm_model | StrOutputParser() | (lambda x: re.split(r'\n+', x)))

    # Now we retrieve the documents
    retrieved_documents = {
        # "docs": itemgetter("question") | retriever,
        "docs": itemgetter("question") | generate_queries | retriever.map() | reciprocal_rank_fusion,
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
        "answer": final_inputs | prompt | llm_model | StrOutputParser(),
        "docs": itemgetter("docs"),
    }
    # And now we put it all together!
    rag_chain  = RunnablePassthrough() | condense_question | retrieved_documents | check_documents | answer
    
    
    return rag_chain