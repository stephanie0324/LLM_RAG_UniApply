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
import logging

logger = logging.getLogger(__name__)

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
    _template = """Please rephrase each follow-up question in the following conversation to be standalone questions only if it is confirmed as a follow-up question. Ensure that each rephrased standalone question retains the original intent and includes all keywords, especially ensuring no words are missing. Output in Traditional Chinese language. Direct to output result. No other text or format is acceptable.

Chat History: 
{chat_history}

Follow Up Input: {question}
Standalone question:
"""

    CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_template(_template)    
    
    _split_question_route_templte="""請你仔細思考最下方的 QUESTION 中是否包含多個保單名稱(例如：全面保、健康樂活、新樂活)，若包含多個保單名稱則回傳yes，若只包含一個或沒有包含則回傳no。
Give a binary score 'yes' or 'no' in english to indicate whether the question contains multiple policy names(e.g. 全面保、健康樂活、新樂活). Direct to output result, and no premable or explaination.

範例：
QUESTION: 請問全面保專案得附加意外傷害一至六級傷害失能補償保險附加條款(*PAED)之規則?
OUTPUT: no
QUESTION: 我投保了全面保專案，可以每個月交付保費嗎?
OUTPUT: no
QUESTION: 最近我把金好專案解約了 可以投保全面保嗎?
OUTPUT: yes
QUESTION: 台北富邦銀行台幣會天天扣款嗎?
OUTPUT: no
QUESTION: 請問MAHUAB繳費期間可以15年嗎?
OUTPUT: no
QUESTION: 請比較健康樂活與新樂活商品繳費期間的差異
OUTPUT: yes
QUESTION: 被保人14歲死亡有何相關規定
OUTPUT: no
QUESTION: 新樂活買計畫7，每次門診手術費用可獲得多少給付金？
OUTPUT: no
QUESTION: 華南銀行台幣帳戶如果我在4/10提出媒體申請，會幾號進行帳戶扣款?
OUTPUT: no
QUESTION: 16歲BDDR保額最高是多少?
OUTPUT: no


QUESTION:{question}
OUTPUT:"""

    QUESTION_SPLIT_ROUTE_PROMPT = ChatPromptTemplate.from_template(_split_question_route_templte)   
    
    question_split_checker = QUESTION_SPLIT_ROUTE_PROMPT | llm_model | StrOutputParser()
    
    _split_question_templte="""請仔細評估使用者問題，並將問題分解成多個簡單且易於查詢的3個不同版本的繁體中文問題，並只用一個換行符分隔，請不要輸出其他描述、資訊或補充文字。
    Direct to output result, and no preamble or explanation.

使用者問題：{question}"""

    _more_example = """以下為範例二:\n\
Question: 全面保專案得附加意外傷害一至六級傷害失能補償保險附加條款(*PAED)之規則?\n\
Document Content:\n\
命中文件片段 0 於 全面保專案_投保規則_20200228-001.pdf\n\
    'file_name': '全面保專案_投保規則_20200228.pdf', 'paragraph': '(5) 【投保限制】：A. 每一被保險人限投保本專案共一張。(已承保過「共好專案」、「新共好專案」、「金好專案」、 「真好專案」、「新真好專案」、「幸福專案」及「鑫真好專案」者，不得再投保本專案；惟未承保、失效、解約、契撤案件不受此限。)B. 要保書上須註明專案名稱或專案代碼CH005；本專案不可和其他主契約商品填寫在同一份要保書 及送金單(如採後印式送金單則不需檢附)。C. 本專案不受理主契約被保險人之配偶及子女附加投保。D. 每一保單之被保險人須投保金好意保險/鑫好意保險/龍滿意保險/元氣御守終身保險/鑫滿意保險/*PAA/*IPA/*VPA 且該保單傷害險保額(含金好意保險/鑫好意保險/龍滿意保險/元氣御守終身保險/鑫滿意保險)須為100 萬(含)以上後，始得附加意外傷害一至六級傷害失能補償保險附 加條款(*PAED)。'\n\
\n\
RESPONSE:\n\
    Answer: 全面保專案得附加意外傷害一至六級傷害失能補償保險附加條款(*PAED)，但須符合規則：每一保單之被保險人須投保金好意保險/鑫好意保險/龍滿意保險/元氣御守終身保險/鑫滿意保險/*PAA/*IPA/*VPA 且該保單傷害險保額(含金好意保險/鑫好意保險/龍滿意保險/元氣御守終身保險/鑫滿意保險)須為100 萬(含)以上後，始得附加意外傷害一至六級傷害失能補償保險附加條款(*PAED)。\n\
    Answer Snippet: 每一保單之被保險人須投保金好意保險/鑫好意保險/龍滿意保險/元氣御守終身保險/鑫滿意保險/*PAA/*IPA/*VPA 且該保單傷害險保額(含金好意保險/鑫好意保險/龍滿意保險/元氣御守終身保險/鑫滿意保險)須為100 萬(含)以上後，始得附加意外傷害一至六級傷害失能補償保險附 加條款(*PAED)。\n\
    File Name: 全面保專案_投保規則_20200228.pdf\n\
    Paragraph Title: 投保限制\n\
    """

    _more_example2 = """以下為範例二:\n\
Question:最近我把金好專案解約了 可以投保全面保嗎?\n\
Document Content:\n\
命中文件片段 0 於 全面保專案_投保規則_20200228-001.pdf\n\
    'file_name': '全面保專案_投保規則_20200228.pdf', 'paragraph': '(5) 【投保限制】：A. 每一被保險人限投保本專案共一張。(已承保過「共好專案」、「新共好專案」、「金好專案」、 「真好專案」、「新真好專案」、「幸福專案」及「鑫真好專案」者，不得再投保本專案；惟未承保、失效、解約、契撤案件不受此限。)B. 要保書上須註明專案名稱或專案代碼CH005；本專案不可和其他主契約商品填寫在同一份要保書 及送金單(如採後印式送金單則不需檢附)。C. 本專案不受理主契約被保險人之配偶及子女附加投保。D. 每一保單之被保險人須投保金好意保險/鑫好意保險/龍滿意保險/元氣御守終身保險/鑫滿意保險/*PAA/*IPA/*VPA 且該保單傷害險保額(含金好意保險/鑫好意保險/龍滿意保險/元氣御守終身保險/鑫滿意保險)須為100 萬(含)以上後，始得附加意外傷害一至六級傷害失能補償保險附 加條款(*PAED)。'\n\
\n\
RESPONSE:\n\
    Answer: 可以。已承保過「共好專案」、「新共好專案」、「金好專案」、「真好專案」、「新真好專案」、「幸福專案」及「鑫真好專案」者，不得再投保本專案；惟未承保、失效、解約、契撤案件不受此限。\n\
    Answer Snippet: (5) 【投保限制】：A. 每一被保險人限投保本專案共一張。(已承保過「共好專案」、「新共好專案」、「金好專案」、「真好專案」、「新真好專案」、「幸福專案」及「鑫真好專案」者，不得再投保本專案；惟未承保、失效、解約、契撤案件不受此限。)\n\
    File Name: 全面保專案_投保規則_20200228.pdf\n\
    Paragraph Title: 投保限制\n\
    """

    QUESTION_SPLIT_PROMPT = ChatPromptTemplate.from_template(_split_question_templte)    
    
    def condense_question_route(input):
        if len(input["chat_history"]) > 0:            
            return RunnablePassthrough.assign(
                chat_history=lambda x: _format_chat_history(input["chat_history"])
            )  | CONDENSE_QUESTION_PROMPT | llm_model | StrOutputParser()
        else: 
            print(input["question"] )
            return input["question"] 
        
    def question_split_route(question):
        try:
            is_need_split = question_split_checker.invoke({"question": question})
            logger.info(f"question: {question}, is_need_split: {is_need_split}")
            if is_need_split == "yes":
                return generate_queries | retriever.map() | reciprocal_rank_fusion
            else:
                return retriever
            
        except Exception as e:
            # 捕获所有异常
            logger.error(f"An error occurred in question_split_route: {e}\n")
            return retriever
        
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

    generate_queries = ( QUESTION_SPLIT_PROMPT | llm_model | StrOutputParser() | (lambda x: re.split(r'\n+', x)))


    # Now we retrieve the documents
    retrieved_documents = {
        # always rewrite
        # "docs": itemgetter("question") | generate_queries | retriever.map() | reciprocal_rank_fusion,
        # auto rewrite
        "docs": itemgetter("question") | RunnableLambda(question_split_route),
        # no rewrite
        # "docs": itemgetter("question") | retriever,
        "question": itemgetter("question")
    }

    # check & reduce ducuments
    check_documents = {
        "docs": lambda x: _reduce_below_limit_token(docs= x["docs"], llm=llm_model, max_tokens_limit=max_input_tokens_limit),
        "question": itemgetter("question"),
    }

    def example_route(question):
        # TODO: 用更好的方式找到更恰當的範例
        if '規則' in question:
            return _more_example
        elif re.fullmatch(".*可以投保.*嗎.*", question):
            return _more_example2
        else:
            return ''

    final_inputs = {
        "context": lambda x: _combine_documents(x["docs"]),
        "question": itemgetter("question"),
        "more_example": itemgetter("question") | RunnableLambda(example_route),
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