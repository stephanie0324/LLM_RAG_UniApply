from langchain.prompts.prompt import PromptTemplate
from langchain.schema import format_document

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    result_string = "未檢索到相關資料"
    if len(doc_strings) > 0:
        result_string = document_separator.join(doc_strings)
    
    return result_string

def _reduce_below_limit_token(docs, llm, max_tokens_limit=7500):
    # print(f"docs:{docs}")
    num_docs = len(docs)
    tokens = [ llm.get_num_tokens(doc.page_content)  for doc in docs]
    token_count = sum(tokens[:num_docs])
    while token_count > max_tokens_limit:
        num_docs -= 1
        token_count -= tokens[num_docs]
    # print(f"num_docs:{num_docs}, token_count:{token_count}")
    return docs[:num_docs]




def get_kb_list():
    return 