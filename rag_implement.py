# %%
# Import Package
import langchain
import os
import openai

import json

from operator import itemgetter

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.vectorstores import FAISS
from langchain.schema.retriever import BaseRetriever
from langchain.docstore.document import Document

from langchain import HuggingFaceTextGenInference

# For Message schemas, 
from langchain.schema import HumanMessage, AIMessage, ChatMessage, FunctionMessage

# %%
langchain.debug = True
openai.api_key = os.environ["OPENAI_API_KEY"]

#%% 
model_name = 'gpt-3.5-turbo-1106'
query = "What is the temperature like in NYC?"

model = ChatOpenAI(
    model=model_name,
    temperature=0,
)

response_ai_message = model.predict_messages([HumanMessage(content=query)])

response_ai_message
# %%
# 建置FAISS索引
# load docs

# faq_json_array = []
# with open('全國工商行政服務網常見FAQ-增加expand_questions欄位.jsonl', 'r', encoding='utf-8') as file:
#     for line in file:
#         data = json.loads(line)
#         faq_json_array.append(data)
