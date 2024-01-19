
from django.shortcuts import render, redirect
from django.http import HttpResponse


import langchain
import os
import openai
import pandas as pd
import json
from typing import List, Dict

from operator import itemgetter

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.vectorstores import FAISS
from langchain.schema.retriever import BaseRetriever
from langchain.docstore.document import Document
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import format_document

from langchain import HuggingFaceTextGenInference

# For Message schemas, 
from langchain.schema import HumanMessage, AIMessage, ChatMessage, FunctionMessage, SystemMessage

def createDoc(file_path:str)->list():
    
    file_list = [os.path.join(file_path, f) for f in os.listdir(file_path)\
                  if os.path.isfile(os.path.join(file_path, f))]
    
    df = pd.DataFrame()
    # combine all the files to a df
    for file in file_list:
        tmp = pd.read_excel(file,engine='openpyxl')
        df = pd.concat([df,tmp],axis = 0)

    docs =[]
    for idx , row in df.iterrows():
        ## TODO: clean the text upon on your collected data
        doc_json = {
            "Q": ','.join(str(row['Tags']).split(',')[:2])+ ' '+str(row["Question"]).replace('\xa0','').replace('\n','') , #text cleaning
            "A": row["Answer"].replace('\xa0','').replace('\n','')
        }
        doc_content = json.dumps(doc_json, ensure_ascii=False)

        current_document = Document(
            page_content=doc_content,  # main doc
            metadata={
                'source': row['Tags'],
                'link':row['Link'],
            }
        )
        docs.append(current_document)
    return docs

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content} \n  from:{link}")
def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)



# Create your views here.
def index(request):
    return render(request, 'UniApply/index.html')

def getResponse(request):

    ## TODO : Change model if you like
    model_name = 'gpt-3.5-turbo-1106' # you can change your model here
    
    # Create FAISS　Index
    docs = createDoc(file_path='./data')
    vectorstore = FAISS.from_documents(
        docs, embedding=OpenAIEmbeddings()
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # k = how many candidates the retriever returns
    ## TODO: You can change your own template
    # template setting
    template = """<|system|>
    Answer User Questions according to the Context Dataset.

    Below are the follwing rules:
    1. Compare all JSON fields named 'Q' in the Context to find the one that is relevant to 'User Question' upon keyowrds or context semantically.
    2. If no similar JSON is found, reply with 'No relevant information found'.
    3. You must directly quote the complete text from the 'A' field of the similar JSON as your answer (you must preserve the original text's line breaks (\r\n) and any formatting), and make no modifications.\
    4. You must return the link
    Contex:
    {context}

    <|user|>
    {question} 
    """
    prompt = ChatPromptTemplate.from_template(template)

    model  = ChatOpenAI(model=model_name,temperature=0)
    translater = ChatOpenAI(model=model_name,temperature=0)

    chain = (
        {"context": retriever | _combine_documents, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    userMessage = request.GET.get('userMessage')

    # translate
    messages = [
        SystemMessage(
            content="You are a helpful assistant that translates any language to English"
        ),
        HumanMessage(
            content=userMessage
        ),
    ]
    userMessage  = translater(messages).content

    response_ai_message = chain.invoke(userMessage)
    return HttpResponse(response_ai_message)
