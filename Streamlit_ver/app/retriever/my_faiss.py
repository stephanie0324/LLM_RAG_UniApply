from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def get_faiss(path):
    # print(f"path:{path}")
    
    embeddings = OpenAIEmbeddings()
    file_path = path
    vector_store = FAISS.load_local(file_path, embeddings) # TODO: 暫時固定用 OpenAI Embedding, 以後 FAISS 會拿掉
    faiss_retriever = vector_store.as_retriever(search_kwargs={'k': 4})
    return faiss_retriever