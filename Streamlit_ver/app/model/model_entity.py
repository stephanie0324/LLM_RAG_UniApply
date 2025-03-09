from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceTextGenInference, VLLMOpenAI, HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace
import os

class ModelEntry(BaseModel):
    type: str
    args: dict
    
    def asInstance(self):        
        if self.type == "ChatOpenAI":
            return ChatOpenAI(**self.args)
        if self.type == "VLLMOpenAI":
            return VLLMOpenAI(**self.args)
        elif self.type == "HuggingFaceTextGenInference":
            # HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
            # print(f"HF_TOKEN: {HF_TOKEN}")
            # llm = HuggingFaceTextGenInference(**self.args)
            # llm = HuggingFaceEndpoint(**self.args)
            # return ChatHuggingFace(llm=llm)
            return HuggingFaceTextGenInference(**self.args)
        elif self.type == "ChatHuggingFace":
            # HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
            # print(f"HF_TOKEN: {HF_TOKEN}")
            llm = HuggingFaceTextGenInference(**self.args)
            return ChatHuggingFace(llm=llm)
        elif self.type == "HuggingFaceEndpoint":
            # HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
            # print(f"HF_TOKEN: {HF_TOKEN}")
            llm = HuggingFaceEndpoint(**self.args)
            return ChatHuggingFace(llm=llm)
        else:
            raise Exception(f"未定義的 model type: {self.type}")