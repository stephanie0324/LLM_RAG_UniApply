from pydantic import BaseModel
from .itri_retriever import ITRIRetriever
from .my_faiss import get_faiss

class RetrieverEntity(BaseModel):
    type: str
    args: dict
    
    def asInstance(self):
        # print(f"type:{self.type}, args: {self.args}")
        
        if self.type == "ITRIRetriever":
            return ITRIRetriever(**self.args)
        elif self.type == "FAISS":
            return get_faiss(**self.args)
        else:
            raise Exception(f"未定義的 retriever typee: {self.type}")