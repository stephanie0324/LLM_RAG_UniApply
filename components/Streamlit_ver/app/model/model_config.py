from pydantic import RootModel, BaseModel
from langchain_openai import ChatOpenAI, OpenAI
from typing import Dict


class ModelEntry(BaseModel):
    type: str
    args: dict

    def as_instance(self):
        if self.type == "ChatOpenAI":
            return ChatOpenAI(**self.args)
        if self.type == "OpenAI":
            return OpenAI(**self.args)

        else:
            raise Exception(f"未定義的 model type: {self.type}")


class ModelConfig(RootModel):
    root: Dict[str, ModelEntry]
