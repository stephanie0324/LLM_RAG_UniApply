from pydantic import RootModel
from .retriever_entity import RetrieverEntity
from typing import Dict

class RetrieverConfig(RootModel):
    root: Dict[str, RetrieverEntity]