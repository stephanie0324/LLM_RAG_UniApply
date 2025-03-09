from pydantic import RootModel
from .model_entity import ModelEntry
from typing import Dict

class ModelConfig(RootModel):
    root: Dict[str, ModelEntry]