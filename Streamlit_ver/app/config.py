from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AnyHttpUrl, validator, Json, model_validator
from typing import List, Union, Optional, Pattern, ClassVar
from model.model_config import ModelConfig
from retriever.retriever_config import RetrieverConfig

import json
import os


class Settings(BaseSettings):
    DEBUG: bool = True
   
    # 要關掉不顯示的 mode
    DEFAULT_CHAT_MODE_DISABLE: List = [""]
    
    DEFAULT_CHAT_MODE: int = 2
    
    DEFAULT_CHAT_HISTORY_WINDOWS: int = 1
    
    CHAT_HISTORY_WINDOWS: int = 2
    
    MODEL_LIST_VISIABLE: bool = True
    
    RAG_REDOUCE_BELOW_LIMIT_TOKEN: int = 7500
    
    TESSDATA_PREFIX: str = "/app/tessdata"
    
    # 這是 改調用 vLLM
    MODEL_CONFIG: ModelConfig = """
    {
        "Meta-Llama-3-70B-Instruct-GPTQ": {
            "type": "ChatOpenAI",
            "args": {
                "model_name": "TechxGenus/Meta-Llama-3-70B-Instruct-GPTQ",
                "openai_api_key": "EMPTY",
                "openai_api_base": "http://140.96.111.198:11127/v1",
                "temperature": 0
            }
        },
        "Llama-3-Taiwan-70B-Instruct-DPO": {
            "type": "ChatOpenAI",
            "args": {
                "model_name": "yentinglin/Llama-3-Taiwan-70B-Instruct-DPO",
                "openai_api_key": "EMPTY",
                "openai_api_base": "http://140.96.111.133:11127/v1",
                "temperature": 0
            }
        },
        "Meta-Llama-3-8B-Instruct": {
            "type": "ChatOpenAI",
            "args": {
                "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
                "openai_api_key": "EMPTY",
                "openai_api_base": "http://61.216.92.21:11125/v1",
                "temperature": 0
            }
        },
        "ITRIv4": {
            "type": "ChatOpenAI",
            "args": {
                "model_name": "ITRIv4",
                "openai_api_key": "EMPTY",
                "openai_api_base": "http://61.216.92.21:11111/v1",
                "temperature": 0
            }
        },
        "GPT-4o": {
            "type": "ChatOpenAI", 
            "args": {
                "temperature": 0,
                "model_name": "gpt-4o"
            }
        },
        "GPT-3.5-Turbo": {
            "type": "ChatOpenAI", 
            "args": {
                "temperature": 0,
                "model_name": "gpt-3.5-turbo-0125"
            }
        }
    }
    """
    
    @model_validator(mode='before')
    def parse_model_config(cls, values):
        model_config = values.get('MODEL_CONFIG', cls.__fields__['MODEL_CONFIG'].default)
        if not isinstance(model_config, dict):
            try:
                model_config_dict = json.loads(model_config)
                model_instance = ModelConfig(model_config_dict)
                values['MODEL_CONFIG'] = model_instance
            except json.JSONDecodeError:
                raise ValueError('MODEL_CONFIG must be a valid JSON string')
        else:
                model_instance = ModelConfig(model_config)
                values['MODEL_CONFIG'] = model_instance
        return values
    
    
    RETRIEVER_CONFIG: RetrieverConfig = """
    {
        "ITRI": {
            "type": "ITRIRetriever", 
            "args": {
                "url": "http://140.96.111.113:32343/v1/knowledge/search",
                "data_types": ["2"],
                "threshold": 0.85,
                "limit": 4
            }
        },
        "OpenAI": {
            "type": "FAISS",
            "args": {
                "path": "./data/taipower_FAISS"
            }
        }
    }
    """
    
    @model_validator(mode='before')
    def parse_retriever_config(cls, values):
        retriever_config = values.get('RETRIEVER_CONFIG', cls.__fields__['RETRIEVER_CONFIG'].default)
        
        if not isinstance(retriever_config, dict):
            try:
                retriever_config_dict = json.loads(retriever_config)
                retriever_instance = RetrieverConfig(retriever_config_dict)
                values['RETRIEVER_CONFIG'] = retriever_instance
            except json.JSONDecodeError:
                raise ValueError('RETRIEVER_CONFIG must be a valid JSON string')
        else:
            retriever_instance = RetrieverConfig(retriever_config)
            values['RETRIEVER_CONFIG'] = retriever_instance
        return values
    
    RETRIEVAL_DOC_VISIABLE: bool = True
    # 定義索引儲存的開頭名稱
    DATA_FOLDER_NAME: ClassVar[str] = os.getenv('DATA_FOLDER_NAME', 'default_date')

    
    RETRIEVER_INDEX_NAME_PREFIX: ClassVar[str] = f"./{DATA_FOLDER_NAME}_ic"
    
    # 定義回傳的 TOP N
    RETRIEVER_RETURN_TOP_N: int = 8
    
    # LLM RAG 檔案來源(模擬IC/RAG共用)
    RAG_FILES: ClassVar[dict] = {
        "twc": f"./data/faq_expand.json"
    }
    
    # LLM RAG 索引前綴名稱
    RAG_INDEX_PREFIX: ClassVar[str] = f"./{DATA_FOLDER_NAME}_expand_4"
    
    # 20240613 RAG demo 固定使用 HF 來源的 embedding model，避免主機網路問題
    RAG_INDEX_HF_EMBEDDING_MODEL_CONFIG: dict = {
        "model_name": "BAAI/bge-m3",
        "model_kwargs": {"device": "cuda:0"},
        "encode_kwargs": {"normalize_embeddings": True}
    }
    

    ITRI_KM_CONFIG: dict = """
    {
        "data_api_base_url": "http://140.96.111.113:31042",
        "admin_api_base_url": "http://140.96.111.113:31043",
        "admin_account": "default_local_user",
        "admin_password": "wszY@7e7Tck="
    }
    """
    
    @model_validator(mode='before')
    def parse_itri_km_config(cls, values):
        itri_km_config = values.get('ITRI_KM_CONFIG', cls.__fields__['ITRI_KM_CONFIG'].default)
        
        if not isinstance(itri_km_config, dict):
            try:
                itri_km_config_dict = json.loads(itri_km_config)
                values['ITRI_KM_CONFIG'] = itri_km_config_dict
            except json.JSONDecodeError:
                raise ValueError('ITRI_KM_CONFIG must be a valid JSON string')

        return values
    
    
    DEMO_WEB_PAGE_TITLE: str = "工研院資通所➰場域諮詢展示"
    DEMO_WEB_TITLE: str = "工研院資通所➰場域諮詢展示"
    DEMO_WEB_DESCRIPTION: str = """<ul><li>請在下方輸入想知道的資訊，然後點擊Enter來進行🏃，並等待約等待一段時間回傳結果🎉</li></<ul>"""
    
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []
    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    # Origins that match this regex OR are in the above list are allowed
    BACKEND_CORS_ORIGIN_REGEX: Optional[Pattern] = '.*'


settings = Settings()