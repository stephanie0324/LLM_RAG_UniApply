from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl, validator
from typing import List, Union, Optional, Pattern, ClassVar, Dict
from model.model_config import ModelConfig, ModelEntry


class Settings(BaseSettings):

    DEBUG: bool = False

    DEFAULT_CHAT_MODE_DISABLE: List = [""]
    DEFAULT_CHAT_MODE: int = 2
    DEFAULT_CHAT_HISTORY_WINDOWS: int = 1
    CHAT_HISTORY_WINDOWS: int = 3
    MODEL_LIST_VISIABLE: bool = True
    RAG_REDOUCE_BELOW_LIMIT_TOKEN: int = 7500

    OPENAI_API_KEY: str = "EMPTY"

    # Model Setting
    MODEL_CONFIG: ModelConfig = {
        "GPT-4o-mini": {
            "type": "ChatOpenAI",
            "args": {
                "max_tokens": 512,
                "temperature": 0,
                "max_retries": 2,
                "model_name": "gpt-4o-mini",
                "api_key": "EMPTY",
            },
        },
        "GPT-3.5": {
            "type": "ChatOpenAI",
            "args": {
                "max_tokens": 512,
                "temperature": 0,
                "max_retries": 2,
                "model_name": "gpt-3.5-turbo-0125",
                "api_key": "EMPTY",
            },
        },
    }

    @validator("MODEL_CONFIG", pre=True, always=True)
    def model_config_convert_to_object_and_add_openai_key(
        cls,
        # NOTE: 無論是用預設值還是用環境變數帶入，v type都是Dict[str, dict]，而非Dict[str, ModelEntry]
        v: Dict[str, dict],
        values,
    ) -> ModelConfig:
        for key, model_conf in v.items():
            # 將dict 轉 ModelEntry obj.
            if isinstance(model_conf, dict):
                model_conf = ModelEntry(**model_conf)
                v[key] = model_conf
            if model_conf.type == "ChatOpenAI":
                model_conf.args["api_key"] = values["OPENAI_API_KEY"]
        print(f"model_config: {v}")
        return v

    # Retriever Setting
    RETRIEVAL_DOC_VISIABLE: bool = True
    RETRIEVER_RETURN_TOP_N: int = 5

    # LLM RAG File Path
    RAG_FILES_FILEPATH: str = "./data/"
    RAG_INDEX_PREFIX: ClassVar[str] = f"uniApply_rag"

    GPU_DEVICE: str = "cuda:0"
    EMBEDDING_MODEL_NAME: str = "ibm-granite/granite-embedding-125m-english"
    RAG_INDEX_HF_EMBEDDING_MODEL_CONFIG: dict = {
        "model_name": "ibm-granite/granite-embedding-125m-english",
        "model_kwargs": {"device": "cuda:0"},
        "encode_kwargs": {"normalize_embeddings": True},
    }

    @validator("RAG_INDEX_HF_EMBEDDING_MODEL_CONFIG", pre=True, always=True)
    def parse_embedding_model_config(
        cls,
        v: Dict[str, dict],
        values,
    ) -> dict:
        for key, val in v.items():
            v["model_name"] = values["EMBEDDING_MODEL_NAME"]
            v["model_kwargs"]["device"] = values["GPU_DEVICE"]
        print(v)
        return v

    DEMO_WEB_PAGE_TITLE: str = "Uni Apply (.◜◡◝)"
    DEMO_WEB_TITLE: str = "Uni Apply (.◜◡◝)"
    DEMO_WEB_DESCRIPTION: str = (
        """<ul><li>Please wnter any question regarding Univ. apply and press Enter🏃. Wait for the response🎉</li></<ul>"""
    )

    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    # Origins that match this regex OR are in the above list are allowed
    BACKEND_CORS_ORIGIN_REGEX: Optional[Pattern] = ".*"

    # Logger format
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = (
        "%(asctime)s.%(msecs)03d %(name)s[%(process)d]: [%(levelname)s] %(message)s"
    )
    LOG_DATE_FORMAT: str = "%Y/%m/%d %H:%M:%S"


settings = Settings()
