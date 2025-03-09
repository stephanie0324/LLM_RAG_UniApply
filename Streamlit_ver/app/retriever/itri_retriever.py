from langchain.schema import BaseRetriever, Document
from typing import TYPE_CHECKING, Any, Dict, List, Optional 
from langchain.callbacks.manager import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import json
from itri_api import ITRI_API

class ITRIRetriever(BaseRetriever):
    url:str
    threshold:float
    skip:int = 0
    limit:int = 10
    data_types: list = []  # 類 collection 的概念
    data_type_ids: list = []  # 類 collection 的概念，新版
    auth: bool = False
    api: ITRI_API = ITRI_API()
    
    def __int__(self):
        pass
    
    def _call_itri_api(self, 
                       url, 
                       query, 
                       threshold=0.75, 
                       skip=0, 
                       limit=10,
                       data_types=[],
                       data_type_ids=[]):
        data = {
            "query": query,
            "data_types_ids": data_type_ids,
            "show_plain_text_data": True,
            "threshold": threshold,
            "skip": skip,
            "limit": limit
        }
        
        # 兼容新舊版而已
        if len(data_types) > 0:
            data['data_types'] = data_types
            
        if len(data_type_ids) > 0:
            data['data_type_ids'] = data_type_ids

        
        retry_strategy = Retry(
            total=5,  # 總共重試次數
            backoff_factor=1,  # 指數退避算法的因子 (這會影響重試之間的間隔時間)
            status_forcelist=[429, 500, 502, 503, 504],  # 哪些 HTTP 狀態碼需要重試
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"] # 哪些 HTTP 方法需要重試
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        http = requests.Session()
        http.mount("http://", adapter)
        response = http.post(url, data=json.dumps(data), headers={'Content-Type': 'application/json'}).json()
        return response
    
    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        實作 _get_relevant_documents
        """
        
        if self.auth:
            response = self.api.search(
                query=query,
                data_type_ids=self.data_type_ids,
                threshold=self.threshold,
                skip=self.skip,
                limit=self.limit,
            )
        else:    
            response = self._call_itri_api(
                url=self.url,
                query=query,
                data_type_ids=self.data_type_ids,
                threshold=self.threshold,
                skip=self.skip,
                limit=self.limit
            )
        

        result_documents = []
        for document in response['records']:
            current_document = Document(
                page_content=document['plain_text_data'],
                metadata={
                  'doc_id': document['_id'],
                  'source': document['doc_name']
                }
            )
            result_documents.append(current_document)
        return result_documents
    
    async def _aget_relevant_documents(
            self,
            query: str,
            *,
            run_manager: AsyncCallbackManagerForRetrieverRun,
            **kwargs: Any,
    ) -> List[Document]:
        raise NotImplementedError()