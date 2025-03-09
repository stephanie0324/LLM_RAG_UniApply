from pydantic import BaseModel

import requests
import json
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import logging

logger = logging.getLogger(__name__)


# 用於 anootation ，自動檢查與登入
def check_auth(method):
    def wrapper(self, *args, **kwargs):
        if 'request_session' in self.session_state:
            self.http_session = self.session_state['request_session']
        
        auth_info = self.auth_info()
        
        if ('status' not in auth_info) or (auth_info['status'] != True):
            logger.info('將進行登入....')
            login_result = self.admin_login()
            
            if ('status' not in login_result) or (login_result['status'] != True):
                logger.info('Login fail.')
                return  # 如果登入失敗，則不繼續執行後面的方法
            else:
                logger.info('登入成功')
        
        return method(self, *args, **kwargs)
    return wrapper

class ITRI_API:   
    def __init__(
            self,
            data_api_base_url: str = "",
            admin_api_base_url: str = "",
            admin_account: str = "",
            admin_password: str = "",
            session_state: dict = {}
        ):
        self.logger = logging.getLogger(__name__)
        self.data_api_base_url = data_api_base_url
        self.admin_api_base_url = admin_api_base_url
        self.admin_account = admin_account
        self.admin_password = admin_password
        self.session_state = session_state
        
        retry_strategy = Retry(
            total=3,  # 總共重試次數
            backoff_factor=1,  # 指數退避算法的因子 (這會影響重試之間的間隔時間)
            status_forcelist=[429, 500, 502, 503, 504],  # 哪些 HTTP 狀態碼需要重試
            allowed_methods=["GET", "POST", "DELETE"] # 哪些 HTTP 方法需要重試
        )
        self.adapter = HTTPAdapter(max_retries=retry_strategy)
        
        self.http_session = requests.Session()
        self.http_session.mount("http://", self.adapter)
        
    
    def admin_login(self):
        login_url = f"{self.admin_api_base_url}/v1/auth/login"
        
        data = {
            "account": f"{self.admin_account}",
            "password": f"{self.admin_password}"
        }

        response = self.http_session.post(login_url, 
                             data=json.dumps(data), 
                             headers={'Content-Type': 'application/json'}).json()
        
        self.session_state['request_session'] = self.http_session
        
        self.logger.info(f"admin login: {response}")
        return response
        
    def auth_info(self):
        auth_info_url = f"{self.admin_api_base_url}/v1/auth/user-info"
        response = self.http_session.get(auth_info_url, 
                             headers={'accept': 'application/json'}).json()
        
        return response
    
    @check_auth
    def get_data_type_list(self):
        """
        取得所有的 data type。
        現為直接 limit=99
        """
        
        list_data_type_url = f"{self.admin_api_base_url}/v1/data_type?limit=99"
        response = self.http_session.get(list_data_type_url, 
                             headers={'accept': 'application/json'}).json()
            
            
        data_type_list_filter = [ record for record in response['records'] if 1 in record['bu_ids']]

        return data_type_list_filter
    
    @check_auth
    def add_data_type(self, name):
        add_data_type_url = f"{self.admin_api_base_url}/v1/data_type"
        data = {
            "data_type_name": name,
            "bu_ids": [1],
            "enable_api": True
        }
        
        add_data_type_response = self.http_session.post(add_data_type_url, 
                             data=json.dumps(data), 
                             headers={'Content-Type': 'application/json'}).json()
        
        
        self.logger.info(f"add_data_type_response: {add_data_type_response}")
        
        if 'status' in add_data_type_response and add_data_type_response['status'] == True:
            data_type_id = add_data_type_response['data_type_id']
            add_app_data_type_result = self.add_app_data_type(data_type_id)
            if 'status' in add_app_data_type_result and add_app_data_type_result['status'] == True:
                return "新增知識庫成功"
            else:
                self.delete_data_type(data_type_id)
        else:
            return "新增知識庫失敗"
    
    def get_app_info(self):
        get_app_data_type_url = f"{self.admin_api_base_url}/v1/app"
        
        response = self.http_session.get(get_app_data_type_url).json()
        
        return response
    
    @check_auth
    def add_app_data_type(self, data_type_id):
        """
        建立 app 和 data_type 的關係
        """
        add_app_data_type_url = f"{self.admin_api_base_url}/v1/app/data_type"
        
        app_infos = self.get_app_info()
        
        app_1_info = [ record for record in app_infos['records'] if record['app_id'] == 1]
        app_1_data_types = app_1_info[0]['data_types']
        
        app_1_data_types[data_type_id] = 2
        data = {
            "app_id": 1,
            "data_type_and_permissions": app_1_data_types
        }
        
        add_app_data_type_response = self.http_session.post(add_app_data_type_url, 
                             data=json.dumps(data), 
                             headers={'Content-Type': 'application/json'}).json()
        
        
        self.logger.info(f"add_app_data_type_response: {add_app_data_type_response}")
        return add_app_data_type_response
    
    @check_auth
    def delete_data_type(self, data_type_id):
        """
        刪除 data_type
        """
        delete_data_type_url = f"{self.admin_api_base_url}/v1/data_type/{data_type_id}"
        
        response = self.http_session.delete(delete_data_type_url, 
                             headers={'Content-Type': 'application/json'}).json()
        
        
        self.logger.info(f"delete_data_type: {response}")
        if 'status' in response and response['status'] == True:
            return '成功刪除知識庫'
        else:
            return '刪除知識庫失敗'
    
    @check_auth
    def add_json_files(self, data_type_id, metadata_list=[], timeout=600):
        """
            metadata_list: [
                {
                    "doc_name": "xxx",
                    "json_object_data": "JSON dump string"
                    "file_path": "yyy"
                }
            ]
        """
        
        add_files_url = f"{self.data_api_base_url}/v1/knowledge/batch_sync"
        
        for current_metadata in metadata_list:
            current_metadata["data_type_id"] = data_type_id
            current_metadata["user_accounts"] = ["default_local_user"]
            current_metadata["overwrite_file"] = True
            current_metadata["has_upload_file"] = False
            
        
        batch_size = 10
        for i in range(0, len(metadata_list), batch_size):
            
            batch_metadata_list = metadata_list[i:i + batch_size]
            current_batch_data = {
                "metadata": json.dumps(batch_metadata_list)
            }
            
            response = self.http_session.post(add_files_url, 
                                data=current_batch_data, 
                                timeout=timeout).json()
        
            self.logger.info(f"add_json_files: {response}")
        return "完成"
    
    @check_auth
    def add_files(self, data_type_id, files, timeout=600):
        """
        直接上傳檔案到 KM
        """
        
        add_files_url = f"{self.data_api_base_url}/v1/knowledge/batch_sync"
        
        metadata_list = []        
        files_to_upload = []
        for i, uploaded_file in enumerate(files, start=0):
            files_to_upload.append(
                (
                    'files', (uploaded_file.name, uploaded_file.read())
                )
            )
            
            file_metadata = {
                "doc_name": uploaded_file.name,
                "data_type_id": data_type_id,
                "file_path": f"/{uploaded_file.name}",
	            "user_accounts": ["default_local_user"],
	            "has_upload_file": True,
                "overwrite_file": True
            }
            metadata_list.append(file_metadata)
        
        data = {
            "metadata": json.dumps(metadata_list)
        }
        
        response = self.http_session.post(add_files_url, 
                             data=data, 
                             files=files_to_upload,
                             timeout=timeout).json()
        
        self.logger.info(f"add_files: {response}")
        return response
    
    @check_auth
    def list_files(self, data_type_id, app_id=1, dir_path='/'):
        """
        列出檔案目錄。
        因 KM API 改版，這邊僅直接先固定 limit=99
        """
        
        list_file_url = f"{self.data_api_base_url}/v1/directory/list"
        payload = {'app_id': app_id, 'data_type_id': data_type_id, 'dir_path': dir_path, 'limit': 99}
        
        respose = self.http_session.get(list_file_url, params=payload).json()
        
        # print(f'respose: {respose}')
        if 'status' in respose and respose['status'] == True:
            files = [ 
                        {
                            'is_dir': record['is_dir'],
                            'id': record['id'],
                            'name': record['name'], 
                            'doc_name': record['doc_name'], 
                            'file_size': record['raw_file_filesize']
                        } for record in respose['records']
                    ]
        else:
            files=[]
        
        return files
            
        
    @check_auth
    def delete_files(self, doc_ids):
        """
        刪除 files
        """
        delete_files_url = f"{self.data_api_base_url}/v1/knowledge/batch"
        data = [ {"id": id} for id in doc_ids ]
        
        response = self.http_session.delete(delete_files_url, 
                             data=json.dumps(data), 
                             headers={'Content-Type': 'application/json'}).json()
        
        self.logger.info(f"delete_directory: {data}, {response}")
        
    @check_auth
    def delete_directory(self, data_type_id, app_id=1, directory_path="/"):
        """
        刪除目錄
        """
        delete_files_url = f"{self.data_api_base_url}/v1/directory"
        data = {
            "data_type_id": data_type_id,
            "dir_path": directory_path
        }
        
        response = self.http_session.delete(delete_files_url, 
                             data=json.dumps(data), 
                             headers={'Content-Type': 'application/json'}).json()
        
        self.logger.info(f"delete_directory: {data}, {response}")
        
    @check_auth
    def search(self, query="", threshold=0.75, data_type_ids=[], skip=0, limit=10, app_id=1):
        logging.info(f'query to api: {query}')


        search_api_url = f"{self.admin_api_base_url}/v1/knowledge/search"
        
        data = {
            "query": query,
            "show_plain_text_data": True,
            "data_type_ids": data_type_ids,
            "threshold": threshold,
            "app_id": app_id,
            "skip": skip,
            "limit": limit
        }
        
        response = self.http_session.post(search_api_url, 
                             data=json.dumps(data), 
                             headers={'Content-Type': 'application/json'}).json()
        
        
        return response