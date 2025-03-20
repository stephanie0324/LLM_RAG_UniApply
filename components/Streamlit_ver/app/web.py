import streamlit as st
from streamlit_option_menu import option_menu
from chat_web import chat_web
from km_web import knowledgebase_management
import os
from itri_api import ITRI_API
from config import settings

import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import logging

logging.basicConfig(
    level=logging.INFO,  # 設定日誌等級為 INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 設定日誌格式
    handlers=[
        logging.StreamHandler()  # 將日誌輸出到控制台
    ]
)

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


itri_api = ITRI_API(**settings.ITRI_KM_CONFIG)

if __name__ == "__main__":
    
    st.set_page_config(
        page_title="Uni Apply",
        page_icon="img/UniApplyLogo.png",
        initial_sidebar_state="expanded",
        menu_items={
            'About': f"""Welcome to Uni Apply"""
        }
    )

    pages = {
            "對話": {
                "icon": "chat",
                "func": chat_web,
            },
            "知識庫管理": {
                "icon": "hdd-stack",
                "func": knowledgebase_management,
            }
        }
        
    with st.sidebar:
        options = list(pages)
        icons = [x["icon"] for x in pages.values()]
        
        selected_page = option_menu(
            "",
            options=options,
            icons=icons,
            # menu_icon="chat-quote",
            default_index=0,
        )
    if selected_page in pages:
        pages[selected_page]["func"](api=itri_api)
