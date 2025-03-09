from config import settings
import streamlit as st
from st_aggrid import AgGrid, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder
from itri_api import ITRI_API
import pandas as pd
import os
from index_preprocess_v1_1c import batch_process as batch_process_v1_1c

knowledge_information = {
    
}

def clear_km_state():
    st.session_state['current_path'] = '/'
    st.session_state['page'] = 0
    

def display_file_browser_with_drill_down(api: ITRI_API, data_type_id):
    # 初始化 session state
    if 'current_path' not in st.session_state:
        st.session_state['current_path'] = '/'
    if 'navigate_to' in st.session_state and st.session_state['navigate_to']:
        st.session_state['current_path'] = st.session_state['navigate_to']
        st.session_state['navigate_to'] = None
    if 'page' not in st.session_state:
        st.session_state['page'] = 0

    # 顯示當前路徑
    st.write(f"Current Page: {st.session_state['page'] + 1}, Path: {st.session_state['current_path']}")
    file_list = api.list_files(data_type_id=data_type_id, dir_path=st.session_state.current_path)
    
    # 表頭
    header_cols = st.columns([3, 1, 1, 1])
    with header_cols[0]:
        st.markdown("**File Name**")
    with header_cols[1]:
        st.markdown("**File Size**")
    with header_cols[2]:
        st.markdown("**Type**")
    with header_cols[3]:
        st.markdown("**Action**")
    
    items_per_page = 20
    # 分頁顯示
    start_index = st.session_state['page'] * items_per_page
    end_index = start_index + items_per_page
    displayed_files = file_list[start_index:end_index]

    for file in displayed_files:
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        with col1:
            st.text(file['name'])
        with col2:
            st.text(f"{file['file_size']} bytes")
        with col3:
            if file['is_dir']:
                action = lambda path=os.path.join(st.session_state['current_path'], file['name']): st.session_state.update({'navigate_to': path})
                st.button(f"Open", key=file['id'], on_click=action)
            else:
                st.text("File")
        with col4:
            if st.button("Delete", key=f"delete_{file['id']}"):
                if not file['is_dir']:
                    api.delete_files([file['id']])
                    st.write(f"File {file['name']} would be deleted.")
                else:
                    delete_directory_action = lambda path=os.path.join(st.session_state['current_path'], file['name']): api.delete_directory(data_type_id=data_type_id, directory_path=path)
                    delete_directory_action()
                st.rerun()

    # 分頁按鈕
    if len(file_list) > items_per_page:
        col_prev, col_next = st.columns(2)
        if col_prev.button("Previous") and st.session_state['page'] > 0:
            st.session_state['page'] -= 1
            st.rerun()
        if col_next.button("Next") and end_index < len(file_list):
            st.session_state['page'] += 1
            st.rerun()

    # 只在非根路徑顯示返回上一級按鈕
    if st.session_state['current_path'] != '/':
        if st.button("Go Back"):
            # st.session_state['navigate_to'] = os.path.dirname(st.session_state['current_path'])
            st.session_state.update({'navigate_to': os.path.dirname(st.session_state['current_path'])})
            st.session_state['page'] = 0  # 返回上一級時重置頁面索引
            st.rerun()

def knowledgebase_management(api: ITRI_API):
    title = f"""知識庫管理"""
    descriptio = f"""<ul>
        <li>請先選擇知識庫，或建立新知識庫</li>
        <li>請勿一次上傳超過10個檔案</li>
        <li>檔案上傳即時建置索引，請耐心等候</li>
        <li>現為同步即時建置索引，若檔案較大請分批上傳</li>
    </<ul>"""
    st.title(title)
    st.caption(descriptio, unsafe_allow_html=True)
    
    data_type_list = api.get_data_type_list()
    
    knowledgebase_names=[ record['data_type_name'] for record in data_type_list ]
    knowledgebase_ids=[ record['data_type_id'] for record in data_type_list ]
    
    if "selected_knowledgebase_name" in st.session_state and st.session_state["selected_knowledgebase_name"] in knowledgebase_names:
        selected_knowledgebase_index = knowledgebase_names.index(st.session_state["selected_knowledgebase_name"])
    else:
        selected_knowledgebase_index = 0   
    
    selected_knowledgebase_name = st.selectbox(
        "請選擇或建立知識庫：",
        knowledgebase_names + ["建立知識庫"],
        index=selected_knowledgebase_index
    )
    
    if selected_knowledgebase_name == "建立知識庫":
        with st.form("建立知識庫"):
            knowledgebase_name = st.text_input(
                "知識庫名稱",
                placeholder="知識庫名稱",
                key="knowledgebase_name",
            )
            knowledgebase_description = st.text_input(
                "知識庫說明",
                placeholder="請描述知識庫內容",
                key="knowledgebase_description",
            )
            cols = st.columns(2)
            
            submit_new_knowledgebase = st.form_submit_button(
                "送出",
                use_container_width=True,
            )
            
        if submit_new_knowledgebase:
            if not knowledgebase_name.strip():
                st.error("請輸入知識庫名稱")
            elif knowledgebase_name in knowledgebase_names:
                st.error("知識庫名稱重覆!")
            else:
                result = api.add_data_type(knowledgebase_name)
                st.session_state["selected_knowledgebase_name"] = knowledgebase_name
                st.toast(result)
                st.rerun()
                
    elif selected_knowledgebase_name:
        if 'current_knlowledgebase' not in st.session_state:
            st.session_state["current_knlowledgebase"] = selected_knowledgebase_name
        elif st.session_state["current_knlowledgebase"] != selected_knowledgebase_name:
            clear_km_state()
            st.session_state["current_knlowledgebase"] = selected_knowledgebase_name
        selected_knowledgebase_index = knowledgebase_names.index(selected_knowledgebase_name)
        knowledgebase_id = knowledgebase_ids[selected_knowledgebase_index]
        
        
        files = st.file_uploader("上傳檔案：",
                                 [],
                                 accept_multiple_files=True,
                                 help="請勿一次上傳超過10個檔案"
                                 )
        knowledgebase_description = st.text_area(
                "知識庫說明",
                placeholder="請描述知識庫內容",
                key="knowledgebase_description",
            )
        
        data_mode = st.selectbox("Data處理模式",
            ["default", "v1.1c"],
            index=1,
            key="data_mode",
        )
        
        if st.button(
                "新增檔案至知識庫",
                disabled=len(files) == 0,
        ):
            # 超過 10 個就禁止上傳
            if len(files) > 10:
                st.toast("請勿上傳超過10個檔案", icon="🔥")
            else:
                # default 直接上傳到 KM，由 KM 進行解析與處理
                if data_mode == "default":
                    result = api.add_files(knowledgebase_id, files)
                # v1.1c 採用表格解析強化，由本地隨意解析
                elif data_mode == "v1.1c":
                    index_json_files = batch_process_v1_1c(knowledgebase_id, files)
                    result = api.add_json_files(knowledgebase_id, index_json_files)
                else:
                    st.info("不存在的資料處理模式")
                    st.stop()
                
                st.toast(result)
        
        st.divider()
        
        display_file_browser_with_drill_down(api, knowledgebase_id)

        st.divider()
        cols = st.columns(3)
        if cols[2].button(
                "刪除知識庫",
                use_container_width=True,
        ):
            ret = api.delete_data_type(knowledgebase_id)
            st.toast(ret)
            st.rerun()