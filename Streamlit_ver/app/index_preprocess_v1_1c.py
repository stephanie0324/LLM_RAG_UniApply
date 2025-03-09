import os
import tempfile
import json

from document_loader.custom_pdf_loader import CustomPDFLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1280,
    chunk_overlap=192,
    length_function=len,
    is_separator_regex=False,
)

def _get_filename_and_extension(file_path):
    # 取得檔名和副檔名
    filename = os.path.basename(file_path)
    name, extension = os.path.splitext(filename)
    
    return name, extension

def parse_pdf(pdf_file):
    loader = CustomPDFLoader(pdf_file, mode='elements')
    loader.use_camelot_table = True
    main_content_doc, footer_or_header_doc= loader.load()
    return main_content_doc, footer_or_header_doc

def chunking(documents):
    split_documents = text_splitter.split_documents(documents)
    return split_documents

def to_index_json_files(documents, header_or_footer: str = None):
    """
    轉換成 KM 的 JSON 新增格式
    """
    
    index_json_files = []
    for doc in documents:
        doc_file_path = doc.metadata["source"]
        doc_chunk_index = doc.metadata["chunk_index"] 
        file_name, file_extension = _get_filename_and_extension(doc_file_path)

        json_object_data = {}
        if header_or_footer is not None:
            json_object_data['header_or_footer'] = header_or_footer
        json_object_data.update({
            "file_name": f"{file_name}{file_extension}",
            "paragraph": doc.page_content
        })

        json_file = {
            "doc_name": f"{file_name}-{doc_chunk_index:03}{file_extension}",
            "json_object_data": json.dumps(json_object_data, ensure_ascii=False),
            "file_path": f"/{file_name}{file_extension}/{file_name}-{doc_chunk_index:03}.json"
        }
        index_json_files.append(json_file)
        

    logger.info(f"index_json_files: {len(index_json_files)}")
    return index_json_files

def batch_process(knowledgebase_id, files):
    """本py主要進入點.
    逐檔批次處理，依序執行，不考慮速度
    1. PDF 解析 (含表格擷取)
    2. chunks
    """
    
    all_split_docs = []
    for file in files:
        temp_dir = f"./km_files/{knowledgebase_id}"
        os.makedirs(temp_dir, exist_ok=True)
        temp_filepath = os.path.join(temp_dir, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        
        main_content_doc, footer_or_header_doc = parse_pdf(temp_filepath)
        current_split_docs = chunking([main_content_doc])

        # current_split_docs = chunk_clean(current_split_docs)
        for index, current_split_doc in enumerate(current_split_docs):
            current_split_doc.metadata["chunk_index"] = index

        all_split_docs.extend(current_split_docs)
    
    logger.info(f"all_split_docs: {len(all_split_docs)}")
    return to_index_json_files(all_split_docs, footer_or_header_doc.page_content)
