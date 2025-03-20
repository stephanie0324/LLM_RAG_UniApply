# RAG DEMO
這是一個支援 streamlit RAG demo 的 LLM API。

## 目錄
[TOC]

## 主要檔案與資料夾說明
- `.env`: 相關環境設定
- `data/`: 給程式讀取資料來源的資料夾
- `index_data/`: 用來建立保存 FAISS 的 index 檔案目錄
- `app/twc_rag.py`: 台水RAG使用langchain route 實作
- `app/twc_web.py`: 台水RAG Demo 之頁面與執行的檔案
- `app/web.py`: 凱基POC執行的檔案

## 執行流程
### 編輯設定檔案
1. 先編輯 `.env` 相關參數
    * 最低需修改 `HOST_PORT`、`OPENAI_API_KEY` (若不帶 `OPENAI_API_KEY` 則需要移除掉 OPENAI 相關的模型)
    * 調整 KM 系統連接參數 `ITRI_KM_CONFIG`
    * 其他參數視情況而定
### docker-compose
- 執行 `docker-compose up -d`
    - `docker` (建議至少24) 或 `docker-compose` (建議至少v2.2.3) 都可能遇到 `* Status: the --mount option requires BuildKit. ***` ，建議**升級版本**或修改 `/etc/docker/daemon.json` 設定檔
    - The embedding model would take some time to load, eveytime you start the container
- 關閉時： `docker-compose down`
### Dockerfile
- CMD 需要更改你要執行的執行檔名稱
## Notes
1. Streamlit 是即時偵測檔案異動

### Dev. Mode
1. Enter the container
`bash ../components/Streamlit_ver/script/run-dev-mode.sh `
2. run app
`streamlit run --server.port 7860 twc_web.py `

