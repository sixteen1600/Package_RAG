import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# 把 src 目錄加入 Python 的模組搜尋路徑
# 這樣一來，src 裡面的檔案就能互相直接 import，不用加 'src.' 了！
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),"src"))

# 匯入自定義模組
from src.docling_parser import AdvancedDocumentParser
from src.hybrid_retriever import AdvancedHybridRetriever
from src.langgraph_workflow import FinancialAgentWorkflow
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 新增VECTOR_DB_DIR
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, VECTOR_DB_DIR


# 載入環境變數 (.env)
load_dotenv()

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DocuMind-Main")


def initialze_system():
    """
    系統初始化檢查：確保資料夾與環境變數就緒
    """
    required_keys = ["OPENAI_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        logger.error(f"缺少必要的環境變數 : {missing_keys}")
        return False
    
    # 建立必要的資料夾結構
    for path in ["data/raw", "data/processed", "data/vector_db"]:
        Path(path).mkdir(parents=True, exist_ok=True)
    
    return True

def run_pipeline(force_rebuild=False):
    """
    執行完整的數據處理與索引建立流程
    """
    # 1. 文件解析 (Parsing)
    parser = AdvancedDocumentParser(raw_dir=RAW_DATA_DIR, processed_dir=PROCESSED_DATA_DIR)
    # 檢查是否已有處理過的檔案，避免重複解析耗時
    processed_files = list(Path("data/processed").glob("*.md"))
    
    if not processed_files or force_rebuild:
        logger.info(">>> 啟動文件解析流程 (Docling) ...")
        parser.batch_process()
    else:
        logger.info(">>> 偵測到已有處理過的 Markdown 檔案，跳過解析步驟。")
        
    # 2. 載入解析後的數據並進行切割 (Chunking)
    logger.info(">>> 正在準備數據進行索引建立...")
    all_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    
    for md_file in Path("data/processed").glob("*.md"):
        with open(md_file, "r" , encoding="utf-8") as f:
            content = f.read()
            # 建立 LangChain Document 物件並切割
            chunks = splitter.create_documents(
                texts=[content],
                metadatas=[{"source":md_file.name}]
            )
            all_docs.extend(chunks)
    
    # 3. 建立混合檢索索引 (Indexing) ??
    retriever_engine = AdvancedHybridRetriever(vector_db_path=str(VECTOR_DB_DIR))
    if not (Path("data/vector_db/faiss_index").exists()) or force_rebuild:
        logger.info(">>> 正在建立 Hybrid Search 索引 (FAISS + BM25)...")
        retriever_engine.build_index(all_docs)
    else:
        logger.info(">>> 載入現有索引庫...")
        retriever_engine.load_index()
    
    return retriever_engine

def start_chat(retriever_engine):
    """
    啟動 Agentic Workflow 互動介面
    """
    # 在建立 Agent 時，直接把 retriever_engine 當作參數傳進去
    workflow_engine = FinancialAgentWorkflow(external_retriever=retriever_engine)
    
    graph = workflow_engine.build_graph()
    
    print("\n" + "="*50)
    print("財報分析 Agent 已就緒")
    print("輸入 'exit' 或 'quit' 結束對話")
    print("="*50, "\n")
    
    while True:
        user_input = input("請輸入欲查詢之資訊 : \n >")
        if user_input.lower() in ['exit', 'quit', 'e', 'q']:
            break
        print("Agent 正在思考與審查中...")
    
        # 執行 LangGraph 工作流
        inputs = {"question":user_input, "revision_count":0}
        final_state = graph.invoke(inputs)
        
        print("\n 分析報告:")
        print("-"*30)
        print(final_state.get("draft_response"))
        print("-"*30 + "\n")

if __name__ == "__main__":
    if initialze_system():
        # 第一次執行或資料更新時，可將 force_rebuild 設為 True
        retriever = run_pipeline(force_rebuild=True)
        start_chat(retriever)