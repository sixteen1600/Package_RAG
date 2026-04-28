import os
from pathlib import Path
from dotenv import load_dotenv


# 載入 .env 檔案中的環境變數
load_dotenv()

# 專案路徑設定
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTOR_DB_DIR = DATA_DIR / "vector_db"


# 模型設定 
# 此處使用小模型以平衡成本與效能，而 Reviewer 則用上較強的模型
GENERATION_MODEL = "gpt-4.1-nano"
REVIEWER_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"

# RAG 與 文本切割參數 
# 針對財報建議 Chunk size 會稍微大一點 (800-1000)，才能保留完整的財務數字背景
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


# 檢索參數 
TOP_K_RECALL = 15
TOP_K_RERANK = 5


# API Keys 檢查 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

