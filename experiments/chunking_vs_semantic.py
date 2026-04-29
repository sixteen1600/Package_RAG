import os
import logging
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# LangChain 相關套件
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# 載入環境變數
load_dotenv()

# 設定日誌
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

class ChunkingExperiment:
    """
    固定長度 (Fixed-size) vs. 語義分段 (Semantic Chunking) 實驗
    量化評估 Chunk 分佈狀況，並對比實際的檢索品質。
    """
    def __init__(self, markdown_path: str):
        self.markdown_path = Path(markdown_path)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.raw_text = self._load_data()
    
    def _load_data(self) -> str:
        """讀取由 Docling 處理好的 Markdown 文件"""
        if not self.markdown_path.exists():
            raise FileNotFoundError(f"找不到檔案: {self.markdown_path}。請確保 Docling 已解析該 PDF。")

        with open(self.markdown_path, "r", encoding="utf-8") as f:
            logger.info(f"成功載入文本: {self.markdown_path.name}")
            return f.read()
    
    def run_fixed_size_chunking(self):
        """
        策略 A: 固定長度切塊
        """
        logger.info("執行策略 A: 固定長度切塊 (Recursive Character)...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
            )
        # LangChain 的 create_documents 會自動轉為 Document 物件
        docs = splitter.create_documents([self.raw_text])
        return docs
    
    def run_semantic_chunking(self):
        """
        策略 B: 語義分段
        透過 Embedding 計算相鄰句子的餘弦相似度 (Cosine Similarity)，
        當相似度低於某個百分位數 (如 80%) 時，即視為「主題切換 ??」並進行切割。
        """
        logger.info("執行策略 B: 語義分段切塊 (Semantic Chunker)...")
        splitter = SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=80
        )
        docs = splitter.create_documents([self.raw_text])
        return docs

    def analyze_distribution(self, name: str, docs: list):
        """利用統計指標分析 Chunk 的長度分佈"""
        lengths = [len(doc.page_content) for doc in docs]
        
        print(f"\n[{name}] Chunk長度統計分佈")
        print(f" - 總 Chunk 數量 : {len(lengths)}個")
        print(f" - 平均長度      : {np.mean(lengths):.2f}字元")
        print(f" - 中位數        : {np.median(lengths):.2f}字元")
        print(f" - 最短/最長     : {np.min(lengths)} / {np.max(lengths)} 字元")
        print(f" - 標準差 (Std)  : {np.std(lengths):.2f} (反映長度變異程度)")
        print(f"-"*40)
    
    def compare_retrieval(self, query: str, docs_fixed: list, docs_semantic: list):
        """對比兩種策略在相同 Query 下的檢索結果 (Top-2)"""
        print(f"\n🔍 【檢索對比測試】 測試問題: '{query}'")
        
        # 建立臨時向量庫
        db_fixed = FAISS.from_documents(docs_fixed, self.embeddings)
        db_semantic = FAISS.from_documents(docs_semantic, self.embeddings)
        
        retriever_fixed = db_fixed.as_retriever(search_kwargs={"k": 2})
        retriever_semantic = db_semantic.as_retriever(search_kwargs={"k": 2})
        
        print("\n--- 策略 A (固定長度) 檢索結果 ---")
        for i, doc in enumerate(retriever_fixed.invoke(query)):
            # 截斷過長的輸出以方便觀看
            content = doc.page_content.replace("\n", " ")[:150] + "..."
            print(f"Rank {i+1} (長度 {len(doc.page_content)}): {content}")

        print("\n--- 策略 B (語義分段) 檢索結果 ---")
        for i,doc in enumerate(retriever_semantic.invoke(query)):
            # 截斷過長的輸出以方便觀看
            content = doc.page_content.replace("\n", " ")[:150] + "..."
            print(f"Rank {i+1} (長度 {len(doc.page_content)}): {content}")

if __name__ == "__main__":
    # 假設 Docling 已經將 PDF 解析到 processed 資料夾下
    BASE_DIR = Path(__file__).parent.parent
    TARGET_MD = BASE_DIR / "data" / "processed" / "2024-TSMC-Sustainability-Report-highlights-c.md"
    
    # 啟動實驗
    experiment = ChunkingExperiment(str(TARGET_MD))
    
    # 取得兩種 Chunking 結果
    docs_fixed = experiment.run_fixed_size_chunking()
    docs_semantic = experiment.run_semantic_chunking()
    
    # 1. 輸出分佈統計 (對齊 JD 中的「統計思維」)
    print("\n" + "="*50)
    print("第一階段：Chunk 分佈統計對比")
    print("="*50)
    experiment.analyze_distribution("策略 A: 固定長度 (Recursive)", docs_fixed)
    experiment.analyze_distribution("策略 B: 語義分段 (Semantic)", docs_semantic)    
    
    # 2. 進行實際檢索對比 (對齊 TSMC 永續報告書的情境)
    print("\n" + "="*50)
    print("第二階段：檢索品質對比")
    print("="*50)
    test_query = "台積電在 2024 年針對水資源管理與海水淡化廠的具體目標與規劃是什麼？"
    experiment.compare_retrieval(test_query, docs_fixed, docs_semantic)
    
    
    
    
    