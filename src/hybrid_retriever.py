import os
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

# 設定日誌
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class AdvancedHybridRetriever:
    """
    進階混合檢索器：
    1. 結合 BM25 (關鍵字精準匹配) 與 FAISS (語意檢索)。
    2. 使用 EnsembleRetriever (RRF 演算法) 融合兩者結果。
    3. 透過 Cross-Encoder 進行最終重排序 (Re-ranking)，提昇 Top-K 準確率。
    """
    def __init__(self, vector_db_path: str = "../data/vector_db"):
        self.vector_db_path = Path(vector_db_path)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化內部變數
        self.faiss_db = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self.reranked_retriever = None
    
    
    def build_index(self, docs: List[Document]) -> None:
        """從 Document 建立混合索引 (FAISS + BM25) 並儲存至本機"""
        logger.info(f"開始建立索引，共 {len(docs)} 個 Chunks...")
        
        # 1. 建立並儲存 Dense Index (FAISS)
        self.faiss_db = FAISS.from_documents(docs, self.embeddings)
        self.faiss_db.save_local(str(self.vector_db_path / "faiss_index"))
        logger.info("FAISS 向量索引建立並儲存完成。")
                
        # 2. 建立並儲存 Sparse Index (BM25)
        self.bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_path = self.vector_db_path / "bm25_index.pkl"
        with open(bm25_path, "wb") as f:
            pickle.dump(self.bm25_retriever, f)
        logger.info("BM25 關鍵字索引建立並儲存完成。")
        
        self._setup_pipeline()
    
    
    def load_index(self) -> bool:
        faiss_path = self.vector_db_path / "faiss_index"
        bm25_path = self.vector_db_path / "bm25_index.pkl"
        
        if not faiss_path.exists() or not bm25_path.exists():
            logger.error("找不到本機索引檔案，請先執行 build_index()。")
            return False
        
        try:
            self.faiss_db = FAISS.load_local(
                str(faiss_path),
                self.embeddings,
                allow_dangerous_deserialization=True # 信任本機檔案
            )
            with open(bm25_path, "rb") as f:
                self.bm25_retriever = pickle.load(f)
                
            logger.info("成功載入 FAISS 與 BM25 索引。")
            self._setup_pipeline()
            return True
        except Exception as e:
            logger.error(f"載入索引時發生錯誤 : {e}")
            return False
        
    def _setup_pipeline(self):
        """配置 Hybrid Search 與 Re-ranking 的檢索管線"""
        # 設定底層檢索器的召回數量 (Recall) - 抓取較多候選文件
        faiss_retriever = self.faiss_db.as_retriever(search_kwargs={"k": 15})
        self.bm25_retriever.k = 15
        
        # 1. 混合檢索 (Ensemble) - 使用 RRF (Reciprocal Rank Fusion) ??
        # weight: 調整語意與關鍵字的權重比例        
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, faiss_retriever],
            weights=[0.4,0.6] 
        )        
        
        # 2. 重排序 (Re-ranking) - 使用開源 Cross-Encoder 模型
        # 這會將前面選出的 30 份候選文件，與使用者的問題進行深度 Attention 計算 
        cross_encoder = HuggingFaceCrossEncoder(
            model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )     
        compressor = CrossEncoderReranker(
            model=cross_encoder,
            top_n=5 # 最終只輸出最精準的 5 份文件給 LLM
        )
        
        # 組合最終檢索器
        self.reranked_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.ensemble_retriever
        )
        logger.info("混合檢索與重排序管線 (Pipeline) 設定完成。")
    
    def retrieve(self, query:str) -> List[Document]:
        """執行檢索並回傳最終文件"""
        if not self.reranked_retriever:
            raise ValueError("檢索管線尚未初始化，請先 load_index() 或 build_index()。")
        logger.info(f"執行檢索 : '{query}'")
        return self.reranked_retriever.invoke(query)

# CLI 測試進入點
if __name__ == "__main__":
    retriever = AdvancedHybridRetriever()
    
    # 假設這是在 parser 解析後，經過 text_splitter 切割好的 documents
    sample_docs = [
        Document(page_content="台積電 2024 年第一季營收達到 5926 億元新台幣。", metadata={"source":"TSM_Q1"}),
        Document(page_content="蘋果 2024 年重點佈局 AI 伺服器與邊緣運算。", metadata={"source":"AAPL_2024"}),
        Document(page_content="台積電先進製程 3nm 產能滿載，帶動毛利率上升。", metadata={"source":"TSM_Q1"})
    ]
    
    # 測試建構
    retriever.build_index(sample_docs)
    
    # 測試檢索 (故意用縮寫與錯字考驗語意與關鍵字混合能力)
    results = retriever.retrieve("TSMC 2024第一季的財務表現如何？")
    for i, doc in enumerate(results):
        print(f"Rank {i+1} [Source: {doc.metadata['source']}] : {doc.page_content}")
