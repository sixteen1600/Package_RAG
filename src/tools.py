import logging
from typing import List, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# 匯入我們之前寫好的混合檢索器
from src.hybrid_retriever import AdvancedHybridRetriever

# 設定日誌
logger = logging.getLogger("DocuMind-Tools")

# 定義工具輸入的結構 (Schema) 
# 這樣做能讓 LLM 更精準地理解應該傳入什麼參數
class SearchInput(BaseModel):
    query: str = Field(description="用於檢索財報或文件的關鍵字或問題")


class FinanceTools:
    """
    封裝所有供 Agent 調用的工具類別
    """    
    def __init__(self, external_retriever=None):
        # 接收外面 (main.py) 傳進來的 external_retriever
        if external_retriever:
            self.retriever_engine = external_retriever
        else:
            # 如果沒有傳進來，才自己去讀取硬碟
            self.retriever_engine = AdvancedHybridRetriever()
            # 嘗試載入現有的索引
            if not self.retriever_engine.load_index():
                logger.warning("檢索索引尚未建立，請確保已執行過資料解析流程。")
            
    def get_retriever_tool(self):
        """
        將 Hybrid Retriever 封裝成 LangChain 標準 Tool 格式。
        使用 @tool 裝飾器能自動將 Docstring 轉化為 LLM 的 Tool Description。
        """   
        
        @tool("financial_document_retriever", args_schema=SearchInput)
        def financial_document_retriever(query: str) -> str:
            """
            這是一個專業的財務文件檢索工具。
            當需要查詢公司財報、營收數字、風險因素或永續發展報告的具體細節時，請使用此工具。
            輸入應為具體的查詢問題或關鍵字。
            """
            try:
                logger.info(f"[Tool Use] 正在檢索 : {query}")
                
                # 執行混合檢索與重排序
                docs = self.retriever_engine.retrieve(query)
                if not docs:
                    return "未找到相關資料，請嘗試調整查詢關鍵字。"
                
                # 將檢索到的多份文件合併為一段文字，供 Agent 閱讀
                formatted_results = "\n\n".join([
                    f"[來源 : {doc.metadata.get('source', '未知')}]:\n{doc.page_content}"
                    for doc in docs
                ])
                return formatted_results
            
            except Exception as e:
                logger.error(f"工具執行發生錯誤 : {e}")
                return f"檢索過程發生錯誤 : {str(e)}"
        return financial_document_retriever
    
            
        # 備註：未來可在此處加入更多工具，ex : 股價 API 查詢等等
        
if __name__ == "__main__":
    tools_factory = FinanceTools()
    retriever_tool = tools_factory.get_retriever_tool()
        
        
        
        
        
        
        