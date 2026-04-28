import os
import logging
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

# 匯入 Docling 核心解析套件
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat

# 設定日誌 (Logging)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"    
)
logger = logging.getLogger(__name__)

class AdvancedDocumentParser:
    """
    進階文件解析器，負責將複雜的 PDF（如財報、永續報告書）
    轉換為高品質的 Markdown 或 JSON 格式，為 RAG 提供乾淨的數據源。
    """
    def __init__(self, raw_dir: str = "../data/raw", processed_dir: str = "../data/processed"):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        
        # 確保輸出目錄存在
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化 Docling 的文件轉換器
        # Docling 預設支援多種格式，且內建強大的版面分析與 OCR 模型
        self.converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF, InputFormat.DOCX, InputFormat.HTML, ]
        )
        logger.info("AdvancedDocumentParser 初始化完成。")
    
    def parse_to_markdown(self, file_path: Path) -> Optional[str]:
        """將單一文件解析並保留版面結構，輸出為 Markdown"""
        try:
            logger.info(f"開始解析文件 : {file_path.name}") # 為什麼file_path可以.name ??
            result = self.converter.convert(file_path)
            
            # 匯出為 Markdown (Docling 會自動處理表格與標題層級)
            markdown_content = result.document.export_to_markdown()
            
            # 儲存到 processed 資料夾
            output_path = self.processed_dir / f"{file_path.stem}.md"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            
            logger.info(f"成功輸出 Markdown:{output_path}")
            return markdown_content
        except Exception as e:
            logger.error(f"解析文件 {file_path.name}")
        
    def parse_to_json(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        將文件解析為 JSON 格式 (Docling 原生文件模型)
        適用於需要嚴格結構化抽取的場景 (例如：只萃取所有表格)
        """
        try:
            logger.info(f"開始萃取結構化 JSON: {file_path.name}")
            result = self.converter.convert(file_path) # 為什麼self可以.converter
            
            # 將 Docling 的 Document 模型轉為字典/JSON
            doc_dict = result.document.export_to_dict()
            
            output_path = self.processed_dir / f"{file_path.stem}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(doc_dict, f, ensure_ascii=False, indent=2)
            
            logger.info(f"成功輸出 JSON : {output_path}")
            return doc_dict
                                                   
        except Exception as e:
            logger.error(f"轉換 JSON 時發生錯誤: {str(e)}")
            return None
    
    def batch_process(self) -> None:
        if not self.raw_dir.exists():
            logger.error(f"找不到原始資料目錄 : {self.raw_dir}")
            return
        
        pdf_files = list(self.raw_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning("在 raw_dir 中沒有找到任何 PDF 檔案。")
            return 
        
        logger.info(f"找到 {len(pdf_files)} 個 PDF 檔案，開始批次處理...")
        
        for file_path in pdf_files:
            # 同時產生 Markdown 與 JSON 雙重格式
            self.parse_to_markdown(file_path)
            self.parse_to_json(file_path)
        
        logger.info("批次處理完成！")

# 讓 CLI 直接測試的進入點
if __name__ == "__main__":
    # 使用相對於 src 目錄的路徑
    BASE_DIR = Path(__file__).parent.parent
    RAW_PATH = BASE_DIR / "data" /"raw"
    PROCESSED_PATH = BASE_DIR / "data" / "processed"
    
    # 執行解析器
    parser = AdvancedDocumentParser(raw_dir=RAW_PATH, processed_dir=PROCESSED_PATH)
    parser.batch_process()