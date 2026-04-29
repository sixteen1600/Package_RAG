

# 🚀 DocuMind-AI: Enterprise-Grade Financial Report Agent

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-1.2.15-green.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-Enabled-orange.svg)
![Docling](https://img.shields.io/badge/Docling-Parser-purple.svg)

DocuMind-AI 是一個專為處理**高複雜度商業文檔，如 10-K 財報、ESG 永續報告書**所設計的多 Agent 協作檢索問答系統。

本專案突破了傳統 RAG 系統在處理跨頁表格、專業術語與數據對齊時的瓶頸，透過導入進階版面解析、混合檢索（Hybrid Search）以及具備反思機制（Reflection）的 Agentic Workflow，實現低幻覺、高準確率的金融數據分析。

---

## ✨ 核心架構與亮點 (Core Features)

本專案針對企業級 AI 應用的三大痛點提出了解決方案：

### 1. 企業級文檔解析 (Advanced Document Parsing)
傳統的 PyPDF2 等工具在面對財報的「跨頁表格」或「雙欄排版」時，文字順序極易錯亂，導致後續 LLM 產生嚴重幻覺。
* **Docling 整合**：本專案採用 `Docling` 進行底層的版面分析（Layout Analysis），能精準將複雜的財務表格與非結構化數據，還原為高語義品質的 Markdown 與原生 JSON 格式。
* **數據清洗與雙軌輸出**：確保提供給向量資料庫與 LLM 的輸入是純淨的結構化數據，為後續的語義 Chunking 打下堅實基礎。

### 2. 混合檢索與重排序 (Hybrid Search & Re-ranking)
在財務領域，單純的向量檢索常因「專有名詞縮寫」或「特定年份」不夠敏感而導致檢索失準。
* **Reciprocal Rank Fusion (RRF)**：結合 **BM25 稀疏矩陣（關鍵字精準匹配）** 與 **FAISS 密集向量（語義理解）**，設定 4:6 權重進行底層召回（Recall），確保不遺漏關鍵數據。
* **Cross-Encoder 漏斗策略**：針對 RRF 召回的 Top-15 候選文檔，掛載 `ms-marco-MiniLM` 進行深度的 Attention 重排序，最終僅輸出 Top-5 最相關文檔給 LLM，大幅提升答題方向準確率（Directional Accuracy）並降低 API 成本。

### 3. 具備反思機制的多 Agent 協作 (Agentic Workflow with Reflection)
捨棄黑盒化的 `AgentExecutor`，採用 **LangGraph (`StateGraph`)** 打造高透明度、可控的多節點狀態機。
* **Planning & Tool Use**：面對複雜提問（如：對比兩年營收與風險），Planner 節點會自主拆解任務步驟，並調用自定義的 Hybrid Retriever 工具進行檢索。
* **多模型協作與自我審查 (Reflection)**：
  * **Drafter Node**：由 OpenAI (GPT-4o-mini) 負責初步資料統整與起草。
  * **Reviewer Node**：由 Anthropic (Claude 3.5 Sonnet) 或 GPT-4o 擔任嚴格的「審計員」，審查草稿是否產生「檢索範圍外」的數據幻覺。若發現瑕疵，將透過 Conditional Edge 退回重寫，確保最終報告的 Evidence-based 可靠性。

---

## 📂 專案架構 (Project Structure)

```text
DocuMind-AI-lite/
│
├── .env                             # 環境變數配置檔 (存放 OpenAI, Anthropic, Cohere 等 API Keys)
├── .gitignore                       # Git 忽略清單 (設定忽略 .env, __pycache__, vector_db 等)
├── requirements.txt                 # 專案依賴套件 (LangChain, LangGraph, Docling, FAISS, BM25 等)
├── README.md                        # 專案說明文件 (系統架構圖、Agent Workflow、成效評估展示)
├── main.py                          # 系統進入點 (可用作 CLI 執行，或未來擴展為 FastAPI 介面)
├── config.py                        # 全域參數設定 (集中管理 Chunk size, 模型名稱如 gpt-4o 等, 檔案路徑)
│
├── data/                            # 資料流管理區
│   ├── raw/                         # 原始未處理的文檔
│   │   ├── AAPL_2024_10K.pdf        # 蘋果年度財報
│   │   └── TSM_Sustainability.pdf   # 台積電永續報告書
│   ├── processed/                   # 經 Docling 處理後的高品質 Markdown 或 JSON 檔案
│   └── vector_db/                   # 向量與稀疏矩陣資料庫本機儲存區 (如 FAISS index 檔)
│
├── src/                             # 系統核心模組
│   ├── docling_parser.py            # 負責處理複雜版面、表格抽取與 Markdown 格式化
│   ├── hybrid_retriever.py          # 實作 Dense (Embedding) + Sparse (BM25) 與 Re-ranking 邏輯
│   ├── langgraph_workflow.py        # 實作具備 Planning, Reflection 審查機制的多節點 Agent
│   └── tools.py                     # 定義供 Agent 調用的外部工具 (如：特定財報檢索器、計算工具)
│
├── experiments/                     # 演算法與策略實驗區
│   ├── chunking_vs_semantic.py      # 固定長度 vs. 語義分段的對照實驗腳本
│   └── advanced_rag_exploration.py  # (進階) 探索 Event-centric 或 GraphRAG 等複雜關聯檢索
│
└── evaluation/                      # 系統成效評估
    ├── dataset.json                 # 事先整理好的 20-50 題財報 QA 測試資料集 (Ground Truth)
    └── metrics_eval.py              # 自動化評估腳本 (計算檢索 Hit Rate、MRR 或 LLM 回答準確率)
```

---

## 🚀 快速開始 (Getting Started)

### 1. 安裝環境依賴
請確保您的 Python 版本 >= 3.10，並執行以下指令安裝所需套件：
```bash
pip install -r requirements.txt
```

### 2. 環境變數設定
複製專案中的 `.env.example` 並重新命名為 `.env`，填入您的 API Keys：
```env
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here  # 選用，用於進階 Reflection
```

### 3. 執行資料解析與建立索引
將目標 PDF 放入 `data/raw/` 目錄後，依序執行：
```bash
# 1. 執行版面解析，產出 Markdown 與 JSON
python src/docling_parser.py

# 2. 建立 Hybrid Search 索引 (FAISS + BM25)
python src/hybrid_retriever.py
```

### 4. 啟動 Agentic Workflow 問答
```bash
python main.py
```

---

## 📊 實驗與評估 (Experiments & Evaluation)

本專案內建 `experiments/` 模組，用於驗證不同策略對檢索結果分佈的影響。我們探討了以下統計與優化思維：
* **Chunking 策略對比**：比較「固定長度切塊 (Fixed-size)」與「語義分段 (Semantic Chunking)」在財報表格檢索上的 Hit Rate。
* **消融實驗 (Ablation Study)**：量化 `Re-ranking` 節點對於最終 LLM 回答準確率的貢獻度，實證拒絕幻覺的成效。

---

> **Note:** 本專案為展示 AI 系統工程、數據解析與 Agentic Workflow 實作能力之概念驗證 (PoC)。
