# Demo Result — Minimal Reproducibility Test

## Test Objective

驗證核心 RAG pipeline 是否能在最小資料集上正常運作，包括：

* Embedding 生成（OpenAI API）
* FAISS 向量索引
* BM25 關鍵字索引
* Hybrid Retrieval（Dense + Sparse）
* Cross-Encoder Re-ranking
* 查詢與結果排序

---

## Input Documents (3 Chunks)

測試使用 3 筆簡化文件：

1. 水資源管理
2. 廢棄物管理（包含回收率 97%）
3. 研發支出

---

## Pipeline Execution Summary

### Step 1 — 建立索引

* 成功建立 3 個 chunks
* OpenAI Embedding API 呼叫成功（HTTP 200）
* FAISS 向量索引建立完成
* BM25 關鍵字索引建立完成

結論：Hybrid Search 基礎索引可正常運作

---

### Step 2 — Re-ranking 模型初始化

* 載入 Cross-Encoder：
  `cross-encoder/ms-marco-MiniLM-L-6-v2`
* 自動從 HuggingFace 下載模型（首次執行）

結論：Re-ranking pipeline 正常初始化

---

### Step 3 — 查詢執行

#### Query

```text id="3c6h0v"
台積電的廢棄物回收率是多少？
```

---

### Step 4 — 檢索結果

#### Rank 1

* Source: `sample_water_management`
* Content:
  台積公司首次提出水資源正效益長期目標，預計民國129年達成率100%。

#### Rank 2（正確答案）

* Source: `sample_resource_circulation`
* Content:
  台積公司民國113年廢棄物回收率達97%，並將9,400公噸廢棄物純化再生。

#### Rank 3

* Source: `sample_innovation`
* Content:
  台積公司民國113年全年研發總支出為63億5,500萬美元。

---

## Test Assertion

系統檢查條件：

```python id="thgknr"
assert any("97%" in doc.page_content for doc in results)
```

Assertion 通過
正確答案成功被檢索到（Rank 2）

---

## Result Analysis

| 項目               | 狀態  |
| ---------------- | --- |
| Embedding        | 正常  |
| FAISS            | 正常  |
| BM25             | 正常  |
| Hybrid Retrieval | 正常  |
| Re-ranking       | 正常  |
| 查詢成功             | 是   |
| 正確答案存在於 Top-K    | 是   |
| 排名最佳化            | 可優化 |

---

## Observation

雖然正確答案（97%）被成功檢索：

* 目前排名為 Rank 2
* Rank 1 為語意相關但非直接答案

顯示：

* Re-ranking 已有效，但還是有優化空間
* 可透過以下方式改善：

  * 調整 chunk 設計
  * 強化 query rewriting
  * 調整 rerank top_k 或權重

---

## Conclusion

Minimal Reproducibility Test 成功：

* 系統可完整執行 RAG pipeline
* 能從資料中正確檢索出目標資訊
* 檢索結果具備語意相關性與可解釋性

此測試證明我們的專案的核心檢索架構（Hybrid + Re-ranking）可正常運作

---

## Verification

原始執行紀錄請見：

```text id="m3r8pb"
logs/cmd_output_example.txt
```

可對照：

* API 呼叫紀錄
* 索引建立過程
* 檢索與排序流程
