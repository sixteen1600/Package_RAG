
import os
import logging
from typing import List, Dict, Any, TypedDict
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from hybrid_retriever import AdvancedHybridRetriever

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==========================================
# 1. 定義 State (狀態字典)：Agent 溝通的共用記憶體
# ==========================================


class AgentState(TypedDict):
    question: str
    plan: List[str]
    context: str
    draft_response: str
    feedback: str
    revision_count: int

# ==========================================
# 2. 定義結構化輸出 (Pydantic Models)
# ==========================================


class Plan(BaseModel):
    """規劃階段的輸出結構"""
    steps: List[str] = Field(description="解決使用者問題的具體步驟列表")


class ReflectionOutput(BaseModel):
    """反思/審查階段的輸出結構"""
    is_acceptable: bool = Field(description="草稿是否合格且無幻覺")
    feedback: str = Field(description="如果不合格，請給出具體修改建議；如果合格則留空")

# ==========================================
# 3. 實作 Agent 節點 (Nodes)
# ==========================================


class FinancialAgentWorkflow:
    # 增加 external_retriever 參數
    def __init__(self, external_retriever=None):
        # 設定 LLM
        self.llm_generator = ChatOpenAI(model="gpt-4.1-nano", temperature=0.2)
        self.llm_reviewer = ChatOpenAI(model="gpt-4o", temperature=0)

        # 模擬載入檢索工具
        # self.retriever = AdvancedHybridRetriever()
        # self.retriever.load_index()
        from src.tools import FinanceTools
        self.tools_factory = FinanceTools(
            external_retriever=external_retriever)
        self.retriever_tool = self.tools_factory.get_retriever_tool()

        # 綁定工具
        self.llm_with_tools = self.llm_generator.bind_tools(
            [self.retriever_tool])

    def plan_node(self, state: AgentState) -> Dict:
        """Planning: 拆解問題"""
        logger.info("[Node: Planner] 正在制定檢索計畫")
        prompt = f"你是一個專業的金融分析師。請將以下問題拆解為最多 3 個具體的檢索步驟：\n問題：{state['question']}"

        # 強制 LLM 輸出 JSON 格式的 Plan
        planner = self.llm_generator.with_structured_output(Plan)
        plan_result = planner.invoke(prompt)

        return {"plan": plan_result.steps}

    def research_node(self, state: AgentState) -> Dict:
        """
        修正後的 Tool Use 節點：實際調用工具進行檢索
        """
        logger.info("[Node: Researcher] 正在調用實體工具執行檢索")

        # 這裡會遍歷 Planning 產生的步驟，並實際執行工具
        all_result = []
        for step in state["plan"]:
            # 使用 .invoke 直接執行工具 ??
            result = self.retriever_tool.invoke({"query": step})
            all_result.append(result)

        return {"context": "\n\n".join(all_result)}

        # """Tool Use: 執行檢索 (此處用假資料模擬)"""
        # # 實務上你會對 state['plan'] 中的每個步驟呼叫 self.retriever.retrieve()
        # # 這裡用 mock_context 模擬
        # mock_context = (
        #     "【檢索結果 1】：台積電 2024 年第一季受惠於 AI 晶片需求，營收達新台幣 5926 億元。\n",
        #     "【檢索結果 2】：風險方面，地緣政治與供應鏈集中度仍是 2024 年主要關注點。"
        # )
        # return {"context":mock_context}

    def draft_node(self, state: AgentState) -> Dict:
        """Drafting: 根據資料撰寫草稿"""
        logger.info("[Node: Drafter] 正在撰寫分析草稿")
        prompt = (
            f"請根據以下資料回答。如果資料不包含答案，請誠實說不知道。\n\n"
            f"資料:{state['context']}\n"
            f"審查建議(若有):{state.get('feedback', '無')}\n\n"
            f"問題 : {state['question']}"
        )
        response = self.llm_generator.invoke(prompt)
        return {"draft_response": response.content}

    def reflection_node(self, state: AgentState) -> Dict:
        """Reflection: 嚴格審查草稿是否產生幻覺或遺漏"""
        logger.info("[Node: Reviewer] 正在進行嚴格的自我審查 (Reflection)")
        prompt = (
            f"你是一個嚴格的審計員。請檢查以下草稿是否完全基於提供的資料，"
            f"有沒有產生資料以外的數字或幻覺?\n\n"
            f"原始資料:{state['context']}\n"
            f"分析草稿:{state['draft_response']}"
        )

        # 你漏掉的執行邏輯
        reviewr = self.llm_reviewer.with_structured_output(ReflectionOutput)
        review_result = reviewr.invoke(prompt)

        count = state.get("revision_count", 0) + 1
        return {
            "feedback": review_result.feedback,
            "revision_count": count,
            "is_acceptable": review_result.is_acceptable,
        }

    # ==========================================
    # 4. 定義條件路由 (Conditional Edges)
    # ==========================================
    def should_continue(self, state: AgentState) -> str:
        """決定是否需要重寫"""
        is_acceptable = state.get("is_acceptable", False)

        # 如果合格，或是重試超過 2 次（避免無限迴圈浪費錢），就結束
        if is_acceptable or state.get("revision_count", 0) >= 2:
            logger.info(">>> 審查通過或達到上限 <<<")
            return END
        else:
            logger.warning(f">>> 發現瑕疵!退回重寫。退回 : {state['feedback']} <<<")
            return "drafter"

    # ==========================================
    # 5. 組裝 LangGraph 工作流
    # ==========================================
    def build_graph(self):
        workflow = StateGraph(AgentState)

        # 加入節點
        workflow.add_node("planner", self.plan_node)
        workflow.add_node("researcher", self.research_node)
        workflow.add_node("drafter", self.draft_node)
        workflow.add_node("reviewer", self.reflection_node)

        # 定義流程線 (Edges)
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "researcher")
        workflow.add_edge("researcher", "drafter")
        workflow.add_edge("drafter", "reviewer")

        # 條件判斷：如果 reviewer 說不 OK，退回 drafter；否則結束
        workflow.add_conditional_edges("reviewer", self.should_continue)

        return workflow.compile()


# CLI 測試
if __name__ == "__main__":
    agent = FinancialAgentWorkflow()
    graph = agent.build_graph()

    print("========================================")
    print("啟動 Agentic Workflow")
    print("========================================\n")

    inputs = {
        "question": "台積電 2024 年第一季的營收是多少？目前面臨哪些主要風險？",
        "revision_count": 0
    }

    # 執行整個 LangGraph 工作流
    for output in graph.stream(inputs):
        for node_name, state_update in output.items():
            pass  # 這裡可以印出每個節點的狀態變化，我們已經在 logger 處理了

    # 取得最終結果
    final_state = graph.get_state(inputs).values
    print("\n最終生成的財報分析")
    print(final_state.get("draft_response"))
