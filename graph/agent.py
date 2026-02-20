import asyncio

from graph import FiFGraph
from query import ORCHESTRATOR_PROMPT
from memory import FiFTraversalContext
from client import request_llm_response

class FiFOrchestrator:
    def __init__(self, fif_graph: FiFGraph, llm_server_url: str) -> None:
        self.fif_graph: FiFGraph = fif_graph
        self.llm_server_url: str = llm_server_url
    
    def one_hop(self, ctx: FiFTraversalContext) -> None:
        complete_prompt: str = ORCHESTRATOR_PROMPT.format(
            query=r"애플의 2023 회계연도 R&D 투자 총액과 전년 대비 증가율을 알려줘",
            serialized_subtasks=r"""- Subtask 1: 애플의 2023년 연례 보고서(10-K) 찾기 (Status: Resolved, Answer: "Apple 2023 Form 10-K retrieved.")
- Subtask 2: 2023년 R&D 지출 금액 추출 (Status: Unresolved)
- Subtask 3: 2022년 R&D 지출 금액 확인 및 증가율 계산 (Status: Unresolved)""",
            serialized_memory=r"""- [Step 0]: document_search_mode="vector search", component_search_mode="vector search". 
  Result: Success. Found Apple 2023 10-K document.
- [Step 1]: Target="2023 R&D spending", anchor=0, document_search_mode="neighbors", component_search_mode="vector search". 
  Result: Failure. 'Neighbors' mode only returned marketing expenses and executive compensation paragraphs. R&D section not found in adjacent blocks.""",
            neighbor_docs=r"""["Executive Compensation 2023", "Marketing and Advertising Strategy", "Shareholder Information"]"""
        )
        result: str = asyncio.run(request_llm_response(self.llm_server_url, complete_prompt))
        print(result)
        pass
    
    def multi_hop(self, top_k: int) -> None:
        pass

class FiFMultiStrategyTraverser:
    def __init__(self) -> None:
        pass
    
    def hop_scope(self) -> None:
        pass
    
    def select_granularity(self) -> None:
        pass
    
    def llm_reasoning(self) -> None:
        pass

class FiFSubqueryPlanner:
    def __init__(self) -> None:
        pass

class FiFReranker:
    def __init__(self) -> None:
        pass
