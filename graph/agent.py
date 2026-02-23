import json
import asyncio
import typing as tp

import numpy as np
from numpy.typing import NDArray

from type import JSONDict, UniqueID
from graph import FiFComponent, FiFDocument, FiFGraph
from memory import FiFMemory, FiFMemoryUnit, FiFTraversalContext
from client import request_llm_response
from query import (
    ORCHESTRATOR_PROMPT,
    DOCUMENT_TRAVERSER_PROMPT, COMPONENT_TRAVERSER_PROMPT,
    SUBQUERY_PLANNER_PROMPT,
    TRAVERSAL_EVALUATOR_PROMPT,
    RERANKER_PROMPT
)

class FiFOrchestrator:
    def __init__(self, fif_graph: FiFGraph, llm_server_url: str, embedding_server_url: str) -> None:
        self.fif_graph: FiFGraph = fif_graph
        self.llm_server_url: str = llm_server_url
        
        self.fif_traverser: FiFMultiStrategyTraverser = FiFMultiStrategyTraverser(fif_graph, llm_server_url, embedding_server_url)
        self.fif_planner: FiFSubqueryPlanner = FiFSubqueryPlanner(fif_graph, llm_server_url)
        self.fif_reranker: FiFReranker = FiFReranker(llm_server_url)
        self.fif_evaluator: FiFTraverseEvaluator = FiFTraverseEvaluator(llm_server_url)
    
    def one_hop(self, ctx: FiFTraversalContext) -> None:
        new_memory_unit: FiFMemoryUnit = FiFMemoryUnit()
        
        complete_prompt: str = ORCHESTRATOR_PROMPT.format(
            query=ctx.fif_memory.query,
            serialized_subtasks=self._subtask_serialization_from_memory(ctx.fif_memory),
            serialized_memory=self._memory_serialization(ctx.fif_memory),
            neighbor_docs=", ".join(self._get_neighbor_doc_title_set_from_memory_unit(ctx.fif_memory.get_last_unit()))
        )
        
        response_result: str = asyncio.run(request_llm_response(self.llm_server_url, complete_prompt))
        response_json: JSONDict = json.loads(response_result)["action"]
        
        new_memory_unit.strategy = (response_json["document_search_mode"], response_json["component_search_mode"], response_json["vector_granularity"])
        
        if response_json["next_action"] == "search":
            self.fif_traverser.run(ctx, new_memory_unit.strategy)
        elif response_json["next_action"] == "replan":
            self.fif_planner.run(ctx)
        elif response_json["next_action"] == "stop":
            self.fif_reranker.run(ctx)
    
    def multi_hop(self, ctx: FiFTraversalContext, max_hop: int) -> None:
        ctx.hop_count = 0
        while ctx.hop_count < max_hop:
            changed = self.one_hop(ctx)
            if not changed: break
            ctx.hop_count += 1
    
    def _subtask_serialization_from_memory(self, memory: FiFMemory) -> str:
        subtask_text_list: tp.List[str] = []
        subquery_status_text_map: tp.Dict[str, str] = dict()
        for memory_unit in memory.history:
            subquery: str = memory_unit.subquery_list[memory_unit.subquery_index]
            status: str = memory_unit.status
            extra = f", Answer: \"{memory_unit.answer}\"" if status == "Resolved" else ""
            subquery_status_text_map[subquery] = f"(Status: {status}{extra})"
        
        for index, subquery in enumerate(memory.get_last_unit().subquery_list):
            subtask_text_list.append(f"- Subtask {index+1}: {subquery} {subquery_status_text_map.get(subquery, '(Status: Unresolved)')}")
        return "\n".join(subtask_text_list)
    
    def _memory_serialization(self, memory: FiFMemory) -> str:
        history_text_list: tp.List[str] = []
        for index, memory_unit in enumerate(memory.history):
            subquery: str = memory_unit.subquery_list[memory_unit.subquery_index]
            document_search_mode, component_search_mode, vector_granularity = memory_unit.strategy
            answer_text: str = f"Success. Found document \"{memory_unit.answer}\"" if memory_unit.answer else f"Failure. Reason: \"{memory_unit.failure_reason}\""
            
            history_text = f"- Unit {index}: Subquery: {subquery}"
            history_text += f", document_search_mode: \"{document_search_mode}\"" if document_search_mode else ""
            history_text += f", component_search_mode: \"{component_search_mode}\"" if component_search_mode else ""
            history_text += f", vector_granularity: \"{vector_granularity}\"" if vector_granularity else ""
            history_text += f", Result: {answer_text}"
            
            history_text_list.append(history_text)
        
        return "\n".join(history_text_list)
    
    def _get_neighbor_doc_title_set_from_memory_unit(self, memory_unit: FiFMemoryUnit) -> tp.Set[str]:
        neighbor_doc_title_set: tp.Set[str] = set()
        anchor_doc_title_set: tp.Set[str] = memory_unit.get_anchor_document_title_set()
        
        for anchor_doc_title in anchor_doc_title_set:
            anchor_document: FiFDocument = self.fif_graph.get_document_from_unique_id((anchor_doc_title, None, None))
            for component_data in anchor_document.component_dict.values():
                neighbor_doc_title_set.update([doc_uid[0] for doc_uid in component_data.linked_document_uid_list])
        
        return neighbor_doc_title_set

from type import Strategy

class FiFMultiStrategyTraverser:
    def __init__(self, fif_graph: FiFGraph, llm_server_url: str, embedding_server_url: str) -> None:
        self.fif_graph: FiFGraph = fif_graph
        self.llm_server_url: str = llm_server_url
        self.embedding_server_url: str = embedding_server_url
    
    def run(self, ctx: FiFTraversalContext, strategy: Strategy) -> None:
        document_search_mode, component_search_mode, vector_granularity = strategy
        
        candidate_document_list: tp.List[FiFDocument] = []
        if document_search_mode == "vector search":
            candidate_document_list = self._document_vector_search(ctx.fif_memory.get_last_unit(), 10) # TODO
        elif document_search_mode == "neighbors":
            self._document_neighbors(ctx) # TODO
        elif document_search_mode == "llm reasoning":
            self._document_llm_reasoning(ctx, 5) # TODO
        
        candidate_component_list: tp.List[FiFComponent] = []
        if component_search_mode == "vector search":
            pass
        elif component_search_mode == "llm reasoning":
            pass
    
    def _document_vector_search(self, memory_unit: FiFMemoryUnit, top_k: int) -> tp.List[FiFDocument]:
        subquery_embedding: NDArray[np.float32] = memory_unit.subquery_embeddings[memory_unit.subquery_index]
        document_score_list: tp.List[float] = []
        for document_data in self.fif_graph.document_list:
            document_score_list.append(float(document_data.embedding @ subquery_embedding.T))
        document_scores: NDArray[np.float32] = np.array(document_score_list)
        document_list: tp.List[FiFDocument] = [self.fif_graph.document_list[doc_index] for doc_index in document_scores.argsort()[:top_k]]
        return document_list
    
    def _document_neighbors(self, ctx: FiFTraversalContext) -> tp.List[FiFDocument]:
        pass
    
    def _document_llm_reasoning(self, ctx: FiFTraversalContext, max_result_count: int) -> tp.List[FiFDocument]:
        last_memory_unit: FiFMemoryUnit = ctx.fif_memory.get_last_unit()
        complete_prompt: str = DOCUMENT_TRAVERSER_PROMPT.format(
            original_query=ctx.fif_memory.query,
            subtask_query=last_memory_unit.subquery_list[last_memory_unit.subquery_index],
            vector_granularity=last_memory_unit.strategy[2],
            candidates=", ".join(self._get_neighbor_doc_title_set_from_memory_unit(last_memory_unit)),
            max_results=str(max_result_count)
        )
        
        response_result: str = asyncio.run(request_llm_response(self.llm_server_url, complete_prompt))
        response_json: JSONDict = json.loads(response_result)
        print(response_json) # TODO
    
    def _component_vector_search(self, ctx: FiFTraversalContext) -> None:
        pass
    
    def _component_llm_reasoning(self, ctx: FiFTraversalContext) -> None:
        pass
    
    def _get_neighbor_doc_title_set_from_memory_unit(self, memory_unit: FiFMemoryUnit) -> tp.Set[str]:
        neighbor_doc_title_set: tp.Set[str] = set()
        anchor_doc_title_set: tp.Set[str] = memory_unit.get_anchor_document_title_set()
        
        for anchor_doc_title in anchor_doc_title_set:
            anchor_document: FiFDocument = self.fif_graph.get_document_from_unique_id((anchor_doc_title, None, None))
            for component_data in anchor_document.component_dict.values():
                neighbor_doc_title_set.update([doc_uid[0] for doc_uid in component_data.linked_document_uid_list])
        
        return neighbor_doc_title_set


class FiFSubqueryPlanner:
    def __init__(self, fif_graph: FiFGraph, llm_server_url: str) -> None:
        self.fif_graph = fif_graph
        self.llm_server_url: str = llm_server_url

    def run(self, ctx: FiFTraversalContext) -> None:
        complete_prompt: str = SUBQUERY_PLANNER_PROMPT.format(
            original_query=ctx.fif_memory.query,
            memory=self._memory_serialization(ctx.fif_memory)
        )
    
    def _memory_serialization(self, memory: FiFMemory) -> str:
        return ""


class FiFTraverseEvaluator:
    def __init__(self, llm_server_url: str) -> None:
        self.llm_server_url: str = llm_server_url
    
    def run(self, ctx: FiFTraversalContext) -> None:
        complete_prompt: str = TRAVERSAL_EVALUATOR_PROMPT.format(
            original_query=ctx.fif_memory.query,
            subtask_query="",
            candidates="",
            subtasks="",
        )


class FiFReranker:
    def __init__(self, llm_server_url: str) -> None:
        self.llm_server_url: str = llm_server_url
    
    def run(self, ctx: FiFTraversalContext) -> None:
        complete_prompt: str = RERANKER_PROMPT.format(
            query=ctx.fif_memory.query,
            candidates="",
            top_k=5
        )
