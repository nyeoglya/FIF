import os
import json
import typing as tp
from math import ceil
from pathlib import Path

import numpy as np
from tqdm import tqdm
from numpy.typing import NDArray

from type import UniqueID, JSONDict
from graph import FiFGraph
from query import MODALITY_AGNOSTIC_QUERY
from lilac import LILaCTraversalContext, LILaCTraverser
from client import request_query_embedding, request_subqueries_embedding

async def embedding_evaluation(
    embedding_server_url: str,
    dev_filepath: str,
    embedding_folderpath: str,
    top_k: int
):
    embedding_filepath_list: tp.List[str] = [
        os.path.join(embedding_folderpath, f) 
        for f in os.listdir(embedding_folderpath) if f.endswith(".npz")
    ]
    
    all_embeddings_list: tp.List[NDArray[np.float32]] = []
    all_ids_list: tp.List[str] = []

    for path in tqdm(embedding_filepath_list, desc="Loading embeddings"):
        with np.load(path, allow_pickle=True) as data:
            doc_title = Path(path).stem
            sub_embs: NDArray[np.float32] = data['subembs']
            metadata: tp.Dict[str, tp.Tuple[int, int]] = data['metadata'].item()
            
            for comp_id, (start, end) in metadata.items():
                for i in range(start, end):
                    all_embeddings_list.append(sub_embs[i])
                    all_ids_list.append(f"{doc_title}_{comp_id}")

    component_embeddings: NDArray[np.float32] = np.array(all_embeddings_list)

    match_count: int = 0
    perfect_match_count: int = 0
    total_mrr: float = 0.0
    total_queries: int = 0
    detailed_count_list: tp.List[int] = [0] * (top_k + 1)

    with open(dev_filepath, "r", encoding="utf-8") as dev_file:
        dev_data_list: tp.List[JSONDict] = [json.loads(line) for line in dev_file]

    for dev_data in tqdm(dev_data_list, desc="MM-Embed Evaluation"):
        query: str = dev_data["question"]
        groundtruth_id_set: tp.Set[str] = set(["_".join(evi) for evi in dev_data["evidence"]])
        
        query_embedding: NDArray[np.float32] = await request_query_embedding(embedding_server_url, MODALITY_AGNOSTIC_QUERY, query)
        similarities: NDArray[np.float32] = component_embeddings @ query_embedding.T
        id_to_max_score: tp.Dict[str, float] = {}
        for idx, score in enumerate(similarities):
            target_id: str = all_ids_list[idx]
            if target_id not in id_to_max_score or score > id_to_max_score[target_id]:
                id_to_max_score[target_id] = score
        
        sorted_results: tp.List[tp.Tuple[str, float]] = sorted(id_to_max_score.items(), key=lambda x: x[1], reverse=True)
        retrieved_ids: tp.List[str] = [item[0] for item in sorted_results[:top_k]]
        
        reciprocal_rank: float = 0.0
        found_at_least_one: bool = False
        for rank, rid in enumerate(retrieved_ids, start=1):
            if rid in groundtruth_id_set:
                if reciprocal_rank == 0.0:
                    detailed_count_list[rank] += 1
                    revised_rank: int = ceil(rank / 3) 
                    reciprocal_rank = 1 / revised_rank
                found_at_least_one = True
        
        total_mrr += reciprocal_rank
        if found_at_least_one:
            match_count += 1
        if groundtruth_id_set.issubset(set(retrieved_ids)):
            perfect_match_count += 1
        total_queries += 1

    print("-" * 30)
    print(f"Total Queries: {total_queries}")
    print(f"Bucketed MRR@{top_k}: {total_mrr / total_queries:.4f}")
    print(f"Top-{top_k} Hit Rate: {match_count / total_queries:.4f}")
    print(f"Top-{top_k} Perfect Match Rate: {perfect_match_count / total_queries:.4f}")
    print(f"Rank Distribution: {detailed_count_list[1:]}")
    print("-" * 30)

async def lilac_retrieval_evaluation(
    graph: FiFGraph,
    llm_server_url: str,
    embedding_server_url: str,
    dev_filepath: str,
    beam_size: int,
    top_k: int,
    max_hop: int,
):
    lilac_traverser: LILaCTraverser = LILaCTraverser(graph)
    
    match_count: int = 0
    perfect_match_count: int = 0
    total_mrr: float = 0.0
    total_queries: int = 0
    detailed_count_list: tp.List[int] = [0] * (top_k + 1)

    with open(dev_filepath, "r", encoding="utf-8") as dev_file:
        for dev_line in tqdm(dev_file, desc="LILaC Retrieval Evaluating"):
            dev_data: JSONDict = json.loads(dev_line)
            query: str = dev_data["question"]
            groundtruth_id_set: tp.Set[str] = set(["_".join(evi) for evi in dev_data["evidence"]])
            
            query_embedding: NDArray[np.float32] = await request_query_embedding(embedding_server_url, MODALITY_AGNOSTIC_QUERY, query)
            subquery_embeddings: NDArray[np.float32] = await request_subqueries_embedding(llm_server_url, embedding_server_url, query)
            
            ctx: LILaCTraversalContext = LILaCTraversalContext(
                query_embedding=query_embedding,
                subquery_embeddings=subquery_embeddings,
                beam_size=beam_size
            )
            lilac_traverser.find_entry(ctx)
            lilac_traverser.multi_hop(ctx, max_hop)
            
            retrieved_component_unique_id_list: tp.List[UniqueID] = lilac_traverser.get_component_unique_id_list(ctx, top_k)
            retrieved_ids: tp.List[str] = [
                f"{uid[0]}_{uid[1]}" 
                for uid in retrieved_component_unique_id_list 
                if uid[1]
            ]
            
            reciprocal_rank: float = 0.0
            found_at_least_one: bool = False
            for rank, retrieved_id in enumerate(retrieved_ids, start=1):
                if retrieved_id in groundtruth_id_set:
                    if reciprocal_rank == 0.0:
                        detailed_count_list[rank] += 1
                        revised_rank: int = ceil(rank / 3)
                        reciprocal_rank = 1 / revised_rank
                    found_at_least_one = True
            
            total_mrr += reciprocal_rank
            if found_at_least_one:
                match_count += 1
            if groundtruth_id_set.issubset(set(retrieved_ids)):
                perfect_match_count += 1
            total_queries += 1
    
    print("-" * 30)
    print(f"Total Queries: {total_queries}")
    print(f"Bucketed MRR@{top_k}: {total_mrr / total_queries:.4f}")
    print(f"Top-{top_k} Hit Rate: {match_count / total_queries:.4f}")
    print(f"Top-{top_k} Perfect Match Rate: {perfect_match_count / total_queries:.4f}")
    print(f"Rank Distribution: {detailed_count_list[1:]}")
    print("-" * 30)

def end_to_end_evaluation(
    graph: FiFGraph,
    llm_server_url: str,
    embedding_server_url: str,
    dev_filepath: str,
    beam_size: int,
    top_k: int,
    max_hop: int,
):
    pass
