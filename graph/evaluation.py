import os
import re
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
from lilac import FiFTraversalContext, LILaCTraverser
from client import request_query_embedding, request_subqueries_embedding

def clean_id_string(s: str) -> str:
    s = re.sub(r'[^a-zA-Z0-9_]', '_', str(s))
    s = re.sub(r'_+', '_', s)
    return s.strip('_')

async def embedding_evaluation(
    embedding_server_url: str,
    dev_filepath: str,
    embedding_folderpath: str,
    top_k: int
):
    embedding_filepath_list = [
        os.path.join(embedding_folderpath, f) 
        for f in os.listdir(embedding_folderpath) if f.endswith(".npz")
    ]
    
    all_embeddings_list: tp.List[NDArray[np.float32]] = []
    all_ids_list: tp.List[str] = []
    for path in tqdm(embedding_filepath_list, desc="Loading component embeddings"):
        with np.load(path, allow_pickle=True) as data:
            doc_title = Path(path).stem
            metadata: tp.Dict[str, tp.Tuple[int, int]] = data['metadata'].item()
            for comp_id in metadata.keys():
                if comp_id in data:
                    all_embeddings_list.append(data[comp_id])
                    all_ids_list.append(f"{clean_id_string(doc_title)}_{clean_id_string(comp_id)}")

    component_embeddings: NDArray[np.float32] = np.array(all_embeddings_list)

    match_count: int = 0
    perfect_match_count: int = 0
    total_mrr: float = 0.0
    total_queries: int = 0
    detailed_count_list: tp.List[int] = [0] * (top_k + 1)

    with open(dev_filepath, "r", encoding="utf-8") as dev_file:
        dev_data_list = [json.loads(line) for line in dev_file]
    
    for dev_data in tqdm(dev_data_list, desc="MMEmbed Baseline Evaluation"):
        query = dev_data["question"]
        groundtruth_id_set = {clean_id_string("_".join(evi)) for evi in dev_data["evidence"]}

        query_embedding = await request_query_embedding(embedding_server_url, MODALITY_AGNOSTIC_QUERY, query)

        similarities = component_embeddings @ query_embedding.T
        
        sorted_indices = np.argsort(-similarities)[:top_k]
        retrieved_ids: list[str] = [all_ids_list[idx] for idx in sorted_indices]
        
        reciprocal_rank: float = 0.0
        found_at_least_one: bool = False
        
        for rank, rid in enumerate(retrieved_ids, start=1):
            if rid in groundtruth_id_set:
                if reciprocal_rank == 0.0:
                    detailed_count_list[rank] += 1
                    revised_rank = ceil(rank / 3) 
                    reciprocal_rank = 1 / revised_rank
                found_at_least_one = True
                break
        
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
    embedding_cache_filepath: str = "",
):
    lilac_traverser: LILaCTraverser = LILaCTraverser(graph)

    query_embedding_cache: tp.Dict[str, NDArray[np.float32]] = {}
    subquery_embedding_cache: tp.Dict[str, NDArray[np.float32]] = {}
    if os.path.exists(embedding_cache_filepath):
        with np.load(embedding_cache_filepath, allow_pickle=True) as cache_data:
            query_embedding_cache = cache_data["query_embedding_cache"].item()
            subquery_embedding_cache = cache_data["subquery_embedding_cache"].item()

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
            groundtruth_id_set = {clean_id_string(gid) for gid in groundtruth_id_set}
            
            if query in query_embedding_cache:
                query_embedding = query_embedding_cache[query]
            else:
                query_embedding = await request_query_embedding(embedding_server_url, MODALITY_AGNOSTIC_QUERY, query)
                query_embedding_cache[query] = query_embedding

            if query in subquery_embedding_cache:
                subquery_embeddings = subquery_embedding_cache[query]
            else:
                subquery_embeddings = await request_subqueries_embedding(llm_server_url, embedding_server_url, query)
                subquery_embedding_cache[query] = subquery_embeddings
            
            ctx: FiFTraversalContext = FiFTraversalContext(
                query_embedding=query_embedding,
                subquery_embeddings=subquery_embeddings, # TODO: better performance when uses query_embedding[None, :]...
                beam_size=beam_size
            )
            lilac_traverser.find_entry(ctx)
            lilac_traverser.multi_hop(ctx, max_hop)
            
            retrieved_component_unique_id_list: tp.List[UniqueID] = lilac_traverser.get_component_unique_id_list(ctx, top_k)
            retrieved_ids: tp.List[str] = []
            for uid in retrieved_component_unique_id_list:
                if uid[1]:
                    clean_doc_title = clean_id_string(uid[0])
                    clean_comp_id = clean_id_string(uid[1])
                    retrieved_ids.append(f"{clean_doc_title}_{clean_comp_id}")

            reciprocal_rank: float = 0.0
            found_at_least_one: bool = False
            for rank, retrieved_id in enumerate(retrieved_ids, start=1):
                if retrieved_id in groundtruth_id_set:
                    if reciprocal_rank == 0.0:
                        detailed_count_list[rank] += 1
                        revised_rank: int = ceil(rank / 3)
                        reciprocal_rank = 1 / revised_rank
                    found_at_least_one = True
                    break
            
            total_mrr += reciprocal_rank
            if found_at_least_one:
                match_count += 1
            if groundtruth_id_set.issubset(set(retrieved_ids)):
                perfect_match_count += 1
            total_queries += 1

    if embedding_cache_filepath:
        np.savez(
            embedding_cache_filepath,
            query_embedding_cache=np.array(query_embedding_cache, dtype=object),
            subquery_embedding_cache=np.array(subquery_embedding_cache, dtype=object),
        )

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
