import os
import json
import asyncio
import typing as tp
from pathlib import Path

import numpy as np
from tqdm import tqdm
from numpy.typing import NDArray

from type import UniqueID
from graph import FiFGraph
from query import MODALITY_AGNOSTIC_QUERY
from lilac import LILaCTraversalContext, LILaCTraverser
from client import request_query_embedding, request_subqueries_embedding

def embedding_evaluation(
    embedding_server_url: str,
    dev_filepath: str,
    embedding_folderpath: str,
    top_k: int
):
    embedding_filepath_list: tp.List[str] = [
        os.path.join(embedding_folderpath, filepath)
        for filepath in os.listdir(embedding_folderpath)
        if filepath.endswith(".npz")
    ]
    
    temp_embedding_list: tp.List[NDArray[np.float32]] = []
    doc_component_id_map: tp.List[str] = []

    for embedding_filepath in tqdm(embedding_filepath_list):
        with np.load(embedding_filepath, allow_pickle=True) as embedding_data:
            doc_title: str = Path(embedding_filepath).stem
            sub_embeddings = embedding_data['subembs'] # (M, D)
            metadata = embedding_data['metadata'].item()
            
            for comp_id, (start, end) in metadata.items():
                for i in range(start, end):
                    temp_embedding_list.append(sub_embeddings[i]) # Each sub embeddings are the individual search target
                    doc_component_id_map.append(f"{doc_title}_{comp_id}")

    component_embeddings = np.array(temp_embedding_list)
    doc_component_id_map_arr = np.array(doc_component_id_map)

    match_count = 0
    perfect_match_count = 0
    total_mrr = 0.0
    total_queries = 0
    with open(dev_filepath, "r", encoding="utf-8") as dev_file:
        for dev_line in tqdm(dev_file, desc="MM-Embed Retrieval Evaluating"):
            dev_data = json.loads(dev_line)
            query = dev_data["question"]
            groundtruth_id_set = set(["_".join(evi) for evi in dev_data["evidence"]])
            
            query_embedding = asyncio.run(
                request_query_embedding(embedding_server_url, MODALITY_AGNOSTIC_QUERY, query)
            )
            similarities = component_embeddings @ query_embedding.T
            
            top_indices = np.argsort(similarities)[::-1][:top_k]
            retrieved_ids = doc_component_id_map_arr[top_indices]
            
            reciprocal_rank = 0
            found_at_least_one = False
            
            for rank, retrieved_id in enumerate(retrieved_ids, start=1):
                if retrieved_id in groundtruth_id_set:
                    if reciprocal_rank == 0:
                        reciprocal_rank = 1 / rank
                    found_at_least_one = True
            
            total_mrr += reciprocal_rank
            if found_at_least_one:
                match_count += 1
            if groundtruth_id_set.issubset(set(retrieved_ids)):
                perfect_match_count += 1
            total_queries += 1
    
    mrr_result = total_mrr / total_queries
    print("-" * 30)
    print(f"Total Queries: {total_queries}")
    print(f"MRR@{top_k}: {mrr_result:.4f}")
    print(f"Top-{top_k} Hit Rate: {match_count / total_queries:.4f}")
    print(f"Top-{top_k} Perfect Match Rate: {perfect_match_count / total_queries:.4f}")
    print("-" * 30)

def lilac_retrieval_evaluation( # TODO: retrieval caching
    graph: FiFGraph,
    llm_server_url: str,
    embedding_server_url: str,
    dev_filepath: str,
    beam_size: int,
    top_k: int,
    max_hop: int,
):
    lilac_traverser = LILaCTraverser(graph)
    
    match_count = 0
    perfect_match_count = 0
    total_mrr = 0.0
    total_queries = 0
    with open(dev_filepath, "r", encoding="utf-8") as dev_file:
        for dev_line in tqdm(dev_file, desc="LILaC Retrieval Evaluating"):
            dev_data = json.loads(dev_line)
            query = dev_data["question"]
            groundtruth_id_set = set(["_".join(evi) for evi in dev_data["evidence"]])
            
            query_embedding = asyncio.run(
                request_query_embedding(embedding_server_url, MODALITY_AGNOSTIC_QUERY, query)
            )
            subquery_embeddings = asyncio.run(
                request_subqueries_embedding(llm_server_url, embedding_server_url, query)
            )
            ctx: LILaCTraversalContext = LILaCTraversalContext(
                query_embedding=query_embedding,
                subquery_embeddings=subquery_embeddings,
                beam_size=beam_size
            )
            lilac_traverser.find_entry(ctx)
            lilac_traverser.multi_hop(ctx, max_hop)
            retrieved_component_unique_id_list: tp.List[UniqueID] = lilac_traverser.get_component_unique_id_list(ctx, top_k)
            retrieved_ids: tp.List[str] = [
                component_unique_id[0] + "_" + component_unique_id[1]
                for component_unique_id in retrieved_component_unique_id_list
                if component_unique_id[1]
            ]
            
            reciprocal_rank = 0
            found_at_least_one = False
            for rank, retrieved_id in enumerate(retrieved_ids, start=1):
                if retrieved_id in groundtruth_id_set:
                    if reciprocal_rank == 0:
                        reciprocal_rank = 1 / rank
                    found_at_least_one = True
            
            total_mrr += reciprocal_rank
            if found_at_least_one:
                match_count += 1
            if groundtruth_id_set.issubset(set(retrieved_ids)):
                perfect_match_count += 1
            total_queries += 1
    
    mrr_result = total_mrr / total_queries
    print("-" * 30)
    print(f"Total Queries: {total_queries}")
    print(f"MRR@{top_k}: {mrr_result:.4f}")
    print(f"Top-{top_k} Hit Rate: {match_count / total_queries:.4f}")
    print(f"Top-{top_k} Perfect Match Rate: {perfect_match_count / total_queries:.4f}")
    print("-" * 30)

def lilac_end_to_end_evaluation(
    graph: FiFGraph,
    llm_server_url: str,
    embedding_server_url: str,
    dev_filepath: str,
    beam_size: int,
    top_k: int,
    max_hop: int,
):
    pass
