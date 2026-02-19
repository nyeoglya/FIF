from __future__ import annotations

import httpx
import numpy as np
from numpy.typing import NDArray

from type import JSONDict
from query import SUBQUERY_DIVIDE_QUERY, SUBQUERY_MODALITY_QUERY, MODALITY_INSTRUCTION_QUERY, MODALITY_AGNOSTIC_QUERY

async def request_embedding(server_url: str, query: str) -> NDArray[np.float32]:
    payload: JSONDict = {
        "text": query,
        "img_path": "",
        "bounding_box": None
    }
    async with httpx.AsyncClient() as client:
        res = await client.post(f"{server_url}/embed", json=payload, timeout=120)
    res.raise_for_status()
    return np.array(res.json()["embedding"], dtype=np.float32)

async def get_llm_response(server_url: str, query: str) -> str:
    payload: JSONDict = {
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": query}]
            }
        ],
        "temperature": 0.2,
        "repetition_penalty": 1.05,
        "top_p": 0.85,
        "max_tokens": 512
    }
    headers = {"Content-Type": "application/json"}
    async with httpx.AsyncClient() as client:
        res = await client.post(server_url, headers=headers, json=payload, timeout=120)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"]

async def request_query_embedding(server_url: str, instruction: str, query_text: str) -> NDArray[np.float32]:
    payload: JSONDict = {
        "instruction": instruction,
        "text": query_text,
        "img_path": "",
    }
    async with httpx.AsyncClient() as client:
        res = await client.post(f"{server_url}/embed/query", json=payload, timeout=120)
    res.raise_for_status()
    return np.array(res.json()["embedding"], dtype=np.float32)

async def request_subqueries_embedding(llm_server_url: str, embedding_server_url: str, query: str) -> NDArray[np.float32]:
    llm_response = await get_llm_response(llm_server_url, SUBQUERY_DIVIDE_QUERY.format(query=query))
    subqueries = llm_response.replace("\n", "").split(";")
    embeddings: list[NDArray[np.float32]] = []
    for subquery in subqueries:
        subquery = subquery.strip()
        if not subquery: continue
        modality = await get_llm_response(llm_server_url, SUBQUERY_MODALITY_QUERY.format(subquery=subquery))
        embedding = await request_query_embedding(embedding_server_url, MODALITY_INSTRUCTION_QUERY.get(modality.strip(), MODALITY_AGNOSTIC_QUERY), subquery)
        embeddings.append(embedding)
    return np.stack(embeddings)
