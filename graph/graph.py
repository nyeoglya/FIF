from __future__ import annotations

import json
import typing as tp
from pathlib import Path

import numpy as np
from tqdm import tqdm
from numpy.typing import NDArray

from type import JSONDict, UniqueID
from common import safe_doc_id


class FiFNode:
    def __init__(self, unique_id: UniqueID, embedding: NDArray[np.float32]) -> None:
        self.unique_id: UniqueID = unique_id
        self.embedding: NDArray[np.float32] = embedding


class FiFSubComponent(FiFNode):
    pass


class FiFComponent(FiFNode):
    def __init__(self, unique_id: UniqueID, embedding: NDArray[np.float32]) -> None:
        super().__init__(unique_id, embedding)
        self.original_json_data: JSONDict = {}
        self.subcomponent_list: list[FiFSubComponent] = []
        self.linked_document_uid_list: list[UniqueID] = []
    
    def get_subcomponent_embeddings(self) -> NDArray[np.float32]:
        embedding_list: list[NDArray[np.float32]] = [
            subcomponent.embedding for subcomponent in self.subcomponent_list
        ]
        return np.array(embedding_list)

class FiFDocument(FiFNode):
    def __init__(self, unique_id: UniqueID, embedding: NDArray[np.float32]) -> None:
        super().__init__(unique_id, embedding)
        self.document_summary: str = ""
        self.component_dict: dict[str, FiFComponent] = {}


class FiFGraph:
    def __init__(self) -> None:
        self.component_embeddings: NDArray[np.float32] = np.array([])
        self.document_list: list[FiFDocument] = []
        self.component_loc_to_id_map: list[UniqueID] = []
        self.doc_title_to_loc_map: dict[str, int] = dict()
        self.subcomponent_embeddings: NDArray[np.float32] = np.array([])
        self.subcomponent_to_component_map: list[UniqueID] = []

    def get_document_from_unique_id(self, unique_id: UniqueID):
        doc_loc: int = self.doc_title_to_loc_map[unique_id[0]]
        return self.document_list[doc_loc]

    def get_component_from_unique_id(self, unique_id: UniqueID) -> FiFComponent:
        assert unique_id[1], "The scope of the UniqueID is too broad."
        document_data: FiFDocument = self.get_document_from_unique_id(unique_id)
        return_component: tp.Optional[FiFComponent] = document_data.component_dict.get(unique_id[1], None)
        assert return_component, "Corrupted graph. Can't find component from UniqueID"
        return return_component

    @staticmethod
    def construct_graph(
        json_folderpath: str,
        embedding_folderpath: str,
        doc_summary_filepath: str,
    ) -> FiFGraph:
        component_json_dict = _load_component_json(Path(json_folderpath))
        doc_summary_dict = _load_doc_summaries(Path(doc_summary_filepath))

        new_graph = FiFGraph()
        temp_component_embedding_list: list[NDArray[np.float32]] = []
        temp_subcomponent_embedding_list: list[NDArray[np.float32]] = []
        temp_subcomponent_to_component_map: list[UniqueID] = []

        embedding_folder = Path(embedding_folderpath)
        embedding_filepaths = sorted(embedding_folder.glob("*.npz"))

        for embedding_filepath in tqdm(embedding_filepaths, desc="Loading Embedding File"):
            with np.load(embedding_filepath, allow_pickle=True) as embedding_data:
                doc_title: str = embedding_filepath.stem
                metadata_dict: dict[str, tuple[int, int]] = embedding_data["metadata"].item()

                new_document = FiFDocument(
                    (doc_title, None, None),
                    embedding_data["doc_emb"],
                )
                new_document.document_summary = doc_summary_dict.get(doc_title, "")

                for key_data in sorted(metadata_dict.keys()):
                    if key_data not in embedding_data:
                        continue
                    component_uid: UniqueID = (doc_title, key_data, None)
                    new_component = FiFComponent(component_uid, embedding_data[key_data])
                    temp_component_embedding_list.append(embedding_data[key_data])
                    new_graph.component_loc_to_id_map.append(component_uid)
                    new_component.original_json_data = component_json_dict[component_uid]
                    new_component.linked_document_uid_list = [
                        (safe_doc_id(link_title), None, None)
                        for link_title in new_component.original_json_data["hyperlinks"]
                    ]
                    start_pos, end_pos = metadata_dict[key_data]
                    subcomponent_embeddings = embedding_data["subembs"][start_pos:end_pos]
                    new_component.subcomponent_list = [
                        FiFSubComponent((doc_title, key_data, str(i)), emb)
                        for i, emb in enumerate(subcomponent_embeddings)
                    ]
                    for emb in subcomponent_embeddings:
                        temp_subcomponent_embedding_list.append(emb)
                        temp_subcomponent_to_component_map.append(component_uid)
                    new_document.component_dict[key_data] = new_component

                new_graph.doc_title_to_loc_map[doc_title] = len(new_graph.document_list)
                new_graph.document_list.append(new_document)

        new_graph.component_embeddings = np.array(temp_component_embedding_list)
        new_graph.subcomponent_embeddings = np.array(temp_subcomponent_embedding_list) if temp_subcomponent_embedding_list else np.array([])
        new_graph.subcomponent_to_component_map = temp_subcomponent_to_component_map
        return new_graph

    @staticmethod
    def save(graph: FiFGraph, filepath: str) -> None:
        base = Path(filepath)
        node_embeddings: list[NDArray[np.float32]] = []

        def _store(emb: NDArray[np.float32]) -> int:
            idx = len(node_embeddings)
            node_embeddings.append(emb)
            return idx

        documents_json: list[dict[str, tp.Any]] = []
        for doc in tqdm(graph.document_list, desc="Saving Graph"):
            components_json: list[dict[str, tp.Any]] = []
            for comp in doc.component_dict.values():
                sub_start = len(node_embeddings)
                for sub in comp.subcomponent_list:
                    node_embeddings.append(sub.embedding)
                components_json.append({
                    "unique_id": list(comp.unique_id),
                    "emb_idx": _store(comp.embedding),
                    "original_json_data": comp.original_json_data,
                    "linked_document_uid_list": [list(uid) for uid in comp.linked_document_uid_list],
                    "sub_uid_list": [list(sub.unique_id) for sub in comp.subcomponent_list],
                    "sub_emb_range": [sub_start, sub_start + len(comp.subcomponent_list)],
                })
            documents_json.append({
                "unique_id": list(doc.unique_id),
                "emb_idx": _store(doc.embedding),
                "document_summary": doc.document_summary,
                "components": components_json,
            })

        graph_json: dict[str, tp.Any] = {
            "component_id_map": [list(uid) for uid in graph.component_loc_to_id_map],
            "doc_title_to_loc_map": graph.doc_title_to_loc_map,
            "documents": documents_json,
            "subcomponent_to_component_map": [list(uid) for uid in graph.subcomponent_to_component_map],
        }

        with open(base.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(graph_json, f, ensure_ascii=False)

        np.savez(
            base.with_suffix(".npz"),
            node_embeddings=np.array(node_embeddings),
            component_embeddings=graph.component_embeddings,
            subcomponent_embeddings=graph.subcomponent_embeddings,
        )

    @staticmethod
    def load(filepath: str) -> FiFGraph:
        base = Path(filepath)
        with open(base.with_suffix(".json"), "r", encoding="utf-8") as f:
            graph_json: dict[str, tp.Any] = json.load(f)
        npz_data = np.load(base.with_suffix(".npz"))
        node_embeddings: NDArray[np.float32] = npz_data["node_embeddings"]

        graph = FiFGraph()
        graph.component_loc_to_id_map = [tuple(uid) for uid in graph_json["component_id_map"]]
        graph.doc_title_to_loc_map = graph_json["doc_title_to_loc_map"]
        graph.component_embeddings = npz_data["component_embeddings"]
        graph.subcomponent_embeddings = npz_data["subcomponent_embeddings"] if "subcomponent_embeddings" in npz_data else np.array([])
        graph.subcomponent_to_component_map = [tuple(uid) for uid in graph_json.get("subcomponent_to_component_map", [])]

        for doc_data in tqdm(graph_json["documents"], desc="Loading Graph"):
            doc = FiFDocument(tuple(doc_data["unique_id"]), node_embeddings[doc_data["emb_idx"]])
            doc.document_summary = doc_data["document_summary"]
            for comp_data in doc_data["components"]:
                comp = FiFComponent(tuple(comp_data["unique_id"]), node_embeddings[comp_data["emb_idx"]])
                comp.original_json_data = comp_data["original_json_data"]
                comp.linked_document_uid_list = [tuple(uid) for uid in comp_data["linked_document_uid_list"]]
                sub_start, _ = comp_data["sub_emb_range"]
                comp.subcomponent_list = [
                    FiFSubComponent(tuple(uid), node_embeddings[sub_start + i])
                    for i, uid in enumerate(comp_data["sub_uid_list"])
                ]
                doc.component_dict[comp_data["unique_id"][1]] = comp
            graph.document_list.append(doc)

        npz_data.close()
        return graph


def _load_doc_summaries(filepath: Path) -> dict[str, str]:
    doc_summary_dict: dict[str, str] = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            entry: JSONDict = json.loads(line)
            doc_summary_dict[entry["doc_title"]] = entry["summary"]
    return doc_summary_dict


def _load_component_json(folderpath: Path) -> dict[UniqueID, JSONDict]:
    component_json_dict: dict[UniqueID, JSONDict] = {}
    json_filepaths = sorted(folderpath.glob("*.json"))
    for json_filepath in tqdm(json_filepaths, desc="Loading JSON File"):
        with open(json_filepath, "r", encoding="utf-8") as f:
            json_data: dict[str, tp.Any] = json.load(f)
            doc_title: str = json_filepath.stem
            for section in ("text", "table", "image"):
                for key, value in json_data[section].items():
                    component_json_dict[(doc_title, key, None)] = value
    
    return component_json_dict
