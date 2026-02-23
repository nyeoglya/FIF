import typing as tp
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from config import BEAM_SIZE
from type import Strategy, UniqueID, SerializedData

@dataclass
class FiFMemoryUnit:
    subquery_list: tp.List[str] = field(default_factory=list) # type: ignore
    subquery_embeddings: NDArray[np.float32] = field(default_factory=lambda: np.array([], dtype=np.float32))
    subquery_index: int = 0
    strategy: Strategy = ("", "", "")
    status: str = "Unresolved"
    answer: str = ""
    failure_reason: str = ""
    component_id_list: tp.List[UniqueID] = field(default_factory=list) # type: ignore

    def get_anchor_document_title_set(self) -> tp.Set[str]:
        anchor_document_set: tp.Set[str] = set()
        for component_id in self.component_id_list:
            anchor_document_set.add(component_id[0])
        return anchor_document_set

class FiFMemory:
    def __init__(self) -> None:
        self.query: str = ""
        self.history: tp.List[FiFMemoryUnit] = []

    def get_last_unit(self) -> FiFMemoryUnit:
        assert self.history
        return self.history[-1]
    
    def reset_memory(self, query: str = "", subquery_list: tp.List[str] = [], subquery_embeddings: NDArray[np.float32] = np.array([])) -> None:
        self.query = query
        self.history = []
        if subquery_list:
            new_memory_unit: FiFMemoryUnit = FiFMemoryUnit(
                subquery_list=subquery_list,
                subquery_embeddings=subquery_embeddings,
                failure_reason="Initial state."
            )
            self.add_new_history(new_memory_unit)

    def add_new_history(self, memory_unit: FiFMemoryUnit):
        self.history.append(memory_unit)

@dataclass
class FiFTraversalContext:
    query_embedding: NDArray[np.float32] = np.array([])
    current_subquery_embeddings: NDArray[np.float32] = np.array([])
    beam_size: int = BEAM_SIZE
    
    component_id_serialized_data_dict: tp.Dict[UniqueID, SerializedData] = field(default_factory=dict, init=False) # type: ignore
    hop_count: int = field(default=0, init=False)
    fif_memory: FiFMemory = field(default_factory=FiFMemory, init=False)
